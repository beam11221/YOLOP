import argparse
import os, sys, gc, shutil, time
import math
import multiprocessing as mp
mp.set_start_method('spawn', force=True) # To. ensure CUDA works in subprocesses
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
import numpy as np
import torch
import torch.nn.parallel
# from torch.cuda import amp
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import random

from tensorboardX import SummaryWriter

from lib.utils import DataLoaderX, torch_distributed_zero_first, log_memory
import lib.dataset as dataset
from lib.config import cfg
from lib.config import update_config
from lib.core.loss import get_loss
from lib.core.function import train
from lib.core.function import validate
from lib.core.general import fitness
from lib.core.fed_buffer import FedBuffBuffer, fedbuff_aggregate
# from lib.core.fed_buffer_disk import FedBuffBuffer, fedbuff_aggregate_streaming
from lib.models import get_net
from lib.utils import is_parallel
from lib.utils.utils import get_optimizer
from lib.utils.utils import save_checkpoint
from lib.utils.utils import create_logger, select_device
from lib.utils import run_anchor, kmean_anchors

class ClientsNote():
    def __init__(self, n_clients: int = 15, global_model=None, final_output_dir: str=None):
        self.n_clients = n_clients
        self.final_output_dir = final_output_dir

        self.client_models = {
            f"client_{i}": {
                "current_global_version": 0
            } for i in range(n_clients)
        }
        self.global_models = {
            "current_version": 0,
            "weight_paths": {
                0: serialize_state_dict(
                    state_dict=global_model,
                    save_root=self.final_output_dir,
                    global_model_version=0,
                    client_id="global_model",
                    saved_name=f"global_model/version_0.pt"
                )
            }
        }
        self.n_updates = 0
        self.max_updates = n_clients * cfg.FED.EPOCHS

    def append_global_model(self, state_dict: str):
        new_version = self.global_models["current_version"] + 1
        self.global_models["current_version"] = new_version

        weight_path = serialize_state_dict(
                    state_dict=state_dict,
                    save_root=self.final_output_dir,
                    global_model_version=new_version,
                    client_id="global_model",
                    saved_name=f"global_model/version_{new_version}.pt"
                )
        self.global_models["weight_paths"][new_version] = weight_path

    def update_global_model(self, new_version: int, weight_path: str):
        self.global_models["current_version"] = new_version
        self.global_models["weight_paths"][new_version] = weight_path

    def get_current_global_model_version(self):
        return self.global_models["current_version"]
    
    def get_global_model_path(self, version: int):
        if version > self.global_models["current_version"]:
            raise f"Requested version {version} exceeds current global model version {self.global_models['current_version']}"

        return self.global_models["weight_paths"].get(version, None)

    def get_global_path_for_model(self, client_id: str):
        client_info = self.client_models.get(client_id, None)
        if client_info is None:
            raise f"Client ID {client_id} not found in ClientsNote."

        version = client_info["current_global_version"]
        return self.get_global_model_path(version)
    
    def get_global_version_for_client(self, client_id: str):
        client_info = self.client_models.get(client_id, None)
        if client_info is None:
            raise f"Client ID {client_id} not found in ClientsNote."

        return client_info["current_global_version"]

    def sync_clients(self, client_ids: list):
        """Update clients' current global model version after aggregation."""
        current_version = self.get_current_global_model_version()
        for client_id in client_ids:
            if client_id in self.client_models:
                self.client_models[client_id]["current_global_version"] = current_version
            else:
                raise f"Client ID {client_id} not found in ClientsNote."


def parse_args():
    parser = argparse.ArgumentParser(description='Train Multitask network with FedAvg')
    parser.add_argument('--modelDir', help='model directory', type=str, default='')
    parser.add_argument('--logDir', help='log directory', type=str, default='runs/')
    parser.add_argument('--dataDir', help='data directory', type=str, default='')
    parser.add_argument('--prevModelDir', help='prev Model directory', type=str, default='')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='IOU threshold for NMS')
    args = parser.parse_args()
    return args

def evaluate(global_model, current_version, cfg, data_loaders, final_output_dir, tb_log_dir, writer_dict, logger, device, rank):
    logger.info(f"Starting evaluation at version {current_version}...")
    global_model.to(device)
    criterion = get_loss(cfg, device=device)
    
    da_segment_results, ll_segment_results, detect_results, total_loss, maps, times = validate(
        current_version, cfg, data_loaders["global_model"]["valid"], 
        data_loaders["global_model"]["valid_dataset"], global_model, criterion,
        final_output_dir, tb_log_dir, writer_dict, logger, device, rank
    )
    
    # Log metrics
    msg = 'Version: [{0}]    Loss({loss:.3f})\n' \
            'Driving area Segment: Acc({da_seg_acc:.3f})    IOU ({da_seg_iou:.3f})    mIOU({da_seg_miou:.3f})\n' \
            'Lane line Segment: Acc({ll_seg_acc:.3f})    IOU ({ll_seg_iou:.3f})  mIOU({ll_seg_miou:.3f})\n' \
            'Detect: P({p:.3f})  R({r:.3f})  mAP@0.5({map50:.3f})  mAP@0.5:0.95({map:.3f})\n'\
            'Time: inference({t_inf:.4f}s/frame)  nms({t_nms:.4f}s/frame)'.format(
                current_version, loss=total_loss, 
                da_seg_acc=da_segment_results[0], da_seg_iou=da_segment_results[1], da_seg_miou=da_segment_results[2],
                ll_seg_acc=ll_segment_results[0], ll_seg_iou=ll_segment_results[1], ll_seg_miou=ll_segment_results[2],
                p=detect_results[0], r=detect_results[1], map50=detect_results[2], map=detect_results[3],
                t_inf=times[0], t_nms=times[1])
    logger.info(msg)
    
    # Write to TensorBoard
    # ==================== ADDED: Write validation metrics to TensorBoard ====================
    writer = writer_dict['writer']
    global_steps = writer_dict['global_model_version']
    
    # Write validation loss
    writer.add_scalar('global_model/total_loss', total_loss, global_steps)
    
    # Write driving area segmentation metrics
    writer.add_scalar('global_model/da_seg_acc', da_segment_results[0], global_steps)
    writer.add_scalar('global_model/da_seg_iou', da_segment_results[1], global_steps)
    writer.add_scalar('global_model/da_seg_miou', da_segment_results[2], global_steps)
    
    # Write lane line segmentation metrics
    writer.add_scalar('global_model/ll_seg_acc', ll_segment_results[0], global_steps)
    writer.add_scalar('global_model/ll_seg_iou', ll_segment_results[1], global_steps)
    writer.add_scalar('global_model/ll_seg_miou', ll_segment_results[2], global_steps)
    
    # Write detection metrics
    writer.add_scalar('global_model/detect_precision', detect_results[0], global_steps)
    writer.add_scalar('global_model/detect_recall', detect_results[1], global_steps)
    writer.add_scalar('global_model/detect_mAP@0.5', detect_results[2], global_steps)
    writer.add_scalar('global_model/detect_mAP@0.5:0.95', detect_results[3], global_steps)
    
    # Update global steps counter for validation
    writer_dict['global_model_version'] = current_version
    # ==================== END OF ADDED CODE ====================
    
    global_model.to("cpu")
    del criterion
    torch.cuda.empty_cache()


def serialize_state_dict(state_dict: dict, save_root: str, global_model_version: str, client_id: str, saved_name: str) -> str:
    """Serialize to bytes - creates a complete copy."""
    state_dict_output_path = os.path.join(save_root, "weight_cache", saved_name)
    os.makedirs(os.path.dirname(state_dict_output_path), exist_ok=True)

    torch.save({
        'state_dict': state_dict,
        'global_model_version': global_model_version,
        "client_id": client_id
    }, state_dict_output_path)

    return state_dict_output_path

def create_data_generator(client_id, rank):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
    
    train_dataset = eval('dataset.' + cfg.DATASET.DATASET)(
        cfg=cfg,
        is_train=True,
        inputsize=cfg.MODEL.IMAGE_SIZE,
        transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]),
        client_id=client_id
    )
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if rank != -1 else None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU * len(cfg.GPUS),
        shuffle=(cfg.TRAIN.SHUFFLE and rank == -1),
        num_workers=cfg.WORKERS,
        sampler=train_sampler,
        pin_memory=cfg.PIN_MEMORY,
        collate_fn=dataset.AutoDriveDataset.collate_fn,
        multiprocessing_context="fork",
        prefetch_factor=2
    )

    valid_dataset = eval('dataset.' + cfg.DATASET.DATASET)(
        cfg=cfg,
        is_train=False,
        inputsize=cfg.MODEL.IMAGE_SIZE,
        transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]),
        client_id=client_id
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU * len(cfg.GPUS),
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY,
        collate_fn=dataset.AutoDriveDataset.collate_fn,
        multiprocessing_context="fork",
        prefetch_factor=2
    )
    
    return {
        "train": train_loader, 
        "valid": valid_loader, 
        "valid_dataset": valid_dataset
        }

def compute_model_delta(global_state_dict, client_state_dict):
    """Compute delta: client - global (only for floating point params)."""
    delta = {}
    for key in global_state_dict:
        if not global_state_dict[key].is_floating_point():
            continue
        
        delta[key] = global_state_dict[key] - client_state_dict[key]

    return delta

def train_client_model_fedbuff(global_model_path, current_version, cfg, client_id, result_queue=None):
    """
    Train a client model for local epochs (FedBuff version)
    Returns: (state_dict on CPU, start_version, end_version)
    """
    try:
        logger, final_output_dir, tb_log_dir = create_logger(
            cfg, cfg.LOG_DIR, 'train', rank=int(os.environ['RANK']) if 'RANK' in os.environ else -1)
        logger.info(f"=> Training client {client_id} model...")
        global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else -1
        rank = global_rank
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        data_loader = create_data_generator(client_id, rank) 

        # Record version when client starts training
        start_version = current_version
        
        # Create a COPY of global model for this client
        global_state = torch.load(global_model_path, map_location=device, weights_only=True)
        global_state = global_state['state_dict']
        client_model = get_net(cfg).to(device)
        client_model.load_state_dict(global_state)

        global_model = get_net(cfg).to("cpu")
        global_model.load_state_dict(global_state)
        
        # Create FRESH optimizer
        criterion = get_loss(cfg, device=device)
        optimizer = get_optimizer(cfg, client_model)

        # Learning rate scheduler
        lf = lambda x: ((1 + math.cos(x * math.pi / cfg.TRAIN.END_EPOCH)) / 2) * \
                    (1 - cfg.TRAIN.LRF) + cfg.TRAIN.LRF
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
        
        begin_epoch = cfg.TRAIN.BEGIN_EPOCH
        # Model configuration
        client_model.gr = 1.0
        client_model.nc = 1

        det = client_model.model[client_model.detector_index]
        logger.info(f"Client {client_id} anchors (should match global):\n{det.anchors}")

        # Training setup
        train_loader = data_loader["train"]
        num_batch = len(train_loader)
        num_warmup = max(round(cfg.TRAIN.WARMUP_EPOCHS * num_batch), 0)
        scaler = torch.amp.GradScaler(enabled=device.type != 'cpu')
        
        logger.info(f'=> Client {client_id} starts training from version {start_version}')
        training_stats = {}
        # Train for ALL local epochs
        for local_epoch in range(begin_epoch + 1, cfg.TRAIN.END_EPOCH + 1):
            if rank != -1:
                train_loader.sampler.set_epoch(local_epoch)
            
            # Train for one epoch
            metric = train(cfg, train_loader, client_model, criterion, optimizer, scaler,
                local_epoch, num_batch, num_warmup, None, logger, device, rank, client_id)
            training_stats[local_epoch] = metric

        # Training complete - get end version (will be provided by caller)
        # Move to CPU to save memory
        client_model.to("cpu")
        delta = compute_model_delta(global_model.state_dict(), client_model.state_dict())
        state_dict_path = serialize_state_dict(delta, final_output_dir, current_version, client_id, f"fed_buffer/{client_id}.pt")

        result = {
                "client_id": client_id,
                "global_model_version": start_version,
                "state_dict": state_dict_path,
                "training_stats": training_stats
            }
        result_queue.put(result)

    finally:
        # 1. Delete model and move GPU memory
        if client_model is not None:
            del client_model
        torch.cuda.empty_cache()
        
        # 2. Shutdown DataLoader workers explicitly
        if train_loader is not None:
            # If using PyTorch's _MultiProcessingDataLoaderIter
            if hasattr(train_loader, '_iterator') and train_loader._iterator is not None:
                try:
                    train_loader._iterator._shutdown_workers()
                except:
                    pass
            del train_loader
            
        if data_loader is not None:
            del data_loader
        
        # 3. Force garbage collection
        gc.collect()
        
        # 4. Brief pause for worker cleanup
        time.sleep(0.1)


def main():
    #### Pre task ####
    args = parse_args()
    update_config(cfg, args)

    global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else -1
    rank = global_rank
    
    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, cfg.LOG_DIR, 'train', rank=rank)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training device: {device}")
    #### End of Pre-task ####

    # Initialize global model on GPU
    data_loaders = dict()
    data_loaders["global_model"] = create_data_generator("global_model", rank)
    global_model = get_net(cfg).to(device)

    #  Model Configuration
    global_model.gr = 1.0  # gradient ratio for loss scaling
    global_model.nc = 1    # number of classes (vehicles only in BDD100K)
    
    # Anchor Configuration
    if cfg.NEED_AUTOANCHOR:
        logger.info("Begin anchor optimization using k-means on training data...")
        # Use training dataset for anchor computation (more representative)
        train_dataset_for_anchors = data_loaders["global_model"]["valid_dataset"]
        run_anchor(logger, train_dataset_for_anchors, model=global_model, 
                   thr=cfg.TRAIN.ANCHOR_THRESHOLD, imgsz=min(cfg.MODEL.IMAGE_SIZE))
    else:
        logger.info("Using default anchors from model config")
        det = global_model.model[global_model.detector_index]
        logger.info(f"Current anchors:\n{det.anchors}")

    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
        'global_model_version': 0
    }

    # ========== FedBuff Initialization ==========
    buffer_size = cfg.FED.get('BUFFER_SIZE', 10)  # K = 10 based on the paper
    fedbuff_buffer = FedBuffBuffer(buffer_size=buffer_size, device='cpu')
    
    logger.info(f"FedBuff initialized with buffer size K={buffer_size}")
    # ============================================

    # Federated Learning Loop
    notes = ClientsNote(
        n_clients=len(cfg.FED.CLIENT_IDS), 
        global_model=global_model.state_dict(),
        final_output_dir=final_output_dir
        )
    
    selected_clients = []
    for fed_round in range(notes.max_updates):
        if not selected_clients:
            selected_clients = random.sample(cfg.FED.CLIENT_IDS, buffer_size)
            logger.info(f"Client selection order for this run: {selected_clients}")

        # Random client selection
        client_id = selected_clients.pop()
        notes.n_updates += 1
        
        logger.info(f"\n--- Training Client {client_id} (Update {notes.n_updates}/{notes.max_updates}) ---")
        logger.info(f"Current global version: {notes.get_current_global_model_version()} | clients's last global version: {notes.client_models[client_id]['current_global_version']}")
        logger.info(f"Buffer status: {fedbuff_buffer.get_buffer_size()}/{buffer_size}")

        # Create queue for receiving results
        result_queue = mp.Queue()
        process = mp.Process(
            target=train_client_model_fedbuff,
            kwargs={
                "global_model_path": notes.get_global_path_for_model(client_id),
                "current_version": notes.get_current_global_model_version(),
                "cfg": cfg,
                "client_id": client_id,
                "result_queue": result_queue
            }
        )
        process.start()
        process.join()

        # Retrieve results, if not empty means finished training
        if not result_queue.empty():
            result = result_queue.get()

            client_delta = torch.load(result["state_dict"], map_location="cpu", weights_only=True)["state_dict"]
            # Add update to buffer
            fedbuff_buffer.add_update(
                state_dict_delta=client_delta,
                client_id=result["client_id"],
                start_version=notes.get_global_version_for_client(result["client_id"]),
                current_version=notes.get_current_global_model_version()
            )
            # Update to TensorBoard
            for local_epoch, stats in result["training_stats"].items():
                writer_dict['writer'].add_scalar(
                    f"clients/{client_id}/{fed_round}/train_loss", 
                    stats["loss_avg"], 
                    local_epoch
                )
        
        if fedbuff_buffer.is_full():
            logger.info(f"\n{'='*60}")
            logger.info(f"Buffer full! Performing staleness-aware aggregation...")
            
            # Get all buffered updates
            buffered_updates = fedbuff_buffer.get_updates()
            
            # Log staleness information
            staleness_values = [u['staleness'] for u in buffered_updates]
            logger.info(f"Staleness values: {staleness_values}")
            logger.info(f"Mean staleness: {np.mean(staleness_values):.2f}")
            logger.info(f"Max staleness: {np.max(staleness_values)}")

            # Perform FedBuff aggregation
            global_model = fedbuff_aggregate(global_model, buffered_updates)

            # Append new global model to notes
            notes.append_global_model(global_model.state_dict())
            notes.sync_clients(fedbuff_buffer.get_clients_in_buffer())
            logger.info(f"Aggregation completed. Updated global model to version {notes.get_current_global_model_version()}")
            logger.info(f"Clients current version: {notes.client_models}")


            fedbuff_buffer.clear()
            # delte buffers directory
            to_del_path = os.path.join(final_output_dir, "weight_cache", "fed_buffer")
            if os.path.exists(to_del_path):
                shutil.rmtree(to_del_path)

            del buffered_updates
            gc.collect()
            torch.cuda.empty_cache()

            evaluate(global_model, notes.get_current_global_model_version(), cfg, data_loaders, final_output_dir, tb_log_dir, writer_dict, logger, device, rank)
            
        logger.info(f"At fed_round: {fed_round} | {log_memory()} GB")
            
if __name__ == '__main__':
    random.seed(42)
    main()