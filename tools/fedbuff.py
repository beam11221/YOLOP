import argparse
import os, sys, gc, shutil
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
from lib.utils import run_anchor


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

    train_loader = DataLoaderX(
        train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU * len(cfg.GPUS),
        shuffle=(cfg.TRAIN.SHUFFLE and rank == -1),
        num_workers=cfg.WORKERS,
        sampler=train_sampler,
        pin_memory=cfg.PIN_MEMORY,
        collate_fn=dataset.AutoDriveDataset.collate_fn
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

    valid_loader = DataLoaderX(
        valid_dataset,
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU * len(cfg.GPUS),
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY,
        collate_fn=dataset.AutoDriveDataset.collate_fn
    )
    
    return {"train": train_loader, "valid": valid_loader, "valid_dataset": valid_dataset}

def compute_model_delta(global_state_dict, client_state_dict):
    """Compute delta: client - global (only for floating point params)."""
    delta = {}
    for key in global_state_dict:
        if not global_state_dict[key].is_floating_point():
            continue
        
        delta[key] = global_state_dict[key] - client_state_dict[key]

    return delta

def train_client_model_fedbuff(global_model_path, current_version, cfg, client_id,result_queue=None):
    """
    Train a client model for local epochs (FedBuff version)
    Returns: (state_dict on CPU, start_version, end_version)
    """
    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, cfg.LOG_DIR, 'train', rank=int(os.environ['RANK']) if 'RANK' in os.environ else -1)
    logger.info(f"=> Training client {client_id} model...")
    logger.info(f"[INNER]RAM usage before client {client_id} training: {log_memory()} GB")
    global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else -1
    rank = global_rank
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    # Record version when client starts training
    start_version = current_version
    
    # Create a COPY of global model for this client
    global_state = torch.load(global_model_path, map_location=device, weights_only=True)
    global_state = global_state['state_dict']
    client_model = get_net(cfg).to(device)
    client_model.load_state_dict(global_state)

    global_model = get_net(cfg).to("cpu")
    global_model.load_state_dict(global_state)
    # client_model.load_state_dict(global_model.state_dict())
    
    # Create FRESH optimizer
    criterion = get_loss(cfg, device=device)
    optimizer = get_optimizer(cfg, client_model)

    # Learning rate scheduler
    lf = lambda x: ((1 + math.cos(x * math.pi / cfg.TRAIN.END_EPOCH)) / 2) * \
                   (1 - cfg.TRAIN.LRF) + cfg.TRAIN.LRF
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    
    begin_epoch = cfg.TRAIN.BEGIN_EPOCH
    logger.info(f"[INNER]RAM usage before#2 client {client_id} training: {log_memory()} GB")
    # Model configuration
    client_model.gr = 1.0
    client_model.nc = 1

    # Training setup
    data_loader = create_data_generator(client_id, rank) 
    train_loader = data_loader["train"]
    num_batch = len(train_loader)
    num_warmup = max(round(cfg.TRAIN.WARMUP_EPOCHS * num_batch), 0)
    scaler = torch.amp.GradScaler(enabled=device.type != 'cpu')
    
    logger.info(f'=> Client {client_id} starts training from version {start_version}')

    # Train for ALL local epochs
    for local_epoch in range(begin_epoch + 1, cfg.TRAIN.END_EPOCH + 1):
        if rank != -1:
            train_loader.sampler.set_epoch(local_epoch)
        
        # Train for one epoch
        logger.info(f"[INNER]RAM usage before call train function {client_id} training: {log_memory()} GB")
        train(cfg, train_loader, client_model, criterion, optimizer, scaler,
              local_epoch, num_batch, num_warmup, None, logger, device, rank, client_id)
        # train(cfg, train_loader, client_model, criterion, optimizer, scaler,
        #       local_epoch, num_batch, num_warmup, writer_dict, logger, device, rank, client_id)

        
        logger.info(f"[INNER]RAM usage after train function {client_id} training: {log_memory()} GB")

    # Training complete - get end version (will be provided by caller)
    # Move to CPU to save memory
    client_model.to("cpu")
    delta = compute_model_delta(global_model.state_dict(), client_model.state_dict())
    state_dict_path = serialize_state_dict(delta, final_output_dir, current_version, client_id, f"fed_buffer/{client_id}.pt")

    
    logger.info(f"Client {client_id} training completed")
    logger.info(f"[INNER]RAM usage after client {client_id} training: {log_memory()} GB")

    result = {
            "client_id": client_id,
            "global_model_version": start_version,
            "state_dict": state_dict_path
        }
    result_queue.put(result)


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
    global_model = get_net(cfg).to("cpu")
    logger.info(f"Global model initialized")

    # Generate dataloaders for each client
    logger.info(f"Creating data loaders for {len(cfg.FED.CLIENT_IDS)} clients...")
    # data_loaders = {
    #     client_id: create_data_generator(client_id, rank) 
    #     for client_id in cfg.FED.CLIENT_IDS
    # }
    data_loaders = {}
    # data_loaders["global_model"] = create_data_generator("global_model", rank)

    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    # ========== FedBuff Initialization ==========
    buffer_size = cfg.FED.get('BUFFER_SIZE', 10)  # K = 10 based on the paper
    fedbuff_buffer = FedBuffBuffer(buffer_size=buffer_size, device='cpu')
    current_version = 0  # Track global model version
    total_updates = 0    # Track total number of client updates
    
    logger.info(f"FedBuff initialized with buffer size K={buffer_size}")
    # ============================================

    # Federated Learning Loop
    max_updates = cfg.FED.EPOCHS * len(cfg.FED.CLIENT_IDS)  # Total updates to perform for global model
    client_idx = 0  # Round-robin client selection
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Starting FedBuff Training")
    logger.info(f"Total planned updates: {max_updates}")
    logger.info(f"Buffer size K: {buffer_size}")
    logger.info(f"{'='*60}\n")

    for _ in range(max_updates):
        # Select next client (round-robin or random)
        client_id = cfg.FED.CLIENT_IDS[client_idx % len(cfg.FED.CLIENT_IDS)]
        client_idx += 1
        total_updates += 1
        
        logger.info(f"\n--- Training Client {client_id} (Update {total_updates}/{max_updates}) ---")
        logger.info(f"Current global version: {current_version}")
        logger.info(f"Buffer status: {fedbuff_buffer.get_buffer_size()}/{buffer_size}")
        logger.info(f"RAM usage before client {client_id} training: {log_memory()} GB")

        # For simplicity, select the same global_model for all client.
        global_model_path = serialize_state_dict(global_model.state_dict(), 
                                                 final_output_dir, 
                                                 current_version, 
                                                 client_id, 
                                                 f"global_model/version_{current_version}.pt"
                                                 )

        # Create queue for receiving results
        result_queue = mp.Queue()
        process = mp.Process(
            target=train_client_model_fedbuff,
            kwargs={
                "global_model_path": global_model_path,
                "current_version": current_version,
                "cfg": cfg,
                # "logger": logger,
                # "writer_dict": writer_dict,
                "client_id": client_id,
                "result_queue": result_queue
            }
        )
        process.start()
        process.join()

        # Retrieve results
        if not result_queue.empty():
            result = result_queue.get()
            logger.info(f"Received results from client {client_id} training process: {result}")

            # Add update to buffer
            fedbuff_buffer.add_update(
                state_dict_delta=torch.load(global_model_path, map_location="cpu", weights_only=True)["state_dict"],
                client_id=result["client_id"],
                start_version=result["global_model_version"],
                current_version=current_version
            )
            logger.info(f"RAM usage after client {client_id} training: {log_memory()} GB")
        
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
            current_version += 1 # Increment global model version
            logger.info(f"Aggregation complete. Updated global model to version {current_version}")

            fedbuff_buffer.clear()
            # delte buffers directory
            to_del_path = os.path.join(final_output_dir, "weight_cache", "fed_buffer")
            if os.path.exists(to_del_path):
                shutil.rmtree(to_del_path)

            del buffered_updates
            gc.collect()
            torch.cuda.empty_cache()
            logger.info(f"RAM usage after cleanup the buffers: {log_memory()} GB")

            
            
if __name__ == '__main__':
    main()