import argparse
import os, sys, gc
import math
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
import numpy as np
import torch
import torch.nn.parallel
from torch.cuda import amp
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

def train_client_model_fedbuff(global_model, current_version, data_loader, cfg, 
                                 logger, writer_dict, device, client_id):
    """
    Train a client model for local epochs (FedBuff version)
    Returns: (state_dict on CPU, start_version, end_version)
    """
    global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else -1
    rank = global_rank
    
    # Record version when client starts training
    start_version = current_version
    
    # Create a COPY of global model for this client
    client_model = get_net(cfg).to(device)
    client_model.load_state_dict(global_model.state_dict())
    
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

    # Training setup
    train_loader = data_loader["train"]
    num_batch = len(train_loader)
    num_warmup = max(round(cfg.TRAIN.WARMUP_EPOCHS * num_batch), 0)
    scaler = amp.GradScaler(enabled=device.type != 'cpu')
    
    logger.info(f'=> Client {client_id} starts training from version {start_version}')

    # Train for ALL local epochs
    for local_epoch in range(begin_epoch + 1, cfg.TRAIN.END_EPOCH + 1):
        if rank != -1:
            train_loader.sampler.set_epoch(local_epoch)
        
        # Train for one epoch
        train(cfg, train_loader, client_model, criterion, optimizer, scaler,
              local_epoch, num_batch, num_warmup, writer_dict, logger, device, rank, client_id)
        
        lr_scheduler.step()
    
    # Training complete - get end version (will be provided by caller)
    # Move to CPU to save memory
    client_model.to("cpu")
    # state_dict = client_model.state_dict()
    delta = compute_model_delta(global_model.state_dict(), client_model.state_dict())
    
    # Cleanup
    del client_model, optimizer, criterion
    torch.cuda.empty_cache()
    
    logger.info(f"Client {client_id} training complete")
    
    return delta, start_version


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
    # Do not create all data loaders at once to save memory
    data_loaders = {}
    data_loaders["global_model"] = create_data_generator("global_model", rank)

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
    
    while total_updates < max_updates:
        # Select next client (round-robin or random)
        client_id = cfg.FED.CLIENT_IDS[client_idx % len(cfg.FED.CLIENT_IDS)]
        client_idx += 1
        total_updates += 1
        
        logger.info(f"\n--- Training Client {client_id} (Update {total_updates}/{max_updates}) ---")
        logger.info(f"Current global version: {current_version}")
        logger.info(f"Buffer status: {fedbuff_buffer.get_buffer_size()}/{buffer_size}")
        logger.info(f"RAM usage before client {client_id} training: {log_memory()} GB")

        
        data_loaders[client_id] = create_data_generator(client_id, rank)

        # Train client and get update
        delta, start_version = train_client_model_fedbuff(
            global_model, current_version, data_loaders[client_id],
            cfg, logger, writer_dict, device, client_id
        )
        
        # Add update to buffer
        fedbuff_buffer.add_update(
            state_dict_delta=delta,
            client_id=client_id,
            start_version=start_version,
            end_version=current_version  # Version hasn't changed yet
        )
        
        logger.info(f"Client {client_id} update added to buffer (staleness: {current_version - start_version})")
        logger.info(f"Buffer: {fedbuff_buffer.get_buffer_size()}/{buffer_size} updates")
        logger.info(f"RAM usage after client {client_id} training: {log_memory()} GB")
        
        # Cleanup client data loader to save memory
        del data_loaders[client_id]
        gc.collect()
        torch.cuda.empty_cache()

        logger.info(f"RAM usage after cleanup: {log_memory()} GB")

        # Check if buffer is full - time to aggregate
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
            
            # Increment version after aggregation
            current_version += 1
            
            logger.info(f"Aggregation complete! New global version: {current_version}")
            logger.info(f"{'='*60}\n")
            
            # Clear buffer
            fedbuff_buffer.clear()
            
            # Save checkpoint every aggregation
            save_path = os.path.join(final_output_dir, f'global_model_version_{current_version}.pth')
            torch.save(global_model.state_dict(), save_path)
            logger.info(f"Checkpoint saved: {save_path}")
            
            # Evaluate model
            if current_version % cfg.FED.get('EVAL_FREQUENCY', 5) == 0:
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
                global_steps = writer_dict['valid_global_steps']
                
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
                writer_dict['valid_global_steps'] = global_steps + 1
                # ==================== END OF ADDED CODE ====================
                
                global_model.to("cpu")
                del criterion
                torch.cuda.empty_cache()
    
    # Save final model
    final_path = os.path.join(final_output_dir, 'final_fedbuff_model.pth')
    torch.save(global_model.state_dict(), final_path)
    logger.info(f"FedBuff training complete! Final model: {final_path}")
    logger.info(f"Total versions: {current_version}")
    logger.info(f"Total client updates: {total_updates}")


if __name__ == '__main__':
    main()