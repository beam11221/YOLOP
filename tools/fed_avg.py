import argparse
import os, sys
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

from lib.utils import DataLoaderX, torch_distributed_zero_first
import lib.dataset as dataset
from lib.config import cfg
from lib.config import update_config
from lib.core.loss import get_loss
from lib.core.function import train
from lib.core.function import validate
from lib.core.general import fitness
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


def fedavg(global_model, client_state_dicts):
    """
    Average the model parameters from all clients (CPU aggregation)
    
    Args:
        global_model: Global model (on GPU)
        client_state_dicts: List of state_dicts (on CPU)
    """
    # Move global model's state_dict to CPU
    global_dict = {k: v.cpu() for k, v in global_model.state_dict().items()}
    
    # Average all client parameters on CPU
    for key in global_dict.keys():
        global_dict[key] = torch.stack(
            [client_state_dicts[i][key].float() for i in range(len(client_state_dicts))], 
            dim=0
        ).mean(dim=0)
    
    # Load averaged parameters back to global model
    global_model.load_state_dict(global_dict)
    
    # Model configuration
    global_model.gr = 1.0
    global_model.nc = 1
    
    return global_model


def train_client_model(global_model, fed_round, data_loader, cfg, logger, writer_dict, device, client_id):
    """
    Train a client model for local epochs
    Returns state_dict on CPU to save GPU memory
    """
    global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else -1
    rank = global_rank
    
    # ✅ Create a COPY of global model for this client
    client_model = get_net(cfg).to(device)
    client_model.load_state_dict(global_model.state_dict())
    
    # ✅ Create FRESH optimizer for this federated round
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

    if rank in [-1, 0]:
        det = client_model.module.model[client_model.module.detector_index] if is_parallel(client_model) \
            else client_model.model[client_model.detector_index]

    # Training setup
    train_loader = data_loader["train"]
    num_batch = len(train_loader)
    # num_warmup = max(round(cfg.TRAIN.WARMUP_EPOCHS * num_batch), 1000)
    num_warmup = max(round(cfg.TRAIN.WARMUP_EPOCHS * num_batch), 5)
    print("SELECT WARMUP: ", num_warmup, num_batch)
    scaler = amp.GradScaler(enabled=device.type != 'cpu')
    
    logger.info(f'=> Start local training for {cfg.TRAIN.END_EPOCH - begin_epoch} epochs...')

    # ✅ Train for ALL local epochs
    for local_epoch in range(begin_epoch + 1, cfg.TRAIN.END_EPOCH + 1):
        if rank != -1:
            train_loader.sampler.set_epoch(local_epoch)
        
        # Train for one epoch
        train(cfg, train_loader, client_model, criterion, optimizer, scaler,
              local_epoch, num_batch, num_warmup, writer_dict, logger, device, rank, client_id)
        
        lr_scheduler.step()
    
    # ✅ AFTER training loop completes, move to CPU
    client_model.to("cpu")
    state_dict = client_model.state_dict()
    
    # Cleanup GPU memory
    del client_model, optimizer, criterion
    torch.cuda.empty_cache()
    
    logger.info(f"Local training complete. Model moved to CPU, GPU memory freed.")
    
    return state_dict


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
    global_model = get_net(cfg).to(device)
    logger.info(f"Global model initialized")

    # Generate dataloaders for each client
    logger.info(f"Creating data loaders for {len(cfg.FED.CLIENT_IDS)} clients...")
    data_loaders = {
        client_id: create_data_generator(client_id, rank) 
        for client_id in cfg.FED.CLIENT_IDS
    }
    data_loaders["global_model"] = create_data_generator("global_model", rank)

    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, cfg.LOG_DIR, 'train', rank=rank)

    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    # Federated Learning Loop
    for fed_round in range(1, cfg.FED.EPOCHS + 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Federated Learning Round {fed_round}/{cfg.FED.EPOCHS}")
        logger.info(f"{'='*60}")
        
        # ✅ Store state_dicts on CPU
        client_state_dicts = []
        
        # Train each client sequentially
        for idx, client_id in enumerate(cfg.FED.CLIENT_IDS):
            logger.info(f"\n--- Training Client {client_id} ({idx+1}/{len(cfg.FED.CLIENT_IDS)}) ---")
            
            # Train and get state_dict (on CPU)
            state_dict = train_client_model(
                global_model, fed_round, data_loaders[client_id], 
                cfg, logger, writer_dict, device, client_id
            )
            criterion = get_loss(cfg, device=device)
            
            client_state_dicts.append(state_dict)
            torch.cuda.empty_cache()
            
            logger.info(f"Client {client_id} complete. State saved in CPU RAM.")
        
        # ✅ Aggregate on CPU
        logger.info(f"\n>>> Aggregating {len(client_state_dicts)} client models...")
        global_model = fedavg(global_model, client_state_dicts)
        logger.info(f">>> Aggregation complete. Global model updated.")
        
        # Free memory
        del client_state_dicts
        torch.cuda.empty_cache()
        
        # Save checkpoint
        save_path = os.path.join(final_output_dir, f'global_model_round_{fed_round}.pth')
        torch.save(global_model.state_dict(), save_path)
        logger.info(f"Checkpoint saved: {save_path}")
        
        logger.info(f"Round {fed_round} complete!\n")

        # Evaluate model on valid dataset
        logger.info(f"Start evaluation...")
        global_model.to("cuda")
        da_segment_results,ll_segment_results,detect_results, total_loss,maps, times = validate(
                fed_round,cfg, data_loaders["global_model"]["valid"], data_loaders["global_model"]["valid_dataset"], global_model, criterion,
                final_output_dir, tb_log_dir, writer_dict,
                logger, device, rank
            )
        fi = fitness(np.array(detect_results).reshape(1, -1))

        msg = 'Epoch: [{0}]    Loss({loss:.3f})\n' \
                    'Driving area Segment: Acc({da_seg_acc:.3f})    IOU ({da_seg_iou:.3f})    mIOU({da_seg_miou:.3f})\n' \
                    'Lane line Segment: Acc({ll_seg_acc:.3f})    IOU ({ll_seg_iou:.3f})  mIOU({ll_seg_miou:.3f})\n' \
                    'Detect: P({p:.3f})  R({r:.3f})  mAP@0.5({map50:.3f})  mAP@0.5:0.95({map:.3f})\n'\
                    'Time: inference({t_inf:.4f}s/frame)  nms({t_nms:.4f}s/frame)'.format(
                        fed_round,  loss=total_loss, da_seg_acc=da_segment_results[0],da_seg_iou=da_segment_results[1],da_seg_miou=da_segment_results[2],
                        ll_seg_acc=ll_segment_results[0],ll_seg_iou=ll_segment_results[1],ll_seg_miou=ll_segment_results[2],
                        p=detect_results[0],r=detect_results[1],map50=detect_results[2],map=detect_results[3],
                        t_inf=times[0], t_nms=times[1])
        logger.info(msg)

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
        
        # Write fitness score
        writer.add_scalar('global_model/fitness', fi, global_steps)
        
        # Write inference times
        writer.add_scalar('global_model/inference_time', times[0], global_steps)
        writer.add_scalar('global_model/nms_time', times[1], global_steps)
        
        # Update global steps counter for validation
        writer_dict['valid_global_steps'] = global_steps + 1
        # ==================== END OF ADDED CODE ====================

        global_model.to("cpu")
        del criterion
    
    # Save final model
    final_path = os.path.join(final_output_dir, 'final_global_model.pth')
    torch.save(global_model.state_dict(), final_path)
    logger.info(f"Training complete! Final model: {final_path}")


if __name__ == '__main__':
    main()