"""
Minimal Example: Subprocess-Based Sequential Training in PyTorch
================================================================

This demonstrates how to train multiple models sequentially while
ensuring complete memory release between training sessions.

Key Concepts:
1. Each training runs in a separate subprocess
2. When subprocess exits, ALL memory (GPU + CPU) is reclaimed by OS
3. Results are passed back via multiprocessing primitives

Author: Example for Federated Learning Research
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import multiprocessing as mp
import os
import time
import psutil

import io

def log_memory():
    process = psutil.Process(os.getpid())
    mem_gb = process.memory_info().rss / 1024**3

    # Count child processes (DataLoader workers)
    children = process.children(recursive=True)
    return "RAM: {} GB | Child processes: {}".format(mem_gb, len(children))

def serialize_state_dict(state_dict: dict, task_id: str) -> bytes:
    """Serialize to bytes - creates a complete copy."""
    delta_output_path = f"state_dict_task_{task_id}.pt"
    torch.save({
        'state_dict': state_dict,
        'task_id': task_id
    }, delta_output_path)

    return delta_output_path

def deserialize_state_dict(data: bytes) -> dict:
    """Deserialize bytes back to state_dict."""
    buffer = io.BytesIO(data)
    return torch.load(buffer, map_location='cpu', weights_only=True)

# =============================================================================
# Simple Models (representing different client models)
# =============================================================================

class SmallCNN(nn.Module):
    """A small CNN for demonstration."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class LargerCNN(nn.Module):
    """A larger CNN to show different memory footprints."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# =============================================================================
# Training Function (runs inside subprocess)
# =============================================================================

def train_in_subprocess(
    model_class: type,
    model_kwargs: dict,
    dataset_size: int,
    num_epochs: int,
    batch_size: int,
    result_queue: mp.Queue,
    task_id: int,
):
    """
    Training function that runs in a separate subprocess.
    
    Args:
        model_class: The class of model to instantiate
        model_kwargs: Arguments to pass to model constructor
        dataset_size: Number of samples in synthetic dataset
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        result_queue: Queue to send results back to main process
        task_id: Identifier for this training task
    """
    try:
        print(f"\n[Task {task_id}] Subprocess started (PID: {os.getpid()})")
        
        # Setup device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Task {task_id}] Using device: {device}")
        
        # Create synthetic dataset (simulating different client data)
        # In real FL, you'd load actual client data here
        X = torch.randn(dataset_size, 3, 32, 32)
        y = torch.randint(0, 10, (dataset_size,))
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=0,  # Important: avoid nested workers in subprocess
            pin_memory=False,
        )
        
        # Create model
        model = model_class(**model_kwargs).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        
        # Print memory before training
        if device.type == "cuda":
            allocated = torch.cuda.memory_allocated() / 1024**2
            print(f"[Task {task_id}] GPU memory allocated: {allocated:.1f} MB")
        
        # Training loop
        model.train()
        for epoch in range(num_epochs):
            total_loss = 0.0
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            print(f"[Task {task_id}] Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
        
        # Move model to CPU before sending back
        model.cpu()
        state_dict = model.state_dict()
        
        # Clear GPU memory before exit
        if device.type == "cuda":
            torch.cuda.empty_cache()
        
        # Send results back
        state_dict_path = serialize_state_dict(model.state_dict(), task_id)
        result = {
            "task_id": task_id,
            "final_loss": avg_loss,
            "state_dict": state_dict_path,
            "success": True,
        }
        result_queue.put(result)

        
    except Exception as e:
        result_queue.put({
            "task_id": task_id,
            "success": False,
            "error": str(e),
        })
        print(f"[Task {task_id}] Error: {e}")


# =============================================================================
# Main Orchestrator
# =============================================================================

def run_sequential_training():
    """
    Main function that orchestrates sequential training tasks.
    Each task runs in a subprocess for complete memory isolation.
    """
    print("=" * 60)
    print("Subprocess-Based Sequential Training Demo")
    print("=" * 60)
    
    # IMPORTANT: Use 'spawn' for CUDA compatibility
    # 'fork' can cause issues with CUDA contexts
    mp.set_start_method('spawn', force=True)
    
    # Define training tasks (simulating different FL clients)
    tasks = [
        {
            "model_class": SmallCNN,
            "model_kwargs": {"num_classes": 10},
            "dataset_size": 1000,
            "num_epochs": 3,
            "batch_size": 32,
        },
        {
            "model_class": LargerCNN,
            "model_kwargs": {"num_classes": 10},
            "dataset_size": 2000,
            "num_epochs": 3,
            "batch_size": 64,
        },
        {
            "model_class": SmallCNN,
            "model_kwargs": {"num_classes": 10},
            "dataset_size": 1500,
            "num_epochs": 3,
            "batch_size": 32,
        },
        {
            "model_class": LargerCNN,
            "model_kwargs": {"num_classes": 10},
            "dataset_size": 2000,
            "num_epochs": 3,
            "batch_size": 64,
        },
        {
            "model_class": SmallCNN,
            "model_kwargs": {"num_classes": 10},
            "dataset_size": 1500,
            "num_epochs": 3,
            "batch_size": 32,
        },
        {
            "model_class": LargerCNN,
            "model_kwargs": {"num_classes": 10},
            "dataset_size": 2000,
            "num_epochs": 3,
            "batch_size": 64,
        },
        {
            "model_class": SmallCNN,
            "model_kwargs": {"num_classes": 10},
            "dataset_size": 1500,
            "num_epochs": 3,
            "batch_size": 32,
        },
    ]
    
    results = []
    
    for task_id, task_config in enumerate(tasks):
        print(f"\n{'='*60}")
        print(f"Starting Task {task_id}: {task_config['model_class'].__name__}")
        print(f"Dataset size: {task_config['dataset_size']}")
        print(f"Before starting subprocess, memory status: {log_memory()}")
        print(f"{'='*60}")
        
        # Create queue for receiving results
        result_queue = mp.Queue()
        
        # Create and start subprocess
        process = mp.Process(
            target=train_in_subprocess,
            kwargs={
                **task_config,
                "result_queue": result_queue,
                "task_id": task_id,
            }
        )
        
        process.start()
        print(f"[Main] Subprocess {process.pid} started")
        
        # Wait for subprocess to complete
        process.join()
        print(f"[Main] Subprocess {process.pid} finished with exit code {process.exitcode}")
        
        # Retrieve results
        if not result_queue.empty():
            result = result_queue.get()
            results.append(result)
            
            if result["success"]:
                print(f"[Main] Task {task_id} succeeded, final loss: {result['final_loss']:.4f}")
                # Here you could aggregate the state_dict into global model
                # global_model.load_state_dict(result['state_dict'])
            else:
                print(f"[Main] Task {task_id} failed: {result['error']}")
        
        # At this point, ALL memory from the subprocess is released!
        print(f"[Main] Memory fully released after Task {task_id}")
        print(f"[Main] After finish training. Current memory status: {log_memory()}")
        time.sleep(2)  # Small delay for demonstration
    
    # Summary
    print("\n" + "=" * 60)
    print("Training Summary")
    print("=" * 60)
    for r in results:
        if r["success"]:
            print(f"Task {r['task_id']}: Loss = {r['final_loss']:.4f}")
        else:
            print(f"Task {r['task_id']}: FAILED - {r['error']}")
    
    del results
    print("\nAll subprocesses completed. Exiting main process.")
    print(f"[Main] Final memory status: {log_memory()}")


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    print("Starting Sequential Training with Subprocesses")
    print(f"[Main] Initial memory status: {log_memory()}")
    # This guard is REQUIRED for multiprocessing with 'spawn'
    run_sequential_training()
    print(f"[Main] Final memory status: {log_memory()}")