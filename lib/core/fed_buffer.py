import torch
import time
from collections import deque

class FedBuffBuffer:
    """
    Buffer for storing client updates in FedBuff
    """
    def __init__(self, buffer_size=10, device='cpu'):
        self.buffer_size = buffer_size  # K in paper
        self.buffer = deque(maxlen=buffer_size)
        self.current_version = 0
        self.device = device
        
    def add_update(self, state_dict_delta, client_id, start_version, end_version):
        """
        Add a client update to the buffer
        
        Args:
            state_dict_delta: Client's trained model - given_global_model
            client_id: Identifier for the client
            start_version: Global model version when client started training
            end_version: Global model version when client finished training
        """
        self.buffer.append({
            'state_dict_delta': state_dict_delta,
            'client_id': client_id,
            'staleness': end_version - start_version,
            'start_version': start_version,
            'end_version': end_version,
            'timestamp': time.time()
        })
        
    def is_full(self):
        """Check if buffer has K updates"""
        return len(self.buffer) >= self.buffer_size
    
    def get_buffer_size(self):
        """Get current number of updates in buffer"""
        return len(self.buffer)
    
    def clear(self):
        """Clear the buffer after aggregation"""
        self.buffer.clear()
    
    def get_updates(self):
        """Get all updates currently in buffer"""
        return list(self.buffer)
    
    @staticmethod
    def staleness_weight(tau):
        """
        Staleness discount function from FedBuff paper
        s(τ) = 1 / (1 + τ)^0.5
        
        Args:
            tau: Staleness value
        Returns:
            Weight between 0 and 1
        """
        return 1.0 / ((1.0 + tau) ** 0.5)

def fedbuff_aggregate(global_model, buffered_updates, server_lr=1.0):
    """
    Aggregate client deltas with staleness weighting
    
    Args:
        global_model: Current global model (on GPU)
        buffered_updates: List of update dictionaries from buffer
        server_lr: η_g in the algorithm
        
    Returns:
        Updated global model
    """
    global_dict = {k: v.cpu() for k, v in global_model.state_dict().items()}

    # Calculate staleness weights
    staleness_weights = []
    for update_info in buffered_updates:
        tau = update_info['staleness']
        weight = FedBuffBuffer.staleness_weight(tau)
        staleness_weights.append(weight)
    
    # Normalize weights to sum to 1
    total_weight = sum(staleness_weights)
    normalized_weights = [w / total_weight for w in staleness_weights]

    # Step 1: Compute weighted average of deltas → Δ̄
    avg_delta = {}
    for key in global_dict.keys():
        avg_delta[key] = torch.zeros_like(global_dict[key])
        
        for update_info, weight in zip(buffered_updates, normalized_weights):
            delta = update_info['state_dict_delta'][key].float()
            avg_delta[key] += weight * delta
    
    # Step 2: Apply to global model → w^(t+1) = w^t - η_g · Δ̄
    for key in global_dict.keys():
        global_dict[key] = global_dict[key] - (server_lr * avg_delta[key])
    
    global_model.load_state_dict(global_dict)

    # Default configuration settings
    global_model.gr = 1.0
    global_model.nc = 1

    return global_model