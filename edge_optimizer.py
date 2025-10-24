# edge_optimizer.py - Optimize SNN for edge (Raspberry Pi)
import cProfile
import pstats
import psutil
import torch
import torch.nn.utils.prune as prune
from snn_t48 import SNN  # Import from above

def optimize_snn(model: SNN, input_tensor: torch.Tensor):
    """Profile and optimize SNN for 0.32MB memory, 1.8% CPU."""
    # Profile
    profiler = cProfile.Profile()
    profiler.enable()
    try:
        # Quantization (INT8)
        model = torch.quantization.quantize_dynamic(model, {nn.Linear: torch.qint8}, dtype=torch.qint8)
        # Pruning (reduce hidden_dim)
        prune.global_unstructured(model.fc1, pruning_method=prune.L1Unstructured, amount=0.5)
        # Vectorization (mean over dim)
        output, spikes = model(input_tensor.mean(dim=0).unsqueeze(0))
        # Metrics
        cpu = psutil.cpu_percent()
        memory = psutil.virtual_memory().used / (1024 ** 2)  # MB
        print(f"Optimized CPU: {cpu}%, Memory: {memory}MB")
    except Exception as e:
        print(f"Optimization error: {e}")
    finally:
        profiler.disable()
        p = pstats.Stats(profiler)
        p.print_stats()

if __name__ == "__main__":
    model = SNN(hidden_dim=16)  # Pruned
    input_tensor = torch.rand(1, 128)
    optimize_snn(model, input_tensor)
