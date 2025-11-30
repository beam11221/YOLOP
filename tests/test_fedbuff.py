"""
Test Script for FedBuff Implementation
=======================================
This script validates that your FedBuff implementation is working correctly
before running full-scale experiments.

Run this BEFORE training on BDD100K dataset!
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import your FedBuff components
try:
    from lib.core.fedbuff_buffer import FedBuffBuffer, fedbuff_aggregate
    print("✓ Successfully imported FedBuff components")
except ImportError as e:
    print(f"✗ Failed to import FedBuff components: {e}")
    print("  Make sure lib/core/fedbuff_buffer.py exists")
    sys.exit(1)


# ============================================================================
# Simple Model for Testing
# ============================================================================

class SimpleModel(nn.Module):
    """Simple model for testing (mimics structure of YOLOP but smaller)"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc = nn.Linear(32, 10)
        
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.mean(x, dim=[2, 3])  # Global average pooling
        x = self.fc(x)
        return x


# ============================================================================
# Test Functions
# ============================================================================

def test_1_buffer_operations():
    """Test 1: Verify buffer basic operations"""
    print("\n" + "="*70)
    print("TEST 1: Buffer Operations")
    print("="*70)
    
    buffer = FedBuffBuffer(buffer_size=3)
    
    # Test 1.1: Initial state
    assert len(buffer.buffer) == 0, "Buffer should start empty"
    assert not buffer.is_full(), "Empty buffer should not be full"
    print("✓ 1.1: Buffer initializes correctly")
    
    # Test 1.2: Add updates
    for i in range(3):
        buffer.add_update(
            state_dict={'param': torch.tensor([i])},
            client_id=f"client_{i}",
            start_version=0
        )
    
    assert len(buffer.buffer) == 3, f"Buffer should have 3 updates, has {len(buffer.buffer)}"
    assert buffer.is_full(), "Buffer should be full"
    print("✓ 1.2: Buffer fills correctly")
    
    # Test 1.3: Get updates
    updates = buffer.get_updates()
    assert len(updates) == 3, "Should retrieve 3 updates"
    assert updates[0]['client_id'] == "client_0", "Updates should be in order"
    print("✓ 1.3: Buffer retrieves updates correctly")
    
    # Test 1.4: Clear buffer
    buffer.clear()
    assert len(buffer.buffer) == 0, "Buffer should be empty after clear"
    assert not buffer.is_full(), "Cleared buffer should not be full"
    print("✓ 1.4: Buffer clears correctly")
    
    print("✅ TEST 1 PASSED: Buffer operations work correctly\n")


def test_2_staleness_calculation():
    """Test 2: Verify staleness calculation"""
    print("\n" + "="*70)
    print("TEST 2: Staleness Calculation")
    print("="*70)
    
    buffer = FedBuffBuffer(buffer_size=5)
    buffer.current_version = 5  # Simulate we're at version 5
    
    # Add updates with different start versions
    test_cases = [
        (5, 0, "Fresh update"),      # staleness = 5 - 5 = 0
        (3, 2, "Slightly stale"),    # staleness = 5 - 3 = 2
        (0, 5, "Very stale"),        # staleness = 5 - 0 = 5
    ]
    
    for start_ver, expected_staleness, desc in test_cases:
        buffer.add_update(
            state_dict={'param': torch.tensor([1.0])},
            client_id=f"client_v{start_ver}",
            start_version=start_ver
        )
        
        last_update = buffer.buffer[-1]
        actual_staleness = last_update['staleness']
        assert actual_staleness == expected_staleness, \
            f"{desc}: Expected staleness {expected_staleness}, got {actual_staleness}"
        
        print(f"✓ {desc}: start_v={start_ver}, staleness={actual_staleness}")
    
    print("✅ TEST 2 PASSED: Staleness calculation correct\n")


def test_3_staleness_weights():
    """Test 3: Verify staleness weight function"""
    print("\n" + "="*70)
    print("TEST 3: Staleness Weight Function")
    print("="*70)
    
    # Test weight formula: s(τ) = 1 / (1 + τ)^0.5
    test_cases = [
        (0, 1.0000, "No staleness → weight = 1"),
        (1, 0.7071, "Staleness 1 → weight = 1/√2"),
        (3, 0.5000, "Staleness 3 → weight = 1/2"),
        (8, 0.3333, "Staleness 8 → weight = 1/3"),
    ]
    
    for tau, expected_weight, desc in test_cases:
        actual_weight = FedBuffBuffer.staleness_weight(tau)
        
        # Allow small floating point error
        assert abs(actual_weight - expected_weight) < 0.001, \
            f"{desc}: Expected {expected_weight:.4f}, got {actual_weight:.4f}"
        
        print(f"✓ {desc}: s({tau}) = {actual_weight:.4f}")
    
    # Test monotonicity: higher staleness → lower weight
    weights = [FedBuffBuffer.staleness_weight(tau) for tau in range(10)]
    for i in range(len(weights)-1):
        assert weights[i] > weights[i+1], \
            f"Weight should decrease with staleness: w({i})={weights[i]:.4f} > w({i+1})={weights[i+1]:.4f}"
    
    print("✓ Weights decrease monotonically with staleness")
    print("✅ TEST 3 PASSED: Staleness weight function correct\n")


def test_4_aggregation_correctness():
    """Test 4: Verify aggregation produces correct results"""
    print("\n" + "="*70)
    print("TEST 4: Aggregation Correctness")
    print("="*70)
    
    # Create simple model
    model = SimpleModel()
    
    # Create mock client updates with known values
    buffer = FedBuffBuffer(buffer_size=3)
    
    # Client updates with different staleness
    client_updates = [
        {'staleness': 0, 'value': 10.0, 'client_id': 'A'},  # Fresh
        {'staleness': 2, 'value': 8.0, 'client_id': 'B'},   # Slightly stale
        {'staleness': 5, 'value': 12.0, 'client_id': 'C'},  # Very stale
    ]
    
    # Create mock state_dicts
    for update in client_updates:
        state_dict = {}
        for name, param in model.named_parameters():
            # Use simple scalar for easy verification
            state_dict[name] = torch.ones_like(param) * update['value']
        
        buffer.add_update(
            state_dict=state_dict,
            client_id=update['client_id'],
            start_version=5 - update['staleness']  # Current version = 5
        )
        buffer.current_version = 5
    
    # Manually calculate expected result
    weights = [FedBuffBuffer.staleness_weight(u['staleness']) for u in client_updates]
    total_weight = sum(weights)
    normalized_weights = [w / total_weight for w in weights]
    
    expected_value = sum(w * u['value'] for w, u in zip(normalized_weights, client_updates))
    
    print(f"Client values: {[u['value'] for u in client_updates]}")
    print(f"Staleness: {[u['staleness'] for u in client_updates]}")
    print(f"Raw weights: {[f'{w:.4f}' for w in weights]}")
    print(f"Normalized weights: {[f'{w:.4f}' for w in normalized_weights]}")
    print(f"Expected aggregated value: {expected_value:.4f}")
    
    # Perform aggregation
    buffered_updates = buffer.get_updates()
    aggregated_model = fedbuff_aggregate(model, buffered_updates)
    
    # Check result
    first_param = next(iter(aggregated_model.parameters()))
    actual_value = first_param[0, 0, 0, 0].item()  # Get one value
    
    print(f"Actual aggregated value: {actual_value:.4f}")
    
    # Allow small floating point error
    assert abs(actual_value - expected_value) < 0.01, \
        f"Aggregation incorrect: expected {expected_value:.4f}, got {actual_value:.4f}"
    
    print("✓ Aggregation produces correct weighted average")
    print("✅ TEST 4 PASSED: Aggregation is correct\n")


def test_5_memory_management():
    """Test 5: Verify proper memory management"""
    print("\n" + "="*70)
    print("TEST 5: Memory Management")
    print("="*70)
    
    if not torch.cuda.is_available():
        print("⚠️  SKIPPED: No GPU available for memory test")
        return
    
    device = torch.device("cuda")
    
    # Create model on GPU
    model = SimpleModel().to(device)
    initial_memory = torch.cuda.memory_allocated() / 1024**2  # MB
    print(f"Initial GPU memory: {initial_memory:.2f} MB")
    
    # Simulate client training
    buffer = FedBuffBuffer(buffer_size=3)
    
    for i in range(3):
        # Create client model on GPU
        client_model = SimpleModel().to(device)
        
        # Simulate training (forward pass)
        dummy_input = torch.randn(2, 3, 32, 32).to(device)
        output = client_model(dummy_input)
        
        # Move to CPU and store (as in real implementation)
        client_model.to("cpu")
        state_dict = client_model.state_dict()
        
        buffer.add_update(
            state_dict=state_dict,
            client_id=f"client_{i}",
            start_version=0
        )
        
        # Free GPU memory
        del client_model, output, dummy_input
        torch.cuda.empty_cache()
    
    final_memory = torch.cuda.memory_allocated() / 1024**2  # MB
    memory_increase = final_memory - initial_memory
    
    print(f"Final GPU memory: {final_memory:.2f} MB")
    print(f"Memory increase: {memory_increase:.2f} MB")
    
    # Should not have significant GPU memory increase
    assert memory_increase < 10, \
        f"GPU memory leak detected: {memory_increase:.2f} MB increase"
    
    print("✓ No GPU memory leaks detected")
    
    # Check that updates are in CPU
    for update in buffer.get_updates():
        for param in update['state_dict'].values():
            assert not param.is_cuda, "State dict should be on CPU"
    
    print("✓ State dicts correctly stored on CPU")
    print("✅ TEST 5 PASSED: Memory management is correct\n")


def test_6_integration():
    """Test 6: Full integration test"""
    print("\n" + "="*70)
    print("TEST 6: Full Integration Test")
    print("="*70)
    
    # Simulate mini training loop
    buffer_size = 3
    buffer = FedBuffBuffer(buffer_size=buffer_size)
    global_model = SimpleModel()
    current_version = 0
    
    num_clients = 5
    updates_per_client = 2
    total_updates = num_clients * updates_per_client
    
    print(f"Simulating {total_updates} updates from {num_clients} clients")
    print(f"Buffer size K = {buffer_size}\n")
    
    aggregation_count = 0
    
    for update_num in range(total_updates):
        client_id = update_num % num_clients
        
        # Simulate client training
        start_version = current_version
        client_model = SimpleModel()
        client_model.load_state_dict(global_model.state_dict())
        
        # Move to CPU (as in real implementation)
        state_dict = client_model.state_dict()
        
        # Add to buffer
        buffer.add_update(
            state_dict=state_dict,
            client_id=f"client_{client_id}",
            start_version=start_version
        )
        
        print(f"Update {update_num+1}/{total_updates}: Client {client_id}, "
              f"Version {start_version}, Buffer: {len(buffer.buffer)}/{buffer_size}")
        
        # Check if buffer is full
        if buffer.is_full():
            aggregation_count += 1
            print(f"  → AGGREGATING (#{aggregation_count})...")
            
            # Get updates and aggregate
            buffered_updates = buffer.get_updates()
            staleness_values = [u['staleness'] for u in buffered_updates]
            
            print(f"     Staleness: {staleness_values}")
            
            global_model = fedbuff_aggregate(global_model, buffered_updates)
            
            # Increment version and clear buffer
            current_version += 1
            buffer.clear()
            
            print(f"     New version: {current_version}\n")
    
    expected_aggregations = total_updates // buffer_size
    assert aggregation_count == expected_aggregations, \
        f"Expected {expected_aggregations} aggregations, got {aggregation_count}"
    
    print(f"✓ Correct number of aggregations: {aggregation_count}")
    print(f"✓ Final version: {current_version}")
    print("✅ TEST 6 PASSED: Integration test successful\n")


# ============================================================================
# Main Test Runner
# ============================================================================

def run_all_tests():
    """Run all tests"""
    print("\n" + "🔬"*35)
    print("FEDBUFF IMPLEMENTATION VALIDATION")
    print("🔬"*35)
    
    tests = [
        ("Buffer Operations", test_1_buffer_operations),
        ("Staleness Calculation", test_2_staleness_calculation),
        ("Staleness Weights", test_3_staleness_weights),
        ("Aggregation Correctness", test_4_aggregation_correctness),
        ("Memory Management", test_5_memory_management),
        ("Full Integration", test_6_integration),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"\n❌ TEST FAILED: {test_name}")
            print(f"   Error: {e}\n")
            failed += 1
        except Exception as e:
            print(f"\n❌ TEST ERROR: {test_name}")
            print(f"   Unexpected error: {e}\n")
            failed += 1
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Total tests: {len(tests)}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED! Your FedBuff implementation is ready!")
        print("\nNext steps:")
        print("  1. Test with your actual YOLOP model")
        print("  2. Run on small subset of BDD100K dataset")
        print("  3. Compare results with FedAvg baseline")
        print("  4. Scale up to full dataset")
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please fix issues before proceeding.")
        print("\nDebugging tips:")
        print("  1. Check lib/core/fedbuff_buffer.py implementation")
        print("  2. Verify staleness_weight formula")
        print("  3. Check aggregation logic")
        print("  4. Review memory management")
    
    print("="*70 + "\n")
    
    return failed == 0


# ============================================================================
# Additional Utilities
# ============================================================================

def benchmark_staleness_weights():
    """Benchmark and visualize staleness weight function"""
    print("\n" + "="*70)
    print("STALENESS WEIGHT VISUALIZATION")
    print("="*70)
    
    print("\nFormula: s(τ) = 1 / (1 + τ)^0.5\n")
    print(f"{'Staleness (τ)':<15} {'Weight s(τ)':<15} {'Visual'}")
    print("-" * 60)
    
    for tau in range(0, 21, 2):
        weight = FedBuffBuffer.staleness_weight(tau)
        bar = "█" * int(weight * 40)
        print(f"{tau:<15} {weight:<15.4f} {bar}")
    
    print("\nKey observations:")
    print("  • τ=0 (fresh): weight = 1.000 (full influence)")
    print("  • τ=3: weight = 0.500 (half influence)")
    print("  • τ=8: weight = 0.333 (one-third influence)")
    print("  • Higher staleness → exponentially lower weight")
    print("="*70 + "\n")


def compare_aggregation_methods():
    """Compare FedAvg vs FedBuff aggregation"""
    print("\n" + "="*70)
    print("AGGREGATION METHOD COMPARISON")
    print("="*70)
    
    # Test scenario
    values = [10.0, 8.0, 12.0]
    staleness = [0, 2, 5]
    
    print(f"\nScenario: 3 clients with updates:")
    for i, (v, s) in enumerate(zip(values, staleness)):
        print(f"  Client {i}: value={v:.1f}, staleness={s}")
    
    # FedAvg (equal weights)
    fedavg_result = sum(values) / len(values)
    print(f"\nFedAvg (equal weights):")
    print(f"  Weights: [0.333, 0.333, 0.333]")
    print(f"  Result: {fedavg_result:.4f}")
    
    # FedBuff (staleness-weighted)
    weights = [FedBuffBuffer.staleness_weight(s) for s in staleness]
    total = sum(weights)
    norm_weights = [w/total for w in weights]
    fedbuff_result = sum(w*v for w, v in zip(norm_weights, values))
    
    print(f"\nFedBuff (staleness-weighted):")
    print(f"  Raw weights: {[f'{w:.4f}' for w in weights]}")
    print(f"  Normalized: {[f'{w:.4f}' for w in norm_weights]}")
    print(f"  Result: {fedbuff_result:.4f}")
    
    print(f"\nDifference: {abs(fedbuff_result - fedavg_result):.4f}")
    print(f"FedBuff {'increases' if fedbuff_result > fedavg_result else 'decreases'} "
          f"weight on stale updates")
    print("="*70 + "\n")


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test FedBuff Implementation')
    parser.add_argument('--benchmark', action='store_true', 
                       help='Run benchmark and visualization')
    parser.add_argument('--compare', action='store_true',
                       help='Compare FedAvg vs FedBuff aggregation')
    args = parser.parse_args()
    
    if args.benchmark:
        benchmark_staleness_weights()
    elif args.compare:
        compare_aggregation_methods()
    else:
        # Run all validation tests
        success = run_all_tests()
        sys.exit(0 if success else 1)