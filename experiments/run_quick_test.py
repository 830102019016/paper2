# Quick test: 3 SNR points x 10 realizations x 3 systems
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from config import config
from src_enhanced.joint_satcon_system import JointOptimizationSATCON

print("=" * 60)
print("Quick Test: Validating Experiment Framework")
print("=" * 60)

# Test parameters
snr_range = np.array([10, 20, 30])
n_real = 10
Bd = 1.2e6

print(f"\nConfiguration:")
print(f"  SNR points: {snr_range}")
print(f"  Realizations: {n_real}")
print(f"  Bandwidth: {Bd/1e6:.1f} MHz")

# Create systems
systems = {
    'Baseline': JointOptimizationSATCON(
        config, Bd,
        use_module1=False, use_module2=False, use_module3=False
    ),
    'Module1': JointOptimizationSATCON(
        config, Bd,
        use_module1=True, use_module2=False, use_module3=False
    ),
    'Full': JointOptimizationSATCON(
        config, Bd,
        use_module1=True, use_module2=True, use_module3=True
    )
}

print(f"  Systems: {len(systems)}")
print(f"  Total simulations: {len(systems)} x {len(snr_range)} x {n_real} = {len(systems) * len(snr_range) * n_real}")
print("\nStarting simulation...")

# Run simulations
results = {}
for name, system in systems.items():
    print(f"\n{name}:")
    mean_rates, mean_se, _, _ = system.simulate_performance(
        snr_db_range=snr_range,
        n_realizations=n_real,
        verbose=True
    )
    results[name] = {'mean_rates': mean_rates, 'mean_se': mean_se}

# Print summary
print("\n" + "=" * 60)
print("Results Summary (SNR=20dB):")
print("-" * 60)

baseline_rate = results['Baseline']['mean_rates'][1]
for name, data in results.items():
    rate = data['mean_rates'][1]
    se = data['mean_se'][1]
    gain = (rate - baseline_rate) / baseline_rate * 100 if name != 'Baseline' else 0
    print(f"{name:12s}: SE={se:5.2f} bits/s/Hz, Gain={gain:+6.2f}%")

print("=" * 60)
print("OK Quick test completed!")
print("\nIf successful, run full experiment:")
print("  python experiments/run_comparison.py")
