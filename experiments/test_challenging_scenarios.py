# -*- coding: utf-8 -*-
"""
Quick Test: Challenging Scenarios

Test hypothesis that more challenging scenarios lead to higher optimization gains

Scenarios tested:
1. Baseline (current): N=32, R=500m, Bd=1.2MHz
2. Large coverage: N=32, R=1000m
3. Many users: N=64, R=500m
4. Limited bandwidth: N=32, Bd=0.4MHz
5. Extreme: N=64, R=1000m, Bd=3.0MHz

Runtime: ~10-15 minutes
"""
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from config import config
from src_enhanced.joint_satcon_system import JointOptimizationSATCON


def test_scenario(name, N, R, Bd, snr=20, elevation=10, n_real=5):
    """Test a specific scenario"""
    print(f"\n{'='*70}")
    print(f"Scenario: {name}")
    print(f"  N={N}, R={R}m, Bd={Bd/1e6:.1f}MHz, SNR={snr}dB, E={elevation}deg")
    print(f"{'='*70}")

    # Temporarily modify config
    original_N = config.N
    original_R = config.coverage_radius

    config.N = N
    config.coverage_radius = R

    # Create systems
    baseline = JointOptimizationSATCON(
        config, Bd,
        use_module1=False, use_module2=False, use_module3=False
    )

    full = JointOptimizationSATCON(
        config, Bd,
        use_module1=True, use_module2=True, use_module3=True
    )

    # Run tests
    print(f"\nRunning {n_real} realizations...")

    rates_baseline = []
    rates_full = []

    for seed in range(n_real):
        print(f"  Realization {seed+1}/{n_real}...", end='', flush=True)

        rate_b, _ = baseline.simulate_single_realization(snr, elevation, seed)
        rate_f, _ = full.simulate_single_realization(snr, elevation, seed)

        rates_baseline.append(rate_b)
        rates_full.append(rate_f)

        print(" Done")

    # Statistics
    mean_b = np.mean(rates_baseline)
    mean_f = np.mean(rates_full)
    std_b = np.std(rates_baseline)
    std_f = np.std(rates_full)

    gain_abs = mean_f - mean_b
    gain_pct = (gain_abs / mean_b) * 100

    # Results
    print(f"\nResults:")
    print(f"  Baseline:    {mean_b/1e6:6.2f} ± {std_b/1e6:4.2f} Mbps")
    print(f"  Full System: {mean_f/1e6:6.2f} ± {std_f/1e6:4.2f} Mbps")
    print(f"  Gain:        {gain_abs/1e6:6.2f} Mbps ({gain_pct:+5.2f}%)")

    # Restore config
    config.N = original_N
    config.coverage_radius = original_R

    return {
        'name': name,
        'baseline_mean': mean_b,
        'baseline_std': std_b,
        'full_mean': mean_f,
        'full_std': std_f,
        'gain_pct': gain_pct
    }


# ==================== Main ====================
if __name__ == "__main__":
    print("="*70)
    print("TESTING CHALLENGING SCENARIOS")
    print("="*70)
    print("\nHypothesis: More challenging scenarios → Higher optimization gains")
    print("Testing 5 scenarios with 5 realizations each")
    print("\nEstimated time: 10-15 minutes")

    results = []

    # Scenario 1: Baseline (current)
    results.append(test_scenario(
        "Baseline (Current)",
        N=32, R=500, Bd=1.2e6
    ))

    # Scenario 2: Large coverage
    results.append(test_scenario(
        "Large Coverage",
        N=32, R=1000, Bd=1.2e6
    ))

    # Scenario 3: Many users
    results.append(test_scenario(
        "Many Users",
        N=64, R=500, Bd=1.2e6
    ))

    # Scenario 4: Limited bandwidth
    results.append(test_scenario(
        "Limited Bandwidth",
        N=32, R=500, Bd=0.4e6
    ))

    # Scenario 5: Extreme challenge
    results.append(test_scenario(
        "Extreme Challenge",
        N=64, R=1000, Bd=3.0e6
    ))

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    print(f"\n{'Scenario':<25} {'Baseline (Mbps)':<18} {'Full (Mbps)':<18} {'Gain (%)'}")
    print("-"*70)

    for r in results:
        print(f"{r['name']:<25} {r['baseline_mean']/1e6:>7.2f} ± {r['baseline_std']/1e6:4.2f}   "
              f"{r['full_mean']/1e6:>7.2f} ± {r['full_std']/1e6:4.2f}   "
              f"{r['gain_pct']:>7.2f}")

    print("\n" + "="*70)

    # Find best
    best_idx = np.argmax([r['gain_pct'] for r in results])
    worst_idx = np.argmin([r['gain_pct'] for r in results])

    print(f"\nBest scenario:  {results[best_idx]['name']} ({results[best_idx]['gain_pct']:+.2f}%)")
    print(f"Worst scenario: {results[worst_idx]['name']} ({results[worst_idx]['gain_pct']:+.2f}%)")

    # Analysis
    gains = [r['gain_pct'] for r in results]
    print(f"\nGain statistics:")
    print(f"  Mean:   {np.mean(gains):.2f}%")
    print(f"  Median: {np.median(gains):.2f}%")
    print(f"  Std:    {np.std(gains):.2f}%")
    print(f"  Range:  [{np.min(gains):.2f}%, {np.max(gains):.2f}%]")

    if np.max(gains) > 8:
        print(f"\n✓ SUCCESS: Found scenario with >{8}% gain!")
        print(f"  Recommendation: Use '{results[best_idx]['name']}' scenario for paper")
    elif np.max(gains) > 5:
        print(f"\n⚠ PARTIAL: Found scenario with >{5}% gain")
        print(f"  Recommendation: Test more extreme parameters")
    else:
        print(f"\n✗ CONCERN: All gains < 5%")
        print(f"  Recommendation: Review algorithm implementations")

    print("\n" + "="*70)
    print("Test complete!")
    print("="*70)
