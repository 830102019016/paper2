# -*- coding: utf-8 -*-
"""
Large-scale Comparison Experiment Script

Run all baseline methods and generate comparison results for paper evaluation

Experiments:
1. Main: 6 methods x 7 SNR points x 1000 realizations
2. Ablation: Individual module contribution analysis
3. Optional: Scalability and bandwidth sweep

Outputs:
- Performance curves (PNG, 300 DPI)
- Raw data (NPZ format)
- Performance tables (CSV)

Author: SATCON Enhancement Project
Date: 2025-12-10
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import yaml
import sys
import time
from pathlib import Path
from datetime import datetime
import pickle

# Add project path
sys.path.append(str(Path(__file__).parent.parent))

from config import config
from src.noma_transmission import SatelliteNOMA
from src_enhanced.joint_satcon_system import JointOptimizationSATCON


class ComparisonFramework:
    """Large-scale Comparison Experiment Framework"""

    def __init__(self, config_file="experiments/config_comparison.yaml"):
        """
        Initialize framework

        Args:
            config_file: Path to experiment configuration file
        """
        # Load configuration
        with open(config_file, 'r', encoding='utf-8') as f:
            self.exp_config = yaml.safe_load(f)

        # Create output directories
        self.results_dir = Path(self.exp_config['output']['results_dir'])
        self.figures_dir = Path(self.exp_config['output']['figures_dir'])
        self.data_dir = Path(self.exp_config['output']['data_dir'])
        self.tables_dir = Path(self.exp_config['output']['tables_dir'])

        for dir_path in [self.results_dir, self.figures_dir, self.data_dir, self.tables_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Initialize systems
        self.systems = {}
        self.results = {}

        print("=" * 80)
        print("Large-scale Comparison Experiment Framework")
        print("=" * 80)
        print(f"Config file: {config_file}")
        print(f"Output dir: {self.results_dir}")
        print(f"Monte Carlo realizations: {self.exp_config['monte_carlo']['n_realizations']}")
        print("=" * 80)

    def create_systems(self):
        """Create all comparison systems"""
        print("\nCreating comparison systems...")

        scenarios = self.exp_config['scenarios']
        Bd = scenarios['abs_bandwidth']

        for baseline_cfg in self.exp_config['baselines']:
            if not baseline_cfg['enabled']:
                continue

            name = baseline_cfg['name']

            if name == "SAT-NOMA":
                # Satellite NOMA only (no ABS)
                self.systems[name] = SatelliteNOMA(config)
            else:
                # SATCON systems (original or enhanced)
                module_flags = baseline_cfg.get('module_flags', {
                    'use_module1': False,
                    'use_module2': False,
                    'use_module3': False
                })

                self.systems[name] = JointOptimizationSATCON(
                    config, Bd,
                    use_module1=module_flags['use_module1'],
                    use_module2=module_flags['use_module2'],
                    use_module3=module_flags['use_module3']
                )

            print(f"  OK {name}")

        print(f"\nTotal: {len(self.systems)} systems")

    def run_main_experiment(self):
        """
        Main experiment: SNR vs Performance

        6 methods x 7 SNR x 1000 realizations = 42000 simulations
        """
        print("\n" + "=" * 80)
        print("Main Experiment: SNR vs Performance")
        print("=" * 80)

        scenarios = self.exp_config['scenarios']
        snr_range = np.array(scenarios['snr_range'])
        elevation = scenarios['elevation_angle']
        n_real = self.exp_config['monte_carlo']['n_realizations']

        print(f"SNR range: {snr_range} dB")
        print(f"Elevation: {elevation} deg")
        print(f"Monte Carlo realizations: {n_real}")
        print(f"Total simulations: {len(self.systems)} x {len(snr_range)} x {n_real} = {len(self.systems) * len(snr_range) * n_real}")

        results = {}
        start_time = time.time()

        for name, system in self.systems.items():
            print(f"\n{'-'*80}")
            print(f"Running: {name}")
            print(f"{'-'*80}")

            sys_start = time.time()

            if name == "SAT-NOMA":
                # Satellite NOMA only
                mean_rates, mean_se, std_rates, _ = system.simulate_performance(
                    snr_db_range=snr_range,
                    elevation_deg=elevation,
                    n_realizations=n_real,
                    verbose=True
                )
                mode_stats = None
            else:
                # SATCON systems
                mean_rates, mean_se, std_rates, mode_stats = system.simulate_performance(
                    snr_db_range=snr_range,
                    elevation_deg=elevation,
                    n_realizations=n_real,
                    use_joint_optimization=True,
                    verbose=True
                )

            sys_time = time.time() - sys_start

            results[name] = {
                'mean_rates': mean_rates,
                'mean_se': mean_se,
                'std_rates': std_rates,
                'mode_stats': mode_stats,
                'time': sys_time
            }

            print(f"OK Completed {name} (time: {sys_time/60:.1f} min)")
            print(f"  SE @ 20dB: {mean_se[4]:.2f} bits/s/Hz")

        total_time = time.time() - start_time

        print("\n" + "=" * 80)
        print(f"Main experiment completed! Total time: {total_time/60:.1f} min ({total_time/3600:.2f} hours)")
        print("=" * 80)

        # Save results
        self.results['main'] = results
        self.save_results('main', results, snr_range)

        return results

    def run_ablation_experiment(self):
        """
        Ablation experiment: Analyze individual module contributions

        Compare: Baseline, +M1, +M2, +M3, +All
        """
        print("\n" + "=" * 80)
        print("Ablation Experiment: Module Contribution Analysis")
        print("=" * 80)

        scenarios = self.exp_config['scenarios']
        snr_range = np.array(scenarios['snr_range'])
        elevation = scenarios['elevation_angle']
        n_real = self.exp_config['monte_carlo']['n_realizations_ablation']

        print(f"Monte Carlo realizations: {n_real} (fewer for ablation)")

        # Select ablation-related systems
        ablation_systems = {
            name: sys for name, sys in self.systems.items()
            if name in ['Original-SATCON', 'SATCON+M1', 'SATCON+M2', 'SATCON+M3', 'Proposed-Full']
        }

        results = {}
        start_time = time.time()

        for name, system in ablation_systems.items():
            print(f"\nRunning: {name}")

            mean_rates, mean_se, std_rates, mode_stats = system.simulate_performance(
                snr_db_range=snr_range,
                elevation_deg=elevation,
                n_realizations=n_real,
                use_joint_optimization=True,
                verbose=True
            )

            results[name] = {
                'mean_rates': mean_rates,
                'mean_se': mean_se,
                'std_rates': std_rates,
                'mode_stats': mode_stats
            }

        total_time = time.time() - start_time

        print(f"\nAblation experiment completed! Time: {total_time/60:.1f} min")

        # Save results
        self.results['ablation'] = results
        self.save_results('ablation', results, snr_range)

        return results

    def save_results(self, exp_name, results, snr_range):
        """
        Save experiment results

        Args:
            exp_name: Experiment name
            results: Results dictionary
            snr_range: SNR range
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save raw data (NPZ format)
        data_file = self.data_dir / f"{exp_name}_{timestamp}.npz"
        save_dict = {
            'snr_range': snr_range,
            'timestamp': timestamp
        }

        for name, data in results.items():
            save_dict[f"{name}_mean_rates"] = data['mean_rates']
            save_dict[f"{name}_mean_se"] = data['mean_se']
            save_dict[f"{name}_std_rates"] = data['std_rates']

        np.savez(data_file, **save_dict)
        print(f"\nOK Raw data saved: {data_file}")

        # Save as pickle (complete information)
        pickle_file = self.data_dir / f"{exp_name}_{timestamp}.pkl"
        with open(pickle_file, 'wb') as f:
            pickle.dump({
                'results': results,
                'snr_range': snr_range,
                'config': self.exp_config
            }, f)
        print(f"OK Complete data saved: {pickle_file}")

        # Save performance table (CSV)
        self.save_performance_table(exp_name, results, snr_range, timestamp)

    def save_performance_table(self, exp_name, results, snr_range, timestamp):
        """Save performance comparison table"""
        # Create DataFrame
        data_rows = []

        for name, data in results.items():
            for i, snr in enumerate(snr_range):
                data_rows.append({
                    'System': name,
                    'SNR (dB)': snr,
                    'Sum Rate (Mbps)': data['mean_rates'][i] / 1e6,
                    'SE (bits/s/Hz)': data['mean_se'][i],
                    'Std Rate (Mbps)': data['std_rates'][i] / 1e6
                })

        df = pd.DataFrame(data_rows)

        # Save CSV
        csv_file = self.tables_dir / f"{exp_name}_performance_{timestamp}.csv"
        df.to_csv(csv_file, index=False)
        print(f"OK Performance table saved: {csv_file}")

        # Calculate and save gain table
        baseline_name = 'Original-SATCON' if 'Original-SATCON' in results else list(results.keys())[0]
        baseline_rates = results[baseline_name]['mean_rates']

        gain_rows = []
        for name, data in results.items():
            if name == baseline_name:
                continue

            for i, snr in enumerate(snr_range):
                gain = (data['mean_rates'][i] - baseline_rates[i]) / baseline_rates[i] * 100
                gain_rows.append({
                    'System': name,
                    'SNR (dB)': snr,
                    'Gain (%)': gain
                })

        if gain_rows:
            df_gain = pd.DataFrame(gain_rows)
            gain_file = self.tables_dir / f"{exp_name}_gains_{timestamp}.csv"
            df_gain.to_csv(gain_file, index=False)
            print(f"OK Gain table saved: {gain_file}")

    def plot_main_results(self):
        """Plot main experiment results"""
        if 'main' not in self.results:
            print("Error: Main experiment results not found")
            return

        print("\n" + "=" * 80)
        print("Plotting Performance Curves")
        print("=" * 80)

        results = self.results['main']
        scenarios = self.exp_config['scenarios']
        snr_range = np.array(scenarios['snr_range'])

        # Set plot style
        plt.style.use('seaborn-v0_8-paper')
        plt.rcParams['font.size'] = 11
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['axes.titlesize'] = 13
        plt.rcParams['legend.fontsize'] = 10

        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Subplot 1: Spectral Efficiency vs SNR
        ax = axes[0, 0]
        for name, data in results.items():
            ax.plot(snr_range, data['mean_se'], marker='o', label=name, linewidth=2)
        ax.set_xlabel('SNR (dB)')
        ax.set_ylabel('Spectral Efficiency (bits/s/Hz)')
        ax.set_title('Spectral Efficiency vs SNR')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        # Subplot 2: Relative Gain vs SNR
        ax = axes[0, 1]
        baseline_name = 'Original-SATCON' if 'Original-SATCON' in results else 'SAT-NOMA'
        baseline_rates = results[baseline_name]['mean_rates']

        for name, data in results.items():
            if name == baseline_name:
                continue
            gain = (data['mean_rates'] - baseline_rates) / baseline_rates * 100
            ax.plot(snr_range, gain, marker='s', label=name, linewidth=2)

        ax.set_xlabel('SNR (dB)')
        ax.set_ylabel('Rate Gain (%)')
        ax.set_title(f'Performance Gain over {baseline_name}')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)

        # Subplot 3: Sum Rate vs SNR
        ax = axes[1, 0]
        for name, data in results.items():
            ax.plot(snr_range, data['mean_rates']/1e6, marker='^', label=name, linewidth=2)
        ax.set_xlabel('SNR (dB)')
        ax.set_ylabel('Sum Rate (Mbps)')
        ax.set_title('System Sum Rate vs SNR')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        # Subplot 4: Standard Deviation vs SNR
        ax = axes[1, 1]
        for name, data in results.items():
            ax.plot(snr_range, data['std_rates']/1e6, marker='d', label=name, linewidth=2)
        ax.set_xlabel('SNR (dB)')
        ax.set_ylabel('Std. Deviation (Mbps)')
        ax.set_title('Performance Variability')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save figure
        fig_file = self.figures_dir / 'main_comparison.png'
        plt.savefig(fig_file, dpi=300, bbox_inches='tight')
        print(f"OK Main experiment figure saved: {fig_file}")

        plt.close()

    def plot_ablation_results(self):
        """Plot ablation experiment results"""
        if 'ablation' not in self.results:
            print("Error: Ablation experiment results not found")
            return

        print("\nPlotting ablation study results...")

        results = self.results['ablation']
        scenarios = self.exp_config['scenarios']
        snr_range = np.array(scenarios['snr_range'])

        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Subplot 1: Spectral Efficiency comparison
        ax = axes[0]
        order = ['Original-SATCON', 'SATCON+M1', 'SATCON+M2', 'SATCON+M3', 'Proposed-Full']
        for name in order:
            if name in results:
                data = results[name]
                ax.plot(snr_range, data['mean_se'], marker='o', label=name, linewidth=2)

        ax.set_xlabel('SNR (dB)')
        ax.set_ylabel('Spectral Efficiency (bits/s/Hz)')
        ax.set_title('Ablation Study: Spectral Efficiency')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        # Subplot 2: Module contribution bar chart at SNR=20dB
        ax = axes[1]
        snr_20_idx = list(snr_range).index(20)
        baseline_rate = results['Original-SATCON']['mean_rates'][snr_20_idx]

        bars_data = []
        labels = []
        for name in order:
            if name in results:
                rate = results[name]['mean_rates'][snr_20_idx]
                gain = (rate - baseline_rate) / baseline_rate * 100
                bars_data.append(gain)
                labels.append(name.replace('SATCON+', '+').replace('Original-SATCON', 'Baseline').replace('Proposed-Full', 'Full'))

        bars = ax.bar(range(len(bars_data)), bars_data, color=['gray', 'blue', 'green', 'orange', 'red'])
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=15, ha='right')
        ax.set_ylabel('Rate Gain over Baseline (%)')
        ax.set_title('Module Contribution @ SNR=20dB')
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)

        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}%',
                   ha='center', va='bottom' if height > 0 else 'top',
                   fontsize=9)

        plt.tight_layout()

        # Save figure
        fig_file = self.figures_dir / 'ablation_study.png'
        plt.savefig(fig_file, dpi=300, bbox_inches='tight')
        print(f"OK Ablation study figure saved: {fig_file}")

        plt.close()

    def print_summary(self):
        """Print experiment summary"""
        print("\n" + "=" * 80)
        print("Experiment Summary")
        print("=" * 80)

        if 'main' in self.results:
            results = self.results['main']
            snr_range = np.array(self.exp_config['scenarios']['snr_range'])
            snr_20_idx = list(snr_range).index(20)

            baseline_name = 'Original-SATCON' if 'Original-SATCON' in results else 'SAT-NOMA'
            baseline_rate = results[baseline_name]['mean_rates'][snr_20_idx]
            baseline_se = results[baseline_name]['mean_se'][snr_20_idx]

            print(f"\nMain experiment results (SNR=20dB):")
            print(f"{'System':<30} {'SE (bits/s/Hz)':<18} {'Gain (%)'}")
            print("-" * 80)

            for name, data in results.items():
                rate = data['mean_rates'][snr_20_idx]
                se = data['mean_se'][snr_20_idx]
                gain = (rate - baseline_rate) / baseline_rate * 100

                print(f"{name:<30} {se:<18.2f} {gain:>7.2f}%")

            print("\n" + "-" * 80)
            print(f"Baseline: {baseline_name}")
            print(f"  Rate: {baseline_rate/1e6:.2f} Mbps")
            print(f"  Spectral Efficiency: {baseline_se:.2f} bits/s/Hz")

            if 'Proposed-Full' in results:
                full_rate = results['Proposed-Full']['mean_rates'][snr_20_idx]
                full_se = results['Proposed-Full']['mean_se'][snr_20_idx]
                full_gain = (full_rate - baseline_rate) / baseline_rate * 100

                print(f"\nFull System:")
                print(f"  Rate: {full_rate/1e6:.2f} Mbps")
                print(f"  Spectral Efficiency: {full_se:.2f} bits/s/Hz")
                print(f"  Overall Gain: {full_gain:.2f}%")

        print("\n" + "=" * 80)

    def run_all_experiments(self):
        """Run all experiments"""
        # Create systems
        self.create_systems()

        # Main experiment
        if self.exp_config['experiments']['main']['enabled']:
            self.run_main_experiment()
            self.plot_main_results()

        # Ablation experiment
        if self.exp_config['experiments']['ablation']['enabled']:
            self.run_ablation_experiment()
            self.plot_ablation_results()

        # Print summary
        self.print_summary()

        print("\nOK All experiments completed!")


# ==================== Main ====================
if __name__ == "__main__":
    print("=" * 80)
    print("SATCON Large-scale Comparison Experiment")
    print("=" * 80)
    print("\nStarting experiments...")

    # Create framework
    framework = ComparisonFramework()

    # Run all experiments
    framework.run_all_experiments()

    print("\n" + "=" * 80)
    print("Experiments Completed!")
    print("=" * 80)
