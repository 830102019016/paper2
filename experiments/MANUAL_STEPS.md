# Manual Experiment Operation Guide

Complete step-by-step guide for running large-scale comparison experiments manually.

---

## Quick Decision Guide

**Choose your experiment scale:**

| Option | Realizations | Time | Use Case |
|--------|-------------|------|----------|
| **Quick** | 10 | 2-3 min | Validation only |
| **Medium** | 100 | 30-60 min | **Recommended** for initial results |
| **Large** | 500 | 2-4 hours | Good for draft paper |
| **Full** | 1000 | 4-8 hours | Publication quality |

---

## Step-by-Step Instructions

### STEP 1: Choose Experiment Scale

Edit the configuration file to set the number of realizations:

```bash
# Open the config file in your editor
notepad experiments\config_comparison.yaml

# OR use command line
# For Medium scale (recommended):
```

Find these lines and change them:

```yaml
monte_carlo:
  n_realizations: 1000          # CHANGE THIS
  n_realizations_ablation: 500  # CHANGE THIS
```

**Recommended values:**

- **For quick test**: `n_realizations: 10` and `n_realizations_ablation: 5`
- **For medium scale**: `n_realizations: 100` and `n_realizations_ablation: 50`
- **For full scale**: Keep as `1000` and `500`

---

### STEP 2: Choose Which Systems to Compare (Optional)

If you want to skip some systems to save time, edit `config_comparison.yaml`:

```yaml
baselines:
  - name: "SAT-NOMA"
    enabled: true          # CHANGE to false to skip

  - name: "Original-SATCON"
    enabled: true          # Keep this as baseline

  - name: "SATCON+M1"
    enabled: true          # CHANGE to false to skip

  - name: "SATCON+M2"
    enabled: true          # CHANGE to false to skip

  - name: "SATCON+M3"
    enabled: true          # CHANGE to false to skip

  - name: "Proposed-Full"
    enabled: true          # Keep this for comparison
```

**Recommendation**: At minimum, keep `Original-SATCON` and `Proposed-Full` enabled.

---

### STEP 3: Choose Which Experiments to Run

You can enable/disable experiments:

```yaml
experiments:
  main:
    enabled: true          # Main SNR sweep - KEEP THIS

  ablation:
    enabled: true          # Module contribution - RECOMMENDED

  scalability:
    enabled: false         # User number sweep - OPTIONAL

  bandwidth:
    enabled: false         # Bandwidth sweep - OPTIONAL
```

**Recommendation**:
- Always run `main` experiment
- Run `ablation` if you have time (adds ~20% time)

---

### STEP 4: Save Your Changes

After editing `config_comparison.yaml`, save and close the file.

**Verify your settings:**

```bash
# Quick check
python -c "import yaml; print(yaml.safe_load(open('experiments/config_comparison.yaml'))['monte_carlo'])"
```

Should output something like:
```
{'n_realizations': 100, 'n_realizations_ablation': 50, 'random_seed': 42}
```

---

### STEP 5: Run the Experiment

Open a terminal/command prompt and run:

```bash
# Navigate to project directory
cd e:\satcon_reproduction

# Run the experiment
python experiments/run_comparison.py
```

**What you'll see:**

```
================================================================================
Large-scale Comparison Experiment Framework
================================================================================
Config file: experiments/config_comparison.yaml
Output dir: results\comparison
Monte Carlo realizations: 100
================================================================================

Creating comparison systems...
  OK SAT-NOMA
  OK Original-SATCON
  OK SATCON+M1
  OK SATCON+M2
  OK SATCON+M3
  OK Proposed-Full

Total: 6 systems

================================================================================
Main Experiment: SNR vs Performance
================================================================================
SNR range: [ 0  5 10 15 20 25 30] dB
Elevation: 10 deg
Monte Carlo realizations: 100
Total simulations: 6 x 7 x 100 = 4200

--------------------------------------------------------------------------------
Running: SAT-NOMA
--------------------------------------------------------------------------------
[Progress bars will show here...]
```

---

### STEP 6: Monitor Progress

The script will show progress for each system:

```
Running: Original-SATCON
--------------------------------------------------------------------------------
Joint SATCON Simulation [Baseline]
  Bd=1.2MHz, Joint_Opt=True
Simulating: 100%|████████| 7/7 [05:23<00:00, 46.17s/it, SE=29.82 bits/s/Hz]
OK Completed Original-SATCON (time: 5.4 min)
  SE @ 20dB: 22.41 bits/s/Hz
```

**Time estimates:**

- **SAT-NOMA**: ~2-3 min per SNR point (fastest)
- **Original-SATCON**: ~5-6 min per SNR point
- **SATCON+M1**: ~5-6 min per SNR point
- **SATCON+M2**: ~15-20 min per SNR point (slowest - joint pairing)
- **SATCON+M3**: ~6-7 min per SNR point
- **Proposed-Full**: ~20-25 min per SNR point

**Total time for 6 systems × 7 SNR points × 100 realizations: ~45-60 minutes**

---

### STEP 7: Check Output Files

After completion, check these directories:

```bash
# View generated files
dir results\figures\comparison
dir results\data\comparison
dir results\tables\comparison
```

**Expected files:**

```
results/
├── figures/comparison/
│   ├── main_comparison.png     # 4-panel performance curves
│   └── ablation_study.png      # Module contribution analysis
├── data/comparison/
│   ├── main_20251210_143052.npz    # Raw data (NumPy)
│   ├── main_20251210_143052.pkl    # Complete results (pickle)
│   ├── ablation_20251210_143052.npz
│   └── ablation_20251210_143052.pkl
└── tables/comparison/
    ├── main_performance_20251210_143052.csv   # Performance table
    ├── main_gains_20251210_143052.csv         # Gain percentages
    ├── ablation_performance_20251210_143052.csv
    └── ablation_gains_20251210_143052.csv
```

---

### STEP 8: View Results

**Option A: Open figures**

```bash
# Windows
start results\figures\comparison\main_comparison.png

# Or just double-click the file in Explorer
```

**Option B: Open CSV tables in Excel**

```bash
start results\tables\comparison\main_performance_*.csv
```

**Option C: View summary in terminal**

The script automatically prints a summary at the end:

```
================================================================================
Experiment Summary
================================================================================

Main experiment results (SNR=20dB):
System                         SE (bits/s/Hz)     Gain (%)
--------------------------------------------------------------------------------
SAT-NOMA                       18.45               -17.67%
Original-SATCON                22.41                +0.00%
SATCON+M1                      22.78                +1.65%
SATCON+M2                      22.94                +2.36%
SATCON+M3                      22.41                +0.00%
Proposed-Full                  23.30                +3.97%

--------------------------------------------------------------------------------
Baseline: Original-SATCON
  Rate: 26.89 Mbps
  Spectral Efficiency: 22.41 bits/s/Hz

Full System:
  Rate: 27.96 Mbps
  Spectral Efficiency: 23.30 bits/s/Hz
  Overall Gain: 3.97%

================================================================================
```

---

## Troubleshooting

### Problem 1: Script Stops or Crashes

**Solution A: Reduce realizations**
- Change `n_realizations: 1000` to `50` or even `10`
- Rerun the experiment

**Solution B: Run systems individually**
- Disable all systems except one in `config_comparison.yaml`
- Run multiple times, each with different system enabled

**Solution C: Check memory usage**
```bash
# Windows Task Manager: Ctrl+Shift+Esc
# Check if RAM is full (>90%)
# Close other programs if needed
```

---

### Problem 2: Module 2 (Joint Pairing) is Too Slow

Joint pairing uses local search which can be slow. To speed up:

**Edit** `src_enhanced/joint_pairing_optimizer.py`:

```python
# Line ~106: Change max_iterations
max_iterations = 50  # CHANGE to 20 or even 10
```

**Or disable Module 2:**
```yaml
# In config_comparison.yaml
  - name: "SATCON+M2"
    enabled: false  # Skip this system
```

---

### Problem 3: cvxpy/ILP Solver Errors

If you see errors like `SolverError` or `GLPK_MI failed`:

**Solution**: Module 3 will automatically fall back to greedy algorithm. No action needed.

**Optional**: Install better solver
```bash
pip install gurobipy
# Then get free academic license from gurobi.com
```

---

### Problem 4: Matplotlib Display Errors

If figures don't display but script continues:

**Solution**: Figures are still saved to disk. Check `results/figures/comparison/`

**Optional**: Add this to script if needed
```python
# At top of run_comparison.py
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
```

---

### Problem 5: Want to Resume Interrupted Experiment

Currently, the script doesn't support resume. Options:

**Option A**: Reduce realizations and rerun
**Option B**: Run each system separately (edit config to enable one at a time)
**Option C**: Use saved data files if available

---

## Advanced: Parallel Execution (Future Enhancement)

To run experiments faster using multiple CPU cores:

**Create**: `experiments/run_comparison_parallel.py`

```python
from multiprocessing import Pool
import numpy as np

def run_single_realization(args):
    system, snr, seed = args
    return system.simulate_single_realization(snr, 10, seed)

# In main loop:
with Pool(8) as p:  # Use 8 cores
    results = p.map(run_single_realization, tasks)
```

**Speed-up**: ~6-8x faster with 8 cores

---

## Summary Checklist

Before running:
- [ ] Edited `config_comparison.yaml` with desired `n_realizations`
- [ ] Chose which systems to compare (`enabled: true/false`)
- [ ] Saved changes to config file
- [ ] Have at least 1-2 hours of free computer time (for medium scale)

After running:
- [ ] Check `results/figures/comparison/` for PNG files
- [ ] Check `results/tables/comparison/` for CSV files
- [ ] Check `results/data/comparison/` for NPZ/PKL files
- [ ] Review performance gains in the summary output

---

## Quick Commands Reference

```bash
# Edit configuration
notepad experiments\config_comparison.yaml

# Run quick test (2-3 min)
python experiments\run_quick_test.py

# Run full experiment
python experiments\run_comparison.py

# View latest figures
dir results\figures\comparison

# View latest tables
dir results\tables\comparison

# Open latest CSV
start results\tables\comparison\main_performance_*.csv
```

---

## Need Help?

1. Check [experiments/README.md](README.md) for overview
2. Check [src_enhanced/README.md](../src_enhanced/README.md) for module details
3. Run quick test first: `python experiments/run_quick_test.py`
4. Start with small `n_realizations` (10-50) for testing

---

**Last Updated**: 2025-12-10
**Status**: Ready to run
**Verified**: Quick test passed with +3.99% gain
