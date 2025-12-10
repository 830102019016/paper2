# Large-scale Comparison Experiment Guide

## Quick Start

### Option 1: Quick Test (2-3 minutes)
```bash
python experiments/run_quick_test.py
```
- 3 SNR points x 10 realizations x 3 systems
- Validates that everything works correctly
- **Result from last test: Full system achieved +3.99% gain @ SNR=20dB**

### Option 2: Medium-scale Test (Recommended, ~30-60 minutes)
```bash
# Edit config_comparison.yaml
# Change n_realizations from 1000 to 100
python experiments/run_comparison.py
```
- 7 SNR points x 100 realizations x 6 systems
- Good balance between accuracy and time
- Generates all figures and tables

### Option 3: Full-scale Experiment (~4-8 hours)
```bash
python experiments/run_comparison.py
```
- 7 SNR points x 1000 realizations x 6 systems = 42000 simulations
- Publication-quality results
- **Recommended to run overnight**

---

## Experiment Configuration

Edit `config_comparison.yaml` to customize:

### Key Parameters

```yaml
monte_carlo:
  n_realizations: 1000          # Main experiment (reduce to 100 for faster)
  n_realizations_ablation: 500  # Ablation study (reduce to 50)

scenarios:
  snr_range: [0, 5, 10, 15, 20, 25, 30]  # 7 SNR points
  elevation_angle: 10                     # Satellite elevation
  abs_bandwidth: 1.2e6                    # ABS bandwidth (Hz)
```

### Systems Compared

1. **SAT-NOMA**: Satellite only (no ABS)
2. **Original-SATCON**: Baseline (paper method)
3. **SATCON+M1**: Baseline + Gradient position optimization
4. **SATCON+M2**: Baseline + Joint pairing optimization
5. **SATCON+M3**: Baseline + Integer programming decision
6. **Proposed-Full**: All three modules enabled

---

## Output Files

### Directory Structure
```
results/
├── comparison/           # Overall results
├── figures/comparison/   # PNG figures (300 DPI)
├── data/comparison/      # NPZ and PKL data files
└── tables/comparison/    # CSV tables
```

### Generated Files

1. **Figures**:
   - `main_comparison.png`: 4-panel performance curves
   - `ablation_study.png`: Module contribution analysis

2. **Data Files**:
   - `main_YYYYMMDD_HHMMSS.npz`: Raw numerical data
   - `main_YYYYMMDD_HHMMSS.pkl`: Complete results (Python pickle)

3. **Tables**:
   - `main_performance_YYYYMMDD_HHMMSS.csv`: Performance data
   - `main_gains_YYYYMMDD_HHMMSS.csv`: Gain over baseline

---

## Performance Optimization Tips

### 1. Parallel Execution (NOT YET IMPLEMENTED)
To enable parallel processing, modify `run_comparison.py`:

```python
from multiprocessing import Pool

def simulate_parallel(args):
    system, snr, seed = args
    return system.simulate_single_realization(snr, 10, seed)

# In run_main_experiment():
with Pool(8) as p:  # 8 cores
    results = p.map(simulate_parallel, tasks)
```

**Speed-up: ~8x with 8 cores**

### 2. Reduce Realizations
For quick testing, reduce Monte Carlo realizations in `config_comparison.yaml`:
- Main experiment: 1000 → 100 (10x faster)
- Ablation study: 500 → 50 (10x faster)

### 3. Reduce SNR Points
Focus on key SNR points: `[10, 15, 20, 25]` instead of full range

### 4. Use Faster Solver
Install Gurobi (academic license free):
```bash
conda install -c gurobi gurobi
```
**Speed-up for Module 3: ~10-100x**

---

## Expected Runtime (Rough Estimates)

Hardware: Intel i7/AMD Ryzen 7, 16GB RAM

| Configuration | Systems | SNR | Realizations | Time |
|--------------|---------|-----|--------------|------|
| Quick Test | 3 | 3 | 10 | **2-3 min** ✓ |
| Small-scale | 6 | 7 | 10 | **10-15 min** |
| Medium-scale | 6 | 7 | 100 | **1-2 hours** |
| Full-scale | 6 | 7 | 1000 | **8-16 hours** |

**Note**: Module 2 (Joint Pairing) is the slowest due to local search iterations

---

## Troubleshooting

### Issue 1: Out of Memory
**Solution**: Reduce n_realizations or run systems one at a time

### Issue 2: cvxpy/GLPK errors
**Solution**: Module 3 will automatically fallback to greedy if ILP solver fails

### Issue 3: Matplotlib display issues
**Solution**: Figures are saved to disk even if display fails

### Issue 4: Slow Joint Pairing
**Solution**: Reduce `max_iterations` in `JointPairingOptimizer` from 50 to 20

---

## Interpreting Results

### Key Metrics @ SNR=20dB

From quick test (10 realizations):
- **Baseline**: 22.41 bits/s/Hz
- **Module 1 (+GradPos)**: 22.78 bits/s/Hz → **+1.65%**
- **Full System (All)**: 23.30 bits/s/Hz → **+3.99%**

Expected with 1000 realizations:
- More stable statistics
- Smaller confidence intervals
- Potentially higher/lower gains depending on scenarios

### Success Criteria

✅ **Good results**:
- Full system > Baseline by at least 2-5%
- Each module contributes positively
- Gains consistent across SNR range

⚠️ **Needs investigation**:
- Module 3 shows 0% gain (may need different scenarios)
- Gains decrease at high SNR
- High variance in results

---

## Advanced: Custom Experiments

### Add New Baseline
Edit `config_comparison.yaml`:

```yaml
baselines:
  - name: "My-Method"
    enabled: true
    module_flags:
      use_module1: true
      use_module2: false
      use_module3: true  # Custom combination
```

### Test Different Parameters
```bash
# Test different bandwidths
python -c "
import experiments.run_comparison as exp
framework = exp.ComparisonFramework()
# Modify framework.exp_config['scenarios']['abs_bandwidth']
# Run experiments
"
```

---

## Next Steps After Experiments

1. **Analyze Results**:
   - Check `results/tables/` for CSV files
   - Open figures in `results/figures/`

2. **Statistical Significance**:
   - Use saved `.pkl` files for t-tests
   - Compare variances across methods

3. **Paper Figures**:
   - Use saved PNG files (300 DPI)
   - Customize plots if needed

4. **Additional Experiments**:
   - Different user numbers (N=16, 32, 64)
   - Different elevations (E=10°, 20°, 40°)
   - Different bandwidths (Bd=0.4, 1.2, 2.0, 3.0 MHz)

---

## Questions?

- Check [src_enhanced/README.md](../src_enhanced/README.md) for module details
- Check [docs/module3_completion_summary.md](../docs/module3_completion_summary.md) for progress
- Run tests: `pytest tests/test_joint_system.py -v`

---

**Last Updated**: 2025-12-10
**Status**: ✅ Quick test validated, ready for large-scale experiments
