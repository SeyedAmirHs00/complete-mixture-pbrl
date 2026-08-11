PyTorch-default reward MLP initialization scale (Monte Carlo)
architecture: d_in -> (128 LReLU) x3 -> 1 Tanh
d_in=[17, 24, 39, 78], seeds=100, n_traj=256
overall mean std(r) ≈ 0.025764  (paper ≈ 0.025)
overall mean std(Δr) ≈ 0.036275
extrapolated std(ΔR), T=50: 0.257637
σ(std(ΔR)) ≈ 0.564055  (paper T=50 ≈ 0.56)
runtime_sec=0.60, torch=1.4.0

Files:
  per_din_summary.csv
  per_seed.csv
  summary.json
