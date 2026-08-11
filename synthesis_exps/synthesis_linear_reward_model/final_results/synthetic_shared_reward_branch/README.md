# Shared reward vs tabular Standard

Supports real mixture-PBRL: a **shared** per-step head with soft `tanh(alpha)` (or zero-head Stabilized) recovers the correct branch under 3R1A even when init `|Delta R|` is large; free tabular `R_i` does not.

config           method                               label  mean_rho  correct_branch_rate  flipped_branch_rate  mean_rms_deltaR_init  mean_coef_rel  mean_coef_adv
  3R1A  tabular_maxnorm       Tabular + max-norm (Standard)  0.107498                 0.58                 0.42              1.422855       0.105110      -0.279844
  3R1A     tabular_soft              Tabular + soft tanh(α)  0.039010                 0.52                 0.48              1.406795       0.032329      -0.021084
  3R1A   shared_maxnorm     Shared head + max-norm (*_over) -0.583694                 0.18                 0.82              6.115836      -0.503597       0.479760
  3R1A      shared_soft          Shared head + soft tanh(α)  0.465451                 0.74                 0.26              6.237559       0.110711      -0.119189
  3R1A shared_zero_head Shared head, zero init (Stabilized)  0.478345                 0.80                 0.20              0.000000       0.271582      -0.430190
  3R1N  tabular_maxnorm       Tabular + max-norm (Standard) -0.179094                 0.32                 0.62              1.401262      -0.171963            NaN
  3R1N     tabular_soft              Tabular + soft tanh(α) -0.079798                 0.44                 0.52              1.379886      -0.059904            NaN
  3R1N   shared_maxnorm     Shared head + max-norm (*_over) -0.447656                 0.24                 0.72              6.204585      -0.412049            NaN
  3R1N      shared_soft          Shared head + soft tanh(α)  0.347955                 0.68                 0.32              6.180866       0.081806            NaN
  3R1N shared_zero_head Shared head, zero init (Stabilized)  0.029679                 0.52                 0.48              0.000000       0.091098            NaN
  1R3A  tabular_maxnorm       Tabular + max-norm (Standard)  0.029507                 0.52                 0.48              1.358822      -0.070934      -0.063777
  1R3A     tabular_soft              Tabular + soft tanh(α) -0.147882                 0.42                 0.58              1.411091      -0.079850       0.109387
  1R3A   shared_maxnorm     Shared head + max-norm (*_over)  0.593614                 0.82                 0.18              6.105121       0.430642      -0.561582
  1R3A      shared_soft          Shared head + soft tanh(α) -0.076326                 0.46                 0.54              6.108479      -0.016159       0.025027
  1R3A shared_zero_head Shared head, zero init (Stabilized) -0.500593                 0.20                 0.80              0.000000      -0.380337       0.287350

## Takeaways

1. `shared_soft`: closest to non-`*_over` mixture models (`logits *= tanh(alpha)`, `init_trust=0.01`) + shared `r_theta`.
2. `shared_zero_head`: Stabilized analogue for trajectory sums.
3. `shared_maxnorm` with large init `|Delta R|` remains hard (same regime as tabular Standard) — so `train_PEBBLE_mixture.py` (`*_over`) is *not* explained by sharing alone; it needs soft early logits, small init `|Delta R|`, or other optimization effects.
4. `1R3A`: majority-following methods prefer the flipped branch.
