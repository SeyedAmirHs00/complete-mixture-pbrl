# Why real TTP / PEBBLE runs work

Paper recipe only: **max-norm** trust, no soft logits.

At **PyTorch-ish** init `rms|Delta R|≈0.25`, a **shared** reward head recovers the correct branch under a reliable majority, while **tabular** free `R_i` at the same Delta R scale does not. Stabilized (≈0) works for both; Standard/large Delta R fails for both.

config      regime                 regime_label   param  target_rms  init_rms_deltaR  mean_rho  correct_branch_rate  flipped_branch_rate
  3R1N  stabilized     Stabilized ($\approx 0$) tabular        0.00         0.000000  0.869661               1.0000               0.0000
  3R1N  stabilized     Stabilized ($\approx 0$)  shared        0.00         0.000000  0.021926               0.5125               0.4875
  3R1N pytorch_ish PyTorch-ish ($\approx 0.25$) tabular        0.25         0.249010 -0.058614               0.4375               0.5375
  3R1N pytorch_ish PyTorch-ish ($\approx 0.25$)  shared        0.25         0.248218  0.871006               0.9625               0.0250
  3R1N    standard     Standard ($\approx 1.4$) tabular        1.40         1.395139 -0.003849               0.3875               0.4250
  3R1N    standard     Standard ($\approx 1.4$)  shared        1.40         1.513627 -0.943746               0.0000               1.0000
  3R1N       large          Large ($\approx 6$) tabular        6.00         5.962355 -0.024949               0.3375               0.4500
  3R1N       large          Large ($\approx 6$)  shared        6.00         6.344526 -0.875412               0.0000               1.0000
  3R1A  stabilized     Stabilized ($\approx 0$) tabular        0.00         0.000000  0.878601               1.0000               0.0000
  3R1A  stabilized     Stabilized ($\approx 0$)  shared        0.00         0.000000  0.929931               0.9875               0.0125
  3R1A pytorch_ish PyTorch-ish ($\approx 0.25$) tabular        0.25         0.253489  0.115270               0.5750               0.4250
  3R1A pytorch_ish PyTorch-ish ($\approx 0.25$)  shared        0.25         0.245497  0.906259               0.9750               0.0250
  3R1A    standard     Standard ($\approx 1.4$) tabular        1.40         1.386833  0.048697               0.4875               0.3000
  3R1A    standard     Standard ($\approx 1.4$)  shared        1.40         1.429307 -0.669009               0.1500               0.8500
  3R1A       large          Large ($\approx 6$) tabular        6.00         6.079709 -0.030734               0.2875               0.4625
  3R1A       large          Large ($\approx 6$)  shared        6.00         6.372233 -0.717357               0.0250               0.9625
  1R3A  stabilized     Stabilized ($\approx 0$) tabular        0.00         0.000000 -0.882504               0.0000               1.0000
  1R3A  stabilized     Stabilized ($\approx 0$)  shared        0.00         0.000000 -0.864656               0.0500               0.9500
  1R3A pytorch_ish PyTorch-ish ($\approx 0.25$) tabular        0.25         0.249320  0.083724               0.5500               0.4500
  1R3A pytorch_ish PyTorch-ish ($\approx 0.25$)  shared        0.25         0.262126 -0.833592               0.0625               0.9375
  1R3A    standard     Standard ($\approx 1.4$) tabular        1.40         1.398213  0.003243               0.4625               0.3875
  1R3A    standard     Standard ($\approx 1.4$)  shared        1.40         1.452884  0.620263               0.8250               0.1750
  1R3A       large          Large ($\approx 6$) tabular        6.00         6.118756 -0.042615               0.2875               0.4625
  1R3A       large          Large ($\approx 6$)  shared        6.00         6.189143  0.777617               0.9875               0.0125
