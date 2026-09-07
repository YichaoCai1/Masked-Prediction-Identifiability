# Direct mode-weight optimization results

## Protocol status

- Overall status: **PASS**.
- Selected N=63 from exact calibration only; fallback used: False.
- One common learning rate eta=0.1 was selected once and held fixed.
- Low-visibility schedules increased the exact a-coordinate curvature: True.
- Population GD: 36/42 runs reached half error within budget.
- SGD: 60/70 runs had sustained half-error crossings; non-crossings remain right-censored.
- Maximum recorded population risk-identity discrepancy: 6.939e-18.

## Acceptance checks

- PASS — exact-only cell selection passed its strict rule.
- PASS — a-coordinate risk identity agreed within 1e-10.
- PASS — every retained low-visibility cell increased curvature.
- PASS — a non-full-mask primary intervention was retained.
- PASS — the common learning rate passed the highest-curvature stability sweep.

## Failures and limitations

- No protocol check failed in this run.

## Supported interpretation

The run isolates optimisation along the coherent mode-reweighting family q_lambda. High visibility supplies weak curvature, while retained low-visibility schedules supply more curvature and are compared through half-times versus inverse curvature. Censored cells are not replaced by their final step.

This study does not claim that arbitrary masked neural networks learn wrong global frequencies, that non-full masks universally identify a joint law, or that fixed-time endpoint slopes test the recovery-radius theorem.
