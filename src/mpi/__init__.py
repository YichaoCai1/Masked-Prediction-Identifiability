"""Experiments for *An Identifiability Theory of Masked Prediction:
Mode Blindness and Mask Schedules*.

Study modules are named for the scientific question they answer:

``mode_blindness``                 exact blindness and sensitivity scaling
``low_visibility_intervention``    curvature and recovery under mask schedules
``mode_weight_optimization``       direct scalar optimization inside q_lambda
``corpus_mode_uncertainty``        residual mode MMSE on labelled corpora
``low_visibility_sampling_cost``   finite-sample cost of the intervention

Supporting modules:

``laws``       weight functions (Law P = two-mode product, Law C = Curie-Weiss)
``core``       exact log-domain machinery: ``log h``, ``ell``, stable binary KL
``estimands``  D, Delta, U, F, rho and the boosted-schedule linearity
``fits``       window OLS, recovery-radius bisection, curvature identity
``brute``      explicit 2^N reference implementation (tests only)
``highprec``   mpmath at 50 digits (T3 and the numerical audit only)
``config``     grids, run matrix, closed-form predicted constants
``tableio``    Parquet/CSV writing with provenance stamping
``style``      colour and mark decisions for every figure
``figures``    semantic figure/table builders from frozen result artifacts
"""

__version__ = "1.0.0"
