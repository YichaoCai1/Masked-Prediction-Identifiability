"""Experiments for *An Identifiability Theory of Masked Prediction:
Mode Blindness and Mask Schedules*.

Module names predate the paper's final ordering and are kept so that the
committed artifacts and their ``config_hash`` values stay valid.  The mapping
is:

``e1``         paper experiment 1 -- mode blindness
``e2``         paper experiment 2 -- low-visibility intervention
``e5``         paper experiment 3 -- does the prediction survive SGD?  [torch]
``realdata``   paper experiment 4 -- residual mode MMSE on labelled corpora
``e4``         paper experiment 5 -- what the intervention costs

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
``figures``    main figure M, table M, appendix figures A1-A10
"""

__version__ = "1.0.0"
