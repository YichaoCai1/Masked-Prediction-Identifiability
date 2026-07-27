"""Experiments for *On the Identifiability of Masked Prediction:
Mode Blindness and Mask Schedules*.

Package layout follows the implementation specification
(``documents/experiment-spec.md``, section 6):

``laws``       weight functions (Law P = two-mode product, Law C = Curie-Weiss)
``core``       exact log-domain machinery: ``log h``, ``ell``, stable binary KL
``estimands``  D, Delta, U, F, rho and the boosted-schedule linearity
``fits``       window OLS, recovery-radius bisection, curvature identity
``e1``         E1 driver -- mode blindness
``e2``         E2 driver -- low-visibility intervention
``realdata``   E3 driver -- residual mode MMSE on labelled corpora
``figures``    main figure M and appendix figures A1-A8
``tables``     main table M
"""

__version__ = "1.0.0"
