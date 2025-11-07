"""Sensitivity analysis rules for GEB models.

This module contains rules for performing sensitivity analysis:
- Parameter sampling (Sobol, Latin Hypercube, etc.)
- Batch model execution
- Sensitivity index computation
"""

# Sensitivity-specific configuration
N_SAMPLES = config.get("N_SAMPLES", 1000)
SAMPLING_METHOD = config.get("SAMPLING_METHOD", "sobol")

# TODO: Implement sensitivity analysis rules
# rule generate_sensitivity_samples:
#     ...

# rule run_sensitivity_sample:
#     ...

# rule compute_sensitivity_indices:
#     ...
