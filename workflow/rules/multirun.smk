"""Multi-run rules for GEB models.

This module contains rules for running multiple model configurations:
- Scenario generation
- Parallel model execution
- Results aggregation and comparison
"""

# Multi-run specific configuration
SCENARIOS = config.get("SCENARIOS", ["baseline", "scenario1", "scenario2"])

# TODO: Implement multi-run rules
# rule setup_scenario:
#     ...

# rule run_scenario:
#     ...

# rule aggregate_scenarios:
#     ...
