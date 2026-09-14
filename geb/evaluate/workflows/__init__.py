"""Helpers used by GEB's evaluation commands.

Hydrology's plot_* commands load scores and select comparisons; discharge_plots
create_* functions draw figures from prepared tables. discharge_metrics owns
metric calculations and their output schema; dashboard owns map preparation.
Across the workflows, load_* reads data, calculate_* computes metrics, build_*
prepares chart data, and write_* saves dashboard output. Browser assets are
packaged with the module.
"""
