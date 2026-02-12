# Flood evaluation

Evaluate simulated flood extents by comparing them with satellite observations or other flood mapping data.

## Basic usage

```python
model.evaluate.hydrology.evaluate_hydrodynamics(run_name="default")
```

This compares modeled flood depths and extents against observed flood maps, calculating metrics like hit rate, false alarm ratio, and critical success index to assess how well the model captures flood events.

