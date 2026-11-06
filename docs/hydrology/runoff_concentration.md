# Runoff concentration

Surface runoff travels from each hillslope to the river through a rolling hourly buffer. Its triangular response is calculated from hillslope length, slope, land-cover retardance and `parameters.runoff_concentration_factor` using the Kerby travel-time relation. Groundwater baseflow is distributed evenly over the day's 24 hours.

Interflow enters the river immediately by default. To represent a shallow subsurface travel time, set:

```yaml
hydrology:
  routing:
    interflow_peak_hours: 48.0
```

The value is the **peak** of a triangular response in hours, rather than a fixed time shift. Zero preserves immediate interflow delivery. Positive values route interflow through the same accounted buffer as surface runoff, preserving volume and carrying undelivered water into subsequent days and checkpoints. The buffer expands to include the full response to the final hour of each day. A 48-hour peak uses a seven-day buffer. Buffer memory grows with both the response duration and the number of grid cells.

Run a fresh spinup when enabling or changing the travel time. A checkpoint with incompatible buffer dimensions is rejected rather than silently losing delayed water. Existing fully expanded configuration files that do not inherit the defaults must add `interflow_peak_hours: 0.0` to retain immediate delivery.

Travel time changes discharge timing and variability; it cannot remove an excessive seasonal water volume by itself. The regional trial below is an experiment in transferring catchment-calibrated settings, not a calibration validated for every European basin.

## Europe_003 trial

The packaged overlay is `geb/examples/regional_trials/model_lithuania_trial.yml`. Copy it beside your **current, version-compatible** Europe_003 `model.yml`. It inherits that file's region, input paths, run dates and reports, while using separate output, preprocessing and checkpoint directories. Your baseline configuration and output remain available for comparison.

```bash
# Set these to your code checkout and active Europe_003 model directory.
GEB_REPOSITORY=/path/to/GEB
EUROPE_MODEL=/path/to/Europe_003/base
cp "$GEB_REPOSITORY/geb/examples/regional_trials/model_lithuania_trial.yml" "$EUROPE_MODEL/"
cd "$EUROPE_MODEL"
PYTHONPATH="$GEB_REPOSITORY" "$GEB_REPOSITORY/.venv/bin/python" -m geb.cli spinup --config model_lithuania_trial.yml
PYTHONPATH="$GEB_REPOSITORY" "$GEB_REPOSITORY/.venv/bin/python" -m geb.cli run --config model_lithuania_trial.yml
PYTHONPATH="$GEB_REPOSITORY" "$GEB_REPOSITORY/.venv/bin/python" -m geb.cli evaluate hydrology.evaluate_discharge --config model_lithuania_trial.yml --run-name default
```

Use the same commands in your usual Slurm scripts and retain your existing CPU allocation. This patch does not update old regional input datasets; use your normal update/build workflow if their version is incompatible with the running model. No new spatial input files are required by these changes.

The overlay reproduces the strongest Guntauninkai setup that retained modeled irrigation geography: consistent infiltration, interflow multiplier **3000**, vegetation demand multiplier **1.25**, and a **48-hour** interflow peak. It reached daily **KGE 0.6915**, correlation **0.7093**, mean ratio **1.0311** and variability ratio **0.9015** over 2000–2010. Summer mean flow was still approximately **3.05 times observed**. The numerical maximum of **0.6946** required allowing all growing crops to access irrigation; that unverified land-use assumption is not included in the regional overlay. The whole-profile wetness gate reduced summer bias but lowered total KGE to **0.6534**, so it is also excluded.

Interflow 3000 and vegetation factor 1.25 are catchment-calibrated values, not new European defaults. Evaluate all gauges over a common period, including seasonal bias and performance in previously successful basins. Compare the corrected code with your regional parameters and zero interflow delay as well, to distinguish the infiltration correction from transferring catchment-specific calibration. Preserve existing irrigation extent and return fractions while making that comparison.

## Code

::: geb.hydrology.runoff_concentration
