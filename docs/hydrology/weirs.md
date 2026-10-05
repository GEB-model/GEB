# Weirs and gates

A weir is a barrier across a river that holds back water. Water flows over
its crest when the upstream water level is high enough. GEB represents it
on a river link, where it changes flow and upstream water depth.

GEB uses Global Dam Watch (GDW) and AMBER data to place these river barriers.
Some structures have gates, which open and close based on upstream water depth.

## Setup

Run `setup_weirs` after `setup_waterbodies` in `build.yml`:

```yaml
setup_waterbodies: {}
setup_weirs: {}
```

## Placement and heights
Barriers placed by `setup_weirs` must be close enough to a valid river line (within 250m distance).
Grid pixels already represented by a waterbody are skipped.

We represent fixed and gated barriers. Fixed barriers let water pass above their crest. Gated barriers open at high upstream
water levels and close at lower levels. By default, the opening and closing
levels are 90% and 70% of the crest height.

Barrier heights are used in metres (if provided). If height is missing, dams and
lake-control dams use bankfull river depth + 1 m; other barriers use half the
bankfull depth. Bankfull depth is the river depth when water reaches the banks.

Set these limits in `model.yml`:

```yaml
hydrology:
  routing:
    gate_opening_level_fraction: 0.9
    gate_closing_level_fraction: 0.7
```

## Check the result

Detailed barrier locations and reasons why some are NOT included is saved in
`routing/barriers`. Each run writes `output/<run>/weir_heights.csv` with
barrier heights and gate settings.

See [Waterbody setup](../getting_started/build/waterbodies.md) for lakes and
reservoirs.
