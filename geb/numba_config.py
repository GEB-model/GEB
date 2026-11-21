"""Numba settings shared by all jitted functions in GEB."""

# Whether numba writes compiled functions to disk and reuses them in later processes,
# which skips compilation at the start of every run. Numba keys its cache on the source
# file of each jitted function only, so a change in a constant or in an inlined function
# from another file is not seen and stale compiled code runs silently: set to False
# while developing such code.
NUMBA_CACHE: bool = True
