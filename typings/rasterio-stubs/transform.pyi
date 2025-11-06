class Affine:
    """A minimal stub of rasterio.transform.Affine for typing."""

    a: float
    b: float
    c: float
    d: float
    e: float
    f: float

    def __init__(
        self, a: float, b: float, c: float, d: float, e: float, f: float
    ) -> None: ...
