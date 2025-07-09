import geopandas as gpd


def export_rivers(model_root, segments, postfix="") -> None:
    segments.to_parquet(model_root / f"segments{postfix}.geoparquet")


def import_rivers(model_root, postfix="") -> gpd.GeoDataFrame:
    return gpd.read_parquet(model_root / f"segments{postfix}.geoparquet")


def export_nodes(model_root, nodes) -> None:
    nodes.to_parquet(model_root / "nodes.geoparquet")


def import_nodes(model_root) -> gpd.GeoDataFrame:
    return gpd.read_parquet(model_root / "nodes.geoparquet")
