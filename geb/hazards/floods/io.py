import geopandas as gpd


def export_rivers(model_root, segments, postfix=""):
    segments.to_parquet(model_root / f"segments{postfix}.geoparquet")


def import_rivers(model_root, postfix=""):
    segments = gpd.read_parquet(model_root / f"segments{postfix}.geoparquet")
    return segments


def export_nodes(model_root, nodes):
    nodes.to_parquet(model_root / "nodes.geoparquet")


def import_nodes(model_root):
    nodes = gpd.read_parquet(model_root / "nodes.geoparquet")
    return nodes
