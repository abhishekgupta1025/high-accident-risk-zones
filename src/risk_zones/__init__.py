# -*- coding: utf-8 -*-
"""
High-Accident Risk Zone Identification via Network-Constrained DBSCAN.
Package initialization and public API with graceful GIS dependency handling.
"""
from .config import PipelineConfig
from .topology.dsu import DisjointSetUnion, group_nearby_roads
from .topology.graph import RoadConnectivityGraph

# Lazy / Guarded imports for GIS-heavy components
try:
    from .io.loader import load_point_data, load_road_data
    from .io.reprojector import reproject_geodataframe, ensure_projected_crs
    from .topology.spatial_index import precompute_segment_intersections, precompute_group_intersections
    from .clustering.dbscan import NetworkConstrainedDBSCAN, calculate_euclidean_distance
    from .clustering.metrics import ClusterMetricsCalculator
    from .visualization.web_map import InteractiveMapBuilder
    from .visualization.static_plot import StaticPlotBuilder
except ImportError:
    pass

__version__ = "2.0.0"
__author__ = "Abhishek Gupta"

__all__ = [
    "PipelineConfig",
    "DisjointSetUnion",
    "group_nearby_roads",
    "RoadConnectivityGraph",
]
