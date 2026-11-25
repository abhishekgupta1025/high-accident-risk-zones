# -*- coding: utf-8 -*-
"""
Topological Graph & Road Network Processing Subpackage.
"""
from .dsu import DisjointSetUnion, group_nearby_roads
from .graph import RoadConnectivityGraph

try:
    from .spatial_index import (
        snap_points_to_roads,
        precompute_segment_intersections,
        precompute_group_intersections,
    )
except ImportError:
    pass

__all__ = [
    "DisjointSetUnion",
    "group_nearby_roads",
    "RoadConnectivityGraph",
]
