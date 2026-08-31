# -*- coding: utf-8 -*-
"""
R-Tree Spatial Indexing utilities for accelerated road segment and corridor intersection queries.
"""
from typing import Dict, Optional, Set
import geopandas as gpd
from tqdm import tqdm


def precompute_segment_intersections(
    road_gdf: gpd.GeoDataFrame,
    intersection_tolerance: float = 0.0,
) -> Optional[Dict[int, Set[int]]]:
    """
    Pre-computes topological road segment intersections using GEOS R-Tree spatial indexing.
    Reduces pairwise intersection checks from O(M^2) to O(M log M).

    Args:
        road_gdf: Road network with 'road_uid' in projected linear CRS.
        intersection_tolerance: Optional geometric buffer (meters) to bridge GIS digitization micro-gaps.

    Returns:
        Dictionary mapping road_uid -> set of intersecting road_uids.
    """
    if road_gdf is None or road_gdf.empty or "road_uid" not in road_gdf.columns:
        raise ValueError("Invalid road GeoDataFrame for intersection precomputation.")

    print("[Spatial Index] Building R-Tree spatial index for road network...")
    # Accessing .sindex builds or verifies the spatial index
    _ = road_gdf.sindex

    segment_intersections: Dict[int, Set[int]] = {uid: set() for uid in road_gdf["road_uid"]}

    # Query bounding box candidate intersections
    left_indices, right_indices = road_gdf.sindex.query(road_gdf.geometry, predicate="intersects")
    print(f"[Spatial Index] Found {len(left_indices)} candidate intersection pairs.")

    processed_pairs = set()
    intersections_count = 0

    for i, j in tqdm(zip(left_indices, right_indices), total=len(left_indices), desc="Checking Intersections"):
        if i == j:
            continue

        road1_uid = road_gdf["road_uid"].iloc[i]
        road2_uid = road_gdf["road_uid"].iloc[j]

        pair_key = tuple(sorted((road1_uid, road2_uid)))
        if pair_key in processed_pairs:
            continue

        geom1 = road_gdf.geometry.iloc[i]
        geom2 = road_gdf.geometry.iloc[j]

        geometries_intersect = False
        if geom1 is not None and geom2 is not None and geom1.is_valid and geom2.is_valid:
            if geom1.intersects(geom2):
                geometries_intersect = True
            elif intersection_tolerance > 0.0:
                buffered_geom1 = geom1.buffer(intersection_tolerance, resolution=2)
                if buffered_geom1 and buffered_geom1.is_valid and not buffered_geom1.is_empty:
                    if buffered_geom1.intersects(geom2):
                        geometries_intersect = True

        if geometries_intersect:
            segment_intersections[road1_uid].add(road2_uid)
            segment_intersections[road2_uid].add(road1_uid)
            processed_pairs.add(pair_key)
            intersections_count += 1

    print(f"[Spatial Index Complete] Verified {intersections_count} intersecting road segment pairs.")
    return segment_intersections


def precompute_group_intersections(
    road_gdf: gpd.GeoDataFrame,
    road_group_map: Dict[int, int],
) -> Optional[Dict[int, Set[int]]]:
    """
    Pre-computes topological intersections between aggregated road groups.

    Args:
        road_gdf: Road network GeoDataFrame.
        road_group_map: Mapping of road_uid -> group_id.

    Returns:
        Dictionary mapping group_id -> set of intersecting group_ids.
    """
    unique_groups = set(road_group_map.values())
    group_intersections: Dict[int, Set[int]] = {gid: set() for gid in unique_groups}

    left_indices, right_indices = road_gdf.sindex.query(road_gdf.geometry, predicate="intersects")
    processed_pairs = set()

    for i, j in tqdm(zip(left_indices, right_indices), total=len(left_indices), desc="Finding Group Intersections"):
        if i == j:
            continue

        road1_uid = road_gdf["road_uid"].iloc[i]
        road2_uid = road_gdf["road_uid"].iloc[j]
        group1_id = road_group_map.get(road1_uid)
        group2_id = road_group_map.get(road2_uid)

        if group1_id is not None and group2_id is not None and group1_id != group2_id:
            pair = tuple(sorted((group1_id, group2_id)))
            if pair not in processed_pairs:
                if road_gdf.geometry.iloc[i].intersects(road_gdf.geometry.iloc[j]):
                    group_intersections[group1_id].add(group2_id)
                    group_intersections[group2_id].add(group1_id)
                    processed_pairs.add(pair)

    return group_intersections
