# -*- coding: utf-8 -*-
"""
R-Tree Spatial Indexing utilities for point-to-road snapping and accelerated intersection queries.
"""
from typing import Dict, List, Optional, Set, Tuple
import geopandas as gpd
import pandas as pd
from tqdm import tqdm


def snap_points_to_roads(
    points_gdf: gpd.GeoDataFrame,
    road_gdf: gpd.GeoDataFrame,
    max_snap_distance: Optional[float] = None,
    road_id_column: str = "road_uid",
) -> gpd.GeoDataFrame:
    """
    Associates incident points with their nearest road segment via R-Tree `sjoin_nearest`.
    Deduplicates equidistant ties (keeping the first match) and assigns `-999` to unmatched points.

    Args:
        points_gdf: Point GeoDataFrame in a projected metric CRS.
        road_gdf: Road network GeoDataFrame containing `road_id_column`.
        max_snap_distance: Optional maximum snap radius in meters (None searches all segments).
        road_id_column: Target column name for the snapped road identifier.

    Returns:
        Copy of `points_gdf` populated with integer `road_id_column` and `dist_to_road`.
    """
    if points_gdf is None or points_gdf.empty:
        raise ValueError("Invalid points GeoDataFrame for road snapping (None or empty).")
    if road_gdf is None or road_gdf.empty or road_id_column not in road_gdf.columns:
        raise ValueError(f"Invalid road GeoDataFrame for snapping (missing '{road_id_column}' or empty).")

    valid_mask = (
        road_gdf.geometry.notna()
        & road_gdf.geometry.is_valid
        & ~road_gdf.geometry.is_empty
    )
    valid_roads = road_gdf[valid_mask]
    if valid_roads.empty:
        raise ValueError("No valid road geometries available for spatial snapping.")

    # Drop any stale join columns so sjoin_nearest assigns fresh road associations cleanly.
    stale_cols = [c for c in (road_id_column, "dist_to_road", "index_road", "index_right") if c in points_gdf.columns]
    points_clean = points_gdf.drop(columns=stale_cols).reset_index(drop=True)

    joined = gpd.sjoin_nearest(
        points_clean,
        valid_roads[["geometry", road_id_column]],
        how="left",
        max_distance=max_snap_distance,
        distance_col="dist_to_road",
        rsuffix="road",
    )

    if joined.index.duplicated().any():
        joined = joined[~joined.index.duplicated(keep="first")].copy()

    joined = joined.sort_index()
    joined[road_id_column] = (
        pd.to_numeric(joined[road_id_column], errors="coerce")
        .fillna(-999)
        .astype(int)
    )
    if "index_road" in joined.columns:
        joined = joined.drop(columns=["index_road"])

    matched = int((joined[road_id_column] != -999).sum())
    print(f"[Road Snapping] Mapped {matched}/{len(joined)} incidents to nearest road segments.")
    return joined


def _find_intersecting_segment_pairs(
    road_gdf: gpd.GeoDataFrame,
    intersection_tolerance: float = 0.0,
) -> List[Tuple[int, int]]:
    """
    Queries the R-Tree spatial index for road segment pairs that intersect directly or within `intersection_tolerance`.
    Uses `dwithin` when tolerance > 0 so micro-gaps are captured without false negatives.
    """
    _ = road_gdf.sindex

    if intersection_tolerance > 0.0:
        left_indices, right_indices = road_gdf.sindex.query(
            road_gdf.geometry,
            predicate="dwithin",
            distance=intersection_tolerance,
        )
    else:
        left_indices, right_indices = road_gdf.sindex.query(
            road_gdf.geometry,
            predicate="intersects",
        )

    geoms = road_gdf.geometry.tolist()
    pairs: List[Tuple[int, int]] = []
    for i, j in zip(left_indices, right_indices):
        if i >= j:
            continue
        geom1 = geoms[i]
        geom2 = geoms[j]
        if geom1 is not None and geom2 is not None and geom1.is_valid and geom2.is_valid:
            pairs.append((int(i), int(j)))

    return pairs


def precompute_segment_intersections(
    road_gdf: gpd.GeoDataFrame,
    intersection_tolerance: float = 0.0,
) -> Dict[int, Set[int]]:
    """
    Pre-computes topological road segment intersections using GEOS R-Tree spatial indexing.
    Reduces pairwise intersection checks from O(M^2) to O(M log M).

    Args:
        road_gdf: Road network with 'road_uid' in projected linear CRS.
        intersection_tolerance: Optional distance tolerance (meters) to bridge digitization micro-gaps.

    Returns:
        Dictionary mapping road_uid -> set of intersecting road_uids.
    """
    if road_gdf is None or road_gdf.empty or "road_uid" not in road_gdf.columns:
        raise ValueError("Invalid road GeoDataFrame for intersection precomputation.")

    print("[Spatial Index] Building R-Tree spatial index for road network...")
    road_uids = road_gdf["road_uid"].tolist()
    segment_intersections: Dict[int, Set[int]] = {uid: set() for uid in road_uids}

    pairs = _find_intersecting_segment_pairs(road_gdf, intersection_tolerance=intersection_tolerance)
    print(f"[Spatial Index] Found {len(pairs)} unique candidate intersection pairs.")

    intersections_count = 0
    for i, j in tqdm(pairs, desc="Checking Intersections"):
        road1_uid = road_uids[i]
        road2_uid = road_uids[j]
        if road1_uid == road2_uid:
            continue
        if road2_uid not in segment_intersections[road1_uid]:
            segment_intersections[road1_uid].add(road2_uid)
            segment_intersections[road2_uid].add(road1_uid)
            intersections_count += 1

    print(f"[Spatial Index Complete] Verified {intersections_count} intersecting road segment pairs.")
    return segment_intersections


def precompute_group_intersections(
    road_gdf: gpd.GeoDataFrame,
    road_group_map: Dict[int, int],
    intersection_tolerance: float = 0.0,
) -> Dict[int, Set[int]]:
    """
    Pre-computes topological intersections between aggregated DSU road corridor groups.

    Args:
        road_gdf: Road network GeoDataFrame with 'road_uid'.
        road_group_map: Mapping of road_uid -> group_id.
        intersection_tolerance: Optional distance tolerance (meters) to bridge micro-gaps.

    Returns:
        Dictionary mapping group_id -> set of intersecting group_ids.
    """
    if road_gdf is None or road_gdf.empty or "road_uid" not in road_gdf.columns:
        raise ValueError("Invalid road GeoDataFrame for group intersection precomputation.")

    road_uids = road_gdf["road_uid"].tolist()
    unique_groups = set(road_group_map.values())
    group_intersections: Dict[int, Set[int]] = {gid: set() for gid in unique_groups}

    pairs = _find_intersecting_segment_pairs(road_gdf, intersection_tolerance=intersection_tolerance)

    for i, j in tqdm(pairs, desc="Finding Group Intersections"):
        group1_id = road_group_map.get(road_uids[i])
        group2_id = road_group_map.get(road_uids[j])
        if group1_id is not None and group2_id is not None and group1_id != group2_id:
            group_intersections[group1_id].add(group2_id)
            group_intersections[group2_id].add(group1_id)

    return group_intersections
