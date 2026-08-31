# -*- coding: utf-8 -*-
"""
Disjoint Set Union (DSU / Union-Find) data structure with Path Compression and Union by Rank.
Used for grouping parallel divided corridors and contiguous road clusters.
"""
from typing import Any, Dict, Hashable, Optional


class DisjointSetUnion:
    """
    High-performance Disjoint Set Union (DSU) implementation with Path Compression
    and Union by Rank, providing O(alpha(N)) near-constant time operations.
    """

    def __init__(self, elements=None):
        self.parent: Dict[Hashable, Hashable] = {}
        self.rank: Dict[Hashable, int] = {}
        if elements:
            for elem in elements:
                self.make_set(elem)

    def make_set(self, item: Hashable) -> None:
        """Initializes a new single-element set."""
        if item not in self.parent:
            self.parent[item] = item
            self.rank[item] = 0

    def find(self, item: Hashable) -> Hashable:
        """Finds the root canonical representative of the set containing `item` with path compression."""
        if item not in self.parent:
            self.make_set(item)
            return item

        if self.parent[item] == item:
            return item

        # Path compression step
        self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, item1: Hashable, item2: Hashable) -> bool:
        """
        Unites the disjoint sets containing item1 and item2 using union by rank.
        Returns True if a merge occurred, False if they were already in the same set.
        """
        root1 = self.find(item1)
        root2 = self.find(item2)

        if root1 == root2:
            return False

        # Union by rank optimization
        if self.rank[root1] < self.rank[root2]:
            self.parent[root1] = root2
        elif self.rank[root1] > self.rank[root2]:
            self.parent[root2] = root1
        else:
            self.parent[root2] = root1
            self.rank[root1] += 1

        return True

    def get_all_sets(self) -> Dict[Hashable, Hashable]:
        """Flattens all parent references and returns a mapping from item -> root representative."""
        return {item: self.find(item) for item in self.parent}


def group_nearby_roads(road_gdf, threshold: float) -> Optional[Dict[int, int]]:
    """
    Groups road segments within a proximity threshold (meters) using DSU and R-Tree spatial indexing.
    Assigns a unique group_id to parallel lanes, dual carriageways, and contiguous corridors.

    Args:
        road_gdf: Road network with 'road_uid' in a projected linear CRS.
        threshold: Max distance in meters to merge into the same logical road group.

    Returns:
        Dictionary mapping road_uid -> group_id.
    """
    import geopandas as gpd
    from tqdm import tqdm

    if road_gdf is None or road_gdf.empty or "road_uid" not in road_gdf.columns:
        raise ValueError("Invalid road GeoDataFrame for grouping (missing road_uid or empty).")

    print(f"[Road Grouping] Grouping segments within {threshold}m threshold...")
    dsu = DisjointSetUnion(road_gdf["road_uid"])

    # Generate buffer geometries for spatial collision query
    buffers = road_gdf.geometry.buffer(threshold / 2.0, resolution=3)

    # Use R-Tree spatial index to find candidate overlapping buffers
    possible_joins = buffers.sindex.query(buffers, predicate="intersects")
    left_indices, right_indices = possible_joins

    merged_count = 0
    for i, j in tqdm(zip(left_indices, right_indices), total=len(left_indices), desc="Merging Road Groups"):
        if i >= j:
            continue

        road_uid1 = road_gdf["road_uid"].iloc[i]
        road_uid2 = road_gdf["road_uid"].iloc[j]

        if road_uid1 != road_uid2 and dsu.union(road_uid1, road_uid2):
            merged_count += 1

    road_group_map = dsu.get_all_sets()
    num_groups = len(set(road_group_map.values()))
    print(f"[Road Grouping Complete] Merged {merged_count} segments into {num_groups} distinct road groups.")
    return road_group_map
