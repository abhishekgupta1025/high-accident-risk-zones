# -*- coding: utf-8 -*-
"""
Network-Constrained Density-Based Spatial Clustering of Applications with Noise (DBSCAN).
Combines R-Tree spatial indexing, metric distance constraints, and the 3-Tier Topological Connectivity Cascade.
"""
import math
from typing import Any, Iterable, List, Optional, Sequence, Tuple, Union
from tqdm import tqdm

from ..topology.graph import RoadConnectivityGraph


def calculate_euclidean_distance(
    point1: Union[List[float], Tuple[float, float]],
    point2: Union[List[float], Tuple[float, float]],
) -> float:
    """Calculates linear 2D Euclidean distance between two points in projected Cartesian space."""
    return math.hypot(point1[0] - point2[0], point1[1] - point2[1])


class NetworkConstrainedDBSCAN:
    """
    DBSCAN clustering algorithm constrained by road network topology.
    Eliminates cross-corridor false merges across divided highways and grade separations.
    """

    def __init__(
        self,
        eps: float = 150.0,
        min_pts: int = 5,
        connectivity_graph: Optional[RoadConnectivityGraph] = None,
    ):
        """
        Args:
            eps: Search radius in meters (linear metric CRS).
            min_pts: Minimum number of incidents to form a dense core cluster.
            connectivity_graph: RoadConnectivityGraph instance enforcing 3-tier reachability.
        """
        if eps <= 0:
            raise ValueError(f"eps must be a positive number, got {eps}")
        if min_pts <= 0:
            raise ValueError(f"min_pts must be a positive integer, got {min_pts}")

        self.eps = eps
        self.min_pts = min_pts
        self.connectivity_graph = connectivity_graph
        self.labels_: List[int] = []

    def find_neighbors(
        self,
        target_idx: int,
        points_coords: Sequence[Tuple[float, float]],
        road_ids: Sequence[Optional[Union[int, str]]],
        candidate_indices: Optional[Iterable[int]] = None,
    ) -> List[int]:
        """
        Finds all points satisfying both Euclidean distance <= eps and network connectivity.
        Uses R-Tree precomputed `candidate_indices` when supplied for O(log N) query cost.
        """
        neighbors: List[int] = []
        target_coord = points_coords[target_idx]
        target_road_id = road_ids[target_idx]
        candidates = candidate_indices if candidate_indices is not None else range(len(points_coords))

        for i in candidates:
            if i == target_idx:
                continue

            if calculate_euclidean_distance(target_coord, points_coords[i]) <= self.eps:
                if self.connectivity_graph is None or self.connectivity_graph.are_connected(
                    target_road_id, road_ids[i]
                ):
                    neighbors.append(i)

        return neighbors

    def expand_cluster(
        self,
        core_idx: int,
        neighbors_indices: List[int],
        cluster_label: int,
        points_coords: Sequence[Tuple[float, float]],
        road_ids: Sequence[Optional[Union[int, str]]],
        labels: List[int],
        spatial_candidates: Optional[Sequence[List[int]]] = None,
    ) -> None:
        """Expands the cluster from a core point using Breadth-First Search (BFS) queue."""
        labels[core_idx] = cluster_label
        queue = list(neighbors_indices)
        processed_or_queued = set(neighbors_indices)
        processed_or_queued.add(core_idx)

        queue_idx = 0
        while queue_idx < len(queue):
            curr_neighbor = queue[queue_idx]
            queue_idx += 1

            if labels[curr_neighbor] in (-1, 0):
                labels[curr_neighbor] = cluster_label

                candidates = (
                    spatial_candidates[curr_neighbor] if spatial_candidates is not None else None
                )
                new_neighbors = self.find_neighbors(
                    curr_neighbor, points_coords, road_ids, candidate_indices=candidates
                )

                if len(new_neighbors) >= (self.min_pts - 1):
                    for n_idx in new_neighbors:
                        if n_idx not in processed_or_queued:
                            processed_or_queued.add(n_idx)
                            queue.append(n_idx)

    def fit_predict(
        self,
        points_gdf: Any,
        road_id_column: str = "road_uid",
    ) -> List[int]:
        """
        Executes network-constrained DBSCAN clustering on the provided GeoDataFrame.

        Args:
            points_gdf: GeoDataFrame containing accident incident points.
            road_id_column: Column name holding road segment IDs or group IDs.

        Returns:
            List of cluster labels (-1 for Noise, 1..K for Cluster IDs).
        """
        if points_gdf is None or len(points_gdf) == 0:
            raise ValueError("Input points dataset is empty or None.")

        points_coords = list(zip(points_gdf.geometry.x, points_gdf.geometry.y))
        road_ids = (
            points_gdf[road_id_column].tolist()
            if road_id_column in points_gdf.columns
            else [None] * len(points_gdf)
        )

        n_points = len(points_coords)
        spatial_candidates: Optional[List[List[int]]] = None
        if hasattr(points_gdf, "sindex"):
            spatial_candidates = [[] for _ in range(n_points)]
            left_idx, right_idx = points_gdf.sindex.query(
                points_gdf.geometry,
                predicate="dwithin",
                distance=self.eps,
            )
            for i, j in zip(left_idx, right_idx):
                if i != j:
                    spatial_candidates[int(i)].append(int(j))

        labels = [0] * n_points  # 0: Unclassified, -1: Noise, >0: Cluster ID
        cluster_id = 0

        print(f"[Clustering] Starting Network-Constrained DBSCAN (eps={self.eps}m, MinPts={self.min_pts})...")
        for i in tqdm(range(n_points), desc="Clustering Incidents"):
            if labels[i] != 0:
                continue

            candidates = spatial_candidates[i] if spatial_candidates is not None else None
            neighbors = self.find_neighbors(
                i, points_coords, road_ids, candidate_indices=candidates
            )

            if len(neighbors) < (self.min_pts - 1):
                labels[i] = -1
            else:
                cluster_id += 1
                self.expand_cluster(
                    i,
                    neighbors,
                    cluster_id,
                    points_coords,
                    road_ids,
                    labels,
                    spatial_candidates=spatial_candidates,
                )

        self.labels_ = labels
        num_clusters = len(set(labels) - {-1, 0})
        num_noise = labels.count(-1)
        print(f"[Clustering Complete] Extracted {num_clusters} high-risk clusters ({num_noise} noise points).")
        return labels
