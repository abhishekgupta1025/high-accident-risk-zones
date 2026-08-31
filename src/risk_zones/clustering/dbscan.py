# -*- coding: utf-8 -*-
"""
Network-Constrained Density-Based Spatial Clustering of Applications with Noise (DBSCAN).
Combines metric distance constraints with the 3-Tier Topological Connectivity Cascade.
"""
import math
from typing import Any, List, Optional, Tuple, Union
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
        points_coords: List[Tuple[float, float]],
        road_ids: List[Optional[Union[int, str]]],
    ) -> List[int]:
        """
        Finds all points that satisfy both:
        1. Spatial constraint: Euclidean distance <= eps
        2. Network constraint: 3-Tier topological connectivity along road network
        """
        neighbors = []
        target_coord = points_coords[target_idx]
        target_road_id = road_ids[target_idx]

        for i, neighbor_coord in enumerate(points_coords):
            if i == target_idx:
                continue

            # 1. Metric Distance Constraint Check
            distance = calculate_euclidean_distance(target_coord, neighbor_coord)
            if distance <= self.eps:
                # 2. Network Topology Constraint Check
                if self.connectivity_graph is None:
                    # Unconstrained fallback
                    neighbors.append(i)
                else:
                    neighbor_road_id = road_ids[i]
                    if self.connectivity_graph.are_connected(target_road_id, neighbor_road_id):
                        neighbors.append(i)

        return neighbors

    def expand_cluster(
        self,
        core_idx: int,
        neighbors_indices: List[int],
        cluster_label: int,
        points_coords: List[Tuple[float, float]],
        road_ids: List[Optional[Union[int, str]]],
        labels: List[int],
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

            if labels[curr_neighbor] in [-1, 0]:
                # Noise point becomes border point, or unassigned gets added
                labels[curr_neighbor] = cluster_label

                # Query neighbors of this point to check if it is also a core point
                new_neighbors = self.find_neighbors(curr_neighbor, points_coords, road_ids)

                # Core point check: requires min_pts points (itself + neighbors >= min_pts)
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

        # Extract coordinates
        points_coords = list(zip(points_gdf.geometry.x, points_gdf.geometry.y))
        road_ids = points_gdf[road_id_column].tolist() if road_id_column in points_gdf.columns else [None] * len(points_gdf)

        n_points = len(points_coords)
        labels = [0] * n_points  # 0: Unclassified, -1: Noise, >0: Cluster ID
        cluster_id = 0

        print(f"[Clustering] Starting Network-Constrained DBSCAN (eps={self.eps}m, MinPts={self.min_pts})...")
        for i in tqdm(range(n_points), desc="Clustering Incidents"):
            if labels[i] != 0:
                continue

            neighbors = self.find_neighbors(i, points_coords, road_ids)

            if len(neighbors) < (self.min_pts - 1):
                # Not a core point -> mark tentatively as Noise
                labels[i] = -1
            else:
                # Core point -> start new cluster
                cluster_id += 1
                self.expand_cluster(i, neighbors, cluster_id, points_coords, road_ids, labels)

        self.labels_ = labels
        num_clusters = len(set(labels) - {-1, 0})
        num_noise = labels.count(-1)
        print(f"[Clustering Complete] Extracted {num_clusters} high-risk clusters ({num_noise} noise points).")
        return labels
