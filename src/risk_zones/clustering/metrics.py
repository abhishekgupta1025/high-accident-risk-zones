# -*- coding: utf-8 -*-
"""
Cluster metrics, centroids, and convex hull polygon area calculations.
"""
from typing import Dict, List, Optional, Tuple
import geopandas as gpd
import pandas as pd
from shapely.geometry import MultiPoint, Point, Polygon
from shapely.ops import unary_union


class ClusterMetricsCalculator:
    """
    Computes spatial metrics for extracted accident clusters:
    - Cluster size (# incidents)
    - Geometric centroids (X, Y in linear metric space)
    - Minimum bounding convex hull polygons
    - Danger zone area in square meters (m^2)
    """

    @staticmethod
    def calculate_centroids_and_hulls(
        points_gdf: gpd.GeoDataFrame,
        cluster_col: str = "cluster",
    ) -> Tuple[pd.DataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
        """
        Calculates cluster summary table, centroid GeoDataFrame, and convex hull GeoDataFrame.

        Args:
            points_gdf: GeoDataFrame containing points with a cluster assignment column.
            cluster_col: Name of the column holding cluster labels.

        Returns:
            Tuple of (summary_df, centroids_gdf, hulls_gdf).
        """
        if points_gdf is None or points_gdf.empty or cluster_col not in points_gdf.columns:
            raise ValueError(f"Invalid GeoDataFrame or missing '{cluster_col}' column.")

        # Filter out noise (-1) and unassigned points
        clustered = points_gdf[points_gdf[cluster_col] > 0].copy()
        if clustered.empty:
            print("[Warning] No clusters identified. Returning empty metrics.")
            empty_gdf = gpd.GeoDataFrame(columns=["cluster", "geometry"], crs=points_gdf.crs)
            return pd.DataFrame(), empty_gdf, empty_gdf

        summary_records: List[Dict] = []
        centroids_list: List[Dict] = []
        hulls_list: List[Dict] = []

        unique_clusters = sorted(clustered[cluster_col].unique())

        for c_id in unique_clusters:
            c_points = clustered[clustered[cluster_col] == c_id]
            size = len(c_points)

            # Calculate geometric centroid
            mean_x = float(c_points.geometry.x.mean())
            mean_y = float(c_points.geometry.y.mean())
            centroid_geom = Point(mean_x, mean_y)

            # Calculate convex hull
            hull_area: Optional[float] = None
            hull_geom = None

            if size >= 3:
                # 3 or more points can form a 2D convex polygon
                points_union = unary_union(c_points.geometry.tolist())
                hull = points_union.convex_hull
                if isinstance(hull, Polygon):
                    hull_geom = hull
                    hull_area = float(hull.area)
                else:
                    hull_geom = hull

            summary_records.append({
                "Cluster_ID": int(c_id),
                "Size": int(size),
                "Centroid_X": mean_x,
                "Centroid_Y": mean_y,
                "Hull_Area": hull_area if hull_area is not None else "",
            })

            centroids_list.append({
                "cluster": int(c_id),
                "size": int(size),
                "geometry": centroid_geom,
            })

            if hull_geom is not None and not hull_geom.is_empty:
                hulls_list.append({
                    "cluster": int(c_id),
                    "size": int(size),
                    "area_m2": hull_area if hull_area is not None else 0.0,
                    "geometry": hull_geom,
                })

        summary_df = pd.DataFrame(summary_records).sort_values(by="Size", ascending=False).reset_index(drop=True)
        centroids_gdf = gpd.GeoDataFrame(centroids_list, crs=points_gdf.crs)
        hulls_gdf = gpd.GeoDataFrame(hulls_list, crs=points_gdf.crs) if hulls_list else gpd.GeoDataFrame(columns=["cluster", "geometry"], crs=points_gdf.crs)

        return summary_df, centroids_gdf, hulls_gdf
