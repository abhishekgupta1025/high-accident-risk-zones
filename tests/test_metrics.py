# -*- coding: utf-8 -*-
"""
Unit tests for ClusterMetricsCalculator (Centroids and Convex Hulls).
"""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

try:
    import geopandas as gpd
    from shapely.geometry import Point
    from risk_zones.clustering.metrics import ClusterMetricsCalculator
    HAS_GIS = True
except ImportError:
    HAS_GIS = False


class TestClusterMetrics(unittest.TestCase):
    """Test suite for ClusterMetricsCalculator."""

    @unittest.skipUnless(HAS_GIS, "GeoPandas/Shapely required for metrics tests")
    def test_metrics_calculation(self):
        # 4 points forming a square (0,0), (10,0), (10,10), (0,10) in Cluster 1
        data = [
            {"geometry": Point(0, 0), "cluster": 1},
            {"geometry": Point(10, 0), "cluster": 1},
            {"geometry": Point(10, 10), "cluster": 1},
            {"geometry": Point(0, 10), "cluster": 1},
            {"geometry": Point(100, 100), "cluster": -1},  # Noise
        ]
        gdf = gpd.GeoDataFrame(data, crs="EPSG:32645")

        summary_df, centroids_gdf, hulls_gdf = ClusterMetricsCalculator.calculate_centroids_and_hulls(gdf)

        # 1 cluster summary record
        self.assertEqual(len(summary_df), 1)
        self.assertEqual(summary_df.iloc[0]["Cluster_ID"], 1)
        self.assertEqual(summary_df.iloc[0]["Size"], 4)

        # Centroid should be at (5.0, 5.0)
        self.assertAlmostEqual(summary_df.iloc[0]["Centroid_X"], 5.0)
        self.assertAlmostEqual(summary_df.iloc[0]["Centroid_Y"], 5.0)

        # Convex hull area should be 10 * 10 = 100.0 m^2
        self.assertAlmostEqual(summary_df.iloc[0]["Hull_Area"], 100.0)


if __name__ == "__main__":
    unittest.main()
