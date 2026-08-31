# -*- coding: utf-8 -*-
"""
Unit tests for NetworkConstrainedDBSCAN.
"""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

try:
    import geopandas as gpd
    from shapely.geometry import Point
    HAS_GIS = True
except ImportError:
    HAS_GIS = False

from risk_zones.clustering.dbscan import NetworkConstrainedDBSCAN, calculate_euclidean_distance
from risk_zones.topology.graph import RoadConnectivityGraph


class TestNetworkConstrainedDBSCAN(unittest.TestCase):
    """Test suite for NetworkConstrainedDBSCAN."""

    def test_euclidean_distance(self):
        self.assertAlmostEqual(calculate_euclidean_distance((0, 0), (3, 4)), 5.0)
        self.assertAlmostEqual(calculate_euclidean_distance((100, 100), (100, 100)), 0.0)

    @unittest.skipUnless(HAS_GIS, "GeoPandas/Shapely required for spatial clustering tests")
    def test_network_constrained_separation(self):
        """
        Verify that 2 points close in 2D Euclidean space (e.g. 20m apart)
        on disconnected road segments (e.g. overpass vs underpass) are NOT clustered.
        """
        # Road 1 and Road 2 have NO intersection
        intersections = {1: set(), 2: set()}
        graph = RoadConnectivityGraph(intersections)

        # 4 points on Road 1 (at x=0..30, y=0)
        # 4 points on Road 2 (at x=0..30, y=20) -> 20m Euclidean distance away
        data = [
            # Road 1 incidents
            {"geometry": Point(0, 0), "road_uid": 1},
            {"geometry": Point(10, 0), "road_uid": 1},
            {"geometry": Point(20, 0), "road_uid": 1},
            {"geometry": Point(30, 0), "road_uid": 1},
            # Road 2 incidents (parallel highway 20m away)
            {"geometry": Point(0, 20), "road_uid": 2},
            {"geometry": Point(10, 20), "road_uid": 2},
            {"geometry": Point(20, 20), "road_uid": 2},
            {"geometry": Point(30, 20), "road_uid": 2},
        ]
        gdf = gpd.GeoDataFrame(data, crs="EPSG:32645")

        dbscan = NetworkConstrainedDBSCAN(eps=50.0, min_pts=3, connectivity_graph=graph)
        labels = dbscan.fit_predict(gdf, road_id_column="road_uid")

        road1_labels = labels[0:4]
        road2_labels = labels[4:8]

        # Both sets formed valid clusters
        self.assertTrue(all(l > 0 for l in road1_labels))
        self.assertTrue(all(l > 0 for l in road2_labels))

        # Road 1 and Road 2 are IN DIFFERENT CLUSTERS (No cross-corridor merging!)
        self.assertNotEqual(road1_labels[0], road2_labels[0])


if __name__ == "__main__":
    unittest.main()
