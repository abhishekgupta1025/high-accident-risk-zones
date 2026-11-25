# -*- coding: utf-8 -*-
"""
Regression tests for road snapping, micro-gap tolerance bridging, DSU corridor grouping,
geometry repair, Matplotlib colormap compatibility, and CLI parsing.
"""
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import geopandas as gpd
from shapely.geometry import LineString, Point

from risk_zones.cli import parse_args
from risk_zones.clustering.dbscan import NetworkConstrainedDBSCAN
from risk_zones.io.loader import load_road_data
from risk_zones.topology.dsu import group_nearby_roads
from risk_zones.topology.graph import RoadConnectivityGraph
from risk_zones.topology.spatial_index import (
    precompute_group_intersections,
    precompute_segment_intersections,
    snap_points_to_roads,
)
from risk_zones.visualization.web_map import InteractiveMapBuilder


class TestPipelineFixes(unittest.TestCase):
    """Regression suite validating network snapping, DSU grouping, and spatial indexing fixes."""

    def test_sjoin_nearest_snapping_and_off_road_isolation(self):
        roads = gpd.GeoDataFrame(
            [
                {"road_uid": 10, "geometry": LineString([(0, 0), (100, 0)])},
                {"road_uid": 20, "geometry": LineString([(0, 100), (100, 100)])},
            ],
            crs="EPSG:32645",
        )
        points = gpd.GeoDataFrame(
            [
                {"FID": 0, "geometry": Point(10, 2)},     # near road 10
                {"FID": 1, "geometry": Point(20, 99)},    # near road 20
                {"FID": 2, "geometry": Point(50, 50)},    # equidistant (50m) from both -> tie broken uniquely
                {"FID": 3, "geometry": Point(500, 500)},  # far off-road -> -999
            ],
            crs="EPSG:32645",
        )

        # With unlimited snap distance all points match, equidistant tie produces 1 row
        snapped_all = snap_points_to_roads(points, roads, max_snap_distance=None)
        self.assertEqual(len(snapped_all), 4)
        self.assertEqual(snapped_all.iloc[0]["road_uid"], 10)
        self.assertEqual(snapped_all.iloc[1]["road_uid"], 20)
        self.assertIn(snapped_all.iloc[2]["road_uid"], (10, 20))

        # With max_snap_distance=20m, far off-road and middle points become -999
        snapped_bounded = snap_points_to_roads(points, roads, max_snap_distance=20.0)
        self.assertEqual(len(snapped_bounded), 4)
        self.assertEqual(snapped_bounded.iloc[0]["road_uid"], 10)
        self.assertEqual(snapped_bounded.iloc[1]["road_uid"], 20)
        self.assertEqual(snapped_bounded.iloc[2]["road_uid"], -999)
        self.assertEqual(snapped_bounded.iloc[3]["road_uid"], -999)

    def test_intersection_tolerance_bridges_micro_gaps(self):
        # Two collinear segments separated by a 1.5m digitization gap at x=50..51.5
        roads = gpd.GeoDataFrame(
            [
                {"road_uid": 0, "geometry": LineString([(0, 0), (50, 0)])},
                {"road_uid": 1, "geometry": LineString([(51.5, 0), (100, 0)])},
            ],
            crs="EPSG:32645",
        )

        exact_map = precompute_segment_intersections(roads, intersection_tolerance=0.0)
        self.assertEqual(exact_map[0], set())
        self.assertEqual(exact_map[1], set())

        bridged_map = precompute_segment_intersections(roads, intersection_tolerance=2.0)
        self.assertIn(1, bridged_map[0])
        self.assertIn(0, bridged_map[1])

    def test_dsu_corridor_grouping_end_to_end(self):
        # Dual carriageway: two parallel carriageways 10m apart (y=0 and y=10)
        roads = gpd.GeoDataFrame(
            [
                {"road_uid": 0, "geometry": LineString([(0, 0), (100, 0)])},
                {"road_uid": 1, "geometry": LineString([(0, 10), (100, 10)])},
            ],
            crs="EPSG:32645",
        )
        points = gpd.GeoDataFrame(
            [
                {"geometry": Point(10, 0)},
                {"geometry": Point(20, 0)},
                {"geometry": Point(30, 0)},
                {"geometry": Point(10, 10)},
                {"geometry": Point(20, 10)},
                {"geometry": Point(30, 10)},
            ],
            crs="EPSG:32645",
        )
        snapped = snap_points_to_roads(points, roads, max_snap_distance=5.0)

        # Segment mode keeps separated dual carriageways in 2 clusters
        seg_intersections = precompute_segment_intersections(roads, intersection_tolerance=0.0)
        seg_labels = NetworkConstrainedDBSCAN(
            eps=40.0,
            min_pts=3,
            connectivity_graph=RoadConnectivityGraph(seg_intersections),
        ).fit_predict(snapped, road_id_column="road_uid")
        self.assertEqual(len(set(seg_labels)), 2)

        # DSU corridor mode merges segments within 15m into 1 corridor cluster
        group_map = group_nearby_roads(roads, threshold=15.0)
        self.assertEqual(group_map[0], group_map[1])
        snapped["group_id"] = snapped["road_uid"].map(group_map)
        grp_intersections = precompute_group_intersections(roads, group_map)
        grp_labels = NetworkConstrainedDBSCAN(
            eps=40.0,
            min_pts=3,
            connectivity_graph=RoadConnectivityGraph(grp_intersections),
        ).fit_predict(snapped, road_id_column="group_id")
        self.assertEqual(len(set(grp_labels)), 1)
        self.assertTrue(all(lbl == 1 for lbl in grp_labels))

    def test_make_valid_preserves_invalid_linestrings_and_webmap_renders(self):
        # Repeated-coordinate LineString is invalid in GEOS; make_valid() keeps geometry non-empty
        # whereas buffer(0) would turn it into POLYGON EMPTY.
        degenerate_line = LineString([(0, 0), (0, 0)])
        normal_line = LineString([(0, 0), (10, 0)])
        self.assertFalse(degenerate_line.is_valid)

        with tempfile.TemporaryDirectory() as tmpdir:
            gpkg_path = Path(tmpdir) / "roads.gpkg"
            gpd.GeoDataFrame(
                [{"geometry": degenerate_line}, {"geometry": normal_line}],
                crs="EPSG:32645",
            ).to_file(gpkg_path, driver="GPKG")

            repaired_gdf, _ = load_road_data(gpkg_path)
            self.assertIsNotNone(repaired_gdf)
            self.assertEqual(len(repaired_gdf), 2)
            self.assertTrue(all(~repaired_gdf.geometry.is_empty))
            self.assertTrue(all(repaired_gdf.geometry.is_valid))

            # Verify InteractiveMapBuilder renders >20 cluster IDs cleanly on Matplotlib 3.9+
            pts = gpd.GeoDataFrame(
                [
                    {"cluster": cid, "road_uid": 0, "geometry": Point(639000 + cid * 10, 2490000)}
                    for cid in range(1, 26)
                ],
                crs="EPSG:32645",
            )
            html_out = Path(tmpdir) / "map.html"
            InteractiveMapBuilder.create_map(pts, output_filepath=html_out, generate_heatmap=False)
            self.assertTrue(html_out.exists())

    def test_cli_parse_args(self):
        cfg = parse_args(["--eps", "120", "--min-pts", "4", "--tolerance", "3.5", "--group-threshold", "20", "--max-snap-distance", "50"])
        self.assertEqual(cfg.eps, 120.0)
        self.assertEqual(cfg.min_pts, 4)
        self.assertEqual(cfg.intersection_tolerance, 3.5)
        self.assertEqual(cfg.group_threshold, 20.0)
        self.assertEqual(cfg.max_snap_distance, 50.0)


if __name__ == "__main__":
    unittest.main()
