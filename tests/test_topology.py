# -*- coding: utf-8 -*-
"""
Unit tests for 3-Tier Road Connectivity Graph.
"""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from risk_zones.topology.graph import RoadConnectivityGraph


class TestRoadConnectivityGraph(unittest.TestCase):
    """Test suite for 3-Tier Topological Connectivity Cascade."""

    def setUp(self):
        # Graph structure:
        # Segment 1 connects to Segment 2
        # Segment 2 connects to Segment 3
        # Segment 4 is isolated (no intersections)
        # 1 <---> 2 <---> 3        4 (Isolated)
        self.intersections = {
            1: {2},
            2: {1, 3},
            3: {2},
            4: set(),
        }
        self.graph = RoadConnectivityGraph(self.intersections)

    def test_tier_1_same_segment(self):
        # Same segment ID -> Connected
        self.assertTrue(self.graph.are_connected(1, 1))
        self.assertTrue(self.graph.are_connected(4, 4))

    def test_tier_2_direct_intersection(self):
        # Direct intersection -> Connected
        self.assertTrue(self.graph.are_connected(1, 2))
        self.assertTrue(self.graph.are_connected(2, 1))
        self.assertTrue(self.graph.are_connected(2, 3))

    def test_tier_3_1_hop_indirect_intersection(self):
        # 1 and 3 are not directly connected, but share intermediate segment 2 -> Connected
        self.assertTrue(self.graph.are_connected(1, 3))
        self.assertTrue(self.graph.are_connected(3, 1))

    def test_disconnected_rejection(self):
        # Segment 4 is disconnected from 1, 2, 3 -> Not connected
        self.assertFalse(self.graph.are_connected(1, 4))
        self.assertFalse(self.graph.are_connected(2, 4))
        self.assertFalse(self.graph.are_connected(3, 4))

    def test_unassociated_points_isolation(self):
        # Unassociated points (NaN or -999) cannot bridge to network segments
        self.assertFalse(self.graph.are_connected(1, -999))
        self.assertFalse(self.graph.are_connected(None, 1))
        # Two unassociated points can connect to each other
        self.assertTrue(self.graph.are_connected(-999, -999))


if __name__ == "__main__":
    unittest.main()
