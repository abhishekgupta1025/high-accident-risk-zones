# -*- coding: utf-8 -*-
"""
Unit tests for Disjoint Set Union (DSU) implementation.
"""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from risk_zones.topology.dsu import DisjointSetUnion


class TestDisjointSetUnion(unittest.TestCase):
    """Test suite for DisjointSetUnion."""

    def test_singletons(self):
        dsu = DisjointSetUnion([1, 2, 3])
        self.assertEqual(dsu.find(1), 1)
        self.assertEqual(dsu.find(2), 2)
        self.assertEqual(dsu.find(3), 3)

    def test_union_and_find(self):
        dsu = DisjointSetUnion([1, 2, 3, 4])
        # Merge (1, 2) and (3, 4)
        self.assertTrue(dsu.union(1, 2))
        self.assertTrue(dsu.union(3, 4))
        self.assertEqual(dsu.find(1), dsu.find(2))
        self.assertEqual(dsu.find(3), dsu.find(4))
        self.assertNotEqual(dsu.find(1), dsu.find(3))

        # Redundant union returns False
        self.assertFalse(dsu.union(1, 2))

        # Merge the two sets together
        self.assertTrue(dsu.union(2, 3))
        self.assertEqual(dsu.find(1), dsu.find(4))

    def test_path_compression(self):
        dsu = DisjointSetUnion([1, 2, 3, 4, 5])
        # Chain: 5 -> 4 -> 3 -> 2 -> 1
        dsu.union(1, 2)
        dsu.union(2, 3)
        dsu.union(3, 4)
        dsu.union(4, 5)

        root = dsu.find(1)
        for i in [1, 2, 3, 4, 5]:
            self.assertEqual(dsu.find(i), root)

    def test_get_all_sets(self):
        dsu = DisjointSetUnion([10, 20, 30])
        dsu.union(10, 20)
        mapping = dsu.get_all_sets()
        self.assertEqual(mapping[10], mapping[20])
        self.assertNotEqual(mapping[10], mapping[30])


if __name__ == "__main__":
    unittest.main()
