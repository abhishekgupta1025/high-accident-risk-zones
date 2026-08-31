# -*- coding: utf-8 -*-
"""
3-Tier Topological Connectivity Graph Engine.
Enforces that incident points are clustered along connected physical corridors:
- Tier 1: Same road segment
- Tier 2: Direct geometric intersection
- Tier 3: 1-hop indirect intersection
"""
import math
from typing import Dict, Hashable, List, Optional, Set


def _is_unassociated(item: Optional[Hashable]) -> bool:
    """Checks if a road identifier is missing or unassociated (-999, None, NaN)."""
    if item is None or item == -999 or item == "-999":
        return True
    if isinstance(item, float) and math.isnan(item):
        return True
    return False


class RoadConnectivityGraph:
    """
    Evaluates topological reachability and connectivity between candidate accident points.
    Prevents false cross-corridor merging across divided highways and grade-separated overpasses.
    """

    def __init__(self, intersections_map: Dict[Hashable, Set[Hashable]]):
        """
        Args:
            intersections_map: Dictionary mapping road_uid (or group_id) -> set of intersecting IDs.
        """
        self.intersections_map = intersections_map

    def are_connected(
        self,
        target_id: Optional[Hashable],
        neighbor_id: Optional[Hashable],
    ) -> bool:
        """
        Evaluates the 3-Tier Topological Connectivity Cascade:
        1. Tier 1: Same segment/group ID
        2. Tier 2: Directly intersecting segment/group
        3. Tier 3: 1-Hop indirect intersection (common intersecting intermediary)
        """
        is_target_unassociated = _is_unassociated(target_id)
        is_neighbor_unassociated = _is_unassociated(neighbor_id)

        if is_target_unassociated:
            # Unassociated incidents connect only to other unassociated incidents
            return bool(is_neighbor_unassociated)

        if is_neighbor_unassociated:
            # Associated incidents never connect to unassociated incidents via network
            return False

        # Tier 1: Same Segment / Group
        if target_id == neighbor_id:
            return True

        # Tier 2: Direct Geometric Intersection
        direct_target_neighbors = self.intersections_map.get(target_id, set())
        if neighbor_id in direct_target_neighbors:
            return True

        # Tier 3: Indirect 1-Hop Intersection (Intersects(Target) ∩ Intersects(Neighbor) ≠ ∅)
        direct_neighbor_neighbors = self.intersections_map.get(neighbor_id, set())
        if not direct_target_neighbors.isdisjoint(direct_neighbor_neighbors):
            return True

        # Disconnected in network topology
        return False
