# -*- coding: utf-8 -*-
import math
import os
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import Point
from shapely.ops import unary_union # Needed for grouping checks and hulls
import folium
from folium.plugins import HeatMap
from branca.colormap import linear, LinearColormap
from branca.element import MacroElement
from jinja2 import Template
import warnings
import tqdm # Added for progress bars

# Ignore specific Shapely warning about comparing non-squares
warnings.filterwarnings("ignore", message="Iteration over multi-part geometry collections is deprecated")
# Ignore UserWarning during CRS reprojection in plotting if needed
warnings.filterwarnings("ignore", category=UserWarning, message=".*CRS mismatch between CRS.*")


# --- Custom Scale Control Class (Unchanged) ---
class ScaleControl(MacroElement):
    """Custom Folium Scale Control based on Leaflet's L.Control.Scale"""
    def __init__(self, position='bottomleft', metric=True, imperial=False, max_width=100):
        super().__init__()
        self._template = Template(u"""
            {% macro script(this, kwargs) %}
                L.control.scale({
                    position: '{{ this.position }}',
                    metric: {{ this.metric | lower }},
                    imperial: {{ this.imperial | lower }},
                    maxWidth: {{ this.max_width }}
                }).addTo({{ this._parent.get_name() }});
            {% endmacro %}
        """)
        self.position = position
        self.metric = metric
        self.imperial = imperial
        self.max_width = max_width

# --- Euclidean Distance (Unchanged) ---
def calculate_distance(point1, point2):
    if not (isinstance(point1, (list, tuple)) and isinstance(point2, (list, tuple))):
        raise TypeError("Both inputs must be lists or tuples.")
    if len(point1) != 2 or len(point2) != 2:
        raise ValueError("Points must have exactly two dimensions (x, y).")
    # Using math.hypot can be slightly more robust against overflow/underflow
    return math.hypot(point1[0] - point2[0], point1[1] - point2[1])

# --- Road Network Preprocessing (MODIFIED for Grouping and Group Intersections) ---

def group_nearby_roads(road_gdf, threshold):
    """
    Groups road segments that are within a given threshold distance of each other.
    Assigns a unique 'group_id' to each group of nearby roads using a Disjoint Set Union approach.

    Args:
        road_gdf (geopandas.GeoDataFrame): Road network with 'road_uid'.
                                           MUST be in a projected CRS.
        threshold (float): The maximum distance (in CRS units, e.g., meters)
                           for roads to be considered part of the same group.

    Returns:
        dict: A dictionary mapping road_uid to group_id.
              Returns None if input is invalid or errors occur.
    """
    if road_gdf is None or road_gdf.empty or 'road_uid' not in road_gdf.columns:
        print("ERROR: Invalid road GeoDataFrame for grouping (None, empty, or missing 'road_uid').")
        return None
    if not road_gdf.crs or road_gdf.crs.is_geographic:
        print("ERROR: Road GeoDataFrame must have a projected CRS for accurate distance grouping.")
        print(f"       Current CRS: {road_gdf.crs}")
        return None
    if not isinstance(threshold, (int, float)) or threshold <= 0:
        print("ERROR: Grouping threshold must be a positive number.")
        return None

    print(f"Grouping nearby roads (threshold = {threshold} {road_gdf.crs.axis_info[0].unit_name})...")

    # --- Disjoint Set Union (DSU) / Union-Find Implementation ---
    parent = {uid: uid for uid in road_gdf['road_uid']}
    rank = {uid: 0 for uid in road_gdf['road_uid']}

    def find_set(item):
        """Find the representative element of the set containing item (with path compression)."""
        if parent[item] == item:
            return item
        parent[item] = find_set(parent[item]) # Path compression
        return parent[item]

    def unite_sets(item1, item2):
        """Unite the sets containing item1 and item2 using union by rank."""
        root1 = find_set(item1)
        root2 = find_set(item2)
        if root1 != root2:
            # Union by rank for better performance
            if rank[root1] < rank[root2]:
                parent[root1] = root2
            elif rank[root1] > rank[root2]:
                parent[root2] = root1
            else:
                parent[root2] = root1
                rank[root1] += 1
            return True # Indicates a merge happened
        return False # Sets were already united

    # --- Grouping Logic using Buffers and Spatial Index ---
    try:
        print("Buffering roads for proximity check...")
        # Buffer by half the threshold: if buffers intersect, originals are within threshold
        # Using resolution=3 for buffer simplification can speed up intersection checks slightly
        buffers = road_gdf.geometry.buffer(threshold / 2.0, resolution=3)

        print("Building spatial index on road buffers...")
        buffer_sindex = buffers.sindex # Use the GeoSeries sindex

        print("Finding and merging nearby road groups...")
        # Efficiently iterate using the spatial index
        possible_joins = buffers.sindex.query(buffers, predicate='intersects')

        # Process the potential intersections
        merged_count = 0
        for i, j in tqdm.tqdm(zip(possible_joins[0], possible_joins[1]),
                         total=len(possible_joins[0]),
                         desc="Merging Groups"):
            # Skip self-matches and ensure i < j to avoid redundant checks
            if i >= j:
                continue

            # Get the road_uids corresponding to the buffer indices i and j
            try:
                 road_uid1 = road_gdf['road_uid'].iloc[i]
                 road_uid2 = road_gdf['road_uid'].iloc[j]
            except IndexError:
                 print(f"Warning: Index out of bounds ({i} or {j}) accessing road_uid. Skipping pair.")
                 continue

            # Check if the buffers *actually* intersect (sindex query can have false positives)
            # This check might be redundant if sindex.query 'intersects' is exact enough,
            # but it's safer to include it. Let's test performance without it first.
            # if buffers.iloc[i].intersects(buffers.iloc[j]):
            #    pass # Proceed with unite_sets

            # Unite the sets if they belong to different roads
            if road_uid1 != road_uid2:
                if unite_sets(road_uid1, road_uid2):
                    merged_count += 1

    except Exception as e:
        print(f"ERROR during road grouping process: {e}")
        import traceback
        traceback.print_exc()
        return None

    # Finalize the road_group_map by assigning the representative ID (root of the set)
    print("Finalizing group assignments...")
    final_road_group_map = {}
    for uid in road_gdf['road_uid']:
        final_road_group_map[uid] = find_set(uid)

    num_groups = len(set(final_road_group_map.values()))
    print(f"Finished grouping. Performed {merged_count} merges.")
    print(f"Identified {num_groups} distinct road groups from {len(road_gdf)} segments.")
    return final_road_group_map


def precompute_group_intersections(road_gdf, road_group_map):
    """
    Pre-computes which road *groups* intersect each other geometrically.

    Args:
        road_gdf (geopandas.GeoDataFrame): Road network with 'road_uid' and geometry.
                                           Must have a spatial index.
        road_group_map (dict): Mapping from road_uid to group_id.

    Returns:
        dict: A dictionary where keys are group_ids and values are sets
              of group_ids that intersect the key group.
              Returns None if input is invalid or errors occur.
    """
    if road_gdf is None or road_gdf.empty or 'road_uid' not in road_gdf.columns or road_group_map is None:
         print("ERROR: Invalid input for group intersection precomputation.")
         return None
    if not hasattr(road_gdf, 'sindex') or road_gdf.sindex is None:
        print("Building spatial index for road geometries (required for intersections)...")
        try:
            road_gdf.sindex # Accessing it usually builds it
        except Exception as e:
            print(f"ERROR: Failed to build spatial index on road_gdf: {e}")
            return None

    print("Pre-computing road group intersections...")
    # Initialize intersection dict using unique group IDs present in the map
    unique_groups = set(road_group_map.values())
    group_intersections = {gid: set() for gid in unique_groups}
    if not unique_groups:
        print("Warning: No groups found in road_group_map. Cannot compute intersections.")
        return group_intersections # Return empty dict

    try:
        # Efficiently query for intersections using the spatial index
        possible_intersections = road_gdf.sindex.query(road_gdf.geometry, predicate='intersects')

        # Process the potential intersections
        processed_pairs = set() # Track group pairs to avoid redundant adds

        for i, j in tqdm.tqdm(zip(possible_intersections[0], possible_intersections[1]),
                         total=len(possible_intersections[0]),
                         desc="Finding Group Intersections"):
            # Skip self-matches
            if i == j:
                continue

            # Get road UIDs and corresponding group IDs
            try:
                road1_uid = road_gdf['road_uid'].iloc[i]
                road2_uid = road_gdf['road_uid'].iloc[j]
                group1_id = road_group_map.get(road1_uid)
                group2_id = road_group_map.get(road2_uid)
            except IndexError:
                print(f"Warning: Index out of bounds ({i} or {j}) accessing road_uid/group_id. Skipping.")
                continue
            except KeyError:
                 print(f"Warning: road_uid {road1_uid} or {road2_uid} not found in road_group_map. Skipping.")
                 continue

            # If groups are valid and different
            if group1_id is not None and group2_id is not None and group1_id != group2_id:
                # Ensure the geometries *actually* intersect (sindex query is approximate)
                if road_gdf.geometry.iloc[i].intersects(road_gdf.geometry.iloc[j]):
                    # Add intersection between groups (symmetrically, avoid duplicates)
                    pair = tuple(sorted((group1_id, group2_id)))
                    if pair not in processed_pairs:
                        group_intersections[group1_id].add(group2_id)
                        group_intersections[group2_id].add(group1_id)
                        processed_pairs.add(pair)

    except Exception as e:
        print(f"ERROR during group intersection computation: {e}")
        import traceback
        traceback.print_exc()
        return None

    intersection_count = sum(len(v) for v in group_intersections.values()) // 2 # Each intersection added twice
    print(f"Finished pre-computing group intersections. Found {intersection_count} intersecting group pairs.")
    return group_intersections

# --- Modified DBSCAN Core Functions (Using Groups) ---

def find_neighbors_constrained_groups(
    point_index,
    eps,
    points_coords,
    points_gdf_with_groups, # GDF with points, 'group_id' (nullable)
    group_intersections     # Precomputed dict: {group_id: {intersecting_group_id, ...}}
    ):
    """
    Finds neighbors using spatial distance and road group connectivity
    (same group, direct intersection, or 1-step indirect intersection).

    Args:
        point_index (int): Index of the target point.
        eps (float): Max spatial distance (in CRS units).
        points_coords (list): List of (x, y) coordinates for all points.
        points_gdf_with_groups (geopandas.GeoDataFrame): Point GDF including 'group_id'.
        group_intersections (dict): Precomputed group intersection data.

    Returns:
        list: Indices of neighboring points. Returns empty list if target point has no group.
    """
    neighbors = []
    target_point_coords = points_coords[point_index]
    target_group_id = points_gdf_with_groups.iloc[point_index]['group_id']

    # If the target point couldn't be associated with a road group, it cannot connect to others via network
    # It might still cluster spatially with other unassociated points if eps is large enough,
    # but it won't bridge clusters across different roads/groups.
    if pd.isna(target_group_id):
        # Find neighbors based *only* on spatial distance IF they ALSO lack a group_id
        for i, point_coords in enumerate(points_coords):
             if i == point_index: continue
             neighbor_group_id = points_gdf_with_groups.iloc[i]['group_id']
             if pd.isna(neighbor_group_id): # Only connect to other points without a group
                 distance = calculate_distance(point_coords, target_point_coords)
                 if distance <= eps:
                     neighbors.append(i)
        return neighbors


    # If the target point *has* a group_id:
    direct_target_neighbors = group_intersections.get(target_group_id, set())

    for i, point_coords in enumerate(points_coords):
        if i == point_index:
            continue

        # 1. Check spatial distance first (optimization)
        distance = calculate_distance(point_coords, target_point_coords)
        if distance <= eps:
            # 2. Check network constraint using groups
            neighbor_group_id = points_gdf_with_groups.iloc[i]['group_id']

            # Skip if neighbor point has no group (cannot connect via network)
            if pd.isna(neighbor_group_id):
                continue

            # Constraint 2a: Same group? (Covers nearby parallel roads case)
            if target_group_id == neighbor_group_id:
                neighbors.append(i)
                continue # Found connection, move to next point

            # Constraint 2b: Direct group intersection?
            if neighbor_group_id in direct_target_neighbors:
                neighbors.append(i)
                continue # Found connection, move to next point

            # Constraint 2c: Indirect group intersection (1 intermediate group)?
            # Find groups intersecting the neighbor's group
            direct_neighbor_neighbors = group_intersections.get(neighbor_group_id, set())
            # Check if any of the target's direct neighbors also intersect the neighbor's group
            # i.e., is there a common intersecting group between the two sets?
            # Use set intersection: `set1.intersection(set2)` or check `not set1.isdisjoint(set2)`
            if not direct_target_neighbors.isdisjoint(direct_neighbor_neighbors):
                 # Ensure this indirect connection isn't just the target/neighbor groups themselves
                 # (already covered by direct check). The logic implicitly handles this
                 # because we wouldn't reach this step if they were direct neighbors or same group.
                 neighbors.append(i)
                 continue # Found connection, move to next point

    return neighbors


def expand_cluster_constrained_groups(
    point_index,
    neighbors_indices, # Initial neighbors of the core point
    cluster_label,
    eps,
    min_pts,
    labels,                 # List of labels for all points
    points_coords,
    points_gdf_with_groups, # Passed through
    group_intersections      # Passed through
    ):
    """Expands cluster using the group-constrained neighbor finding."""
    labels[point_index] = cluster_label
    i = 0 # Index for iterating through the list of neighbors to process
    # Use a list as a queue; set keeps track of who has been added to avoid duplicates
    neighbors_queue = list(neighbors_indices)
    processed_or_queued = set(neighbors_indices)
    processed_or_queued.add(point_index)

    while i < len(neighbors_queue):
        current_neighbor_index = neighbors_queue[i]
        i += 1 # Move to the next neighbor in the queue

        # Process point if it's noise or unclassified
        if labels[current_neighbor_index] in [-1, 0]:
            # If it was noise, reclassify to current cluster
            if labels[current_neighbor_index] == -1:
                 labels[current_neighbor_index] = cluster_label

            # If unclassified, tentatively assign to cluster
            elif labels[current_neighbor_index] == 0:
                 labels[current_neighbor_index] = cluster_label

                 # Find *its* neighbors using the GROUP constrained method
                 # This neighbor now becomes a potential core point
                 current_neighbors_neighbors = find_neighbors_constrained_groups(
                    current_neighbor_index, eps, points_coords, points_gdf_with_groups, group_intersections
                 )

                 # If it meets the core point criteria (MinPts)
                 if len(current_neighbors_neighbors) >= min_pts:
                     # Add its neighbors to the queue if they haven't been processed/queued
                     for nn_idx in current_neighbors_neighbors:
                         if nn_idx not in processed_or_queued:
                             processed_or_queued.add(nn_idx)
                             neighbors_queue.append(nn_idx)
                             # Mark as reachable/part of cluster border if unclassified
                             # Note: Actual assignment happens when it's processed from queue
                             # if labels[nn_idx] == 0:
                             #    labels[nn_idx] = cluster_label # Tentative assignment is fine

        # If the point was already part of another cluster (label > 0), do nothing.

def dbscan_constrained_groups(
    eps,
    min_pts,
    points_coords,
    points_gdf_with_groups, # Needs 'group_id' column
    group_intersections
    ):
    """Performs DBSCAN constrained by road group connectivity."""
    if not points_coords or points_gdf_with_groups is None or points_gdf_with_groups.empty:
        print("Warning: Input data for constrained DBSCAN is missing or empty.")
        return []
    if len(points_coords) != len(points_gdf_with_groups):
         raise ValueError("Mismatch between coordinates list and GeoDataFrame length.")
    if group_intersections is None:
        print("Warning: Group intersection data not provided. Cannot perform constrained DBSCAN.")
        # Return all as noise if constraints can't be checked? Or maybe run unconstrained?
        # Let's return noise as the constraint is the core feature here.
        return [-1] * len(points_coords)
    if 'group_id' not in points_gdf_with_groups.columns:
         raise ValueError("Missing 'group_id' column in points_gdf_with_groups for group-constrained DBSCAN.")


    n_points = len(points_coords)
    labels = [0] * n_points # 0: Undefined, -1: Noise, >0: Cluster ID
    cluster_label = 0

    print("Starting group-constrained DBSCAN...")
    for point_index in tqdm.tqdm(range(n_points), desc="Clustering Points (Groups)"):
        # If point is already classified (part of a cluster or noise), skip
        if labels[point_index] != 0:
            continue

        # Find neighbors using the group-constrained method
        neighbors_indices = find_neighbors_constrained_groups(
            point_index, eps, points_coords, points_gdf_with_groups, group_intersections
        )

        # Check if core point criteria met (MinPts neighbors)
        if len(neighbors_indices) < min_pts:
             # Not a core point, mark as Noise for now
             # It might become a border point later if reached by a cluster expansion
             labels[point_index] = -1
        else:
            # Core point found, start a new cluster
            cluster_label += 1
            expand_cluster_constrained_groups(
                point_index, neighbors_indices, cluster_label, eps, min_pts,
                labels, points_coords, points_gdf_with_groups, group_intersections
            )
            # Note: expand_cluster handles assigning the label to the core point itself
            # and recursively expanding.

    print(f"Group-constrained DBSCAN complete. Found {cluster_label} potential clusters.")
    return labels


# --- Data Reading Functions ---
def read_point_shp_data(filename):
    """Reads point geometries from a shapefile. Resets index."""
    if not isinstance(filename, str): raise TypeError("Filename must be a string.")
    if not os.path.exists(filename): raise FileNotFoundError(f"Shapefile not found: {filename}")
    try:
        gdf = gpd.read_file(filename)
        if gdf.empty: print(f"Warning: Shapefile '{filename}' is empty."); return None
        if 'geometry' not in gdf.columns: raise ValueError(f"Shapefile '{filename}' lacks 'geometry'.")

        # Filter for actual Point geometries, ignore others if mixed types
        original_count = len(gdf)
        gdf_points = gdf[gdf.geometry.geom_type == 'Point'].copy()
        if gdf_points.empty:
            print(f"Warning: No Point geometries found in '{filename}'. Check geometry types.");
            return None
        if len(gdf_points) < original_count:
            print(f"Warning: Filtered {original_count - len(gdf_points)} non-Point features from '{filename}'.")

        print(f"Successfully read {len(gdf_points)} points from '{filename}'.")
        print(f"Point Data Coordinate Reference System (CRS): {gdf_points.crs}")
        if gdf_points.crs is None:
             print("ERROR: Point shapefile is missing CRS information. Cannot proceed.")
             return None
        # Reset index to ensure it's contiguous (0, 1, 2...)
        gdf_points = gdf_points.reset_index(drop=True)
        return gdf_points
    except ImportError: raise ImportError("Requires 'geopandas'. Install: `pip install geopandas`.")
    except Exception as e: raise Exception(f"Error reading point shapefile '{filename}': {e}")


def read_line_shp_data(filename):
    """Reads line geometries, assigns unique 'road_uid', checks/fixes invalid geometry."""
    if not isinstance(filename, str): raise TypeError("Filename must be a string.")
    if not os.path.exists(filename): raise FileNotFoundError(f"Shapefile not found: {filename}")
    try:
        gdf = gpd.read_file(filename)
        if gdf.empty: print(f"Warning: Shapefile '{filename}' is empty."); return None, None
        if 'geometry' not in gdf.columns: raise ValueError(f"Shapefile '{filename}' lacks 'geometry'.")

        # Filter for LineString/MultiLineString, ignore others if mixed
        original_count = len(gdf)
        gdf_lines = gdf[gdf.geometry.geom_type.isin(['LineString', 'MultiLineString'])].copy()
        if gdf_lines.empty:
            print(f"Warning: No LineString or MultiLineString geometries found in '{filename}'. Check geometry types.");
            return None, None
        if len(gdf_lines) < original_count:
            print(f"Warning: Filtered {original_count - len(gdf_lines)} non-Line features from '{filename}'.")

        # Reset index before assigning UID based on index
        gdf_lines = gdf_lines.reset_index(drop=True)
        gdf_lines['road_uid'] = gdf_lines.index # Use clean index as unique ID

        # Check and fix invalid geometries
        invalid_geom = gdf_lines[~gdf_lines.geometry.is_valid]
        if not invalid_geom.empty:
            print(f"Warning: Found {len(invalid_geom)} invalid road geometries. Attempting fix using buffer(0)...")
            original_types = gdf_lines.geometry.geom_type
            gdf_lines.geometry = gdf_lines.geometry.buffer(0) # Attempt fix
            new_types = gdf_lines.geometry.geom_type

            # Check if buffer(0) changed lines to polygons (problematic)
            mismatched = (original_types.str.contains("Line")) & (~new_types.str.contains("Line"))
            if mismatched.any():
                 print(f"ERROR: buffer(0) changed geometry type for {mismatched.sum()} roads (e.g., LineString to Polygon). Cannot reliably proceed.")
                 return None, None # Indicate failure

            # Re-check validity after buffer(0)
            still_invalid = gdf_lines[~gdf_lines.geometry.is_valid]
            if not still_invalid.empty:
                 print(f"ERROR: {len(still_invalid)} road geometries remain invalid after buffer(0). Cannot proceed.")
                 return None, None # Indicate failure
            else:
                 print("Successfully fixed invalid road geometries.")

        print(f"Successfully read {len(gdf_lines)} line features from '{filename}'. Assigned 'road_uid'.")
        print(f"Line Data Coordinate Reference System (CRS): {gdf_lines.crs}")
        if gdf_lines.crs is None:
             print("ERROR: Road shapefile is missing CRS information. Cannot proceed.")
             return None, None

        return gdf_lines, gdf_lines.crs
    except ImportError: raise ImportError("Requires 'geopandas'. Install: `pip install geopandas`.")
    except Exception as e: raise Exception(f"Error reading line shapefile '{filename}': {e}")


# --- Plotting Functions (MODIFIED interactive plot popup) ---

def plot_clusters_with_roads_interactive(
    points_gdf, # Expects GDF with 'cluster', and potentially 'group_id'
    road_gdf,
    centroids_gdf,
    hulls_gdf,
    output_filename="cluster_map.html",
    generate_heatmap=True
    ):
    """
    Generates interactive map (colored by cluster size). Adds group_id to popup if present.
    """
    if points_gdf is None or points_gdf.empty or 'cluster' not in points_gdf.columns:
        print("Warning: Cannot plot clusters due to invalid/missing points GeoDataFrame or 'cluster' column.")
        return

    # Check if group_id exists for popup info
    has_group_id = 'group_id' in points_gdf.columns
    if not has_group_id:
        print("Info: 'group_id' column not found in points GDF for interactive plot popup.")

    print(f"Generating interactive map: '{output_filename}'...")

    # Target CRS for Folium maps
    target_crs = "EPSG:4326" # WGS 84 Lat/Lon

    # Project points for mapping, keep necessary columns
    points_cols_to_keep = ['geometry', 'cluster']
    if has_group_id: points_cols_to_keep.append('group_id')
    try:
        points_gdf_proj = points_gdf[points_cols_to_keep].to_crs(target_crs)
        lats = points_gdf_proj.geometry.y
        lons = points_gdf_proj.geometry.x
        if lats.empty or lons.empty:
             print("Warning: No valid coordinates after projecting points. Cannot create map.")
             return
    except Exception as e:
        print(f"Error reprojecting point data to {target_crs}: {e}. Cannot create map.")
        return

    # --- Reproject Roads, Centroids, Hulls (if they exist) ---
    def reproject_or_warn(gdf, name):
        if gdf is None or gdf.empty: return None
        if gdf.crs == target_crs: return gdf
        try:
            print(f"Reprojecting {name} to {target_crs} for map...")
            return gdf.to_crs(target_crs)
        except Exception as e:
            print(f"Warning: Error reprojecting {name}: {e}. {name} layer might be missing or misaligned.")
            return None

    roads_gdf_proj = reproject_or_warn(road_gdf, "Roads")
    centroids_gdf_proj = reproject_or_warn(centroids_gdf, "Centroids")
    hulls_gdf_proj = reproject_or_warn(hulls_gdf, "Hulls")

    # --- Create Folium Map & Add Layers ---
    map_center = [lats.mean(), lons.mean()]
    m = folium.Map(location=map_center, zoom_start=12, tiles=None)
    # Add Basemaps
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(m)
    folium.TileLayer("CartoDB positron", name="CartoDB Positron", show=False).add_to(m)
    folium.TileLayer("CartoDB dark_matter", name="CartoDB Dark", show=False).add_to(m)

    # Add Roads Layer
    if roads_gdf_proj is not None:
        road_fg = folium.FeatureGroup(name='Road Network', show=True).add_to(m)
        try:
            # Prepare tooltip fields dynamically, excluding geometry and potentially sensitive IDs if needed
            road_tooltip_fields = [col for col in roads_gdf_proj.columns if col not in ['geometry', 'road_uid']]
            road_tooltip_aliases = [f'{col}:' for col in road_tooltip_fields]
            folium.GeoJson(
                roads_gdf_proj,
                style_function=lambda x: {'color': 'gray', 'weight': 1.5, 'opacity': 0.7},
                tooltip=folium.features.GeoJsonTooltip(
                    fields=road_tooltip_fields, aliases=road_tooltip_aliases, localize=True, sticky=False)
            ).add_to(road_fg)
        except Exception as e: print(f"Error adding roads GeoJson layer: {e}")

    # --- Prepare for Size-Based Coloring ---
    unique_labels = sorted(points_gdf_proj['cluster'].unique())
    cluster_ids = sorted([l for l in unique_labels if l != -1 and pd.notna(l)]) # Exclude noise and potential NaN
    num_clusters = len(cluster_ids)
    size_colormap = None
    cluster_sizes = {}
    min_size, max_size = 0, 0
    if num_clusters > 0:
        valid_clusters_gdf = points_gdf_proj[points_gdf_proj['cluster'].isin(cluster_ids)]
        if not valid_clusters_gdf.empty:
             cluster_sizes = valid_clusters_gdf.groupby('cluster').size().to_dict()
             if cluster_sizes:
                 min_size = min(cluster_sizes.values())
                 max_size = max(cluster_sizes.values())
                 if min_size == max_size:
                     # Single color if all clusters same size
                     size_colormap = LinearColormap(['yellow'], vmin=min_size, vmax=max_size)
                     size_colormap.caption = f'Cluster Size (All={min_size})'
                 else:
                     # Gradient for varying sizes
                     size_colormap = LinearColormap(['green', 'yellow', 'red'], vmin=min_size, vmax=max_size)
                     size_colormap.caption = f'Cluster Size (# Points)'
             else: num_clusters = 0 # Handle case where grouping resulted in empty sizes

    # --- Add Points Layer (Cluster/Noise points) ---
    # Create feature groups for toggling layers
    cluster_points_fg = {}
    for label in unique_labels:
        if pd.isna(label): continue # Skip potential NaN label if DBSCAN failed somehow
        label_int = int(label)
        group_name = f"Cluster {label_int}" if label_int != -1 else "Noise Points"
        # Show clusters by default, hide noise
        show_layer = (label_int != -1)
        cluster_points_fg[label] = folium.FeatureGroup(name=group_name, show=show_layer).add_to(m)

        # Get points for this label
        points_in_label = points_gdf_proj[points_gdf_proj['cluster'] == label]

        # Determine color and size
        cluster_size = 0
        if label_int == -1:
            color, radius = 'grey', 2 # Noise points
        elif size_colormap and label_int in cluster_sizes:
            cluster_size = cluster_sizes[label_int]
            color = size_colormap(cluster_size)
            radius = 4 # Clustered points
        else:
            color, radius = 'black', 3 # Fallback or single-size clusters
            cluster_size = len(points_in_label)

        # Add points to their layer
        for idx, point in points_in_label.iterrows():
            # Build Popup HTML
            popup_html = f"<b>{group_name}</b><br>"
            popup_html += f"Lat: {point.geometry.y:.6f}<br>"
            popup_html += f"Lon: {point.geometry.x:.6f}<br>"
            popup_html += f"Cluster ID: {point['cluster']:.0f}<br>" # Format as integer
            if label_int != -1:
                popup_html += f"Cluster Size: {cluster_size}<br>"
            # Add group_id if available and not NaN
            if has_group_id:
                group_id_val = point.get('group_id', 'N/A') # Use .get for safety
                if pd.notna(group_id_val):
                    popup_html += f"Road Group ID: {group_id_val:.0f}<br>" # Format as integer
                else:
                     popup_html += f"Road Group ID: Not Associated<br>"

            # Add CircleMarker
            folium.CircleMarker(
                location=[point.geometry.y, point.geometry.x],
                radius=radius,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.7,
                popup=folium.Popup(popup_html, max_width=350) # Use folium.Popup
            ).add_to(cluster_points_fg[label])

    # --- Add Centroids, Hulls, Heatmap, Controls ---
    if centroids_gdf_proj is not None:
        centroid_fg = folium.FeatureGroup(name='Cluster Centroids', show=False).add_to(m)
        for idx, centroid in centroids_gdf_proj.iterrows():
            cluster_id = centroid.get('cluster', 'N/A')
            cluster_id_int = int(cluster_id) if pd.notna(cluster_id) else 'N/A'
            popup_text = f"Centroid for Cluster {cluster_id_int}"
            if cluster_id_int != 'N/A' and cluster_id_int in cluster_sizes:
                 popup_text += f" (Size: {cluster_sizes[cluster_id_int]})"
            folium.Marker(location=[centroid.geometry.y, centroid.geometry.x], popup=popup_text,
                          icon=folium.Icon(color='blue', icon='info-sign')).add_to(centroid_fg)

    if hulls_gdf_proj is not None:
        hull_fg = folium.FeatureGroup(name='Cluster Convex Hulls', show=False).add_to(m)
        if size_colormap and cluster_sizes:
            # Style hulls based on cluster size color
            try:
                folium.GeoJson(hulls_gdf_proj, style_function=lambda feature: {
                        'fillColor': size_colormap(cluster_sizes.get(int(feature['properties'].get('cluster', -999)), min_size)) # Use get with default
                                     if size_colormap and feature['properties'].get('cluster') is not None
                                        and int(feature['properties']['cluster']) in cluster_sizes else 'grey',
                        'color': 'black', 'weight': 1, 'fillOpacity': 0.3, },
                    tooltip=folium.features.GeoJsonTooltip(fields=['cluster'], aliases=['Cluster ID:'], localize=True)
                ).add_to(hull_fg)
            except Exception as e: print(f"Error adding size-colored hulls GeoJson: {e}")
        else: # Fallback default color if colormap failed
             try:
                 folium.GeoJson(hulls_gdf_proj, style_function=lambda f: {'fillColor': 'blue', 'color': 'black', 'weight': 1, 'fillOpacity': 0.2},
                     tooltip=folium.features.GeoJsonTooltip(fields=['cluster'], aliases=['Cluster ID:'], localize=True)
                 ).add_to(hull_fg)
             except Exception as e: print(f"Error adding default hulls GeoJson: {e}")

    if generate_heatmap:
        heatmap_fg = folium.FeatureGroup(name='Accident Heatmap', show=False).add_to(m)
        try:
            HeatMap(list(zip(lats, lons))).add_to(heatmap_fg)
        except Exception as e: print(f"Error adding heatmap layer: {e}")

    # Add colormap legend if created
    if size_colormap:
        m.add_child(size_colormap)
    # Add custom scale bar
    custom_scale = ScaleControl(position='bottomleft', metric=True, imperial=True)
    m.add_child(custom_scale)
    # Add Layer control
    folium.LayerControl(collapsed=False).add_to(m)

    # --- Save Map ---
    try:
        m.save(output_filename)
        print(f"Interactive map saved successfully to '{output_filename}'.")
    except Exception as e: print(f"Error saving Folium map: {e}")


def generate_static_plot(points_gdf, road_gdf, output_filename="static_cluster_map.png"):
    """Generates static map (colored by cluster ID). Unchanged logic, uses 'cluster' column."""
    if points_gdf is None or points_gdf.empty or 'cluster' not in points_gdf.columns:
        print("Warning: Cannot generate static plot due to invalid points GeoDataFrame or missing 'cluster' column.")
        return

    print(f"Generating static map: '{output_filename}'...")
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    # Plot roads first (if available and CRS matches or can be plotted)
    if road_gdf is not None and not road_gdf.empty:
        try:
            # Check CRS consistency for plotting - plot regardless but warn if different
            if points_gdf.crs and road_gdf.crs != points_gdf.crs:
                 print(f"Static Plot Warning: Road CRS ({road_gdf.crs}) differs from Point CRS ({points_gdf.crs}). Plotting may be misaligned if not visually compatible.")
            road_gdf.plot(ax=ax, color='darkgrey', linewidth=0.6, alpha=0.7, zorder=1, label='_nolegend_') # roads beneath points
        except Exception as e: print(f"Warning: Could not plot roads on static map: {e}")

    # Plot points, colored by cluster ID
    try:
        # Use a copy for plotting modifications
        plot_gdf = points_gdf.copy()
        # Ensure cluster column is integer, handle potential NaNs/noise (-1) for categorical plotting
        # Map NaN to a distinct value like -2 if necessary, or handle during legend creation
        plot_gdf['cluster_plot'] = plot_gdf['cluster'].fillna(-2).astype(int)

        # Plot points, using the integer cluster column for color
        plot_gdf.plot(column='cluster_plot', ax=ax, categorical=True, legend=True,
                      markersize=8, # Slightly larger for visibility
                      cmap='viridis', # Or choose another suitable cmap like 'tab20' for many clusters
                      legend_kwds={'title': "Cluster ID", 'loc': 'upper left', 'bbox_to_anchor': (1.02, 1)}, # Legend outside plot
                      zorder=2) # Points above roads

        # Customize legend labels
        handles, labels = ax.get_legend_handles_labels()
        new_labels, new_handles = [], []
        # Sort labels numerically for consistent legend order
        try:
            sorted_indices = np.argsort([int(float(l)) for l in labels])
            handles = [handles[i] for i in sorted_indices]
            labels = [labels[i] for i in sorted_indices]
        except ValueError: pass # Keep original order if conversion fails

        for h, l in zip(handles, labels):
             try:
                 label_val = int(float(l)) # Convert label text to int
                 if label_val == -1: l_new = 'Noise'
                 elif label_val == -2: l_new = 'Unassociated/NA' # Label for potential NaN points
                 else: l_new = f'Cluster {label_val}'
                 new_labels.append(l_new)
                 new_handles.append(h)
             except ValueError:
                 # Keep original label if it's not a number
                 new_labels.append(l)
                 new_handles.append(h)
        # Replace legend with custom labels
        ax.legend(new_handles, new_labels, title="Cluster ID / Type", loc='upper left', bbox_to_anchor=(1.02, 1), frameon=False)

    except Exception as e:
        print(f"Error plotting clustered points on static map: {e}")
        # Fallback: plot all points in a single color if clustering plot fails
        points_gdf.plot(ax=ax, color='blue', markersize=5, label='Accident Points (Plotting Error)', zorder=2)
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1))

    ax.set_title('Accident Clusters (Group Constrained) and Road Network')
    # Use CRS information for labels if available
    try: ax.set_xlabel(f"Easting ({points_gdf.crs.axis_info[0].abbreviation})")
    except: ax.set_xlabel("Easting / Longitude")
    try: ax.set_ylabel(f"Northing ({points_gdf.crs.axis_info[1].abbreviation})")
    except: ax.set_ylabel("Northing / Latitude")

    ax.set_axis_on() # Keep axes for projected coordinates
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend

    # Save the plot
    try:
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        print(f"Static map saved successfully to '{output_filename}'.")
    except Exception as e: print(f"Error saving static map: {e}")
    plt.close(fig) # Close the plot figure to free memory


# --- Main Analysis Function (MODIFIED for Grouping) ---

def analyze_accident_clusters_with_roads(
    accident_shp_filename,
    road_shp_filename,
    eps,
    min_pts,
    grouping_threshold, # New parameter for grouping nearby roads
    output_dir=".",
    generate_heatmap=True,
    generate_static=True
    ):
    """
    Main analysis workflow using road-group-constrained DBSCAN.
    Includes road grouping, group intersection checks, and modified DBSCAN.
    """
    print("\n--- Starting Analysis Workflow ---")
    # --- Validate Inputs ---
    if not isinstance(accident_shp_filename, str): raise TypeError("Accident shapefile filename must be a string.")
    if not isinstance(road_shp_filename, str): raise TypeError("Road shapefile filename must be a string.")
    if not isinstance(eps, (int, float)) or eps <= 0: raise ValueError("eps must be a positive number.")
    if not isinstance(min_pts, int) or min_pts <= 1: raise ValueError("min_pts must be an integer greater than 1.")
    if not isinstance(grouping_threshold, (int, float)) or grouping_threshold <= 0:
        raise ValueError("grouping_threshold must be a positive number.")

    os.makedirs(output_dir, exist_ok=True)
    # Define output filenames using the new method name
    summary_report_path = os.path.join(output_dir, "cluster_summary_group_constrained.csv")
    interactive_map_path = os.path.join(output_dir, "cluster_map_group_constrained.html")
    static_map_path = os.path.join(output_dir, "static_cluster_map_group_constrained.png")
    # Optional: Save intermediate GDFs for debugging
    # roads_grouped_path = os.path.join(output_dir, "roads_with_groups.gpkg")
    # points_associated_path = os.path.join(output_dir, "points_associated_with_groups.gpkg")

    try:
        # 1. Read Accident Data (Points)
        print(f"\n[Step 1/9] Reading accident data: {accident_shp_filename}")
        points_gdf = read_point_shp_data(accident_shp_filename)
        if points_gdf is None: print("ERROR: Failed to read point data. Aborting."); return
        if points_gdf.crs is None: print("ERROR: Point data missing CRS. Aborting."); return
        original_point_count = len(points_gdf)
        print(f"Read {original_point_count} accident points.")

        # 2. Read Road Network Data (Lines)
        print(f"\n[Step 2/9] Reading road network data: {road_shp_filename}")
        road_gdf, road_crs = read_line_shp_data(road_shp_filename)
        if road_gdf is None: print("ERROR: Failed to read valid road network data. Aborting."); return
        if road_crs is None: print("ERROR: Road data missing CRS. Aborting."); return
        print(f"Read {len(road_gdf)} road segments.")

        # 3. CRS Alignment & Projection Check (CRITICAL)
        print("\n[Step 3/9] Checking and aligning Coordinate Reference Systems (CRS)...")
        if points_gdf.crs != road_gdf.crs:
             print(f"   - CRS Mismatch Detected: Points ({points_gdf.crs}) vs Roads ({road_gdf.crs})")
             print(f"   - Attempting to reproject roads to match points CRS: {points_gdf.crs}...")
             try:
                 road_gdf = road_gdf.to_crs(points_gdf.crs)
                 print(f"   - Road network successfully reprojected.")
                 road_crs = road_gdf.crs # Update road_crs variable
             except Exception as e:
                 print(f"   - ERROR: Failed to reproject road network: {e}.")
                 print("   - CRS must match for spatial operations. Aborting.")
                 return
        else:
             print(f"   - Point and road CRS match: {points_gdf.crs}")

        # Check if the final CRS is projected (essential for distance calcs)
        if points_gdf.crs.is_geographic:
             print("\n   --- CRITICAL ERROR: GEOGRAPHIC CRS DETECTED ---")
             print(f"   - Current CRS: {points_gdf.crs}")
             print("   - Data MUST be in a PROJECTED CRS (e.g., UTM, State Plane) for accurate distance calculations (eps, grouping_threshold).")
             print("   - Please reproject BOTH shapefiles to a suitable PROJECTED CRS before running this script.")
             print("   - Aborting analysis.")
             return # Stop execution
        else:
             print(f"   - Data is in a projected CRS: {points_gdf.crs.name}")
             try:
                crs_unit = points_gdf.crs.axis_info[0].unit_name
                print(f"   - Ensure 'eps' ({eps}) and 'grouping_threshold' ({grouping_threshold}) are specified in '{crs_unit}'.")
             except: print("   - Ensure 'eps' and 'grouping_threshold' are in the correct units for this CRS.")


        # 4. Group Nearby Roads
        print(f"\n[Step 4/9] Grouping nearby roads (threshold: {grouping_threshold})...")
        road_group_map = group_nearby_roads(road_gdf, grouping_threshold)
        if road_group_map is None:
             print("ERROR: Failed to group nearby roads. Cannot proceed. Aborting."); return
        # Optional: Add group_id to road_gdf itself for inspection/saving
        try:
             road_gdf['group_id'] = road_gdf['road_uid'].map(road_group_map)
             # Optional Save: road_gdf.to_file(roads_grouped_path, driver='GPKG')
        except Exception as e:
            print(f"Warning: Could not add 'group_id' column to road_gdf: {e}")


        # 5. Pre-compute Road GROUP Intersections
        print("\n[Step 5/9] Pre-computing road group intersections...")
        group_intersections = precompute_group_intersections(road_gdf, road_group_map)
        if group_intersections is None:
             print("ERROR: Failed to compute group intersections. Cannot proceed. Aborting."); return


        # 6. Associate Points with Nearest Roads & Assign Group ID
        print("\n[Step 6/9] Associating accidents with nearest roads and assigning group IDs...")
        if not hasattr(road_gdf, 'sindex') or road_gdf.sindex is None:
             print("   - Building spatial index for roads (for sjoin_nearest)...")
             road_gdf.sindex

        try:
            # Use sjoin_nearest to find the index of the nearest road feature
            points_gdf_joined = gpd.sjoin_nearest(points_gdf, road_gdf[['geometry', 'road_uid', 'group_id']],
                                                 how='left', max_distance=None, # Find nearest regardless of distance initially
                                                 distance_col='dist_to_road') # Keep distance info if needed

            # Check for points that might not have joined (shouldn't happen with max_distance=None unless road_gdf is empty)
            failed_join_count = points_gdf_joined['index_right'].isna().sum()
            if failed_join_count > 0:
                 print(f"Warning: {failed_join_count} points could not be joined to any road feature (unexpected).")
                 # These points will have NaN group_id, handled by DBSCAN

            # The 'group_id' column should now be directly available from the join
            # Rename 'group_id_right' if geopandas adds suffix, otherwise it's just 'group_id'
            if 'group_id_right' in points_gdf_joined.columns:
                points_gdf_joined.rename(columns={'group_id_right': 'group_id'}, inplace=True)
            elif 'group_id' not in points_gdf_joined.columns:
                 print("ERROR: 'group_id' column not found after sjoin_nearest. Check join results.")
                 return

             # Keep only essential columns for clustering + original point index if needed later
            cols_to_keep = list(points_gdf.columns) + ['group_id', 'dist_to_road']
            # Remove potential duplicates created by sjoin if a point is equidistant to multiple nearest roads
            # Keep the first match (usually arbitrary but consistent)
            # Check for duplicated index values after the join. Keep the 'first' occurrence.
            # This handles cases where one point might be equidistant to multiple road segments.
            is_duplicate_index = points_gdf_joined.index.duplicated(keep='first')

            # Keep only the rows where the index is NOT duplicated
            points_gdf_for_clustering = points_gdf_joined[~is_duplicate_index].copy()

            # Check association quality
            points_associated_count = points_gdf_for_clustering['group_id'].notna().sum()
            unassociated_count = original_point_count - points_associated_count
            print(f"   - Associated {points_associated_count} of {original_point_count} points with a road group.")
            if unassociated_count > 0:
                 print(f"   - Note: {unassociated_count} points remain unassociated (likely no nearby road found within limits, or join issue). They will be treated as potential noise.")
            # Optional Save: points_gdf_for_clustering.to_file(points_associated_path, driver='GPKG')

        except Exception as e:
            print(f"ERROR during point-road association (sjoin_nearest): {e}")
            import traceback
            traceback.print_exc()
            return


        # 7. Perform Group-Constrained DBSCAN
        print(f"\n[Step 7/9] Running Road-Group-Constrained DBSCAN (eps={eps}, min_pts={min_pts})...")
        # Ensure coordinates are extracted correctly
        if points_gdf_for_clustering.geometry.empty:
             print("ERROR: No point geometries available for clustering after association. Aborting.")
             return
        try:
            points_coords = list(zip(points_gdf_for_clustering.geometry.x, points_gdf_for_clustering.geometry.y))
            if not points_coords:
                 print("ERROR: Failed to extract point coordinates for DBSCAN. Aborting.")
                 return
        except Exception as e:
             print(f"ERROR extracting point coordinates: {e}. Aborting.")
             return

        cluster_labels = dbscan_constrained_groups(
            eps=eps,
            min_pts=min_pts,
            points_coords=points_coords,
            points_gdf_with_groups=points_gdf_for_clustering, # Pass the GDF with group_id
            group_intersections=group_intersections
        )

        if not cluster_labels or len(cluster_labels) != len(points_gdf_for_clustering):
            print("ERROR: Group-constrained DBSCAN did not return valid labels or label count mismatch. Aborting."); return

        # Add cluster labels back to the main GeoDataFrame for analysis/plotting
        points_gdf_for_clustering['cluster'] = cluster_labels


        # 8. Analyze and Summarize Results
        print("\n[Step 8/9] Analyzing and summarizing clustering results...")
        print("\n--- Group-Constrained DBSCAN Results ---")
        noise_count = (points_gdf_for_clustering['cluster'] == -1).sum()
        clustered_points_gdf = points_gdf_for_clustering[points_gdf_for_clustering['cluster'] > 0].copy() # Filter > 0 for clusters
        num_clusters = clustered_points_gdf['cluster'].nunique()

        print(f" - Total accident points processed: {len(points_gdf_for_clustering)}")
        print(f" - Number of clusters found: {num_clusters}")
        print(f" - Number of points classified as Noise: {noise_count}")
        unclassified_count = (points_gdf_for_clustering['cluster'] == 0).sum() # Should be 0 after DBSCAN
        if unclassified_count > 0: print(f" - Warning: {unclassified_count} points remained unclassified (label 0).")

        summary_data = []
        centroids_gdf = None
        hulls_gdf = None

        if num_clusters > 0:
            print("\n--- Calculating Cluster Characteristics ---")
            cluster_centroids_list = []
            cluster_hulls_list = []

            # Group by the calculated cluster label
            try:
                cluster_groups = clustered_points_gdf.groupby('cluster')

                for cluster_id, group in tqdm.tqdm(cluster_groups, total=num_clusters, desc="Summarizing Clusters"):
                    cluster_size = len(group)
                    centroid = None
                    hull_geom = None

                    # Calculate Centroid
                    try:
                         # Using unary_union first is safer for centroid calculation on multi-part clusters
                         cluster_geom_union = unary_union(group.geometry)
                         centroid = cluster_geom_union.centroid
                         cluster_centroids_list.append({'cluster': cluster_id, 'geometry': centroid, 'size': cluster_size})
                    except Exception as e: print(f"  Warning: Centroid calculation failed for Cluster {cluster_id}: {e}")

                    # Calculate Convex Hull (only if >= 3 points)
                    if cluster_size >= 3:
                        try:
                            # Use unary_union before convex_hull for robustness
                            hull_geom = unary_union(group.geometry).convex_hull
                            # Ensure the hull is a Polygon (could be Line or Point for collinear points)
                            if hull_geom and hull_geom.geom_type == 'Polygon':
                                cluster_hulls_list.append({'cluster': cluster_id, 'geometry': hull_geom, 'area': hull_geom.area})
                            else: hull_geom = None # Reset if not a polygon
                        except Exception as e: print(f"  Warning: Convex Hull calculation failed for Cluster {cluster_id}: {e}")

                    # Append summary data for CSV report
                    summary_row = {'Cluster_ID': cluster_id,
                                   'Size': cluster_size,
                                   'Centroid_X': centroid.x if centroid else None,
                                   'Centroid_Y': centroid.y if centroid else None,
                                   'Hull_Area': hull_geom.area if hull_geom else None}
                    summary_data.append(summary_row)

                # Create GeoDataFrames for centroids and hulls if any were created
                if cluster_centroids_list:
                    centroids_gdf = gpd.GeoDataFrame(cluster_centroids_list, crs=points_gdf_for_clustering.crs)
                if cluster_hulls_list:
                    hulls_gdf = gpd.GeoDataFrame(cluster_hulls_list, crs=points_gdf_for_clustering.crs)

                # Save summary CSV report
                if summary_data:
                    summary_df = pd.DataFrame(summary_data).sort_values(by='Size', ascending=False).set_index('Cluster_ID')
                    try:
                        summary_df.to_csv(summary_report_path)
                        print(f"\nGroup-constrained cluster summary report saved to '{summary_report_path}'")
                    except Exception as e: print(f"\nError saving summary report CSV: {e}")
                else: print("\nNo cluster summary data generated.")

            except Exception as e:
                 print(f"ERROR during cluster analysis (grouping/summarizing): {e}")
                 traceback.print_exc()
        else:
            print("\nNo clusters found (num_clusters = 0). Skipping summary and hull/centroid generation.")


        # 9. Plotting and Saving Outputs
        print("\n[Step 9/9] Generating output maps...")
        plot_clusters_with_roads_interactive(
            points_gdf=points_gdf_for_clustering, # Pass GDF with 'cluster' and 'group_id'
            road_gdf=road_gdf,
            centroids_gdf=centroids_gdf,
            hulls_gdf=hulls_gdf,
            output_filename=interactive_map_path,
            generate_heatmap=generate_heatmap
        )

        if generate_static:
            generate_static_plot(
                points_gdf=points_gdf_for_clustering, # Pass GDF with 'cluster'
                road_gdf=road_gdf,
                output_filename=static_map_path
            )

    except (TypeError, ValueError, FileNotFoundError, ImportError, Exception) as e:
        print(f"\n--- CRITICAL ERROR ENCOUNTERED IN WORKFLOW ---")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Message: {e}")
        print("Traceback:")
        import traceback
        traceback.print_exc()
        print("-----------------------------------------------")
        print("Analysis aborted due to error.")

    finally:
        print("\n--- Analysis Workflow Finished ---")


# --- Main Execution ---
if __name__ == "__main__":
    print("========================================================")
    print(" Starting Road-Group-Constrained Accident Clustering ")
    print("========================================================")

    # --- Configuration ---
    # >>> CRITICAL: Use PROJECTED Shapefiles with matching CRS <<<
    # Verify these files exist and are in a suitable PROJECTED CRS (e.g., UTM Meters)
    accident_shapefile = "Accidents_Projected_Meters.shp"
    road_shapefile = "Selected_Road_Features.shp"

    # Define output directory name based on the method
    output_directory = "group_constrained_clustering_results_v3" # Changed version

    # --- Road Grouping Parameter ---
    # Max distance between road segments to be considered part of the same group (e.g., parallel lanes)
    # UNITS MUST MATCH THE PROJECTED CRS (e.g., meters)
    road_grouping_threshold = 20.0 # Example: 20 meters

    # --- DBSCAN Parameters ---
    # Max spatial distance between points to be considered neighbors
    # UNITS MUST MATCH THE PROJECTED CRS (e.g., meters)
    epsilon = 120.0       # Example: 100 meters

    # Minimum number of points required to form a dense region (core point)
    # A point needs MinPts neighbors (within eps AND connected via groups) to be a core point.
    minimum_points = 5 # Example: Core point needs itself + 5 neighbors meeting criteria

    # --- Output Generation Options ---
    create_heatmap_layer = True # Include heatmap layer in interactive map (can be toggled off)
    create_static_map = True    # Generate the static PNG map output

    # --- End Configuration ---

    print("\n--- Configuration ---")
    print(f"Accident data:          '{accident_shapefile}'")
    print(f"Road network:         '{road_shapefile}'")
    print(f"Output directory:     '{output_directory}'")
    print(f"Road Group Threshold: {road_grouping_threshold} (units of CRS)")
    print(f"DBSCAN eps:           {epsilon} (units of CRS)")
    print(f"DBSCAN min_pts:       {minimum_points}")
    print(f"Generate Heatmap:     {create_heatmap_layer}")
    print(f"Generate Static Map:  {create_static_map}")
    print("---------------------\n")

    # --- Library Check ---
    print("Checking required libraries...")
    try:
        import folium, geopandas, pandas, shapely, branca, jinja2, tqdm
        from shapely.ops import unary_union
        from matplotlib import pyplot # Check specific submodule
        print("All required libraries seem to be installed.")
    except ImportError as e:
        print(f"\n--- MISSING LIBRARY ---: {e}")
        print("Please install the required libraries:")
        print("pip install geopandas folium matplotlib numpy pandas shapely branca Jinja2 tqdm")
        exit(1) # Exit if libraries are missing

    # --- Run the Main Analysis Function ---
    analyze_accident_clusters_with_roads(
        accident_shp_filename=accident_shapefile,
        road_shp_filename=road_shapefile,
        eps=epsilon,
        min_pts=minimum_points,
        grouping_threshold=road_grouping_threshold, # Pass the new threshold
        output_dir=output_directory,
        generate_heatmap=create_heatmap_layer,
        generate_static=create_static_map
    )

    print("\n========================================================")
    print(f"Analysis script finished.")
    print(f"Check the output directory: '{output_directory}'")
    print("Review the .html map and .csv summary.")
    print("========================================================")

    # --- Usage Notes Reminder ---
    # 1. **CRITICAL:** Both shapefiles MUST be in the same Projected CRS.
    # 2. The `road_grouping_threshold` defines how close roads (like lanes) must be to form a group.
    # 3. `epsilon` is the spatial search radius for DBSCAN points.
    # 4. Connectivity requires points within `epsilon` AND their associated road groups must be connected (same group, direct intersect, or 1-step indirect intersect).
    # 5. Performance depends on data size. Grouping and intersection steps can take time.
    # 6. Check popups on the interactive map for `Road Group ID`. Noise points may show 'Not Associated'.