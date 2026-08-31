# -*- coding: utf-8 -*-
import math
import os
import geopandas as gpd
# ----- NEW IMPORTS for Categorical Colors -----
import matplotlib.cm as cm
import matplotlib.colors as mcolors
# --------------------------------------------
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import Point
from shapely.ops import unary_union
import folium
from folium.plugins import HeatMap
from branca.colormap import linear, LinearColormap # Keep import for potential future use, but won't add size legend to map
from branca.element import MacroElement
from jinja2 import Template
import warnings
import tqdm
import traceback

# --- Standard Warnings Configuration ---
warnings.filterwarnings("ignore", message="Iteration over multi-part geometry collections is deprecated")
warnings.filterwarnings("ignore", category=UserWarning, message=".*CRS mismatch between CRS.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*`buffer` is deprecated.*")
# Ignore specific Matplotlib warning about cmap registration if it occurs
warnings.filterwarnings("ignore", message="The get_cmap function was deprecated")


# --- CONFIGURATION FOR BUFFERED INTERSECTION CHECK ---
# Tolerance (in projected CRS units, e.g., meters) to bridge small gaps
# between intersecting road segments. Set to 0 for exact intersections only.
INTERSECTION_BUFFER_TOLERANCE = 0 # Example: Use 0 meters tolerance for exact


# --- Custom Scale Control Class (Unchanged) ---
class ScaleControl(MacroElement):
    """Adds a scale bar to the Folium map."""
    def __init__(self, position='bottomleft', metric=True, imperial=False, max_width=100):
        super().__init__()
        self._template = Template(u"""
            {% macro script(this, kwargs) %}
                L.control.scale({
                    position: '{{ this.position }}', metric: {{ this.metric | lower }},
                    imperial: {{ this.imperial | lower }}, maxWidth: {{ this.max_width }}
                }).addTo({{ this._parent.get_name() }});
            {% endmacro %}
        """)
        self.position, self.metric, self.imperial, self.max_width = position, metric, imperial, max_width

def calculate_distance(point1, point2):
    """Calculates the Euclidean distance between two 2D points."""
    if not (isinstance(point1, (list, tuple)) and isinstance(point2, (list, tuple))): raise TypeError("Inputs must be lists/tuples.")
    if len(point1) != 2 or len(point2) != 2: raise ValueError("Points must be 2D.")
    return math.hypot(point1[0] - point2[0], point1[1] - point2[1])

# --- Road Network Preprocessing ---

# Renamed and Modified: Precomputes SEGMENT intersections
def precompute_segment_intersections(road_gdf, intersection_tolerance=0.0):
    """
    Pre-computes which road *segments* intersect each other geometrically,
    using a spatial index and optional buffer tolerance.
    Assumes road_gdf is in a PROJECTED CRS. Tolerance is in CRS units.

    Args:
        road_gdf (geopandas.GeoDataFrame): Road network with 'road_uid' and geometry.
        intersection_tolerance (float): Buffer distance for checking near intersections.

    Returns:
        dict: Maps road_uid to a set of intersecting road_uids. None on error.
    """
    if road_gdf is None or road_gdf.empty or 'road_uid' not in road_gdf.columns:
         print("ERROR: Invalid input for segment intersection precomputation."); return None
    if not road_gdf.crs or not road_gdf.crs.is_projected:
         print(f"ERROR: Road GDF must be projected. Found: {road_gdf.crs}"); return None

    print("\nPerforming geometry/index checks before intersection precomputation...")
    invalid_mask = ~road_gdf.geometry.is_valid
    if invalid_mask.any(): print(f"  WARNING: Found {invalid_mask.sum()} invalid geometries.")
    empty_mask = road_gdf.geometry.is_empty
    if empty_mask.any(): print(f"  WARNING: Found {empty_mask.sum()} empty geometries.")
    if not isinstance(road_gdf.index, pd.RangeIndex): print(f"  WARNING: road_gdf index not RangeIndex ({type(road_gdf.index)}).")

    if not hasattr(road_gdf, 'sindex') or road_gdf.sindex is None:
        print("  Building spatial index...")
        try: road_gdf.sindex # Trigger index creation
        except Exception as e: print(f"ERROR: Failed to build spatial index: {e}"); return None

    print("Pre-computing road segment intersections using spatial index...")
    try: unit_name = road_gdf.crs.axis_info[0].unit_name
    except: unit_name = "CRS units"
    if intersection_tolerance > 0: print(f"   - Using buffered check: tolerance = {intersection_tolerance:.2f} {unit_name}")
    else: print("   - Using exact geometric intersection check.")

    # Initialize dictionary mapping each road_uid to an empty set
    segment_intersections = {uid: set() for uid in road_gdf['road_uid']}

    try:
        # Query spatial index for potential candidates (predicate='intersects')
        print("  Querying spatial index (predicate='intersects')...")
        possible_intersections_indices = road_gdf.sindex.query(road_gdf.geometry, predicate='intersects')

        left_indices, right_indices = possible_intersections_indices
        print(f"  Spatial index query found {len(left_indices)} potential intersection pairs.")

        intersections_found_count = 0
        processed_segment_pairs = set() # Track (uid1, uid2) pairs added

        for i, j in tqdm.tqdm(zip(left_indices, right_indices), total=len(left_indices), desc="  Checking Segment Intersections"):
            if i == j: continue # Skip self-comparison

            try: # Access data using positional index from sindex result
                geom1 = road_gdf.geometry.iloc[i]
                geom2 = road_gdf.geometry.iloc[j]
                road1_uid = road_gdf['road_uid'].iloc[i]
                road2_uid = road_gdf['road_uid'].iloc[j]
            except Exception as e: continue # Skip if data access fails

            # Ensure pair is processed only once (e.g., add (10, 20) but not (20, 10))
            pair_key = tuple(sorted((road1_uid, road2_uid)))
            if pair_key in processed_segment_pairs: continue

            # --- ACTUAL INTERSECTION CHECK ---
            geometries_intersect = False
            try:
                if geom1 is None or geom2 is None or not geom1.is_valid or not geom2.is_valid: pass # Skip invalid
                elif geom1.intersects(geom2): geometries_intersect = True
                elif intersection_tolerance > 0:
                    # Buffer and check intersection only if exact check fails and tolerance > 0
                    buffered_geom1 = geom1.buffer(intersection_tolerance, resolution=2) # Lower resolution for speed
                    if buffered_geom1 and buffered_geom1.is_valid and not buffered_geom1.is_empty:
                        if buffered_geom1.intersects(geom2): geometries_intersect = True
            except Exception: continue # Skip if geometry check errors

            # --- Record result if intersection found ---
            if geometries_intersect:
                segment_intersections.setdefault(road1_uid, set()).add(road2_uid)
                segment_intersections.setdefault(road2_uid, set()).add(road1_uid)
                processed_segment_pairs.add(pair_key)
                intersections_found_count += 1

    except Exception as e: print(f"ERROR during segment intersection computation: {e}"); traceback.print_exc(); return None

    # Calculate the number of unique pairs found
    final_pair_count = intersections_found_count # This already counts unique pairs added
    print(f"Finished pre-computing segment intersections. Found {final_pair_count} intersecting segment pairs.")
    if final_pair_count == 0: print("  WARNING: No intersecting segment pairs found with current tolerance.")
    return segment_intersections


# --- Modified DBSCAN Core Functions (Segment Constrained) ---
def find_neighbors_segment_constrained(point_index, eps, points_coords, points_gdf_with_segments, segment_intersections):
    """
    Finds neighbors based on eps distance and road segment connectivity.
    Connectivity: same segment, directly intersecting segment, or indirectly
                  intersecting segment (1 hop).
    """
    neighbors = []
    try: target_point_coords = points_coords[point_index]
    except IndexError: print(f"ERROR find_neighbors: index {point_index} invalid coords."); return []
    try: target_road_uid = points_gdf_with_segments.iloc[point_index]['road_uid'] # Assumes column 'road_uid' exists
    except IndexError: print(f"ERROR find_neighbors: index {point_index} invalid GDF."); return []
    except KeyError: print("ERROR find_neighbors: 'road_uid' column missing."); return[]

    # Treat NaN or -999 road_uid as unassociated
    is_target_unassociated = pd.isna(target_road_uid) or target_road_uid == -999
    # Get directly intersecting segments ONLY if target is associated
    direct_neighbors_uids = set() if is_target_unassociated else segment_intersections.get(target_road_uid, set())

    for i, point_coords in enumerate(points_coords):
        if i == point_index: continue # Don't compare point to itself
        distance = calculate_distance(point_coords, target_point_coords)
        if distance <= eps:
            # Check connectivity constraint ONLY if distance constraint met
            try: neighbor_road_uid = points_gdf_with_segments.iloc[i]['road_uid']
            except IndexError: continue # Skip if neighbor index invalid
            is_neighbor_unassociated = pd.isna(neighbor_road_uid) or neighbor_road_uid == -999

            # --- Connectivity Logic ---
            if is_target_unassociated:
                # Connect unassociated points only to other unassociated points within eps
                if is_neighbor_unassociated: neighbors.append(i)
            else: # Target point IS associated with a road segment
                if is_neighbor_unassociated: continue # Don't connect associated points to unassociated ones via network constraint

                # Check constraints:
                # 1. Same segment?
                if target_road_uid == neighbor_road_uid: neighbors.append(i); continue
                # 2. Directly intersecting segment? (segment_intersections maps uid -> {intersecting_uids})
                if neighbor_road_uid in direct_neighbors_uids: neighbors.append(i); continue
                # 3. Indirectly intersecting segment (1 hop)?
                # Check if any segment intersecting the neighbor also intersects the target
                neighbor_direct_uids = segment_intersections.get(neighbor_road_uid, set())
                if not direct_neighbors_uids.isdisjoint(neighbor_direct_uids): # Efficient check for common elements
                    neighbors.append(i); continue
                # --- End Connectivity Logic ---
    return neighbors

def expand_cluster_segment_constrained(point_index, neighbors_indices, cluster_label, eps, min_pts, labels, points_coords, points_gdf_with_segments, segment_intersections):
    """Expands cluster using segment-constrained neighbor finding."""
    labels[point_index] = cluster_label
    i = 0
    # Use a queue for Breadth-First Search (BFS) exploration
    neighbors_queue = list(neighbors_indices)
    processed_or_queued = set(neighbors_indices) # Keep track of points already in queue or processed
    processed_or_queued.add(point_index) # Add starting point

    while i < len(neighbors_queue):
        current_neighbor_index = neighbors_queue[i]; i += 1 # Dequeue
        if current_neighbor_index >= len(labels): print(f"ERROR expand_cluster: index {current_neighbor_index} invalid."); continue # Safety check

        # Process the neighbor if it's noise or unclassified
        if labels[current_neighbor_index] in [-1, 0]: # If noise or unclassified
            if labels[current_neighbor_index] == -1:
                 labels[current_neighbor_index] = cluster_label # Change noise to border point of current cluster
            elif labels[current_neighbor_index] == 0: # If unclassified
                 labels[current_neighbor_index] = cluster_label
                 # Find *its* segment-constrained neighbors
                 current_neighbors_neighbors = find_neighbors_segment_constrained(current_neighbor_index, eps, points_coords, points_gdf_with_segments, segment_intersections)

                 # If it's a core point, add its neighbors to the queue
                 if len(current_neighbors_neighbors) >= (min_pts - 1): # Core point check (need min_pts - 1 neighbors + itself)
                     for nn_idx in current_neighbors_neighbors:
                         if nn_idx not in processed_or_queued:
                             processed_or_queued.add(nn_idx) # Mark as queued
                             neighbors_queue.append(nn_idx) # Enqueue
        # If neighbor already belongs to another cluster (labels[current_neighbor_index] > 0), do nothing

def dbscan_segment_constrained(eps, min_pts, points_coords, points_gdf_with_segments, segment_intersections):
    """Performs DBSCAN constrained by road segment connectivity."""
    if not points_coords or points_gdf_with_segments is None or points_gdf_with_segments.empty:
        print("Warning: DBSCAN input data missing."); return []

    # Ensure consistent indexing
    points_gdf_with_segments = points_gdf_with_segments.reset_index(drop=True)
    if len(points_coords) != len(points_gdf_with_segments):
        raise ValueError(f"DBSCAN coord/GDF length mismatch: {len(points_coords)} vs {len(points_gdf_with_segments)}.")
    if segment_intersections is None:
        print("Warning: Segment intersections missing. Returning all noise.");
        return [-1] * len(points_coords)
    if not any(segment_intersections.values()):
        print("INFO: 0 segment intersections found. Constraint limited to points on the same segment.")
    if 'road_uid' not in points_gdf_with_segments.columns:
        raise ValueError("Missing 'road_uid' column for segment-constrained DBSCAN.")

    n_points = len(points_coords)
    labels = [0] * n_points # 0: unclassified, -1: noise, >0: cluster ID
    cluster_label = 0

    print("Starting segment-constrained DBSCAN...")
    for point_index in tqdm.tqdm(range(n_points), desc="Clustering Points (Segments)"):
        if labels[point_index] != 0: continue # Skip already classified points

        # Find neighbors based on distance AND segment connectivity
        neighbors_indices = find_neighbors_segment_constrained(point_index, eps, points_coords, points_gdf_with_segments, segment_intersections)

        # Check if core point (itself + neighbors >= min_pts)
        if len(neighbors_indices) < (min_pts - 1): # Need min_pts - 1 neighbors
             # Not a core point, mark as noise (might be changed later if border point)
             if labels[point_index] == 0: labels[point_index] = -1
        else:
            # Core point, start a new cluster
            cluster_label += 1
            expand_cluster_segment_constrained(point_index, neighbors_indices, cluster_label, eps, min_pts, labels, points_coords, points_gdf_with_segments, segment_intersections)

    print(f"Segment-constrained DBSCAN complete. Found {cluster_label} potential clusters.")
    return labels


# --- Data Reading Functions (Unchanged) ---
def read_point_shp_data(filename):
    """Reads point shapefile, checks validity, ensures Point geometry, resets index."""
    if not isinstance(filename, str): raise TypeError("Filename must be a string.")
    if not os.path.exists(filename): raise FileNotFoundError(f"Point shapefile not found: {filename}")
    try:
        gdf = gpd.read_file(filename)
        if gdf.empty: print(f"Warning: Point shapefile '{filename}' is empty."); return None
        if 'geometry' not in gdf.columns: raise ValueError(f"'{filename}' lacks 'geometry'.")

        original_count = len(gdf)
        # Ensure Point geometry type
        gdf = gdf[gdf.geometry.geom_type == 'Point'].copy()
        if gdf.empty: print(f"Warning: No Point geometries found in '{filename}'."); return None
        initial_point_count = len(gdf)

        # Check validity and remove empty/invalid points
        gdf = gdf[gdf.geometry.is_valid & ~gdf.geometry.is_empty].reset_index(drop=True)
        valid_point_count = len(gdf)
        if valid_point_count < initial_point_count:
            print(f"Warning: Removed {initial_point_count - valid_point_count} invalid/empty points.")
        if valid_point_count < original_count:
             print(f"Warning: Filtered {original_count - valid_point_count} non-Point/invalid features.")
        if gdf.empty: print(f"Warning: No valid Point geometries remain in '{filename}'."); return None

        print(f"Successfully read {len(gdf)} valid points from '{filename}'.")
        print(f"  Point CRS: {gdf.crs}")
        if gdf.crs is None: print("ERROR: Point shapefile missing CRS. Cannot proceed."); return None
        return gdf
    except ImportError: raise ImportError("Requires 'geopandas'. Install with 'pip install geopandas'.")
    except Exception as e: raise Exception(f"Error reading point shapefile '{filename}': {e}")

def read_line_shp_data(filename, fix_geometry=True):
    """Reads line shapefile, checks/fixes validity, assigns 'road_uid', resets index."""
    if not isinstance(filename, str): raise TypeError("Filename must be a string.")
    if not os.path.exists(filename): raise FileNotFoundError(f"Road shapefile not found: {filename}")
    try:
        gdf = gpd.read_file(filename)
        if gdf.empty: print(f"Warning: Road shapefile '{filename}' is empty."); return None, None
        if 'geometry' not in gdf.columns: raise ValueError(f"'{filename}' lacks 'geometry'.")

        original_count = len(gdf)
        # Ensure LineString or MultiLineString geometry types
        line_types = ['LineString', 'MultiLineString']
        gdf = gdf[gdf.geometry.geom_type.isin(line_types)].copy()
        if gdf.empty: print(f"Warning: No LineString/MultiLineString geometries found in '{filename}'."); return None, None
        if len(gdf) < original_count:
            print(f"Warning: Filtered {original_count - len(gdf)} non-Line features.")

        gdf = gdf.reset_index(drop=True)
        gdf['road_uid'] = gdf.index # Assign unique ID based on cleaned index

        # Check for invalid geometries
        invalid_geom = gdf[~gdf.geometry.is_valid]
        num_invalid = len(invalid_geom)
        if num_invalid > 0:
            print(f"Warning: Found {num_invalid} invalid road geometries.")
            if fix_geometry:
                print("  Attempting fix using buffer(0)...")
                try:
                     # Check if buffer(0) changes geometry types inappropriately
                     original_types = gdf.geometry.geom_type.copy()
                     gdf.geometry = gdf.geometry.buffer(0)
                     new_types = gdf.geometry.geom_type
                     # Identify rows that were LineString/MultiLineString but are no longer
                     mismatched = (original_types.isin(line_types)) & (~new_types.isin(line_types))
                     if mismatched.any():
                         print(f"ERROR: buffer(0) changed geometry type for {mismatched.sum()} roads (e.g., to Polygon). Cannot proceed with fix."); return None, None

                     # Check if any geometries are still invalid or became empty after buffer(0)
                     still_invalid = gdf[~gdf.geometry.is_valid | gdf.geometry.is_empty]
                     if not still_invalid.empty:
                         print(f"ERROR: {len(still_invalid)} roads remain invalid/empty after buffer(0). Cannot proceed with fix."); return None, None
                     else:
                         print(f"  Successfully fixed {num_invalid} invalid geometries.")
                except Exception as buffer_err:
                    print(f"ERROR during buffer(0) fix: {buffer_err}. Cannot proceed with fix."); return None, None
            else:
                print("  Fixing disabled. Proceeding with invalid geometries may cause errors.")

        # Final check for empty geometries after potential fixing
        gdf = gdf[~gdf.geometry.is_empty].reset_index(drop=True)
        if gdf.empty: print(f"Warning: No valid geometries remain in roads after processing."); return None, None

        print(f"Successfully read {len(gdf)} line features from '{filename}'. Assigned 'road_uid'.")
        initial_crs = gdf.crs
        print(f"  Initial Road CRS: {initial_crs}")
        if initial_crs is None: print("ERROR: Road shapefile missing CRS. Cannot proceed."); return None, None
        return gdf, initial_crs
    except ImportError: raise ImportError("Requires 'geopandas'. Install with 'pip install geopandas'.")
    except Exception as e: raise Exception(f"Error reading line shapefile '{filename}': {e}")


# --- Plotting Functions (MODIFIED Interactive Plot for Categorical Colors) ---
def plot_clusters_with_roads_interactive(
    points_gdf, road_gdf, centroids_gdf, hulls_gdf,
    output_filename="cluster_map.html", generate_heatmap=True
    ):
    """
    Generates interactive map using CATEGORICAL colors for clusters.
    Shows Road UID in point popup. Shows cluster size in hull tooltip.
    """
    if points_gdf is None or points_gdf.empty or 'cluster' not in points_gdf.columns:
        print("Warning: Cannot plot clusters."); return
    has_road_uid = 'road_uid' in points_gdf.columns
    if not has_road_uid: print("Info: 'road_uid' column not found in points GDF for interactive plot popup.")

    print(f"Generating interactive map (Categorical Colors): '{output_filename}'...")
    target_crs_map = "EPSG:4326" # Folium requires Lat/Lon

    # Select and reproject points
    points_cols_to_keep = ['geometry', 'cluster']
    if has_road_uid: points_cols_to_keep.append('road_uid')
    try:
        points_gdf_map = points_gdf[points_cols_to_keep].to_crs(target_crs_map)
        lats = points_gdf_map.geometry.y; lons = points_gdf_map.geometry.x
        if lats.empty or lons.empty: print("Warning: No valid coordinates for map."); return
    except Exception as e: print(f"Error reprojecting points for map: {e}"); return

    # Helper function to reproject other GeoDataFrames
    def reproject_for_map(gdf, name):
        if gdf is None or gdf.empty: return None
        try:
             if gdf.crs is None: print(f"Warning: CRS missing for {name} (map)."); return None
             if gdf.crs == target_crs_map: return gdf
             print(f"  Reprojecting {name} to {target_crs_map} for map...")
             return gdf.to_crs(target_crs_map)
        except Exception as e: print(f"Warning: Error reprojecting {name} for map: {e}"); return None

    roads_gdf_map = reproject_for_map(road_gdf, "Roads")
    centroids_gdf_map = reproject_for_map(centroids_gdf, "Centroids")
    hulls_gdf_map = reproject_for_map(hulls_gdf, "Hulls")

    # Create base map
    map_center = [lats.mean(), lons.mean()]
    m = folium.Map(location=map_center, zoom_start=12, tiles=None)
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(m)
    folium.TileLayer("CartoDB positron", name="CartoDB Positron", show=False).add_to(m)
    folium.TileLayer("CartoDB dark_matter", name="CartoDB Dark", show=False).add_to(m)

    # Add Roads Layer
    if roads_gdf_map is not None:
        road_fg = folium.FeatureGroup(name='Road Network', show=True).add_to(m)
        try:
            # Tooltip for roads - Exclude geometry and UID
            road_tooltip_fields = [col for col in roads_gdf_map.columns if col not in ['geometry', 'road_uid']]
            road_tooltip_aliases = [f'{col}:' for col in road_tooltip_fields]
            if not roads_gdf_map.empty:
                folium.GeoJson(
                    roads_gdf_map,
                    style_function=lambda x: {'color': 'gray', 'weight': 1.5, 'opacity': 0.7},
                    tooltip=folium.features.GeoJsonTooltip(fields=road_tooltip_fields, aliases=road_tooltip_aliases, localize=True, sticky=False)
                ).add_to(road_fg)
        except Exception as e: print(f"Error adding roads GeoJson layer: {e}")

    # --- Calculate cluster sizes (still needed for popups/tooltips) ---
    valid_labels_mask = pd.notna(points_gdf_map['cluster'])
    points_gdf_map_valid = points_gdf_map[valid_labels_mask]
    # Ensure cluster IDs are integers for sorting and lookup
    points_gdf_map_valid['cluster'] = points_gdf_map_valid['cluster'].astype(int)
    unique_labels_valid = sorted(points_gdf_map_valid['cluster'].unique())
    # Get actual cluster IDs (sorted, excluding noise -1)
    cluster_ids = sorted([l for l in unique_labels_valid if l != -1])
    num_clusters = len(cluster_ids)
    cluster_sizes = {}
    if num_clusters > 0:
        valid_clusters_gdf = points_gdf_map_valid[points_gdf_map_valid['cluster'] != -1]
        if not valid_clusters_gdf.empty:
             cluster_sizes = valid_clusters_gdf.groupby('cluster').size().to_dict()

    # --- Generate Categorical Colors for Clusters ---
    cluster_id_to_color = {}
    noise_color = 'grey' # Standard color for noise
    if num_clusters > 0:
        # Choose a colormap suitable for many categories ('turbo' recommended)
        cmap_name = 'turbo'
        try:
            colormap = cm.get_cmap(cmap_name, num_clusters)
            # Generate colors, mapping index (0 to num_clusters-1) to 0-1 range
            # Handle num_clusters=1 case for normalization
            cluster_colors_rgba = [colormap(i / max(1, num_clusters - 1)) for i in range(num_clusters)]
            # Convert RGBA to hex (Folium prefers hex)
            cluster_colors_hex = [mcolors.to_hex(c) for c in cluster_colors_rgba]
            # Map actual cluster IDs (which might not be sequential starting from 1) to colors
            cluster_id_to_color = {cluster_id: cluster_colors_hex[i] for i, cluster_id in enumerate(cluster_ids)}
            print(f"  Generated {len(cluster_id_to_color)} categorical colors using '{cmap_name}' colormap.")
        except Exception as e:
            print(f"Warning: Failed to generate categorical colors using Matplotlib cmap '{cmap_name}': {e}. Falling back to single color.")
            # Fallback if colormap fails
            fallback_color = 'blue'
            cluster_id_to_color = {cluster_id: fallback_color for cluster_id in cluster_ids}

    # --- Create Feature Groups and Add Points (Using Categorical Colors) ---
    cluster_points_fg = {}
    for label in unique_labels_valid: # Iterates through -1 and all positive cluster IDs
        label_int = int(label) # Should already be int, but ensure
        group_name = f"Cluster {label_int}" if label_int != -1 else "Noise Points"
        # Show clusters by default, hide noise layer initially
        cluster_points_fg[label] = folium.FeatureGroup(name=group_name, show=(label_int != -1)).add_to(m)

        points_in_label = points_gdf_map_valid[points_gdf_map_valid['cluster'] == label]
        if points_in_label.empty: continue

        # Get size for popup info (use previously calculated dict)
        cluster_size_popup = cluster_sizes.get(label_int, len(points_in_label)) # Fallback to actual count

        # --- Determine color and radius based on cluster ID (categorical) ---
        if label_int == -1:
            color = noise_color
            radius = 2
        else:
            # Look up the specific color for this cluster ID
            color = cluster_id_to_color.get(label_int, 'black') # Fallback to black if ID somehow missing
            radius = 4
        # --- End Color Determination ---

        # Add points to the appropriate feature group
        for idx, point in points_in_label.iterrows():
            popup_html = f"<b>{group_name}</b><br>Lat: {point.geometry.y:.6f}<br>Lon: {point.geometry.x:.6f}<br>Cluster ID: {int(point['cluster'])}<br>"
            if label_int != -1: popup_html += f"Cluster Size: {cluster_size_popup}<br>" # Use calculated size
            # Show associated Road Segment UID if available
            if has_road_uid:
                road_uid_val = point.get('road_uid', 'N/A') # Get value, default to N/A
                # Display UID or 'N/A' / 'Unassociated'
                road_uid_display = int(road_uid_val) if pd.notna(road_uid_val) and road_uid_val != -999 else 'N/A'
                popup_html += f"Road Segment UID: {road_uid_display}<br>"

            folium.CircleMarker(
                location=[point.geometry.y, point.geometry.x],
                radius=radius,
                color=color, # Use categorical color
                fill=True,
                fill_color=color, # Use categorical color
                fill_opacity=0.7,
                popup=folium.Popup(popup_html, max_width=350) # Embed HTML in popup
            ).add_to(cluster_points_fg[label])

    # Add Centroids Layer (styling unchanged, using default blue icon)
    if centroids_gdf_map is not None and not centroids_gdf_map.empty:
        centroid_fg = folium.FeatureGroup(name='Cluster Centroids', show=False).add_to(m)
        for idx, centroid in centroids_gdf_map.iterrows():
            try: cluster_id_int = int(centroid.get('cluster', 'N/A'))
            except: cluster_id_int = 'N/A'
            # Get size from the 'size' column if it exists in centroids_gdf_map, else from dict
            size = centroid.get('size', cluster_sizes.get(cluster_id_int, 'N/A')) # Use pre-calculated size if available
            popup_text = f"Centroid Cluster {cluster_id_int}{f' (Size: {size})' if size != 'N/A' else ''}"
            folium.Marker(location=[centroid.geometry.y, centroid.geometry.x], popup=popup_text, icon=folium.Icon(color='blue', icon='info-sign')).add_to(centroid_fg)

    # Add Hulls Layer (Using Categorical Colors for Fill)
    if hulls_gdf_map is not None and not hulls_gdf_map.empty:
        # --- Add cluster size to hulls_gdf_map for tooltip (if not already present) ---
        # This assumes hulls_gdf was created with a 'cluster' column
        if 'size' not in hulls_gdf_map.columns and cluster_sizes and 'cluster' in hulls_gdf_map.columns:
             try:
                 # Map cluster ID to size, fill missing with 0, ensure integer
                 hulls_gdf_map['cluster'] = hulls_gdf_map['cluster'].astype(int) # Ensure cluster ID is int for mapping
                 hulls_gdf_map['size'] = hulls_gdf_map['cluster'].map(cluster_sizes).fillna(0).astype(int)
             except Exception as e:
                 print(f"Warning: Could not map cluster sizes to hulls GDF: {e}")
                 if 'size' not in hulls_gdf_map.columns: hulls_gdf_map['size'] = 'N/A' # Fallback column
        elif 'size' not in hulls_gdf_map.columns: # Fallback if cluster_sizes is empty or 'cluster' column missing
             hulls_gdf_map['size'] = 'N/A'

        hull_fg = folium.FeatureGroup(name='Cluster Convex Hulls', show=False).add_to(m)
        try:
            def hull_style_func(feature):
                """Styles hull using the pre-generated categorical colormap based on cluster ID."""
                try: cluster_id_int = int(feature['properties'].get('cluster', -999))
                except: cluster_id_int = -999 # Default for missing or invalid ID

                # Get fill color from the categorical map
                fill_color = noise_color if cluster_id_int == -1 else cluster_id_to_color.get(cluster_id_int, 'black') # Use black as fallback

                return {'fillColor': fill_color, 'color': 'black', 'weight': 1, 'fillOpacity': 0.4} # Adjusted opacity slightly

            # --- Define tooltip fields and aliases (Conditional) ---
            tooltip_fields = ['cluster']
            tooltip_aliases = ['Cluster ID:']
            # Check if the 'size' column exists and is numeric-like
            if 'size' in hulls_gdf_map.columns and pd.api.types.is_numeric_dtype(hulls_gdf_map['size']):
                tooltip_fields.append('size')
                tooltip_aliases.append('# Points:')
            elif 'size' in hulls_gdf_map.columns: # If size column exists but isn't numeric (e.g., 'N/A')
                tooltip_fields.append('size')
                tooltip_aliases.append('Size Info:') # More generic alias

            folium.GeoJson(
                hulls_gdf_map,
                style_function=hull_style_func, # Use updated style function
                tooltip=folium.features.GeoJsonTooltip(
                    fields=tooltip_fields,    # Use the updated list
                    aliases=tooltip_aliases,   # Use the updated list
                    localize=True,            # Format numbers based on locale
                    sticky=False              # Tooltip disappears on mouse-off
                )
            ).add_to(hull_fg)
        except Exception as e: print(f"Error adding hulls GeoJson layer: {e}")


    # Add Heatmap Layer (Optional)
    if generate_heatmap and not lats.empty:
        try: HeatMap(list(zip(lats, lons)), name='Accident Heatmap', show=False).add_to(m)
        except Exception as e: print(f"Error adding heatmap layer: {e}")

    # --- REMOVED Size Colormap Legend ---
    # Since coloring is now categorical by ID, the size-based legend is not appropriate.
    # if size_colormap: m.add_child(size_colormap)

    # Add Controls
    m.add_child(ScaleControl(position='bottomleft', metric=True, imperial=True)) # Add scale bar
    folium.LayerControl(collapsed=False).add_to(m) # Add layer control panel

    # Save the map
    try:
        m.save(output_filename)
        print(f"Interactive map saved successfully to '{output_filename}'.")
    except Exception as e: print(f"Error saving Folium map: {e}")


# --- Static Plot Function (Unchanged - Already uses categorical colors) ---
def generate_static_plot(points_gdf, road_gdf, output_filename="static_cluster_map.png"):
    """Generates static map using PROJECTED coordinates and categorical cluster colors."""
    if points_gdf is None or points_gdf.empty or 'cluster' not in points_gdf.columns:
        print("Warning: Cannot generate static plot."); return
    if not points_gdf.crs or not points_gdf.crs.is_projected:
        print(f"Warning: Points GDF for static plot not projected ({points_gdf.crs}). Cannot generate reliable static plot."); return
    if road_gdf is not None and (not road_gdf.crs or not road_gdf.crs.is_projected):
        print(f"Warning: Roads GDF for static plot not projected ({road_gdf.crs}). Roads will not be plotted."); road_gdf = None
    # Ensure road GDF CRS matches points GDF CRS if both are projected
    if road_gdf is not None and points_gdf.crs != road_gdf.crs:
         print(f"Warning: CRS mismatch between points ({points_gdf.crs}) and roads ({road_gdf.crs}) for static plot. Roads will not be plotted."); road_gdf = None

    print(f"Generating static map (using projected CRS): '{output_filename}'...")
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    # Plot Roads (if available and valid)
    if road_gdf is not None and not road_gdf.empty:
        try: road_gdf.plot(ax=ax, color='darkgrey', linewidth=0.6, alpha=0.7, zorder=1, label='_nolegend_')
        except Exception as e: print(f"Warning: Could not plot roads: {e}")

    # Plot Clustered Points
    try:
        plot_gdf = points_gdf.copy()
        # Convert cluster ID to numeric, handling potential non-numeric values gracefully
        # Assign -2 to NA/non-numeric values after conversion attempt
        plot_gdf['cluster_plot'] = pd.to_numeric(plot_gdf['cluster'], errors='coerce').fillna(-2).astype(int)

        # Separate valid clusters from noise/unassociated for color mapping
        clusters_only = plot_gdf[plot_gdf['cluster_plot'] >= 0]
        noise_only = plot_gdf[plot_gdf['cluster_plot'] == -1]
        unassoc_only = plot_gdf[plot_gdf['cluster_plot'] == -2]

        num_static_clusters = clusters_only['cluster_plot'].nunique()

        # Choose colormap based on number of clusters
        cmap_static = 'viridis' # Default
        if num_static_clusters > 20: cmap_static = 'turbo' # Good for many distinct colors
        elif num_static_clusters > 10: cmap_static = 'tab20' # Good for up to 20
        elif num_static_clusters > 0: cmap_static = 'tab10'  # Good for up to 10

        # Plot actual clusters using the chosen colormap
        if not clusters_only.empty:
            clusters_only.plot(column='cluster_plot', ax=ax, categorical=True, legend=False, # Generate legend manually later
                               markersize=8, cmap=cmap_static, zorder=3)

        # Plot noise points
        if not noise_only.empty:
            noise_only.plot(ax=ax, color='grey', markersize=5, alpha=0.6, label='Noise', zorder=2)

        # Plot unassociated/NA points (if any resulted from coercion)
        if not unassoc_only.empty:
            unassoc_only.plot(ax=ax, color='lightgrey', marker='x', markersize=5, alpha=0.5, label='Unassociated/NA', zorder=2)

        # --- Create Custom Legend ---
        handles, labels = ax.get_legend_handles_labels() # Get handles/labels from noise/unassoc plots
        # Create handles for the actual clusters
        if num_static_clusters > 0:
            cmap_obj = plt.get_cmap(cmap_static)
            # Get unique cluster IDs present in the data, sorted
            unique_cluster_ids = sorted(clusters_only['cluster_plot'].unique())
            # Create normalized color mapping based on the number of unique clusters
            norm = plt.Normalize(vmin=0, vmax=max(1, num_static_clusters - 1)) # Normalize index 0 to N-1
            # Add a legend entry for each cluster ID
            for i, cluster_id in enumerate(unique_cluster_ids):
                 color = cmap_obj(norm(i)) # Get color based on index
                 handles.append(plt.Line2D([0], [0], marker='o', color='w', label=f'Cluster {cluster_id}',
                                           markerfacecolor=color, markersize=8, linestyle='None'))
                 labels.append(f'Cluster {cluster_id}')

        # Display the combined legend outside the plot area
        ax.legend(handles, labels, title="Cluster ID / Type", loc='upper left', bbox_to_anchor=(1.02, 1), frameon=False)

    except Exception as e: print(f"Error plotting clustered points: {e}")

    # Add titles and labels
    ax.set_title('Accident Clusters (Segment Constrained) and Road Network')
    try: ax.set_xlabel(f"Easting ({points_gdf.crs.axis_info[0].abbreviation})")
    except: ax.set_xlabel("Easting / Longitude")
    try: ax.set_ylabel(f"Northing ({points_gdf.crs.axis_info[1].abbreviation})")
    except: ax.set_ylabel("Northing / Latitude")
    ax.set_axis_on(); ax.grid(True, linestyle='--', alpha=0.5)
    try: plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend
    except Exception: pass

    # Save the plot
    try:
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        print(f"Static map saved successfully to '{output_filename}'.")
    except Exception as e: print(f"Error saving static map: {e}")
    plt.close(fig) # Close the figure to free memory


# --- Main Analysis Function (Calls modified plotting function) ---
def analyze_accident_clusters_with_roads(
    accident_shp_filename,
    road_shp_filename,
    eps,
    min_pts,
    intersection_tolerance,
    output_dir=".",
    generate_heatmap=True,
    generate_static=True
    ):
    """Main analysis workflow using segment-constrained DBSCAN."""
    print("\n--- Starting Analysis Workflow (Segment Constrained) ---")
    # --- Validate Inputs ---
    if not isinstance(accident_shp_filename, str): raise TypeError("Accident file must be string.")
    if not isinstance(road_shp_filename, str): raise TypeError("Road file must be string.")
    if not isinstance(eps, (int, float)) or eps <= 0: raise ValueError("eps must be positive.")
    if not isinstance(min_pts, int) or min_pts < 2: raise ValueError("min_pts must be >= 2.")
    if not isinstance(intersection_tolerance, (int, float)) or intersection_tolerance < 0: raise ValueError("intersection_tolerance must be non-negative.")

    # --- Setup Output ---
    os.makedirs(output_dir, exist_ok=True)
    # Define output filenames based on parameters
    method_suffix = f"seg_intol{intersection_tolerance}_eps{eps}_mp{min_pts}"
    summary_report_path = os.path.join(output_dir, f"cluster_summary_{method_suffix}.csv")
    interactive_map_path = os.path.join(output_dir, f"cluster_map_{method_suffix}.html")
    static_map_path = os.path.join(output_dir, f"static_cluster_map_{method_suffix}.png")
    roads_projected_path = os.path.join(output_dir, f"roads_projected_{method_suffix}.gpkg") # Save projected roads
    points_associated_path = os.path.join(output_dir, f"points_associated_{method_suffix}.gpkg") # Save points with road UID

    try:
        # === STEP 1: Read Accident Data (Points) ===
        print(f"\n[Step 1/7] Reading accident data: {accident_shp_filename}")
        points_gdf = read_point_shp_data(accident_shp_filename)
        if points_gdf is None: return # Error handled in read function
        # CRITICAL: Ensure input points are projected
        if not points_gdf.crs or not points_gdf.crs.is_projected:
             print(f"ERROR: Accident data MUST be in a projected CRS. Found: {points_gdf.crs}. Aborting."); return
        target_crs = points_gdf.crs # Use the CRS of the points data
        print(f"  Using target projected CRS: {target_crs.name} ({target_crs})")

        # === STEP 2: Read & Reproject Road Network Data ===
        print(f"\n[Step 2/7] Reading & Reprojecting roads: {road_shp_filename}")
        # Read roads, attempting geometry fixes
        road_gdf, initial_road_crs = read_line_shp_data(road_shp_filename, fix_geometry=True)
        if road_gdf is None: return # Error handled in read function

        # Reproject roads if their CRS doesn't match the points CRS
        if road_gdf.crs != target_crs:
             print(f"  Reprojecting roads from {initial_road_crs.name} to {target_crs.name}...")
             try: road_gdf = road_gdf.to_crs(target_crs)
             except Exception as e: print(f"ERROR: Failed to reproject roads: {e}. Aborting."); return
        else: print(f"  Road CRS matches target CRS ({target_crs.name}). No reprojection needed.")

        # Final check on projected road data validity
        road_gdf = road_gdf[road_gdf.geometry.is_valid & ~road_gdf.geometry.is_empty].reset_index(drop=True)
        if road_gdf.empty: print("ERROR: No valid roads remain after potential reprojection/cleaning. Aborting."); return
        print(f"  Processed {len(road_gdf)} valid, projected road segments.")

        # Save the processed (projected, cleaned) road data
        try:
             print(f"  Saving projected roads to '{roads_projected_path}'...")
             road_gdf.to_file(roads_projected_path, driver='GPKG')
        except Exception as e: print(f"Warning: Could not save projected roads: {e}")


        # === STEP 3: Precompute Road SEGMENT Intersections ===
        print("\n[Step 3/7] Pre-computing road segment intersections...")
        road_gdf = road_gdf.reset_index(drop=True) # Ensure clean default index for iloc access later
        segment_intersections = precompute_segment_intersections(road_gdf, intersection_tolerance)
        if segment_intersections is None: print("ERROR: Failed computing segment intersections. Aborting."); return

        # === STEP 4: Build Road Spatial Index ===
        print("\n[Step 4/7] Ensuring spatial index exists for roads...")
        try:
            if not hasattr(road_gdf, 'sindex') or road_gdf.sindex is None:
                 print("  Building spatial index for roads...")
                 road_gdf.sindex # Trigger index creation
            print("  Spatial index available.")
        except Exception as e: print(f"ERROR: Failed building spatial index for roads: {e}. Aborting."); return

        # === STEP 5: Associate Points with Nearest Road SEGMENT ===
        print("\n[Step 5/7] Associating accidents with nearest road segments...")
        try:
            # Use only valid road geometries for joining
            valid_road_geoms_gdf = road_gdf[road_gdf.geometry.is_valid & ~road_gdf.geometry.is_empty]
            if len(valid_road_geoms_gdf) < len(road_gdf):
                print(f"  Warning: Joining against {len(valid_road_geoms_gdf)} valid roads (out of {len(road_gdf)} total).")
            if valid_road_geoms_gdf.empty: print("ERROR: No valid roads available for spatial join. Aborting."); return

            points_gdf = points_gdf.reset_index(drop=True) # Ensure clean index
            print("  Performing spatial join (sjoin_nearest)...")
            # Join only necessary columns: geometry and the unique road ID
            points_gdf_joined = gpd.sjoin_nearest(
                points_gdf,
                valid_road_geoms_gdf[['geometry', 'road_uid']], # Join target must have geometry and key
                how='left',                   # Keep all points, add road info if found
                max_distance=None,            # Search indefinitely for nearest
                distance_col='dist_to_road',  # Add column with distance to nearest road
                rsuffix='road'                # Suffix for joined columns (e.g., index_road, road_uid_road)
            )
            print(f"  Spatial join complete.")

            # --- Process Join Results ---
            final_road_col = 'road_uid' # Desired final column name for the road ID in points_gdf
            road_col_suffix = 'road_uid_road' # Default suffix added by sjoin_nearest

            # Rename the joined road UID column if it exists
            if road_col_suffix in points_gdf_joined.columns:
                points_gdf_joined.rename(columns={road_col_suffix: final_road_col}, inplace=True)
            elif final_road_col not in points_gdf_joined.columns:
                # If the original points GDF already had 'road_uid' and it wasn't overwritten,
                # or if the join failed to add it, ensure the column exists, initializing with NaN
                 if final_road_col not in points_gdf_joined.columns:
                     points_gdf_joined[final_road_col] = np.nan
            # else: final_road_col likely existed and was populated by join directly

            # Handle potential duplicates arising from sjoin_nearest if a point is equidistant
            # Keep the first match found
            is_duplicate = points_gdf_joined.index.duplicated(keep='first')
            if is_duplicate.any():
                print(f"  Removing {is_duplicate.sum()} duplicates introduced by spatial join (keeping first match).")
            points_gdf_assoc = points_gdf_joined[~is_duplicate].copy()

            # Assign -999 for failed joins (where index_road is NaN, indicating no nearest road found)
            # or if the road_uid itself was NaN after the join
            failed_join_mask = points_gdf_assoc['index_road'].isna() # Check if the index from roads was joined
            points_gdf_assoc.loc[failed_join_mask, final_road_col] = -999
            # Also fill any remaining NaNs in the road_uid column (e.g., if original road had NaN ID)
            points_gdf_assoc[final_road_col].fillna(-999, inplace=True)
            # Convert to integer type safely
            points_gdf_assoc[final_road_col] = pd.to_numeric(points_gdf_assoc[final_road_col], errors='coerce').fillna(-999).astype(int)


            assoc_count = (points_gdf_assoc[final_road_col] != -999).sum()
            unassoc_count = len(points_gdf_assoc) - assoc_count
            print(f"  Associated {assoc_count} of {len(points_gdf_assoc)} points with a road segment UID.")
            if unassoc_count > 0: print(f"  {unassoc_count} points could not be associated (assigned UID -999).")

            # Save the points with associated road UIDs
            print(f"  Saving associated points to '{points_associated_path}'...")
            # Define columns to save - original point columns + new association info
            cols_to_save = list(points_gdf.columns) + [final_road_col, 'dist_to_road']
            # Ensure only columns that actually exist are selected (in case original GDF changed)
            cols_present = [c for c in cols_to_save if c in points_gdf_assoc.columns]
            # Ensure geometries are valid before saving
            points_gdf_assoc_valid = points_gdf_assoc[points_gdf_assoc.geometry.is_valid & ~points_gdf_assoc.geometry.is_empty]
            if not points_gdf_assoc_valid.empty:
                 points_gdf_assoc_valid[cols_present].to_file(points_associated_path, driver='GPKG')
            else: print("Warning: No valid points remain after association to save.")

        except Exception as e: print(f"ERROR during point-road association: {e}"); traceback.print_exc(); return

        # === STEP 6: Perform Segment-Constrained DBSCAN ===
        print(f"\n[Step 6/7] Running Segment-Constrained DBSCAN...")
        points_gdf_final = points_gdf_assoc.reset_index(drop=True) # Use associated points GDF
        if points_gdf_final.empty or 'geometry' not in points_gdf_final.columns:
             print("ERROR: No points available for clustering after association step. Aborting."); return

        try: # Extract coordinates, ensuring valid geometries
            valid_geom_mask = points_gdf_final.geometry.is_valid & ~points_gdf_final.geometry.is_empty
            if (~valid_geom_mask).any():
                print(f"Warning: Removing {(~valid_geom_mask).sum()} points with invalid geometry before DBSCAN.")
            points_gdf_final = points_gdf_final[valid_geom_mask].reset_index(drop=True) # Keep only valid points
            if points_gdf_final.empty: print("ERROR: No valid points remain for DBSCAN. Aborting."); return

            # Extract coordinates as list of tuples
            points_coords = list(zip(points_gdf_final.geometry.x, points_gdf_final.geometry.y))
            if not points_coords: print("ERROR: Failed extracting coordinates for DBSCAN. Aborting."); return
            print(f"  Input points for DBSCAN: {len(points_coords)}")
        except Exception as e: print(f"ERROR extracting coordinates for DBSCAN: {e}. Aborting."); return

        # --- Run DBSCAN ---
        # Pass the prepared points GDF (with road_uid) and intersection dictionary
        cluster_labels = dbscan_segment_constrained(
            eps=eps,
            min_pts=min_pts,
            points_coords=points_coords,
            points_gdf_with_segments=points_gdf_final, # GDF must have 'road_uid'
            segment_intersections=segment_intersections
        )
        if not isinstance(cluster_labels, list) or len(cluster_labels) != len(points_gdf_final):
            print(f"ERROR: DBSCAN returned invalid labels (type: {type(cluster_labels)}, len: {len(cluster_labels)} vs expected {len(points_gdf_final)}). Aborting."); return

        # Add cluster labels back to the GeoDataFrame
        points_gdf_final['cluster'] = cluster_labels

        # === STEP 7: Analyze, Summarize, Plot ===
        print("\n[Step 7/7] Analyzing, summarizing, and plotting results...")
        noise_count = (points_gdf_final['cluster'] == -1).sum()
        clustered_points_gdf = points_gdf_final[points_gdf_final['cluster'] >= 0] # Include cluster 0 if it exists (shouldn't with DBSCAN)
        num_clusters = clustered_points_gdf['cluster'].nunique()
        # Refine cluster count to exclude potential 0 label if DBSCAN output it (usually only -1 or >0)
        num_actual_clusters = clustered_points_gdf[clustered_points_gdf['cluster'] > 0]['cluster'].nunique()


        print("\n--- Segment-Constrained DBSCAN Results ---")
        print(f" - Total points processed: {len(points_gdf_final)}")
        print(f" - Clusters found (IDs > 0): {num_actual_clusters}")
        print(f" - Noise points (ID = -1): {noise_count}")
        # Check for any points left unclassified (ID = 0), which shouldn't happen in standard DBSCAN
        unclassified = (points_gdf_final['cluster'] == 0).sum()
        if unclassified > 0: print(f" - WARNING: {unclassified} points remained unclassified (ID = 0).")

        summary_data, centroids_gdf, hulls_gdf = [], None, None
        if num_actual_clusters > 0:
            print("\n--- Calculating Cluster Characteristics ---")
            cluster_centroids_list, cluster_hulls_list = [], []
            try:
                # Work with points belonging to actual clusters (ID > 0)
                valid_clusters_gdf = points_gdf_final[points_gdf_final['cluster'] > 0].copy()
                # Ensure cluster ID is integer for grouping
                valid_clusters_gdf['cluster'] = valid_clusters_gdf['cluster'].astype(int)
                cluster_groups = valid_clusters_gdf.groupby('cluster')

                for cluster_id, group in tqdm.tqdm(cluster_groups, total=num_actual_clusters, desc="Summarizing Clusters"):
                    cluster_size, centroid, hull_geom, hull_area = len(group), None, None, None
                    # Calculate Centroid
                    try:
                         valid_geoms = group.geometry[group.geometry.is_valid & ~group.geometry.is_empty]
                         if not valid_geoms.empty:
                             # Use unary_union for potentially multi-part geometries before centroid
                             cluster_union = unary_union(valid_geoms)
                             if cluster_union and not cluster_union.is_empty:
                                 centroid = cluster_union.centroid
                                 # Append centroid info including size
                                 cluster_centroids_list.append({'cluster': cluster_id, 'geometry': centroid, 'size': cluster_size})
                    except Exception as e: print(f"  Warn: Could not calculate centroid for C{cluster_id}: {e}")

                    # Calculate Convex Hull (requires >= 3 points)
                    if cluster_size >= 3:
                        try:
                            valid_geoms_hull = group.geometry[group.geometry.is_valid & ~group.geometry.is_empty]
                            if len(valid_geoms_hull) >= 3:
                                 # Use unary_union first in case of MultiPoints, then convex hull
                                 hull_geom = unary_union(valid_geoms_hull).convex_hull
                                 # Ensure hull is a valid Polygon
                                 if hull_geom and hull_geom.geom_type == 'Polygon' and hull_geom.is_valid and not hull_geom.is_empty:
                                     hull_area = hull_geom.area
                                     # Append hull info including size and area
                                     cluster_hulls_list.append({'cluster': cluster_id, 'geometry': hull_geom, 'area': hull_area, 'size': cluster_size})
                                # else: print(f"  Warn: Hull for C{cluster_id} not a valid Polygon.") # Optional warning
                        except Exception as e: print(f"  Warn: Could not calculate hull for C{cluster_id}: {e}")

                    # Append summary data regardless of hull/centroid success
                    summary_data.append({
                        'Cluster_ID': cluster_id,
                        'Size': cluster_size,
                        'Centroid_X': centroid.x if centroid else None,
                        'Centroid_Y': centroid.y if centroid else None,
                        'Hull_Area': hull_area # Will be None if hull failed or size < 3
                    })

                # Create GeoDataFrames for centroids and hulls using the points' projected CRS
                common_crs = points_gdf_final.crs
                if cluster_centroids_list:
                     centroids_gdf = gpd.GeoDataFrame(cluster_centroids_list, crs=common_crs)
                     # Ensure 'size' column is integer
                     if 'size' in centroids_gdf.columns: centroids_gdf['size'] = centroids_gdf['size'].astype(int)
                if cluster_hulls_list:
                     hulls_gdf = gpd.GeoDataFrame(cluster_hulls_list, crs=common_crs)
                     # Ensure 'size' and 'cluster' columns are integer
                     if 'size' in hulls_gdf.columns: hulls_gdf['size'] = hulls_gdf['size'].astype(int)
                     if 'cluster' in hulls_gdf.columns: hulls_gdf['cluster'] = hulls_gdf['cluster'].astype(int)

                # Create and save summary DataFrame
                if summary_data:
                    summary_df = pd.DataFrame(summary_data).sort_values(by='Size', ascending=False).set_index('Cluster_ID')
                    try:
                         summary_df.to_csv(summary_report_path)
                         print(f"\nCluster summary saved to '{summary_report_path}'")
                    except Exception as e: print(f"\nError saving summary CSV: {e}")

            except Exception as e: print(f"ERROR during cluster characteristics calculation: {e}"); traceback.print_exc()
        else: print("\nNo clusters (IDs > 0) found. Skipping summary calculations.")

        # --- Plotting ---
        # Pass the GDFs which now contain the 'size' column where applicable
        plot_clusters_with_roads_interactive(
            points_gdf=points_gdf_final,    # Includes 'cluster' and 'road_uid'
            road_gdf=road_gdf,              # Projected roads
            centroids_gdf=centroids_gdf,    # Includes 'cluster' and 'size'
            hulls_gdf=hulls_gdf,            # Includes 'cluster', 'size', 'area'
            output_filename=interactive_map_path,
            generate_heatmap=generate_heatmap
        )
        if generate_static:
            generate_static_plot(
                points_gdf=points_gdf_final, # Clustered points in projected CRS
                road_gdf=road_gdf,           # Roads in same projected CRS
                output_filename=static_map_path
            )

    except (TypeError, ValueError, FileNotFoundError, ImportError, Exception) as e:
        print(f"\n--- CRITICAL ERROR ENCOUNTERED IN WORKFLOW ---")
        print(f"Error Type: {type(e).__name__}: {e}")
        print("Traceback:"); traceback.print_exc()
        print("-----------------------------------------------"); print("Analysis aborted due to error.")
    finally:
        print("\n--- Analysis Workflow Finished ---")


# --- Main Execution Block ---
if __name__ == "__main__":
    print("========================================================")
    print(" Starting Segment-Constrained Accident Clustering ")
    print("         (Projected CRS Workflow)                 ")
    print("========================================================")

    # --- Configuration ---
    # IMPORTANT: Ensure these files exist and the accident shapefile IS PROJECTED (e.g., UTM)
    accident_shapefile = "Accidents_Projected_Meters.shp" # MUST be projected
    road_shapefile = "Selected_Road_Features.shp"       # Can be geographic or projected

    # Output directory for results
    output_directory = "segment_constrained_clustering_results_v3_cat_colors" # Updated dir name

    # --- Parameters in Projected Units (e.g., METERS if using UTM) ---
    # Tolerance for considering segments as intersecting (0 for exact geometry intersection)
    intersection_check_tolerance = INTERSECTION_BUFFER_TOLERANCE # Use value defined at top (e.g., 0)

    # DBSCAN Parameters (tune based on data density and scale)
    epsilon = 150.0                 # Max distance between samples for one to be considered as in the neighborhood of the other (in CRS units, e.g., meters)
    minimum_points = 5              # Number of samples in a neighborhood for a point to be considered as a core point.

    # Output Options
    create_heatmap_layer = True     # Include a heatmap layer in the interactive map
    create_static_map = True        # Generate a static PNG map using Matplotlib
    # --- End Configuration ---

    print("\n--- Configuration ---")
    print(f"Accident data (Projected): '{accident_shapefile}'")
    print(f"Road network (Initial):    '{road_shapefile}'")
    print(f"Output directory:          '{output_directory}'")
    print(f"Intersection Tolerance:    {intersection_check_tolerance} (projected units)")
    print(f"DBSCAN eps:                {epsilon} (projected units)")
    print(f"DBSCAN min_pts:            {minimum_points}")
    print(f"Generate Heatmap:          {create_heatmap_layer}")
    print(f"Generate Static Map:       {create_static_map}")
    print("---------------------\n")

    # --- Pre-run Checks ---
    print("Checking input file existence...")
    if not os.path.exists(accident_shapefile): print(f"ERROR: Accident file not found: '{accident_shapefile}'"); exit(1)
    if not os.path.exists(road_shapefile): print(f"ERROR: Road file not found: '{road_shapefile}'"); exit(1)
    print("Input files found.")

    print("Checking required libraries...")
    try:
        import folium, geopandas, pandas, shapely, branca, jinja2, tqdm, matplotlib, numpy # Check all major ones
        print("All required libraries seem installed.")
    except ImportError as e:
        print(f"\n--- MISSING LIBRARY ---")
        print(f"Error: {e}")
        print("Please install missing libraries (e.g., using 'pip install geopandas folium matplotlib pandas tqdm branca jinja2')")
        exit(1)
    # --- End Pre-run Checks ---


    # --- Run the Main Analysis Function ---
    analyze_accident_clusters_with_roads(
        accident_shp_filename=accident_shapefile,
        road_shp_filename=road_shapefile,
        eps=epsilon,
        min_pts=minimum_points,
        intersection_tolerance=intersection_check_tolerance,
        output_dir=output_directory,
        generate_heatmap=create_heatmap_layer,
        generate_static=create_static_map
    )

    # --- Completion Message ---
    print("\n========================================================")
    print(f"Analysis script finished.")
    print(f"Check the output directory: '{output_directory}'")
    print("Method Used: Segment-constrained DBSCAN (No Grouping)")
    print("NOTE: Interactive map uses CATEGORICAL colors for clusters ('turbo' colormap).") # Updated note
    print("NOTE: Hull tooltips include cluster size.")
    print("Review Step 3 intersection count and final cluster results in summary CSV.")
    print("========================================================")