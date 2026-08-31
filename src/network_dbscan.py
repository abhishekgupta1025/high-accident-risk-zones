import math
import geopandas as gpd
import matplotlib.pyplot as plt # Still potentially useful for non-interactive fallback or other plots
import numpy as np
from shapely.geometry import Point
import folium # For interactive maps
# Removed: from folium.plugins import ScaleControl
from branca.colormap import linear # For cluster colors on the map
from branca.element import MacroElement # <<< IMPORT ADDED for custom scale
from jinja2 import Template # <<< IMPORT ADDED for custom scale
import warnings # To manage geopandas warnings during reprojection

# --- Custom Scale Control Class (Provided by User) ---
class ScaleControl(MacroElement):
    """Custom Folium Scale Control based on Leaflet's L.Control.Scale"""
    def __init__(self, position='bottomleft', metric=True, imperial=False, max_width=100):
        """
        Initializes the custom ScaleControl.

        Args:
            position (str): Position of the control (e.g., 'topleft', 'bottomright').
            metric (bool): Whether to show metric scale (m/km).
            imperial (bool): Whether to show imperial scale (mi/ft).
            max_width (int): Max width of the control in pixels.
        """
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

# --- Core DBSCAN Functions (Unchanged) ---

def calculate_distance(point1, point2):
    """
    Calculates the Euclidean distance between two points.
    Assumes points are in a Cartesian coordinate system.
    May be inaccurate for geographic coordinates (lat/lon) over large distances.

    Args:
        point1 (list or tuple): The coordinates of the first point (x, y).
        point2 (list or tuple): The coordinates of the second point (x, y).

    Returns:
        float: The Euclidean distance between the two points.
    """
    if not (isinstance(point1, (list, tuple)) and isinstance(point2, (list, tuple))):
        raise TypeError("Both inputs must be lists or tuples.")
    if len(point1) != 2 or len(point2) != 2:
        raise ValueError("Points must have exactly two dimensions (x, y).")
    distance_sq = (point1[0] - point2[0])**2 + (point1[1] - point2[1])**2
    return math.sqrt(distance_sq)


def find_neighbors(data, point_index, eps):
    """
    Finds the neighbors of a point within a given radius (eps).

    Args:
        data (list of tuples): The dataset, where each element is a point (x, y).
        point_index (int): The index of the point for which to find neighbors.
        eps (float): The radius within which to search for neighbors.
                       *** IMPORTANT: Ensure 'eps' is appropriate for the
                       *** data's coordinate system units (e.g., meters, degrees).

    Returns:
        list: A list of indices of the neighbors of the point.
    """
    if not isinstance(data, list):
        raise TypeError("Data must be a list of points.")
    if not isinstance(point_index, int):
        raise TypeError("point_index must be an integer.")
    if not isinstance(eps, (int, float)):
        raise TypeError("eps must be a number.")
    if point_index < 0 or point_index >= len(data):
        raise IndexError("point_index is out of bounds.")
    if eps <= 0:
        raise ValueError("eps must be a positive value.")

    neighbors = []
    target_point = data[point_index]
    for i, point in enumerate(data):
        if i != point_index:
            distance = calculate_distance(point, target_point)
            if distance <= eps:
                neighbors.append(i)
    return neighbors

def expand_cluster(data, labels, point_index, neighbors, cluster_label, eps, min_pts):
    """
    Expands a cluster starting from a core point (used internally by DBSCAN).

    Args:
        data (list of tuples): The dataset (x, y coordinates).
        labels (list): A list of cluster labels for each point (-1=Noise, 0=Unvisited, >0=Cluster ID).
        point_index (int): The index of the current core point to expand from.
        neighbors (list): A list of indices of the neighbors of the current point.
        cluster_label (int): The label ID for the new cluster being formed.
        eps (float): The radius within which to search for neighbors.
        min_pts (int): The minimum number of points required to form a dense region (core point).

    Returns:
        list: The updated list of cluster labels.
    """
    labels[point_index] = cluster_label
    i = 0
    processed_neighbors = set(neighbors) # Keep track to avoid redundant checks
    processed_neighbors.add(point_index)

    while i < len(neighbors):
        neighbor_index = neighbors[i]

        if labels[neighbor_index] == -1: # Noise point becomes border point
            labels[neighbor_index] = cluster_label
        elif labels[neighbor_index] == 0: # Unvisited point
            labels[neighbor_index] = cluster_label
            new_neighbors = find_neighbors(data, neighbor_index, eps)
            if len(new_neighbors) >= min_pts: # Neighbor is also a core point
                # Add new neighbors that haven't been processed or added yet
                for nn_idx in new_neighbors:
                    if nn_idx not in processed_neighbors:
                        neighbors.append(nn_idx)
                        processed_neighbors.add(nn_idx)
        i += 1
    return labels

def dbscan(data, eps, min_pts):
    """
    Performs DBSCAN clustering on the given data.

    Args:
        data (list of tuples): The dataset, where each element is a point (x, y).
        eps (float): The radius for DBSCAN. **Crucial: Match units to data CRS.**
        min_pts (int): The minimum points for DBSCAN.

    Returns:
        list: A list of cluster labels for each point (-1 for noise). Empty list for empty data.
    """
    if not isinstance(data, list):
        raise TypeError("Data must be a list of points.")
    if not isinstance(eps, (int, float)) or eps <= 0:
        raise ValueError("eps must be a positive number.")
    if not isinstance(min_pts, int) or min_pts <= 0:
        raise ValueError("min_pts must be a positive integer.")
    if not data:
        print("Warning: Input data is empty.")
        return []

    n_points = len(data)
    labels = [0] * n_points # 0: Unvisited, -1: Noise, >0: Cluster ID
    cluster_label = 0

    for point_index in range(n_points):
        if labels[point_index] == 0: # Process only unvisited points
            neighbors = find_neighbors(data, point_index, eps)
            if len(neighbors) < min_pts:
                labels[point_index] = -1 # Mark as Noise
            else:
                cluster_label += 1 # Start a new cluster
                labels = expand_cluster(data, labels, point_index, neighbors, cluster_label, eps, min_pts)
    return labels

# --- Data Reading Functions (Unchanged) ---

def read_point_shp_data(filename):
    """
    Reads point data (e.g., accidents) from a Shapefile.

    Args:
        filename (str): The path to the .shp file.

    Returns:
        list: A list of (x, y) tuples representing point coordinates.
        object: The Coordinate Reference System (CRS) of the shapefile.
        geopandas.GeoDataFrame: The full GeoDataFrame read from the file.
        Returns ([], None, None) if the file is empty, not found,
        or contains no valid point geometries.
    """
    if not isinstance(filename, str):
        raise TypeError("Filename must be a string.")
    try:
        gdf = gpd.read_file(filename)
        if gdf.empty:
            print(f"Warning: Shapefile '{filename}' is empty.")
            return [], None, None
        if 'geometry' not in gdf.columns:
            raise ValueError(f"Shapefile '{filename}' lacks a 'geometry' column.")

        # Ensure we only use Point geometries for clustering
        gdf_points = gdf[gdf.geometry.type == 'Point'].copy() # Use .copy() to avoid SettingWithCopyWarning
        if gdf_points.empty:
            print(f"Warning: No Point geometries found in '{filename}'.")
            return [], gdf.crs, gdf # Return original gdf even if no points

        # Extract coordinates
        data = list(zip(gdf_points.geometry.x, gdf_points.geometry.y))
        print(f"Successfully read {len(data)} points from '{filename}'.")
        print(f"Point Data Coordinate Reference System (CRS): {gdf_points.crs}")
        return data, gdf_points.crs, gdf_points

    except ImportError:
        raise ImportError("The 'geopandas' library is required. Please install it (`pip install geopandas`).")
    except Exception as e:
        raise Exception(f"Error reading point shapefile '{filename}': {e}")

def read_line_shp_data(filename):
    """
    Reads line data (e.g., road network) from a Shapefile.

    Args:
        filename (str): The path to the .shp file.

    Returns:
        geopandas.GeoDataFrame: GeoDataFrame containing the line features.
        object: The Coordinate Reference System (CRS) of the shapefile.
        Returns (None, None) if the file is empty, not found, or has no geometry.
    """
    if not isinstance(filename, str):
        raise TypeError("Filename must be a string.")
    try:
        gdf = gpd.read_file(filename)
        if gdf.empty:
            print(f"Warning: Shapefile '{filename}' is empty.")
            return None, None
        if 'geometry' not in gdf.columns:
            raise ValueError(f"Shapefile '{filename}' lacks a 'geometry' column.")

        # Optional: Filter for LineString geometries if needed
        # gdf = gdf[gdf.geometry.type == 'LineString']
        # if gdf.empty:
        #     print(f"Warning: No LineString geometries found in '{filename}'.")
        #     return None, None

        print(f"Successfully read {len(gdf)} features from '{filename}'.")
        print(f"Line Data Coordinate Reference System (CRS): {gdf.crs}")
        return gdf, gdf.crs

    except ImportError:
        raise ImportError("The 'geopandas' library is required. Please install it (`pip install geopandas`).")
    except Exception as e:
        raise Exception(f"Error reading line shapefile '{filename}': {e}")


# --- Interactive Plotting Function using Folium (MODIFIED) ---

def plot_clusters_with_roads_interactive(point_data, labels, point_crs, road_gdf, output_filename="cluster_map.html"):
    """
    Generates an interactive HTML map of clusters and roads using Folium.
    Assumes point_data contains (x, y) coordinates matching point_crs.
    Reprojects data to EPSG:4326 for Folium compatibility if needed.
    Includes the custom scale bar on the map.

    Args:
        point_data (list of tuples): The original point dataset (x, y coordinates).
        labels (list): A list of cluster labels for each point.
        point_crs (object): The CRS of the original point data.
        road_gdf (geopandas.GeoDataFrame or None): GeoDataFrame containing road network lines,
                                                  assumed to be in the same CRS as point_data.
        output_filename (str): The name of the HTML file to save the map to.
    """
    if not point_data or not labels or len(point_data) != len(labels):
        print("Warning: Cannot plot clusters due to invalid point data or labels.")
        return
    if not point_crs:
        print("Warning: Point CRS is missing. Cannot reliably create interactive map.")
        return

    print(f"Generating interactive map: '{output_filename}'...")

    # --- Prepare Data for Folium (Requires EPSG:4326 - Latitude/Longitude) ---
    target_crs = "EPSG:4326" # Standard CRS for web maps

    # Create a GeoDataFrame for points to facilitate reprojection
    point_geometries = [Point(p) for p in point_data]
    points_gdf = gpd.GeoDataFrame(geometry=point_geometries, crs=point_crs)
    points_gdf['label'] = labels # Add cluster labels

    # Reproject points if necessary
    try:
        if points_gdf.crs != target_crs:
            print(f"Reprojecting points from {points_gdf.crs} to {target_crs} for Folium map...")
            with warnings.catch_warnings(): # Suppress potential Shapely warnings during transform
                 warnings.simplefilter("ignore", category=UserWarning)
                 points_gdf_proj = points_gdf.to_crs(target_crs)
        else:
            points_gdf_proj = points_gdf
        # Extract lat/lon coordinates
        lats = points_gdf_proj.geometry.y
        lons = points_gdf_proj.geometry.x
    except Exception as e:
        print(f"Error reprojecting point data to {target_crs}: {e}")
        print("Cannot create Folium map without lat/lon coordinates.")
        return

    # Reproject roads if necessary
    roads_gdf_proj = None
    if road_gdf is not None and not road_gdf.empty:
        try:
            if road_gdf.crs != target_crs:
                print(f"Reprojecting roads from {road_gdf.crs} to {target_crs} for Folium map...")
                with warnings.catch_warnings():
                     warnings.simplefilter("ignore", category=UserWarning)
                     roads_gdf_proj = road_gdf.to_crs(target_crs)
            else:
                roads_gdf_proj = road_gdf
        except Exception as e:
            print(f"Error reprojecting road data to {target_crs}: {e}")
            print("Proceeding with map generation, but roads may be missing or misaligned.")
            roads_gdf_proj = None # Ensure it's None if reprojection failed

    # --- Create Folium Map ---
    # Calculate map center (use mean of projected point coordinates)
    map_center = [lats.mean(), lons.mean()]
    m = folium.Map(location=map_center, zoom_start=12, tiles="OpenStreetMap")

    # Add Roads Layer (if available and projected)
    if roads_gdf_proj is not None and not roads_gdf_proj.empty:
        print("Adding road network layer...")
        try:
            # Use GeoJson for better performance with lines
            folium.GeoJson(
                roads_gdf_proj,
                name='Road Network',
                style_function=lambda x: {'color': 'gray', 'weight': 1.5, 'opacity': 0.7},
                tooltip=folium.features.GeoJsonTooltip(fields=list(roads_gdf_proj.columns.drop('geometry', errors='ignore')), aliases=['Property:']*len(roads_gdf_proj.columns), localize=True)
            ).add_to(m)
        except Exception as e:
            print(f"Error adding roads GeoJson to map: {e}")


    # Add Points Layer (Clustered)
    print("Adding clustered points layer...")
    unique_labels = sorted(points_gdf_proj['label'].unique())
    n_clusters = len([l for l in unique_labels if l != -1])

    # Define colormap for clusters (excluding noise)
    if n_clusters > 0:
        cluster_ids = sorted([l for l in unique_labels if l != -1])
        # Use a linear colormap (e.g., Viridis)
        colormap = linear.viridis.scale(min(cluster_ids), max(cluster_ids))
        # Or define specific colors: colors = ['red', 'blue', 'green', ...]
    else:
        colormap = None # No clusters, only noise

    # Create feature groups for toggling layers
    feature_groups = {}

    # Add points to the map, grouped by cluster label
    for label in unique_labels:
        group_name = f"Cluster {label}" if label != -1 else "Noise"
        feature_groups[label] = folium.FeatureGroup(name=group_name, show=True) # Show by default

        points_in_label = points_gdf_proj[points_gdf_proj['label'] == label]

        if label == -1:
            color = 'grey'
            radius = 2
        elif colormap:
            color = colormap(label)
            radius = 4
        else: # Should not happen if there are clusters, but fallback
            color = 'black'
            radius = 3

        # Add each point as a CircleMarker
        for idx, point in points_in_label.iterrows():
            # Find the original index in the unprojected data
            # This assumes the order of points is preserved during reprojection
            original_index = points_gdf.index[points_gdf['label'] == label].tolist()[points_in_label.index.get_loc(idx)]

            folium.CircleMarker(
                location=[point.geometry.y, point.geometry.x],
                radius=radius,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.7,
                popup=f"{group_name}<br>Original Coords: ({point_data[original_index][0]:.4f}, {point_data[original_index][1]:.4f})<br>Lat/Lon: ({point.geometry.y:.6f}, {point.geometry.x:.6f})" # Show original coords too
            ).add_to(feature_groups[label])

        feature_groups[label].add_to(m) # Add the group to the map

    # Add Layer Control to toggle layers
    folium.LayerControl(collapsed=False).add_to(m)

    # Add Colormap legend if clusters exist
    if colormap:
        colormap.caption = 'Cluster ID'
        m.add_child(colormap)

    # <<< ADD CUSTOM SCALE CONTROL TO THE MAP >>>
    custom_scale = ScaleControl(position='bottomleft', metric=True, imperial=False)
    m.add_child(custom_scale)
    print("Added custom scale bar to the map.")

    # Save the map to HTML
    try:
        m.save(output_filename)
        print(f"Interactive map saved successfully to '{output_filename}'. Open this file in a web browser.")
    except Exception as e:
        print(f"Error saving Folium map: {e}")


# --- Main Analysis Function (Unchanged) ---

def analyze_accident_clusters_with_roads(accident_shp_filename, road_shp_filename, eps, min_pts):
    """
    Reads accident points and road lines, performs DBSCAN, automatically
    reprojects roads if needed, prints results, and generates an interactive
    Folium map saved to HTML.

    Args:
        accident_shp_filename (str): Path to the accident points .shp file.
        road_shp_filename (str): Path to the road network lines .shp file.
        eps (float): The radius for DBSCAN. **Crucial: Set based on accident data CRS units.**
        min_pts (int): The minimum points for DBSCAN.
    """
    # Input validation
    if not isinstance(accident_shp_filename, str):
        raise TypeError("Accident shapefile filename must be a string.")
    if not isinstance(road_shp_filename, str):
        raise TypeError("Road shapefile filename must be a string.")
    if not isinstance(eps, (int, float)) or eps <= 0:
        raise ValueError("eps must be a positive number.")
    if not isinstance(min_pts, int) or min_pts <= 0:
        raise ValueError("min_pts must be a positive integer.")

    try:
        # 1. Read Accident Data (Points)
        print(f"\nReading accident data from: {accident_shp_filename}")
        point_data, point_crs, _ = read_point_shp_data(accident_shp_filename) # Don't need the gdf here
        if not point_data or not point_crs:
            print("No point data found or CRS missing. Cannot perform analysis.")
            return

        # 2. Read Road Network Data (Lines)
        print(f"\nReading road network data from: {road_shp_filename}")
        road_gdf, road_crs = read_line_shp_data(road_shp_filename)
        if road_gdf is None:
            print("Warning: Could not read road network data. Map will not include roads.")
        elif road_crs != point_crs:
             # *** AUTOMATIC REPROJECTION ***
             print("\n--- CRS Mismatch Detected ---")
             print(f"Attempting to reproject roads from CRS '{road_crs}' to match points CRS '{point_crs}'...")
             try:
                 with warnings.catch_warnings(): # Suppress potential Shapely warnings during transform
                     warnings.simplefilter("ignore", category=UserWarning)
                     road_gdf = road_gdf.to_crs(point_crs)
                 print("Road network successfully reprojected.")
                 road_crs = road_gdf.crs # Update road_crs variable
             except Exception as e:
                 print(f"ERROR: Failed to reproject road network: {e}")
                 print("Proceeding, but roads might be misaligned or missing from the final map.")
                 # Keep original road_gdf, plotting function will handle potential issues
             print("---------------------------\n")


        # 3. Parameter Check (Based on Accident Data CRS) - Crucial for DBSCAN `eps`
        print("\n--- Parameter Check ---")
        print(f"Using eps = {eps}")
        print(f"Using min_pts = {min_pts}")
        if point_crs:
            print(f"Accident Data CRS: {point_crs}")
            if point_crs.is_geographic:
                 print("WARNING: CRS is geographic (lat/lon). 'eps' should be in decimal degrees.")
                 print("         Euclidean distance used by DBSCAN; may be inaccurate. Consider reprojecting points to a projected CRS *before* DBSCAN for 'eps' in meters.")
            else:
                 try: unit_name = point_crs.axis_info[0].unit_name
                 except: unit_name = 'units'
                 print(f"INFO: CRS is projected. 'eps' should be in CRS units (e.g., meters - detected: {unit_name}).")
        else:
            print("WARNING: Could not determine Accident CRS. Ensure 'eps' is appropriate.")
        print("-----------------------\n")


        # 4. Perform DBSCAN Clustering on Accident Points (using original coordinates)
        print("Running DBSCAN on accident data...")
        cluster_labels = dbscan(point_data, eps, min_pts)
        print("DBSCAN complete.")

        if not cluster_labels:
            print("DBSCAN did not produce any labels.")
            return

        # 5. Analyze and Print Results
        print("\nDBSCAN Clustering Results Summary:")
        noise_count = cluster_labels.count(-1)
        cluster_ids = sorted([label for label in set(cluster_labels) if label != -1])
        num_clusters = len(cluster_ids)
        print(f" - Total accident points analyzed: {len(point_data)}")
        print(f" - Number of clusters found: {num_clusters}")
        print(f" - Number of noise points: {noise_count}")
        if num_clusters > 0:
            print("\nCluster Sizes:")
            for cluster_id in cluster_ids:
                size = cluster_labels.count(cluster_id)
                print(f" - Cluster {cluster_id}: {size} points")


        # 6. Plotting (Interactive Folium Map)
        # Call the interactive plotting function
        plot_clusters_with_roads_interactive(
            point_data=point_data,
            labels=cluster_labels,
            point_crs=point_crs,
            road_gdf=road_gdf # Pass the potentially reprojected road_gdf
            # output_filename="my_custom_map_name.html" # Optional: customize output file
        )

    except (TypeError, ValueError, FileNotFoundError, ImportError, Exception) as e:
        print(f"\n--- ERROR ---")
        print(f"An critical error occurred during analysis: {e}")
        import traceback
        traceback.print_exc() # Print detailed traceback for debugging
        print("---------------")

# --- Main Execution ---
if __name__ == "__main__":
    # --- Configuration ---
    # >>> IMPORTANT: Set the correct paths to your Shapefiles <<<

    # MODIFIED: Use the projected accident shapefile
    accident_shapefile = "Accidents_Projected_Meters.shp" # USE THE NEW PROJECTED FILE
    road_shapefile = "Selected_Road_Features.shp"       # REPLACE WITH YOUR FILE (Ensure this is also projected or matches accident CRS)

    # --- CRITICAL DBSCAN PARAMETERS (Adjust based on ACCIDENT data CRS) ---
    # CHECK YOUR ACCIDENT SHAPEFILE'S CRS! Use a GIS tool (like QGIS) or print(point_crs)
    # Since we are using "Accidents_Projected_Meters.shp", assume the CRS is projected in METERS.
    # 'eps' should now be specified in METERS.

    # MODIFIED: Set epsilon in meters (e.g., 100m search radius)
    epsilon = 230 # EXAMPLE: Search radius of 100 meters. **ADJUST THIS BASED ON YOUR DATA DENSITY**

    # Minimum number of points to form a cluster.
    minimum_points = 5 # EXAMPLE VALUE - **ADJUST THIS BASED ON YOUR DATA**
    # --- End Configuration ---

    print(f"Starting analysis...")
    print(f"Accident data: '{accident_shapefile}'")
    print(f"Road network: '{road_shapefile}'")

    # Ensure you have installed folium, branca, jinja2: pip install folium branca Jinja2
    try:
        import folium
        from branca.element import MacroElement # Check imports needed here too
        from jinja2 import Template
    except ImportError as e:
        print("\n--- MISSING LIBRARY ---")
        print(f"Required library missing: {e}")
        print("Please install required libraries: pip install geopandas folium matplotlib numpy shapely branca Jinja2")
        print("-----------------------\n")
        exit() # Exit if required libraries are not installed

    analyze_accident_clusters_with_roads(
        accident_shp_filename=accident_shapefile,
        road_shp_filename=road_shapefile,
        eps=epsilon,
        min_pts=minimum_points
    )

    print("\nAnalysis finished.")

    # --- Usage Notes ---
    # 1. Ensure shapefiles (.shp, .dbf, .shx, .prj) are accessible.
    # 2. Install required libraries: `pip install geopandas folium matplotlib numpy shapely branca Jinja2`
    # 3. **CRITICAL:** Adjust `epsilon` (now in meters) and `minimum_points` based on your projected data.
    # 4. The script now automatically tries to reproject the road network if its CRS
    #    differs from the accident points' CRS before plotting. Ensure the road shapefile path is correct.
    # 5. The output is an interactive HTML file (default: 'cluster_map.html')
    #    which you need to open in a web browser.
    # 6. The script assumes "Accidents_Projected_Meters.shp" uses a projected CRS with meters as units.
