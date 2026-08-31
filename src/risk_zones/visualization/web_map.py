# -*- coding: utf-8 -*-
"""
Interactive Folium / Leaflet Web GIS Map Builder.
Renders road network vectors, categorical cluster points, centroids, convex hull polygons, and heatmaps.
"""
from pathlib import Path
from typing import Dict, Optional, Union
import folium
from folium.plugins import HeatMap
import geopandas as gpd
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

from .scale_control import ScaleControl


class InteractiveMapBuilder:
    """Builds multi-layer interactive Leaflet maps with custom popups, tooltips, and scale bars."""

    @staticmethod
    def create_map(
        points_gdf: gpd.GeoDataFrame,
        road_gdf: Optional[gpd.GeoDataFrame] = None,
        centroids_gdf: Optional[gpd.GeoDataFrame] = None,
        hulls_gdf: Optional[gpd.GeoDataFrame] = None,
        output_filepath: Union[str, Path] = "cluster_map.html",
        cluster_col: str = "cluster",
        generate_heatmap: bool = True,
    ) -> folium.Map:
        """
        Creates and saves an interactive Folium map.

        Args:
            points_gdf: Point incidents with cluster labels.
            road_gdf: Road network GeoDataFrame.
            centroids_gdf: Cluster centroids GeoDataFrame.
            hulls_gdf: Convex hulls GeoDataFrame.
            output_filepath: Destination HTML file path.
            cluster_col: Column name containing cluster labels.
            generate_heatmap: Whether to include the KDE HeatMap layer.
        """
        output_path = Path(output_filepath)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        target_crs = "EPSG:4326"  # Standard WGS84 for Leaflet map tiles

        # Reproject layers to WGS84
        points_proj = points_gdf.to_crs(target_crs) if points_gdf.crs != target_crs else points_gdf
        lats = points_proj.geometry.y
        lons = points_proj.geometry.x

        map_center = [float(lats.mean()), float(lons.mean())]
        m = folium.Map(location=map_center, zoom_start=12, tiles=None)

        # Add basemaps
        folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(m)
        folium.TileLayer("CartoDB positron", name="CartoDB Positron (Light)", show=False).add_to(m)
        folium.TileLayer("CartoDB dark_matter", name="CartoDB Dark", show=False).add_to(m)

        # 1. Road Network Layer
        if road_gdf is not None and not road_gdf.empty:
            road_proj = road_gdf.to_crs(target_crs) if road_gdf.crs != target_crs else road_gdf
            road_fg = folium.FeatureGroup(name="Road Network", show=True).add_to(m)

            road_tooltip_fields = [col for col in road_proj.columns if col not in ["geometry", "road_uid"]]
            road_tooltip_aliases = [f"{col}: " for col in road_tooltip_fields]

            folium.GeoJson(
                road_proj,
                style_function=lambda _: {"color": "gray", "weight": 1.5, "opacity": 0.7},
                tooltip=folium.features.GeoJsonTooltip(
                    fields=road_tooltip_fields,
                    aliases=road_tooltip_aliases,
                    localize=True,
                    sticky=False,
                ) if road_tooltip_fields else None,
            ).add_to(road_fg)

        # 2. Categorical Colormap for Clusters
        unique_labels = sorted(points_proj[cluster_col].unique())
        cluster_ids = [l for l in unique_labels if l != -1 and pd.notna(l)]

        cmap = cm.get_cmap("tab20", max(len(cluster_ids), 1))
        color_map: Dict[int, str] = {}
        for idx, cid in enumerate(cluster_ids):
            color_map[int(cid)] = mcolors.rgb2hex(cmap(idx % 20))

        # 3. Clustered Points Layer
        cluster_sizes = points_proj[points_proj[cluster_col].isin(cluster_ids)].groupby(cluster_col).size().to_dict()

        for label in unique_labels:
            if pd.isna(label):
                continue
            label_int = int(label)
            group_name = f"Cluster {label_int}" if label_int != -1 else "Noise Points"
            show_layer = (label_int != -1)

            fg = folium.FeatureGroup(name=group_name, show=show_layer).add_to(m)
            points_in_label = points_proj[points_proj[cluster_col] == label]

            if label_int == -1:
                color, radius = "gray", 2
                cluster_size = len(points_in_label)
            else:
                color = color_map.get(label_int, "blue")
                radius = 4
                cluster_size = cluster_sizes.get(label_int, len(points_in_label))

            for _, point in points_in_label.iterrows():
                popup_html = f"<b>{group_name}</b><br>"
                popup_html += f"Lat: {point.geometry.y:.6f}<br>"
                popup_html += f"Lon: {point.geometry.x:.6f}<br>"
                popup_html += f"Cluster ID: {label_int}<br>"
                if label_int != -1:
                    popup_html += f"Cluster Size: {cluster_size} incidents<br>"
                if "road_uid" in point and pd.notna(point["road_uid"]):
                    popup_html += f"Road UID: {int(point['road_uid'])}<br>"

                folium.CircleMarker(
                    location=[point.geometry.y, point.geometry.x],
                    radius=radius,
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.75,
                    popup=folium.Popup(popup_html, max_width=300),
                ).add_to(fg)

        # 4. Centroids Layer
        if centroids_gdf is not None and not centroids_gdf.empty:
            centroids_proj = centroids_gdf.to_crs(target_crs) if centroids_gdf.crs != target_crs else centroids_gdf
            centroid_fg = folium.FeatureGroup(name="Cluster Centroids", show=False).add_to(m)

            for _, c in centroids_proj.iterrows():
                c_id = int(c["cluster"])
                c_size = c.get("size", "")
                popup_text = f"Centroid for Cluster {c_id}" + (f" ({c_size} incidents)" if c_size else "")
                folium.Marker(
                    location=[c.geometry.y, c.geometry.x],
                    popup=popup_text,
                    icon=folium.Icon(color="blue", icon="info-sign"),
                ).add_to(centroid_fg)

        # 5. Convex Hulls Layer
        if hulls_gdf is not None and not hulls_gdf.empty:
            hulls_proj = hulls_gdf.to_crs(target_crs) if hulls_gdf.crs != target_crs else hulls_gdf
            hulls_fg = folium.FeatureGroup(name="Cluster Convex Hulls (Danger Zones)", show=False).add_to(m)

            folium.GeoJson(
                hulls_proj,
                style_function=lambda f: {
                    "fillColor": color_map.get(int(f["properties"].get("cluster", -1)), "blue"),
                    "color": "black",
                    "weight": 1.5,
                    "fillOpacity": 0.25,
                },
                tooltip=folium.features.GeoJsonTooltip(
                    fields=["cluster", "size", "area_m2"] if "area_m2" in hulls_proj.columns else ["cluster"],
                    aliases=["Cluster ID: ", "Size: ", "Area (m²): "] if "area_m2" in hulls_proj.columns else ["Cluster ID: "],
                    localize=True,
                ),
            ).add_to(hulls_fg)

        # 6. Heatmap Layer
        if generate_heatmap:
            heatmap_fg = folium.FeatureGroup(name="Accident Heatmap (KDE)", show=False).add_to(m)
            HeatMap(list(zip(lats, lons)), radius=15, blur=20).add_to(heatmap_fg)

        # Controls
        m.add_child(ScaleControl(position="bottomleft", metric=True, imperial=True))
        folium.LayerControl(collapsed=False).add_to(m)

        m.save(str(output_path))
        print(f"[Map Exported] Interactive Leaflet map saved to: {output_path}")
        return m
