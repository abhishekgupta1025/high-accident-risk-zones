# -*- coding: utf-8 -*-
"""
Data loading and geometric validation utilities for geospatial datasets.
"""
from pathlib import Path
from typing import Optional, Tuple, Union
import geopandas as gpd
import pandas as pd


def load_point_data(filepath: Union[str, Path]) -> Optional[gpd.GeoDataFrame]:
    """
    Reads point data (e.g. accident incident records) from a GIS shapefile or geopackage.
    Filters out non-point geometries, fixes invalid entries, and resets dataframe indexing.

    Args:
        filepath: Path to the point vector file.

    Returns:
        Cleaned geopandas.GeoDataFrame with Point geometries, or None on failure.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Point data file not found: {path}")

    try:
        gdf = gpd.read_file(path)
        if gdf.empty:
            print(f"[Warning] Point file '{path.name}' is empty.")
            return None

        if "geometry" not in gdf.columns:
            raise ValueError(f"File '{path.name}' lacks a 'geometry' column.")

        # Ensure Point geometry type
        original_count = len(gdf)
        gdf_points = gdf[gdf.geometry.geom_type == "Point"].copy()
        if gdf_points.empty:
            print(f"[Warning] No Point geometries found in '{path.name}'.")
            return None

        # Filter invalid or empty geometries
        valid_mask = gdf_points.geometry.is_valid & ~gdf_points.geometry.is_empty
        gdf_points = gdf_points[valid_mask].reset_index(drop=True)

        if len(gdf_points) < original_count:
            print(f"[Info] Filtered {original_count - len(gdf_points)} non-point or invalid geometries.")

        if gdf_points.crs is None:
            raise ValueError(f"Point dataset '{path.name}' is missing CRS metadata.")

        print(f"[Loaded] {len(gdf_points)} valid point features from '{path.name}' (CRS: {gdf_points.crs}).")
        return gdf_points

    except Exception as e:
        print(f"[Error] Failed reading point file '{path}': {e}")
        raise e


def load_road_data(filepath: Union[str, Path]) -> Tuple[Optional[gpd.GeoDataFrame], Optional[object]]:
    """
    Reads road network line data, assigns unique 'road_uid' identifiers, and repairs invalid geometries.

    Args:
        filepath: Path to the road linestring vector file.

    Returns:
        Tuple of (Cleaned road GeoDataFrame with 'road_uid', CRS object).
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Road network file not found: {path}")

    try:
        gdf = gpd.read_file(path)
        if gdf.empty:
            print(f"[Warning] Road file '{path.name}' is empty.")
            return None, None

        if "geometry" not in gdf.columns:
            raise ValueError(f"Road file '{path.name}' lacks a 'geometry' column.")

        # Filter for LineString / MultiLineString
        original_count = len(gdf)
        gdf_lines = gdf[gdf.geometry.geom_type.isin(["LineString", "MultiLineString"])].copy()
        if gdf_lines.empty:
            print(f"[Warning] No LineString features found in '{path.name}'.")
            return None, None

        # Reset index and assign deterministic road_uid
        gdf_lines = gdf_lines.reset_index(drop=True)
        gdf_lines["road_uid"] = gdf_lines.index

        # Attempt to repair invalid geometries using buffer(0) if present
        invalid_mask = ~gdf_lines.geometry.is_valid
        if invalid_mask.any():
            print(f"[Warning] Found {invalid_mask.sum()} invalid road geometries. Attempting buffer(0) repair...")
            gdf_lines.geometry = gdf_lines.geometry.buffer(0)

        if gdf_lines.crs is None:
            raise ValueError(f"Road dataset '{path.name}' is missing CRS metadata.")

        print(f"[Loaded] {len(gdf_lines)} valid road segments from '{path.name}' (CRS: {gdf_lines.crs}).")
        return gdf_lines, gdf_lines.crs

    except Exception as e:
        print(f"[Error] Failed reading road file '{path}': {e}")
        raise e
