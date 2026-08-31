# -*- coding: utf-8 -*-
"""
Coordinate Reference System (CRS) transformation utilities.
"""
from typing import Union
import geopandas as gpd


def reproject_geodataframe(gdf: gpd.GeoDataFrame, target_crs: Union[str, int]) -> gpd.GeoDataFrame:
    """
    Safely reprojects a GeoDataFrame to a target Coordinate Reference System.

    Args:
        gdf: Input GeoDataFrame.
        target_crs: Desired CRS (e.g. "EPSG:32645" or "EPSG:4326").

    Returns:
        Reprojected GeoDataFrame.
    """
    if gdf is None or gdf.empty:
        return gdf

    if gdf.crs == target_crs:
        return gdf

    print(f"[CRS Transformation] Reprojecting from {gdf.crs} -> {target_crs}...")
    return gdf.to_crs(target_crs)


def ensure_projected_crs(gdf: gpd.GeoDataFrame, default_utm_crs: str = "EPSG:32645") -> gpd.GeoDataFrame:
    """
    Ensures that the GeoDataFrame is projected into a linear metric space (not angular degrees).
    If geographic (e.g. EPSG:4326), reprojects to the default UTM CRS.
    """
    if gdf is None or gdf.empty:
        return gdf

    if gdf.crs is None or gdf.crs.is_geographic:
        print(f"[Warning] Detected geographic CRS ({gdf.crs}). Converting to projected metric space: {default_utm_crs}")
        return reproject_geodataframe(gdf, default_utm_crs)

    return gdf
