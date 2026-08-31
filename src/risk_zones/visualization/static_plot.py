# -*- coding: utf-8 -*-
"""
Static publication-grade figure generator using Matplotlib.
"""
from pathlib import Path
from typing import Optional, Union
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class StaticPlotBuilder:
    """Generates publication-quality static cartographic maps of clustered accident zones."""

    @staticmethod
    def create_plot(
        points_gdf: gpd.GeoDataFrame,
        road_gdf: Optional[gpd.GeoDataFrame] = None,
        output_filepath: Union[str, Path] = "static_cluster_map.png",
        cluster_col: str = "cluster",
    ) -> None:
        """
        Renders and saves a high-resolution static PNG map.

        Args:
            points_gdf: Clustered incident points.
            road_gdf: Road network vector geometries.
            output_filepath: Destination PNG file path.
            cluster_col: Column name containing cluster labels.
        """
        output_path = Path(output_filepath)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"[Static Plot] Generating map: '{output_path.name}'...")
        fig, ax = plt.subplots(1, 1, figsize=(12, 10), dpi=300)

        # Plot road network underneath
        if road_gdf is not None and not road_gdf.empty:
            road_gdf.plot(ax=ax, color="darkgrey", linewidth=0.7, alpha=0.6, zorder=1)

        # Plot clustered points
        plot_gdf = points_gdf.copy()
        plot_gdf["cluster_plot"] = plot_gdf[cluster_col].fillna(-2).astype(int)

        plot_gdf.plot(
            column="cluster_plot",
            ax=ax,
            categorical=True,
            legend=True,
            markersize=9,
            cmap="turbo",
            legend_kwds={"title": "Cluster ID / Type", "loc": "upper left", "bbox_to_anchor": (1.02, 1), "frameon": False},
            zorder=2,
        )

        ax.set_title("Accident Clusters (Network-Constrained DBSCAN) and Road Network", fontsize=13, fontweight="bold", pad=15)
        ax.set_xlabel("Easting / Longitude (meters)", fontsize=10)
        ax.set_ylabel("Northing / Latitude (meters)", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        plt.savefig(output_path, bbox_inches="tight")
        plt.close(fig)
        print(f"[Static Plot Exported] High-res map saved to: {output_path}")
