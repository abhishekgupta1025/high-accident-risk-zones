#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Command-Line Interface & Pipeline Runner for Network-Constrained High-Accident Risk Zone Analysis.
"""
import argparse
from pathlib import Path
from typing import Optional, Sequence
import sys
import time

from .clustering.dbscan import NetworkConstrainedDBSCAN
from .clustering.metrics import ClusterMetricsCalculator
from .config import PipelineConfig
from .io.loader import load_point_data, load_road_data
from .io.reprojector import ensure_projected_crs
from .topology.dsu import group_nearby_roads
from .topology.graph import RoadConnectivityGraph
from .topology.spatial_index import (
    precompute_group_intersections,
    precompute_segment_intersections,
    snap_points_to_roads,
)
from .visualization.static_plot import StaticPlotBuilder
from .visualization.web_map import InteractiveMapBuilder


def run_pipeline(config: PipelineConfig) -> None:
    """Executes the full network-constrained accident hotspot detection pipeline."""
    start_time = time.time()
    print("=" * 75)
    print("🚦 High-Accident Risk Zone Analysis: Network-Constrained DBSCAN Engine")
    print("=" * 75)
    print(
        f"Parameters: eps={config.eps}m | MinPts={config.min_pts} | "
        f"Tolerance={config.intersection_tolerance}m | "
        f"GroupThreshold={config.group_threshold}m | "
        f"MaxSnap={config.max_snap_distance if config.max_snap_distance is not None else 'unlimited'}"
    )

    # 1. Load Datasets
    print("\n[Step 1/7] Ingesting geospatial vector datasets...")
    points_gdf = load_point_data(config.accident_shapefile)
    road_gdf, _ = load_road_data(config.road_shapefile)

    if points_gdf is None or road_gdf is None:
        print("[Error] Failed loading required datasets. Exiting pipeline.")
        sys.exit(1)

    # 2. Coordinate System Metric Reprojection
    print("\n[Step 2/7] Verifying metric Coordinate Reference Systems...")
    points_gdf = ensure_projected_crs(points_gdf, config.utm_crs)
    road_gdf = ensure_projected_crs(road_gdf, config.utm_crs)

    # 3. Snap Incident Points to Nearest Road Segments
    print("\n[Step 3/7] Snapping incident points to nearest road segments via R-Tree...")
    points_gdf = snap_points_to_roads(
        points_gdf=points_gdf,
        road_gdf=road_gdf,
        max_snap_distance=config.max_snap_distance,
        road_id_column="road_uid",
    )

    # 4. Topological Connectivity & Optional DSU Corridor Grouping
    if config.group_threshold > 0.0:
        print(
            f"\n[Step 4/7] Running DSU corridor grouping (threshold={config.group_threshold}m) "
            "& building group topology..."
        )
        road_group_map = group_nearby_roads(road_gdf=road_gdf, threshold=config.group_threshold)
        road_gdf["group_id"] = road_gdf["road_uid"].map(road_group_map)
        points_gdf["group_id"] = points_gdf["road_uid"].map(
            lambda uid: -999 if uid == -999 else road_group_map.get(uid, -999)
        )
        intersections = precompute_group_intersections(
            road_gdf=road_gdf,
            road_group_map=road_group_map,
            intersection_tolerance=config.intersection_tolerance,
        )
        cluster_road_col = "group_id"
    else:
        print("\n[Step 4/7] Building 3-Tier Topological Connectivity Graph & R-Tree Index...")
        intersections = precompute_segment_intersections(
            road_gdf=road_gdf,
            intersection_tolerance=config.intersection_tolerance,
        )
        cluster_road_col = "road_uid"

    connectivity_graph = RoadConnectivityGraph(intersections_map=intersections)

    # 5. Network-Constrained DBSCAN Clustering
    print("\n[Step 5/7] Executing Network-Constrained DBSCAN...")
    dbscan = NetworkConstrainedDBSCAN(
        eps=config.eps,
        min_pts=config.min_pts,
        connectivity_graph=connectivity_graph,
    )
    labels = dbscan.fit_predict(points_gdf, road_id_column=cluster_road_col)
    points_gdf["cluster"] = labels

    # 6. Calculate Metrics, Centroids, and Convex Hulls
    print("\n[Step 6/7] Computing spatial metrics, centroids, and convex hull danger zones...")
    summary_df, centroids_gdf, hulls_gdf = ClusterMetricsCalculator.calculate_centroids_and_hulls(
        points_gdf=points_gdf,
        cluster_col="cluster",
    )

    # 7. Save Deliverables and Cartographic Visualizations
    print("\n[Step 7/7] Exporting results, GIS layers, and interactive maps...")
    suffix = f"eps{config.eps}_mp{config.min_pts}"

    csv_path = config.results_dir / "reports" / f"cluster_summary_{suffix}.csv"
    summary_df.to_csv(csv_path, index=False)
    print(f"  [Saved] Numerical summary CSV: {csv_path}")

    points_gpkg = config.results_dir / "spatial_layers" / f"points_clustered_{suffix}.gpkg"
    roads_gpkg = config.results_dir / "spatial_layers" / f"roads_projected_{suffix}.gpkg"
    points_gdf.to_file(points_gpkg, driver="GPKG")
    road_gdf.to_file(roads_gpkg, driver="GPKG")
    print(f"  [Saved] Clustered points GeoPackage: {points_gpkg}")
    print(f"  [Saved] Projected road network GeoPackage: {roads_gpkg}")

    if config.generate_interactive_map:
        html_path = config.results_dir / "interactive_maps" / f"cluster_map_{suffix}.html"
        InteractiveMapBuilder.create_map(
            points_gdf=points_gdf,
            road_gdf=road_gdf,
            centroids_gdf=centroids_gdf,
            hulls_gdf=hulls_gdf,
            output_filepath=html_path,
            cluster_col="cluster",
            generate_heatmap=config.generate_heatmap,
        )

    if config.generate_static_map:
        png_path = config.results_dir / "static_plots" / f"static_cluster_map_{suffix}.png"
        StaticPlotBuilder.create_plot(
            points_gdf=points_gdf,
            road_gdf=road_gdf,
            output_filepath=png_path,
            cluster_col="cluster",
        )

    elapsed = time.time() - start_time
    num_clusters = len(summary_df)
    total_accidents = len(points_gdf)
    clustered_accidents = len(points_gdf[points_gdf["cluster"] > 0])

    print("\n" + "=" * 75)
    print(f"✅ Pipeline Successfully Completed in {elapsed:.2f}s")
    print(f"  • Total Incidents Analyzed:  {total_accidents}")
    print(f"  • Hotspot Clusters Formed:   {num_clusters}")
    print(
        f"  • Clustered Incidents:       {clustered_accidents} "
        f"({(clustered_accidents / max(total_accidents, 1) * 100):.1f}%)"
    )
    print("=" * 75)

    if not summary_df.empty:
        print("\n🏆 Top 5 Identified High-Accident Risk Zones:")
        print(summary_df.head(5).to_string(index=False))
        print("=" * 75)


def parse_args(argv: Optional[Sequence[str]] = None) -> PipelineConfig:
    """Parses command-line arguments and returns a populated PipelineConfig object."""
    parser = argparse.ArgumentParser(
        description="High-Accident Risk Zone Identification via Network-Constrained DBSCAN",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--eps", type=float, default=150.0, help="DBSCAN search radius (meters)")
    parser.add_argument("--min-pts", type=int, default=5, help="Minimum points to form a cluster core")
    parser.add_argument("--tolerance", type=float, default=0.0, help="Intersection buffer tolerance (meters)")
    parser.add_argument(
        "--group-threshold",
        type=float,
        default=0.0,
        help="DSU road corridor grouping threshold in meters (0.0 disables grouping)",
    )
    parser.add_argument(
        "--max-snap-distance",
        type=float,
        default=None,
        help="Maximum snap distance in meters to associate incidents with road segments",
    )
    parser.add_argument("--road-shp", type=str, default=None, help="Path to road network shapefile")
    parser.add_argument("--accident-shp", type=str, default=None, help="Path to accident incident shapefile")
    parser.add_argument("--output-dir", type=str, default=None, help="Custom results output directory")
    parser.add_argument("--no-interactive-map", action="store_true", help="Disable Folium HTML generation")
    parser.add_argument("--no-static-map", action="store_true", help="Disable Matplotlib PNG generation")

    args = parser.parse_args(argv)

    return PipelineConfig(
        eps=args.eps,
        min_pts=args.min_pts,
        intersection_tolerance=args.tolerance,
        group_threshold=args.group_threshold,
        max_snap_distance=args.max_snap_distance,
        road_shapefile=Path(args.road_shp) if args.road_shp else None,
        accident_shapefile=Path(args.accident_shp) if args.accident_shp else None,
        results_dir=Path(args.output_dir) if args.output_dir else None,
        generate_interactive_map=not args.no_interactive_map,
        generate_static_map=not args.no_static_map,
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI entrypoint for `risk-zones` and `run_pipeline.py`."""
    cfg = parse_args(argv)
    run_pipeline(cfg)


if __name__ == "__main__":
    main()
