#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main Execution Script for Network-Constrained High-Accident Risk Zone Analysis.
CLI Entrypoint orchestrating data ingestion, topological graph building, DBSCAN clustering, and cartographic rendering.
"""
import argparse
from pathlib import Path
import sys
import time

# Ensure src/ is on pythonpath
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR / "src"))

from risk_zones import (
    ClusterMetricsCalculator,
    DisjointSetUnion,
    InteractiveMapBuilder,
    NetworkConstrainedDBSCAN,
    PipelineConfig,
    RoadConnectivityGraph,
    StaticPlotBuilder,
    ensure_projected_crs,
    group_nearby_roads,
    load_point_data,
    load_road_data,
    precompute_group_intersections,
    precompute_segment_intersections,
    reproject_geodataframe,
)


def run_pipeline(config: PipelineConfig) -> None:
    """Executes the full network-constrained accident hotspot detection pipeline."""
    start_time = time.time()
    print("=" * 75)
    print("🚦 High-Accident Risk Zone Analysis: Network-Constrained DBSCAN Engine")
    print("=" * 75)
    print(f"Parameters: Search Radius (eps) = {config.eps}m | MinPts = {config.min_pts} | Tolerance = {config.intersection_tolerance}m")

    # 1. Load Datasets
    print("\n[Step 1/6] Ingesting geospatial vector datasets...")
    points_gdf = load_point_data(config.accident_shapefile)
    road_gdf, _ = load_road_data(config.road_shapefile)

    if points_gdf is None or road_gdf is None:
        print("[Error] Failed loading required datasets. Exiting pipeline.")
        sys.exit(1)

    # 2. Coordinate System Metric Reprojection
    print("\n[Step 2/6] Verifying metric Coordinate Reference Systems...")
    points_gdf = ensure_projected_crs(points_gdf, config.utm_crs)
    road_gdf = ensure_projected_crs(road_gdf, config.utm_crs)

    # 3. Topological Connectivity & R-Tree Precomputation
    print("\n[Step 3/6] Building 3-Tier Topological Connectivity Graph & R-Tree Index...")
    segment_intersections = precompute_segment_intersections(
        road_gdf=road_gdf,
        intersection_tolerance=config.intersection_tolerance,
    )
    connectivity_graph = RoadConnectivityGraph(intersections_map=segment_intersections)

    # 4. Network-Constrained DBSCAN Clustering
    print("\n[Step 4/6] Executing Network-Constrained DBSCAN...")
    dbscan = NetworkConstrainedDBSCAN(
        eps=config.eps,
        min_pts=config.min_pts,
        connectivity_graph=connectivity_graph,
    )
    labels = dbscan.fit_predict(points_gdf, road_id_column="road_uid")
    points_gdf["cluster"] = labels

    # 5. Calculate Metrics, Centroids, and Convex Hulls
    print("\n[Step 5/6] Computing spatial metrics, centroids, and convex hull danger zones...")
    summary_df, centroids_gdf, hulls_gdf = ClusterMetricsCalculator.calculate_centroids_and_hulls(
        points_gdf=points_gdf,
        cluster_col="cluster",
    )

    # 6. Save Deliverables and Cartographic Visualizations
    print("\n[Step 6/6] Exporting results, GIS layers, and interactive maps...")
    suffix = f"eps{config.eps}_mp{config.min_pts}"

    # Export CSV Summary Report
    csv_path = config.results_dir / "reports" / f"cluster_summary_{suffix}.csv"
    summary_df.to_csv(csv_path, index=False)
    print(f"  [Saved] Numerical summary CSV: {csv_path}")

    # Export Spatial GeoPackages
    points_gpkg = config.results_dir / "spatial_layers" / f"points_clustered_{suffix}.gpkg"
    roads_gpkg = config.results_dir / "spatial_layers" / f"roads_projected_{suffix}.gpkg"
    points_gdf.to_file(points_gpkg, driver="GPKG")
    road_gdf.to_file(roads_gpkg, driver="GPKG")
    print(f"  [Saved] Clustered points GeoPackage: {points_gpkg}")
    print(f"  [Saved] Projected road network GeoPackage: {roads_gpkg}")

    # Render Interactive Leaflet Map
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

    # Render Static Matplotlib Figure
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
    print(f"  • Clustered Incidents:       {clustered_accidents} ({(clustered_accidents / max(total_accidents, 1) * 100):.1f}%)")
    print("=" * 75)

    if not summary_df.empty:
        print("\n🏆 Top 5 Identified High-Accident Risk Zones:")
        print(summary_df.head(5).to_string(index=False))
        print("=" * 75)


def parse_args() -> PipelineConfig:
    """Parses command-line arguments and returns a populated PipelineConfig object."""
    parser = argparse.ArgumentParser(
        description="High-Accident Risk Zone Identification via Network-Constrained DBSCAN",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--eps", type=float, default=150.0, help="DBSCAN search radius (meters)")
    parser.add_argument("--min-pts", type=int, default=5, help="Minimum points to form a cluster core")
    parser.add_argument("--tolerance", type=float, default=0.0, help="Intersection buffer tolerance (meters)")
    parser.add_argument("--road-shp", type=str, default=None, help="Path to road network shapefile")
    parser.add_argument("--accident-shp", type=str, default=None, help="Path to accident incident shapefile")
    parser.add_argument("--output-dir", type=str, default=None, help="Custom results output directory")
    parser.add_argument("--no-interactive-map", action="store_true", help="Disable Folium HTML generation")
    parser.add_argument("--no-static-map", action="store_true", help="Disable Matplotlib PNG generation")

    args = parser.parse_args()

    config = PipelineConfig(
        eps=args.eps,
        min_pts=args.min_pts,
        intersection_tolerance=args.tolerance,
        road_shapefile=Path(args.road_shp) if args.road_shp else None,
        accident_shapefile=Path(args.accident_shp) if args.accident_shp else None,
        results_dir=Path(args.output_dir) if args.output_dir else None,
        generate_interactive_map=not args.no_interactive_map,
        generate_static_map=not args.no_static_map,
    )
    return config


if __name__ == "__main__":
    cfg = parse_args()
    run_pipeline(cfg)
