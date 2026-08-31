# -*- coding: utf-8 -*-
"""
Configuration management for the Network-Constrained High-Accident Risk Zone Analysis.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class PipelineConfig:
    """Configuration dataclass holding pipeline thresholds, CRS, and filesystem paths."""

    # Spatial clustering parameters
    eps: float = 150.0  # Search radius in meters (UTM Zone 45N)
    min_pts: int = 5  # Minimum incidents to form a dense core cluster
    intersection_tolerance: float = 0.0  # Buffer distance (meters) to bridge digitization gaps

    # Coordinate Reference Systems
    wgs84_crs: str = "EPSG:4326"
    utm_crs: str = "EPSG:32645"  # UTM Zone 45N metric projection

    # Base Directories
    base_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent.parent)
    data_dir: Optional[Path] = None
    results_dir: Optional[Path] = None

    # Input Data Paths
    road_shapefile: Optional[Path] = None
    accident_shapefile: Optional[Path] = None

    # Visual Mapping Settings
    generate_interactive_map: bool = True
    generate_static_map: bool = True
    generate_heatmap: bool = True

    def __post_init__(self):
        if self.data_dir is None:
            self.data_dir = self.base_dir / "data"
        if self.results_dir is None:
            self.results_dir = self.base_dir / "results"

        # Set default input shapefiles if not explicitly provided
        if self.road_shapefile is None:
            # Check projected path first, fallback to WGS84 path
            proj_road = self.data_dir / "shapefiles_projected_utm" / "Selected_Road_Features.shp"
            wgs_road = self.data_dir / "shapefiles_wgs84" / "Selected_Road_Features.shp"
            root_road = self.base_dir / "Selected_Road_Features.shp"
            self.road_shapefile = proj_road if proj_road.exists() else (wgs_road if wgs_road.exists() else root_road)

        if self.accident_shapefile is None:
            proj_acc = self.data_dir / "shapefiles_projected_utm" / "Accidents_Projected_Meters.shp"
            wgs_acc = self.data_dir / "shapefiles_wgs84" / "Accident_Data_Mapped_Roads.shp"
            root_acc = self.base_dir / "Accidents_Projected_Meters.shp"
            self.accident_shapefile = proj_acc if proj_acc.exists() else (wgs_acc if wgs_acc.exists() else root_acc)

        # Ensure output directories exist
        self.results_dir.mkdir(parents=True, exist_ok=True)
        (self.results_dir / "interactive_maps").mkdir(parents=True, exist_ok=True)
        (self.results_dir / "static_plots").mkdir(parents=True, exist_ok=True)
        (self.results_dir / "spatial_layers").mkdir(parents=True, exist_ok=True)
        (self.results_dir / "reports").mkdir(parents=True, exist_ok=True)
