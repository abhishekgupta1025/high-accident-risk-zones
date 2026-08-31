import geopandas as gpd

input_shp = "Accident_Data_Mapped_Roads.shp" # Your original file
output_shp = "Accidents_Projected_Meters.shp" # New file name
target_crs = "EPSG:32645" # UTM Zone 45N

try:
    print(f"Reading {input_shp}...")
    gdf = gpd.read_file(input_shp)

    if gdf.crs == target_crs:
        print("Data is already in the target CRS. No reprojection needed.")
    else:
        print(f"Original CRS: {gdf.crs}")
        print(f"Reprojecting to {target_crs}...")
        gdf_proj = gdf.to_crs(target_crs)
        print(f"Saving reprojected data to {output_shp}...")
        gdf_proj.to_file(output_shp)
        print("Reprojection complete.")

except Exception as e:
    print(f"An error occurred: {e}")
