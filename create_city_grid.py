import osmnx as ox
import geopandas as gpd
from shapely.geometry import box

utrecht = ox.geocode_to_gdf("Utrecht, Netherlands")
utrecht = utrecht.to_crs(epsg=3857)

minx, miny, maxx, maxy = utrecht.total_bounds
cell_size = 200 # In meters
cols = list(range(int(minx), int(maxx), cell_size))
rows = list(range(int(miny), int(maxy), cell_size))
polygons = []
for x in cols:
    for y in rows:
        polygons.append(box(x, y, x + cell_size, y + cell_size))
grid = gpd.GeoDataFrame({"cell_id": range(len(polygons)), 
                         "geometry": polygons}, 
                         crs=utrecht.crs)

# Only use up to city bounds
grid_clipped = gpd.clip(grid, utrecht)
grid_clipped.to_file('city_files/utrecht.shp')

