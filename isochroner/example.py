import isochroner as iso
import geopandas as gpd

gdf = gpd.read_file('ti_2015_chi_only.shp')
iso.isochrone_batch(gdf, key='AIzaSyAaimJsTXHE1WjiEM-ARZxSMmMQpsY_4xQ',
                    duration=20, keep_cols=['GEOID'], batch_size=3)

# df = pd.read_csv('isochrones.csv')
# iso.isochrones_to_shp(df, os.path.join('shapefiles', 'isochrones.shp'), crs=4269)

