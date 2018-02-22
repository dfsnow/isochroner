from shapely import geometry, wkt
from . import isocronut
import pandas as pd
import geopandas as gpd
import statistics as st
import fiona
import os


def batch(iterable, n=1):
    """ Batch process an iterable list """
    li = len(iterable)
    for ndx in range(0, li, n):
        yield iterable[ndx:min(ndx + n, li)]


def get_centroids(gdf):
    """
    Get the centroids of the polygons in a specified dataframe

    :param gdf: GeoDataFrame to extract centroids from
    :return: Returns list of lat, lng coordinates
    """
    x = gdf['geometry'].centroid.x
    y = gdf['geometry'].centroid.y
    coords = [list(i) for i in zip(y, x)]
    return coords


def check_isochrones(points, std_devs=2):
    """
    Remove points which are erroneously added to isochrones (by std deviation)

    :param points: Isochrone points in nested lists which are to be corrected
    :param std_devs: Toss points this many std devs from the mean
    :return: Fixed points with outliers removed
    """
    corrected = []
    for iso_num in range(0, len(points)):
        mean = [st.mean(x) for x in zip(*points[iso_num])]
        sd = [st.stdev(x) for x in zip(*points[iso_num])]
        fixed_isos = [x for x in points[iso_num] if abs(mean[0] - x[0]) / sd[0]
                      < std_devs and abs(mean[1] - x[1]) / sd[1] < std_devs]
        corrected.append(fixed_isos)
    return corrected


def iterate_isochrones(coords, key, duration, swap_xy):
    """
    Iterate over a list of coordinates and returns isochrones for each duration

    :param coords: Coordinates to iterate over
    :param key: Google Maps API key passed within function
    :param durations: Single or list of durations to calculate isochrones with
    :param swap_xy: Swap final lat and long values
    :return: Isochrones as polygons
    """
    iso_points = [isocronut.get_isochrone(
        origin=x, key=key, duration=duration, tolerance=2) for x in coords]
    iso_points = check_isochrones(iso_points)
    if swap_xy:
        iso_polys = [geometry.Polygon([[p[1], p[0]] for p in x])
                     for x in iso_points]
    else:
        iso_polys = [geometry.Polygon([[p[0], p[1]] for p in x])
                     for x in iso_points]
    return iso_polys


def shp_to_isochrones(gdf, key, duration, keep_cols=None, swap_xy=True):
    """
    Get a dataframe of origin coordinates and their corresponding
    isochrone polygons for different durations

    :param gdf: Name of the GeoDataFrame to use for use with centroids
    :param key: Google Maps API key passed within function
    :param duration: List of durations to create isochrones from
    :param keep_cols: Columns to keep from the original dataframe
    :param swap_xy: Swap output lat and long values
    :return: Dataframe of origin coords and their respective isochrones
    """
    df = pd.DataFrame()
    df['coords'] = get_centroids(gdf)
    if type(duration) is int or (type(duration) is list and len(duration) == 1):
        df['duration'] = duration
        if keep_cols is not None:
            df = df.set_index(gdf.index)
            df[keep_cols] = gdf[keep_cols]
        df['geometry'] = iterate_isochrones(df['coords'], key, duration, swap_xy)
        return df
    else:
        for length in duration:
            iso = pd.Series(iterate_isochrones(df['coords'], key, length, swap_xy))
            df[length] = iso.values
            df = df.set_index(gdf.index)
            df[keep_cols] = gdf[keep_cols]
    if swap_xy:
        df['coords'] = [x[::-1] for x in df['coords']]
    id_vars = ['coords']
    if keep_cols is not None:
        if keep_cols is not isinstance(keep_cols, list):
            id_vars += [keep_cols]
        else:
            id_vars += keep_cols
    cols = [col for col in df.columns if isinstance(col, int)]
    df = pd.melt(df, id_vars=id_vars, value_vars=cols,
                 var_name='duration', value_name='geometry')
    return df


def isochrones_to_shp(df, filename, crs, format='ESRI Shapefile'):
    """
    Convert a dataframe of isochrones to a shapefile or GeoJSON

    :param df: Input dataframe
    :param key: Google Maps API key passed within function
    :param filename: Filename to save as
    :param crs: CRS to use when creating the output
    :param format: Format to use for output, geojson or esri
    :return: Shapefile or GeoJSON with attached duration and origin point data
    """
    df['geometry'] = df['geometry'].astype(str).map(wkt.loads)
    crs = {'init': 'epsg:' + str(crs)}
    df = gpd.GeoDataFrame(df, crs=crs, geometry=df['geometry'])

    iso_schema = {
        'geometry': 'Polygon',
        'properties': {'id': 'int',
                       'coords': 'str',
                       'duration': 'int'}
    }

    with fiona.open(filename, 'w', format, iso_schema) as c:
        for index, geo in df.iterrows():
            c.write({
                'geometry': geometry.mapping(geo['geometry']),
                'properties': {'id': index,
                               'coords': str(geo['coords']),
                               'duration': int(geo['duration'])}
            })


def isochrone_batch(gdf, key, out_filename='isochrones.csv',
                    matching_var='GEOID', duration=15,
                    keep_cols=None, batch_size=5):
    """
    Batch processor for shapefiles to find centroid isochrones

    :param gdf: GeoDataFrame to find centroids for
    :param key: Google Maps API key passed within function
    :param out_filename: CSV to output to
    :param matching_var: Variable to match CSV against shapefile
    :param duration: Duration or list of durations to create isochrones for
    :param keep_cols: Keep data columns for GeoDataFrame
    :param batch_size: Batch size to process by, keep as low as possible
    :return: Outputs a CSV of isochrones for each geometry
    """
    if not os.path.exists(out_filename):
        out_df = pd.DataFrame()
        out_df[matching_var] = ""
        out_df.to_csv(out_filename, index=False)
    else:
        out_df = pd.read_csv(out_filename)
        gdf[matching_var] = gdf[matching_var].astype(int)
    unfinished_idx = [item for item in gdf[matching_var]
                      if item not in set(out_df[matching_var])]
    gdf_unfinished = gdf[gdf[matching_var].isin(unfinished_idx)]
    for batch_items in batch(gdf_unfinished, batch_size):
        batch_df = shp_to_isochrones(batch_items, key=key,
                                     duration=duration, keep_cols=keep_cols)
        print(batch_df)
        pd.read_csv(out_filename).append(batch_df).to_csv(out_filename, index=False)
