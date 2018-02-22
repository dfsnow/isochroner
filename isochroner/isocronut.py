from __future__ import division
from math import cos, sin, pi, radians, degrees, asin, atan2
import requests


def build_url(key='', origin='', destination=''):
    """ Determine the url to pass for the desired search """
    # origin is either an address string (like '1 N State St Chicago IL') or a [lat, lng] 2-list
    if origin == '':
        raise Exception('origin cannot be blank.')
    elif isinstance(origin, str):
        origin_str = origin.replace(' ', '+')
    elif isinstance(origin, list) and len(origin) == 2:
        origin_str = ','.join(map(str, origin))
    else:
        raise Exception('origin should be a list [lat, lng] or an address string.')
    # destination is similar, although it can be a list of address strings or [lat, lng] 2 lists
    if destination == '':
        raise Exception('destination cannot be blank.')
    elif isinstance(destination, str):
        destination_str = destination.replace(' ', '+')
    elif isinstance(destination, list):
        destination_str = ''
        for element in destination:
            if isinstance(element, str):
                destination_str = '{0}|{1}'.format(destination_str, element.replace(' ', '+'))
            elif isinstance(element, list) and len(element) == 2:
                destination_str = '{0}|{1}'.format(destination_str, ','.join(map(str, element)))
            else:
                raise Exception('destination must be a list of lists [lat, lng] or a list of strings.')
        destination_str = destination_str.strip('|')
    else:
        raise Exception('destination must be a a list of lists [lat, lng] or a list of strings.')

    prefix = 'https://maps.googleapis.com/maps/api/distancematrix/json?mode=walking&units=imperial&avoid=tolls|ferries'
    full_url = prefix + '&origins={0}&destinations={1}&key={2}'.format(origin_str, destination_str, key)
    return full_url


def parse_json(url=''):
    """
    Parse the json response from the API
    """
    req = requests.get(url)
    d = req.json()

    if not d['status'] == 'OK':
        raise Exception('Error. Google Maps API return status: {}'.format(d['status']))

    addresses = d['destination_addresses']

    i = 0
    durations = [0] * len(addresses)
    for row in d['rows'][0]['elements']:
        if not row['status'] == 'OK':
            durations[i] = 9999
        else:
            if 'duration_in_traffic' in row:
                durations[i] = row['duration_in_traffic']['value'] / 60
            else:
                durations[i] = row['duration']['value'] / 60
        i += 1
    return [addresses, durations]


def geocode_address(key='', address=''):
    """ For use in calculating distances between 2 locations, the [lat, lng] is needed instead of the address """
    # Convert origin and destination to URL-compatible strings
    if address == '':
        raise Exception('address cannot be blank.')
    elif isinstance(address, str):
        address_str = address.replace(' ', '+')
    else:
        raise Exception('address should be a string.')

    # Convert the URL string to a URL, which we can parse
    # using the urlparse() function into path and query
    # Note that this URL should already be URL-encoded
    prefix = 'https://maps.googleapis.com/maps/api/geocode/json'
    full_url = prefix + '?address={0}&key={1}'.format(address_str, key)
    print(full_url)
    # Request geocode from address
    req = requests.get(full_url)
    d = req.json()

    # Parse the json to pull out the geocode
    if not d['status'] == 'OK':
        print('Error. Google Maps API return status: {}'.format(d['status']))
        geocode = None
    else:
        geocode = [d['results'][0]['geometry']['location']['lat'],
                   d['results'][0]['geometry']['location']['lng']]
        print(geocode)
    return geocode


def select_destination(key='', origin='', angle='', radius=''):
    """ Given a distance and polar angle, calculate the geocode of a destination point from the origin """
    if origin == '':
        raise Exception('origin cannot be blank.')
    if angle == '':
        raise Exception('angle cannot be blank.')
    if radius == '':
        raise Exception('radius cannot be blank.')

    if isinstance(origin, str):
        origin_geocode = geocode_address(key=key, address=origin)
    elif isinstance(origin, list) and len(origin) == 2:
        origin_geocode = origin
    else:
        raise Exception('origin should be a list [lat, lng] or a string address.')

    # Find the location on a sphere a distance 'radius' along a bearing 'angle' from origin
    # This uses haversines rather than simple Pythagorean distance in Euclidean space
    # because spheres are more complicated than planes.
    r = 3963.1676  # Radius of the Earth in miles
    bearing = radians(angle)  # Bearing in radians converted from angle in degrees
    lat1 = radians(origin_geocode[0])
    lng1 = radians(origin_geocode[1])
    lat2 = asin(sin(lat1) * cos(radius / r) + cos(lat1) * sin(radius / r) * cos(bearing))
    lng2 = lng1 + atan2(sin(bearing) * sin(radius / r) * cos(lat1), cos(radius / r) - sin(lat1) * sin(lat2))
    lat2 = degrees(lat2)
    lng2 = degrees(lng2)
    return [lat2, lng2]


def get_bearing(origin='', destination=''):
    """ Calculate the bearing from origin to destination """
    if origin == '':
        raise Exception('origin cannot be blank')
    if destination == '':
        raise Exception('destination cannot be blank')

    bearing = atan2(sin((destination[1] - origin[1]) * pi / 180) *
                    cos(destination[0] * pi / 180),
                    cos(origin[0] * pi / 180) * sin(destination[0] * pi / 180) -
                    sin(origin[0] * pi / 180) * cos(destination[0] * pi / 180) *
                    cos((destination[1] - origin[1]) * pi / 180))
    bearing = bearing * 180 / pi
    bearing = (bearing + 360) % 360
    return bearing


def sort_points(key='', origin='', iso=''):
    """ Put the isochrone points in a proper order """
    if origin == '':
        raise Exception('origin cannot be blank.')
    if iso == '':
        raise Exception('iso cannot be blank.')

    if isinstance(origin, str):
        origin_geocode = geocode_address(key=key, address=origin)
    elif isinstance(origin, list) and len(origin) == 2:
        origin_geocode = origin
    else:
        raise Exception('origin should be a list [lat, lng] or a string address.')

    bearings = []
    for row in iso:
        bearings.append(get_bearing(origin_geocode, row))

    points = zip(bearings, iso)
    sorted_points = sorted(points)
    sorted_iso = [point[1] for point in sorted_points]
    return sorted_iso


def get_isochrone(origin='', key='', duration='', number_of_angles=12, tolerance=0.1):
    """
    Get isochrone as a list of lat, long points

    :param origin: string address or [lat, lng] 2-list
    :param duration: minutes that the isochrone contour value should map
    :param number_of_angles: how many bearings to calculate this contour for (think of this like resolution)
    :param tolerance: how many minutes within the exact answer for the contour is good enough
    :return Nested list of lat, long points for an isochrone given an origin point
    """
    if origin == '':
        raise Exception('origin cannot be blank')
    if duration == '':
        raise Exception('duration cannot be blank')
    if not isinstance(number_of_angles, int):
        raise Exception('number_of_angles must be an int')

    # Make a radius list, one element for each angle,
    #   whose elements will update until the isochrone is found
    rad1 = [duration / 12] * number_of_angles  # initial r guess based on 5 mph speed
    phi1 = [i * (360 / number_of_angles) for i in range(number_of_angles)]
    data0 = [0] * number_of_angles
    rad0 = [0] * number_of_angles
    rmin = [0] * number_of_angles
    rmax = [1.25 * duration] * number_of_angles  # rmax based on 75 mph speed
    iso = [[0, 0]] * number_of_angles

    # Counter to ensure we're not getting out of hand
    j = 0

    # Here's where the binary search starts
    while sum([a - b for a, b in zip(rad0, rad1)]) != 0:
        rad2 = [0] * number_of_angles
        for i in range(number_of_angles):
            iso[i] = select_destination(key=key, origin=origin,
                                        angle=phi1[i], radius=rad1[i])
        url = build_url(key, origin, iso)
        data = parse_json(url)
        for i in range(number_of_angles):
            if (data[1][i] < (duration - tolerance)) & (data0[i] != data[0][i]):
                rad2[i] = (rmax[i] + rad1[i]) / 2
                rmin[i] = rad1[i]
            elif (data[1][i] > (duration + tolerance)) & (data0[i] != data[0][i]):
                rad2[i] = (rmin[i] + rad1[i]) / 2
                rmax[i] = rad1[i]
            else:
                rad2[i] = rad1[i]
            data0[i] = data[0][i]
        rad0 = rad1
        rad1 = rad2
        j += 1
        if j > 50:
            raise Exception("This is taking too long, so I'm just going to quit.")

    for i in range(number_of_angles):
        iso[i] = geocode_address(key=key, address=data[0][i])

    iso = [x for x in iso if x is not None]
    iso = sort_points(key, origin=origin, iso=iso)
    return iso

