import json
import requests
from shapely.geometry import Point
from DeTour.credentials import google_maps_API_key

def text_search_request(query, key=google_maps_API_key, **kwargs):
    # Set base URL for request
    request = "https://maps.googleapis.com/maps/api/place/textsearch/"

    # Set required parameters
    # Set output form defaulted to json
    output = 'json'
    # Ensure the input query is a string value and replace spaces by '+' - catch possible errors?
    query = str(query)
    query = query.replace(' ','+')
    # Append required parameters to request URL
    request = request + "{}?query={}&key={}".format(output,query,key)

    # Set optional parameters
    #opt_parameters = ['region','location','radius','language','minprice','maxprice','opennow','pagetoken','type']
    #kwargs_keys = kwargs.

    return request

def get_location_point(query, **kwargs):
    request = text_search_request(query=query)
    json_response = requests.get(request).json()
    response_data = json_response.get('results')
    location = response_data[0]['geometry']['location']
    point = (location['lat'],location['lng'])
    return point

def set_node_location_data(node, **kwargs):
    if node['geographic_crs'] is None:
        node['gapi_request'] = text_search_request(query=node['query'])
        json_response = requests.get(node['gapi_request']).json()
        if json_response['status']=='ZERO_RESULTS':
            raise Exception('Location not found in Google Maps. Please try another location')
        else:
            response_data = json_response.get('results')
            if len(response_data)>0:
                node['gapi_response'] = response_data[0]
                location = response_data[0]['geometry']['location']
                node['geographic_crs'] = (location['lat'],location['lng'])
    return node
