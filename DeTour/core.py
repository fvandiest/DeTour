# Overpass API
network_type = 'walk'
network_multiplier = 1.2
bbox_distance = 1000
fig_dpi = 100
timeout = 180
memory = None
maxsize = None

# Foursquare api
fs_intent = "browse"
fs_limit = 50

def fs_data_dict():
    return {'name': {'location': ['name'],
                     'values': []},
            'address': {'location': ['location', 'address'],
                        'values': []},
            'postalCode': {'location': ['location', 'postalCode'],
                           'values': []},
            'city': {'location': ['location', 'city'],
                     'values': []},
            'state': {'location': ['location', 'state'],
                      'values': []},
            'country': {'location': ['location', 'country'],
                        'values': []},
            'id': {'location': ['id'],
                   'values': []},
            'lat': {'location': ['location', 'lat'],
                    'values': []},
            'lon': {'location': ['location', 'lng'],
                    'values': []},
            'rating': {'location': ['details', 'rating'],
                       'values': []},
            'ratingSignals': {'location': ['details', 'ratingSignals'],
                              'values': []},
            'price_tier': {'location': ['details', 'price', 'tier'],
                           'values': []},
            'chain': {'location': ['venueChains'],
                      'values': []},
            'category': {'location': ['recommendation_category'],
                         'values': []}}

graph_item_dict = dict(
    walk=dict(
        # Query parameters
        key='highway', value=None, footprint=False, filter_type='walk',
        way=True, relation=False, node=False,
        # Plot parameters
        bgcolor='#333333', fig_length = 8, stat_fig_height = 6,
        edge_color='w', edge_alpha=1, edge_zorder=4, folder_name = './images/',
        node_color='w', node_alpha=1, node_zorder=4,
        # Route ploting parameters
        route=dict(edge_color='r', edge_linewidth=4, edge_alpha=0.8, edge_zorder=5,
                   node_color='r', node_size=500, node_alpha=1, node_zorder=5)
    ),
    rail=dict(
        # Query parameters
        key='railway', value=None, footprint=False, filter_type='rail',
        way=True, relation=True, node=False,
        # Plot parameters
        edge_color='lightslategray', edge_linewidth=4, edge_alpha=1, edge_zorder=5,
        node_color='lightslategray', node_size=4, node_alpha=1, node_zorder=5
    ),
    park=dict(
        # Query parameters
        key='leisure', value=None, footprint=True, filter_type='park',
        way=True, relation=True, node=False,
        # Plot parameters
        facecolor='greenyellow', edge_color='greenyellow', linewidth=0,
        alpha=1, zorder=1
    ),
    water=dict(
        # Query parameters
        key='natural', value='water', footprint=True,
        way=True, relation=True, node=False,
        # Plot parameters
        facecolor='dodgerblue', edge_color='dodgerblue', linewidth=0,
        alpha=1, zorder=1
    ),
    waterway=dict(
        # Query parameters
        key='waterway', value=None, footprint=True,
        way=True, relation=True, node=False,
        # Plot parameters
        facecolor='dodgerblue', edge_color='dodgerblue', linewidth=0,
        alpha=1, zorder=1
    ),
    religious=dict(
        # Query parameters
        # key='building', value=None, relation=True, footprint=True, filter_type='building',
        key='building', value=None, footprint=True, filter_type='religious',
        way=True, relation=True, node=False,
        # Plot parameters
        facecolor='darkorange', edge_color='darkorange', linewidth=0,
        alpha=1, zorder=1
    ),
    historic=dict(
        # Query parameters
        # key='building', value=None, relation=True, footprint=True, filter_type='building',
        key='historic', value=None, footprint=True, filter_type='historic',
        way=True, relation=False, node=False,
        # Plot parameters
        facecolor='yellow', edge_color='yellow', linewidth=0,
        alpha=1, zorder=1
    ),
    historic_node=dict(
        # Query parameters
        # key='building', value=None, relation=True, footprint=True, filter_type='building',
        key='historic', value=None, footprint=False, filter_type='historic',
        way=False, relation=False, node=True,
        # Plot parameters
        node_color='yellow', node_size=20, node_alpha=1, node_zorder=4
    ),
    tourism=dict(
        # Query parameters
        # key='building', value=None, relation=True, footprint=True, filter_type='building',
        key='tourism', value=None, footprint=True, filter_type='tourism',
        way=True, relation=False, node=False,
        # Plot parameters
        facecolor='deeppink', edge_color='deeppink', linewidth=0,
        alpha=1, zorder=2
    ),
    tourism_node=dict(
        # Query parameters
        # key='building', value=None, relation=True, footprint=True, filter_type='building',
        key='historic', value=None, footprint=False, filter_type='tourism',
        way=False, relation=False, node=True,
        # Plot parameters
        node_color='deeppink', node_size=20, node_alpha=1, node_zorder=2
    ),
    gastronomy=dict(
        # Query parameters
        # key='building', value=None, relation=True, footprint=True, filter_type='building',
        key='amenity', value=None, footprint=True, filter_type='gastronomy',
        way=True, relation=False, node=False,
        # Plot parameters
        facecolor='magenta', edge_color='magenta', linewidth=0,
        alpha=1, zorder=2
    ),
    gastronomy_node=dict(
        # Query parameters
        # key='building', value=None, relation=True, footprint=True, filter_type='building',
        key='amenity', value=None, footprint=False, filter_type='gastronomy',
        way=False, relation=False, node=True,
        # Plot parameters
        node_color='magenta', node_size=20, node_alpha=1, node_zorder=2
    ),
    tree=dict(
        # Query parameters
        key='natural', value='tree', footprint=False,
        way=False, relation=False, node=True,
        # Plot parameters
        node_color='limegreen', node_size=20, node_alpha=1, node_zorder=4
    )
)

user_profiles=dict(
    Base=dict(
        # Four square profile specifics
        fs_categories={
            # https://developer.foursquare.com/docs/resources/categories
            'coffee':{'Coffee Shop':'4bf58dd8d48988d1e0931735'},
            'restaurant':{'Vietnamese Restaurant':'4bf58dd8d48988d14a941735',
                          'Indian Restaurant':'4bf58dd8d48988d10f941735',
                          'Italian Restaurant':'4bf58dd8d48988d110941735',
                          'Middle Eastern Restaurant':'4bf58dd8d48988d115941735',
                          'Vegetarian / Vegan Restaurant':'4bf58dd8d48988d1d3941735'},
            'bar':{'Beer Bar':'56aa371ce4b08b9a8d57356c',
                    'Pub':'4bf58dd8d48988d11b941735',
                    'Cocktail Bar':'4bf58dd8d48988d11e941735',
                    'Speakeasy':'4bf58dd8d48988d1d4941735'},
            'lunch':{'Bagel Shop':'4bf58dd8d48988d179941735',
                    'Salad Place':'4bf58dd8d48988d1bd941735',
                    'Sandwich Place':'4bf58dd8d48988d1c5941735',
                    'Soup Place':'4bf58dd8d48988d1dd931735',
                    'Vegetarian / Vegan Restaurant':'4bf58dd8d48988d1d3941735'}
        },
        fs_category_order = ['coffee','lunch', 'bar', 'restaurant'],
        # Does the user want to avoid venues which are part of a chain?
        fs_accept_chains=False,
        # Set the price tier of the user: 1 (least pricey) - 4 (most pricey)
        fs_price_tier=2,
        # Set the factors of how the user scores the importance of types of locations
        user_scoring_factors = dict(park=1,
                                    historic=1,
                                    tree=1,
                                    gastronomy=1,
                                    water=1,
                                    tourism=1,
                                    religious=1)
    )
)
