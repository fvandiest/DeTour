import time
import threading
import foursquare
import numpy as np
import pandas as pd
from osmnx.utils import log
import multiprocessing as mp
from DeTour.core import fs_intent, fs_limit, fs_data_dict
from DeTour.utils import get_nested, process_time, get_quadrants
from DeTour.credentials import foursquare_API_client_ID, foursquare_API_client_secret

class Recommender(object):
    """
    The Recommender class captures the Foursquare recommendation engine

    """


    def __init__(self, user_profile, fs_client_ID=foursquare_API_client_ID,
                 fs_client_secret=foursquare_API_client_secret, intent=fs_intent,
                 limit=fs_limit):
        """
        Initializes the Recommender object instance

        Parameters
        ----------

        user_profile: dictionary
            Dictionary with user preferences
        fs_client_ID: string
            Foursquare API client ID
        fs_client_secret: string
            Foursquare API client password

        """

        # Instantiate Foursquare API wrapper
        self.fs_client = foursquare.Foursquare(client_id=fs_client_ID,
                                               client_secret=fs_client_secret)
        self.user_profile = user_profile
        self.intent = intent
        self.limit = limit
        self.recommendations = None

    @process_time
    def set_recommendations(self, bbox):
        '''
        Makes recommendation on venue category in accordance with user_profile
        Request docs: https://developer.foursquare.com/docs/api/venues/search

        Parameters
        ----------
        bbox : tuple of tuples
            Tuple with (0) south-west and (1) north_east lat/lon coordinates

        Returns
        -------
        DataFrame row with top recommendations
        '''

        # Split bbox in four quadrants
        quad_bbox = get_quadrants(bbox)
        key_order = self.user_profile['fs_category_order']
        bbox_mapping = {cat:qb for cat,qb in zip(key_order, quad_bbox)}

        #Request all venues across all user categories within boundary box
        venues = self.multithreading_get_venues(self.user_profile['fs_categories'],
                                                bbox_mapping)

        # Request details from all requested values
        venue_df = self.multithreading_get_details(venues)

        if venue_df is not None:
            recommendations = venue_df.groupby(['category'],
                                               as_index=False).apply(self.make_recommendation_per_category)
            recommendations = recommendations.dropna()
            self.recommendations = recommendations

        return self


    def multithreading_get_venues(self, category_dict, bbox_mapping):
        '''
        Multi-processes the get_venues Foursquare API requests

        Parameters
        ----------
        categories_dict : dict
            Category dictionary from user profile
        bbox : tuple of tuples
            Tuple with (0) south-west and (1) north_east lat/lon coordinates

        Returns
        -------
        List of all venues across subcategories in category dictionary
        '''

        log('Multiprocessing get_venue search with Foursquare API...')
        start_time = time.time()

        # Function to multi-process
        def single_thread_get_venues(category, subcategories, bb):
            try:
                result = self.fs_client.venues.search(params={'intent': self.intent,
                                                      'sw': ','.join(str(i) for i in bb[0]),
                                                      'ne': ','.join(str(i) for i in bb[1]),
                                                      'categoryId': ','.join(i for i in subcategories.values()),
                                                      'limit': self.limit})
                for venue in result.get('venues', None):
                    venue['recommendation_category'] = category
            except:
                result = {}
                log('get_venues request fails for category {}...'.format(category))
            return_dict[category] = result

        jobs = []
        manager = mp.Manager()
        return_dict = manager.dict()
        for category, subcategories in category_dict.items():
            thread = threading.Thread(name=category,
                                      target=single_thread_get_venues,
                                      args=(category, subcategories,
                                            bbox_mapping[category]))
            jobs.append(thread)
            thread.start()

        for j in jobs:
            j.join()

        # Get process results from the output queue
        venues = [venue
                  for category in return_dict.values()
                  for venue in category.get('venues', None)]

        log('Downloaded {:,} venues in {:,.2f} seconds'.format(len(venues),
                                                                time.time()-start_time))

        return venues


    def multithreading_get_details(self, venues):
        '''
        Multi-processes the get_details Foursquare API requests

        Parameters
        ----------
        client: Foursquare client
            Foursquare api client to do requests
        venues: list
            List of venues retrieved from Foursquare API get_venues request

        Returns
        -------
        DataFrame with venue details used for recommendations
        '''

        log('Multiprocessing get_details search with Foursquare API...')
        start_time = time.time()

        # Define process function
        def single_thread_get_details(venue):
            data_dict = fs_data_dict()
            venue_id = venue.get('id')
            try:
                result = self.fs_client.venues(venue_id).get('venue', None)
            except:
                result = {}
                log('get_details request fails for venue_id {}...'.format(venue_id))

            venue['details'] = result
            for key, value in data_dict.items():
                insert_value = get_nested(venue, value['location'])
                data_dict[key]['values'] = insert_value
            venue = None
            return_list.append(data_dict)

        jobs = []
        manager = mp.Manager()
        return_list = manager.list()
        for venue in venues:
            thread = threading.Thread(name=venue.get('id'),
                                      target=single_thread_get_details,
                                      args=(venue, ))
            jobs.append(thread)
            thread.start()

        for j in jobs:
            j.join()

        # Generate data dict to convert to dataframe
        data_dict = fs_data_dict()
        for venue in return_list:
            for key, value in data_dict.items():
                insert_value = get_nested(venue, value['location'])
                data_dict[key]['values'].append(venue[key]['values'])
        venue_df = pd.DataFrame({key: value['values']
                                 for key, value in data_dict.items()})
        venue_df['chain'] = venue_df['chain'].map(lambda x: 1 if len(x) > 0 else 0)
        log('Downloaded details for {:,} venues in {:,.2f} seconds'.format(venue_df.shape[0],
                                                                time.time()-start_time))

        return venue_df


    def make_recommendation_per_category(self, category_df):
        '''
        Makes one recommendation for each category

        Parameters
        ----------
        category_df: dataframe
            Dataframe of venues corresponding to one category

        Returns
        -------
        DataFrame row with recommendation
        '''

        # Drop row with any values missing
        category_df = category_df.dropna()

        # Add column in which ratings are floored to nearest integer
        category_df.loc[:, 'rating_floor'] = category_df.loc[:,
                                                             'rating'].map(lambda x: np.floor(x))

        # Drop rows with price tier above user preferences
        category_df = category_df.loc[category_df['price_tier']
                                      <= self.user_profile['fs_price_tier'], :]

        # Sort values based on bucketed ratings, price and number of ratings
        sorting_list = [('rating_floor', False),
                        ('price_tier', True),
                        ('ratingSignals', False)]

        # If the user does not accept chains, put it as first order preference
        if not self.user_profile['fs_accept_chains']:
            sorting_list = [('chain', True)] + sorting_list
        sort_category_df = category_df.sort_values(by=[i[0]
                                                       for i in sorting_list],
                                                   ascending=[i[1]
                                                              for i in sorting_list]).reset_index(drop=True)

        if len(sort_category_df)>0:
            # Return venue at top of the list
            return sort_category_df.iloc[0,:]
        else:
            return None

    def __str__(self):
        return('DeTour.Recommender')
