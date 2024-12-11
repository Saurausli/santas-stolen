
north_pole = [90.0,0.0]
weight_limit = 1000
sleigh_weight = 10.0
GRAD_RAD = 3.141592653589793 / 180
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from math import radians, cos, sin, asin, sqrt

def get_trips_meta(df:pd.DataFrame):
    uniq_trips = df.TripId.unique().tolist()
    trips = pd.DataFrame()
    trips["Id"] = uniq_trips
    trips["Weight"] = 0.0
    trips["Latitude"] = 0.0
    trips["Longitude"] = 0.0
    # trips["Cost"] = 0.0
    for t in uniq_trips:
        this_trip = df[df.TripId==t]
        trips.loc[trips.Id == t,"Weight"] = float(this_trip.Weight.sum())
        trips.loc[trips.Id == t,"Cost"] = float(weighted_trip_length_tuned(this_trip['Latitude'].tolist(),
                                                                           this_trip['Longitude'].tolist(), 
                                                                           this_trip.Weight.tolist()))
        trips.loc[trips.Id == t,"Latitude"], trips.loc[trips.Id == t,"Longitude"] = calculate_middle_point(this_trip['Latitude'].to_numpy(),
                                                                                                           this_trip['Longitude'].to_numpy())
    trips["Weight_To_Cost"] = trips["Weight"] / trips["Cost"] *1000
    return trips

def weighted_trip_length(stops, weights): 
    tuples = [tuple(x) for x in stops.values]
    # adding the last trip back to north pole, with just the sleigh weight
    tuples.append(north_pole)
    weights.append(sleigh_weight)
    dist = 0.0
    prev_stop = north_pole
    prev_weight = sum(weights)

    for location, weight in zip(tuples, weights):
        dist = dist + haversine(location, prev_stop) * prev_weight
        prev_stop = location
        prev_weight = prev_weight - weight

    return dist

def weighted_trip_length_tuned(lat:list,lon:list, weights:list): 
    # print(len(lat))
    
    deg_to_rad = np.pi / 180

    weights = np.array([*weights,sleigh_weight])
    end_lat= np.array([north_pole[0],*lat])* deg_to_rad
    end_lon = np.array([north_pole[1],*lon])* deg_to_rad

    start_lat  = np.array([*lat,north_pole[0]])* deg_to_rad
    start_lon = np.array([*lon,north_pole[1]])* deg_to_rad

    prev_weight = np.cumsum(weights[::-1])[::-1]

    dist = array_haversin(start_lat,start_lon,end_lat,end_lon)
    dist = np.sum(dist*prev_weight)

    return dist


class ToHeavy(Exception):
    """Exception raised for custom error scenarios.

    Attributes:
        message -- explanation of the error
    """
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

def weighted_reindeer_weariness(_all_trips):
    all_trips = _all_trips.copy()
    # check_sum = len(all_trips.GiftId.unique())
    # if check_sum != 100000:
    #     raise Exception(f"Gifts missing {check_sum}")
    uniq_trips = all_trips.TripId.unique()
    
    if any(all_trips.groupby('TripId').Weight.sum() > weight_limit):
        raise ToHeavy(f"One of the sleighs over weight limit! {all_trips.groupby('TripId').Weight.sum()}")

 
    dist = 0.0
    for t in uniq_trips:
        this_trip = all_trips[all_trips.TripId==t]
        dist = dist + weighted_trip_length(this_trip[['Latitude','Longitude']], this_trip.Weight.tolist())

    return dist   



def weighted_reindeer_weariness_single_trip(trip:pd.DataFrame):
    if trip.Weight.sum() > weight_limit:
        raise ToHeavy(f"One of the sleighs over weight limit!")
    dist_opt = weighted_trip_length_tuned(trip['Latitude'].to_list(),trip['Longitude'].to_list(), trip.Weight.to_list())
    return dist_opt   

t_sum = 0
t_it = 0
t1, t2 = [0],[0]



def haversine(pos1: tuple, pos2: tuple):
    """
    Calculate the great circle distance in kilometers between two points 
    on the earth (specified in decimal degrees) using NumPy.
    """
    # convert decimal degrees to radians 
    # lon1, lat1, lon2, lat2 = np.radians([*pos1, *pos2])
    # pos = pos* GRAD_RAD
    lat1 = pos1[0]* GRAD_RAD
    lon1 = pos1[1]* GRAD_RAD
    lat2 = pos2[0]* GRAD_RAD
    lon2 = pos2[1]* GRAD_RAD
    # haversine formula 
    
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371  # Radius of earth in kilometers
    return c * r


def array_haversin(start_lat,start_lon,end_lat,end_lon):
    dlon = (start_lon - end_lon) 
    dlat = (start_lat - end_lat)
    a = np.sin(dlat / 2)**2 + np.cos(end_lat) * np.cos(start_lat) * np.sin(dlon / 2)**2
    c = 2 * np.asin(np.sqrt(a)) 
    r = 6371  
    return  r *c


from os import system, name

# import sleep to show output for some time period
from time import sleep

def clear():

    # for windows
    if name == 'nt':
        _ = system('cls')

    # for mac and linux(here, os.name is 'posix')
    else:
        _ = system('clear')

def start_meas():
    global t1
    global t2
    t1 = time.perf_counter(), time.process_time()

def end_meas():
    global t1
    global t2
    global t_sum
    global t_it
    t2 = time.perf_counter(), time.process_time()
    t_sum = t_sum + t2[0] - t1[0]
    t_it += 1

def print_meas():
    global t_sum
    global t_it
    
    print(f" Total time: {(t_sum)*1000:.2f} msec")
    if t_it > 0:
        print(f" Cycle time: {(t_sum)/t_it*1e6:.2f} usec")
    t_sum = 0
    t_it = 0

    
def distribute_gifts_to_trips(inital_step,next_steps,gifts):

    trips = inital_step['TripId'].unique().tolist()
    random.shuffle(trips)
    for t in trips:
        
        last_step = inital_step[inital_step['TripId']==t].iloc[-1]
        # print(last_step)

        wrw_min_row = []
        start_lat = float(last_step['Latitude'])* GRAD_RAD
        start_lon = float(last_step['Longitude'])* GRAD_RAD
        

        end_lat = next_steps['Latitude'].to_numpy()* GRAD_RAD
        end_lon = next_steps['Longitude'].to_numpy()* GRAD_RAD
        weights = next_steps['Weight'].to_numpy()
        if len(weights)==0:
            break
        dlon = (start_lon - end_lon) 
        dlat = (start_lat - end_lat)
        a = np.sin(dlat / 2)**2 + np.cos(end_lat) * np.cos(start_lat) * np.sin(dlon / 2)**2
        # print(a)
        c = 2 * np.asin(np.sqrt(a)) 
        r = 6371  
        dist =  r * c * weights
        wrw_min_row = next_steps.iloc[dist.argmin()].copy()
        wrw_min_row['TripId'] = t
        wrw_min_row= wrw_min_row.to_frame().T
        # if t == 0:
        # print(wrw_min_row)
        next_steps.drop(index=wrw_min_row.index, inplace = True)
        gifts.drop(index = wrw_min_row.index, inplace = True)
        inital_step = pd.concat([inital_step,wrw_min_row], ignore_index=True)
         
    return inital_step

def calculate_middle_point(latitude, longitude):
    x = y = z = 0.0

    lat_rad = np.radians(latitude)
    lon_rad = np.radians(longitude)
    # Convert to Cartesian coordinates
    x = np.sum(np.cos(lat_rad) * np.cos(lon_rad))
    y = np.sum(np.cos(lat_rad) * np.sin(lon_rad))
    z = np.sum(np.sin(lat_rad))
    # Average Cartesian coordinates
    total = len(latitude)
    x /= total
    y /= total
    z /= total
    # Convert back to latitude and longitude
    lon_center = np.atan2(y, x)
    hyp = np.sqrt(x * x + y * y)
    lat_center = np.atan2(z, hyp)
    # Convert back to degrees
    return np.degrees(lat_center), np.degrees(lon_center)
    





        # inital_steps = pd.concat([inital_steps,wrw_min_row], ignore_index=True)
        # print(wrw_min_trip)
        
        # wrw_min_row['TripId'] = t
        # wrw_min_row= wrw_min_row.to_frame().T
    #     # if t == 0:
    #     # print(wrw_min_row)
    #     next_steps.drop(index=wrw_min_row.index, inplace = True)
    #     gifts.drop(index = wrw_min_row.index, inplace = True)
    #     inital_step = pd.concat([inital_step,wrw_min_row], ignore_index=True)
    #     end_meas()   
    
    # return inital_step