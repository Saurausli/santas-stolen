
north_pole = [90.0,0.0]
weight_limit = 1000
sleigh_weight = 10.0

import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from math import radians, cos, sin, asin, sqrt

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

def weighted_reindeer_weariness(all_trips):
    uniq_trips = all_trips.TripId.unique()
    
    if any(all_trips.groupby('TripId').Weight.sum() > weight_limit):
        raise Exception("One of the sleighs over weight limit!")
 
    dist = 0.0
    for t in uniq_trips:
        this_trip = all_trips[all_trips.TripId==t]
        dist = dist + weighted_trip_length(this_trip[['Latitude','Longitude']], this_trip.Weight.tolist())
    
    return dist    
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
    # pos = pos* (3.141592653589793 / 180)
    lon1 = pos1[0]* (3.141592653589793 / 180)
    lat1 = pos1[1]* (3.141592653589793 / 180)
    lon2 = pos2[0]* (3.141592653589793 / 180)
    lat2 = pos2[1]* (3.141592653589793 / 180)
    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371  # Radius of earth in kilometers
    return c * r

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
    print(f" Total time: {(t_sum):.2f} sec")
    print(f" Cycle time: {(t_sum)/t_it*1e6:.2f} usec")
    t_sum = 0
    t_it = 0

    
def calcutlate_df_loop(inital_step,next_steps,gifts):
    # print(inital_step[inital_step["GiftId"]==99980])
    # print(next_steps[next_steps["GiftId"]==99980])
    trips = inital_step['TripId'].unique().tolist()
    random.shuffle(trips)
    for t in trips:
        start_meas()
        last_step = inital_step[inital_step['TripId']==t].iloc[-1]
        # print(last_step)

        wrw_min_row = []
        start_lat = float(last_step['Latitude'])* (3.141592653589793 / 180)
        start_lon = float(last_step['Longitude'])* (3.141592653589793 / 180)
        

        end_lat = next_steps['Latitude'].to_numpy()* (3.141592653589793 / 180)
        end_lon = next_steps['Longitude'].to_numpy()* (3.141592653589793 / 180)
        weights = next_steps['Weight'].to_numpy()
        if len(weights)==0:
            end_meas()
            break
        dlon = (start_lon - end_lon) 
        dlat = (start_lat - end_lat)
        a = np.sin(dlat / 2)**2 + np.cos(end_lat) * np.cos(start_lat) * np.sin(dlon / 2)**2
        # print(a)
        c = 2 * np.asin(np.sqrt(a)) 
        r = 6371
        
        dist =  r * c * weights
        # print(dist.argmin())
        # print(next_steps)
        # try:
        wrw_min_row = next_steps.iloc[dist.argmin()].copy()
        # except ValueError:
        #     print(weights)
        #     print(c)
        #     print(dist)
        #     print(next_steps)
        wrw_min_row['TripId'] = t
        wrw_min_row= wrw_min_row.to_frame().T
        # if t == 0:
        # print(wrw_min_row)
        next_steps.drop(index=wrw_min_row.index, inplace = True)
        gifts.drop(index = wrw_min_row.index, inplace = True)

        inital_step = pd.concat([inital_step,wrw_min_row], ignore_index=True)
        

        end_meas()   
    
    return inital_step