north_pole = [90.0,0.0]
weight_limit = 1000
sleigh_weight = 10.0


import pandas as pd
import numpy as np
from haversine import haversine as hv
import matplotlib.pyplot as plt
import time
from math import radians, cos, sin, asin, sqrt


import numba as nb

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

def custom_map_radians(coords):
    """
    Custom implementation of map(radians, iterable) using plain Python.
    Converts each coordinate in the input to radians.
    """
    result = []
    for coord in coords:
        result.append(coord * (3.141592653589793 / 180))  # Manual radians conversion
    return result

# @nb.jit(nopython=True, cache=True)
# def haversine(pos1:tuple,pos2:tuple):
#     """
#     Calculate the great circle distance in kilometers between two points 
#     on the earth (specified in decimal degrees)
#     """
#     # convert decimal degrees to radians 
#     lon1 = pos1[0]* (3.141592653589793 / 180)
#     lat1 = pos1[1]* (3.141592653589793 / 180)
#     lon2 = pos2[0]* (3.141592653589793 / 180)
#     lat2 = pos2[1]* (3.141592653589793 / 180)
        
#     # haversine formula 
#     dlon = lon2 - lon1 
#     dlat = lat2 - lat1 
#     a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
#     c = 2 * asin(sqrt(a)) 
#     r = 6371
#     return c * r
# @nb.jit()
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

# @nb.jit(nopython=True, cache=True)
def weighted_trip_length_custom(tuples, weights): 
    
    dist = 0.0
    prev_stop = north_pole
    
    prev_weight = sum(weights)
    start_meas()
    for location, weight in zip(tuples, weights):

        dlon = (prev_stop[0] - location[0])* (3.141592653589793 / 180) 
        dlat = (prev_stop[1] - location[1]) * (3.141592653589793 / 180)
        a = sin(dlat / 2)**2 + cos(location[1]* (3.141592653589793 / 180) ) * cos(prev_stop[1]* (3.141592653589793 / 180) ) * sin(dlon / 2)**2
        # print(a)
        c = 2 * asin(sqrt(a)) 
        r = 6371
        
        dist = dist + r * c * prev_weight
        
        prev_stop = location
        prev_weight = prev_weight - weight
    end_meas()
    return dist

# @nb.jit(nopython=True, cache=True)
# def weighted_trip_length_custom(tuples, weights): 
#     # global t1
#     # global t2
#     dist = 0.0
#     prev_stop = north_pole
#     weights = np.array(weights)
#     prev_weight = np.sum(weights)
#     # print(tuples)
#     position = np.asmatrix(tuples)* (3.141592653589793 / 180)

#     # print(position) 
#     lon1 = position[0,:-1]  
#     lat1 =  position[1,:-1]  
#     lon2 = position[0,1:] 
#     lat2 =position[1,1:]
#     # haversine formula 
#     # dlon = position[0,1:] - position[0,:-1]  
#     # dlat = position[1,1:] - position[1,:-1]  
#     dlon = lon2 - lon1 
#     dlat = lat2 - lat1 
#     # print(lon1) 
#     a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
#     # print(a)
#     c = 2 * np.arcsin(np.sqrt(a)) 
#     r = 6371  # Radius of earth in kilometers
#     haversine =  c * r
#     for location, weight in zip(haversine, weights):
        
        
#         dist = dist + haversine * prev_weight
        
#         prev_stop = location
#         prev_weight = prev_weight - weight
    
#     return dist



trips= 5000
gifts = pd.read_csv('gifts.csv')
sample_sub = pd.read_csv('sample_submission.csv')
gifts = gifts.sort_values(by=['Weight'], ascending=False)
solution = gifts.head(trips).copy()
gifts.drop(index = solution.index, inplace = False)
# plt.scatter(solution["Longitude"],solution["Latitude"])

solution['TripId'] = range(0, len(solution))

next_chunk = gifts.head(trips).copy()
gifts.drop(index = next_chunk.index, inplace = False)

for t in solution['TripId'].unique():
    part_solution = solution[solution['TripId']==t]
    wrw_min = np.inf
    wrw_min_row = []

    postion = [[float(part_solution['Latitude'].values[0]),float(part_solution['Longitude'].values[0])]]
    # print(stops)
    # stops = part_solution[['Latitude','Longitude']]
    # postion = [tuple(x) for x in stops.values]

    postion.append([0,0])
    postion.append(north_pole)

    weights = part_solution.Weight.to_list()
    
    weights.append(0)
    weights.append(sleigh_weight)

    # t1 = time.perf_counter(), time.process_time()
    
    for i, row_next_chunk in next_chunk.iterrows():
        
        rl = row_next_chunk.to_list()
        # print(rl[1:3])
        
        # pos= [row_next_chunk['Latitude'],row_next_chunk['Longitude']]
        postion[len(postion)-2] = rl[1:3]
        weights[len(weights)-2] = rl[3]
        
        # weights[len(weights)-2] = row_next_chunk['Weight']
        
        wrw = weighted_trip_length_custom(postion, weights)
        t2 = time.perf_counter(), time.process_time()
        
        if wrw < wrw_min:
            wrw_min_row = row_next_chunk
            wrw_min = wrw


        
        
    # print(f" Total time: {(t_sum)*1e3:.2f} msec")
    print(f" Cycle time: {(t_sum)/t_it*1e6:.2f} usec")

    # print(f" Real time: {1e6*(t2[0] - t1[0])/trips:.2f} usec")
    #print(f" CPU time: {1e6*(t2[1] - t1[1])/trips:.2f} usec")
    wrw_min_row= wrw_min_row.to_frame().T
    next_chunk.drop(index=wrw_min_row.index, inplace = False)
    solution = pd.concat([solution,wrw_min_row], ignore_index=True)
print(weighted_reindeer_weariness(solution))
plt.show()
# all_trips = sample_sub.merge(gifts, on='GiftId')
# # print(all_trips)
# print(weighted_reindeer_weariness(all_trips))