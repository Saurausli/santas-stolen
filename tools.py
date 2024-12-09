
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

class ToHeavy(Exception):
    """Exception raised for custom error scenarios.

    Attributes:
        message -- explanation of the error
    """
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

def weighted_reindeer_weariness(all_trips):
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
    if t_it > 0:
        print(f" Cycle time: {(t_sum)/t_it*1e6:.2f} usec")
    t_sum = 0
    t_it = 0

    
def distribute_gifts_to_trips(inital_step,next_steps,gifts):

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
        wrw_min_row = next_steps.iloc[dist.argmin()].copy()
        wrw_min_row['TripId'] = t
        wrw_min_row= wrw_min_row.to_frame().T
        # if t == 0:
        # print(wrw_min_row)
        next_steps.drop(index=wrw_min_row.index, inplace = True)
        gifts.drop(index = wrw_min_row.index, inplace = True)
        inital_step = pd.concat([inital_step,wrw_min_row], ignore_index=True)
        end_meas()   
        print_meas()
    return inital_step


def array_haversin(start_lat,start_lon,end_lat,end_lon):
    dlon = (start_lon - end_lon) 
    dlat = (start_lat - end_lat)
    a = np.sin(dlat / 2)**2 + np.cos(end_lat) * np.cos(start_lat) * np.sin(dlon / 2)**2
    # 
    c = 2 * np.asin(np.sqrt(a)) 
    r = 6371  
    
    return  r * c

def distribute_trips_to_gifts(inital_steps:pd.DataFrame,next_steps:pd.DataFrame,gifts:pd.DataFrame):
    start_meas()
    last_steps = inital_steps.groupby('TripId').tail(1)
    last_steps.sort_values('TripId')

    start_lat = last_steps['Latitude'].to_numpy()
    start_lon = last_steps['Longitude'].to_numpy()
    trips = []
    for i, next_gift in next_steps.iterrows():

        
        end_lat = next_gift.Latitude
        end_lon = next_gift.Longitude
        weights = next_gift.Weight

        dist = array_haversin(start_lat,start_lon,end_lat,end_lon)
        dist = dist* weights

        
        next_gift = next_gift.to_frame().T
        t = dist.argmin()
        trips.append(t)
        start_lat[t] = end_lat
        start_lon[t] = end_lon
    next_steps['TripId'] = trips
    inital_steps = pd.concat([inital_steps,next_steps], ignore_index=True)
    gifts.drop(index = next_steps.index, inplace = True)
    end_meas() 
    print_meas()
    return inital_steps

def calculate_middle_point(latitude, longitude):
    
    x = y = z = 0.0
    for lat, lon in zip(latitude, longitude):
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)
        
        # Convert to Cartesian coordinates
        x += np.cos(lat_rad) * np.cos(lon_rad)
        y += np.cos(lat_rad) * np.sin(lon_rad)
        z += np.sin(lat_rad)
    
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


def get_trips_meta(df:pd.DataFrame):
    uniq_trips = df.TripId.unique().tolist()
    trips = pd.DataFrame()
    trips["Id"] = uniq_trips
    trips["Weight"] = 0.0
    trips["Latitude"] = 0.0
    trips["Longitude"] = 0.0
    trips["Cost"] = 0.0
    for t in uniq_trips:
        this_trip = df[df.TripId==t]
        trips.loc[trips.Id == t,"Weight"] = float(this_trip.Weight.sum())
        trips.loc[trips.Id == t,"Cost"] = float(weighted_trip_length(this_trip[['Latitude','Longitude']], this_trip.Weight.tolist()))
        trips.loc[trips.Id == t,"Latitude"], trips.loc[trips.Id == t,"Longitude"] = calculate_middle_point(this_trip['Latitude'].to_numpy(),this_trip['Longitude'].to_numpy())
    trips["Weight_To_Cost"] = trips["Weight"] / trips["Cost"] *1000
    return trips


def combine(df):
    
    # print(trips)
    old_score = weighted_reindeer_weariness(df)
    trips = get_trips_meta(df)
    trips_ids = trips["Id"].to_list()

    for trip_id in trips_ids:
        old_id = trip_id
        t_start = trips[trips.Id == old_id]
        trips["Distance"] = array_haversin(t_start["Latitude"].to_numpy()[0],t_start["Longitude"].to_numpy()[0],trips["Latitude"].to_numpy(),trips["Longitude"].to_numpy())
        trips = trips.sort_values("Distance",ascending=True)

        for i,t in trips.iloc[1:100].iterrows():
            new_id = t.Id
            trip1 = df[df.TripId == old_id].copy()
            trip2 = df[df.TripId == new_id].copy()
            cost1 = weighted_reindeer_weariness(trip1)
            cost2 = weighted_reindeer_weariness(trip2)
            # print(cost1,len(trip1))
            # print(cost2,len(trip2))
            trip2.TripId = old_id
            combined_trip = pd.concat([trip1,trip2], ignore_index=True)
            new_cost = cost1 + cost2 +10
            try:
                new_cost = weighted_reindeer_weariness(combined_trip)
            except ToHeavy:
                # print("to heavy")
                # break
                pass

            if new_cost < cost1 + cost2:
                print(f"merge {new_id} {old_id}")
                df.drop(index=df[df["GiftId"].isin(combined_trip.GiftId)].index, inplace=True)
                df = pd.concat([df,combined_trip], ignore_index=True)
                print(f"{i} score {new_cost -(cost1 + cost2):20.0f}")
                trips_ids.remove(new_id)
        trips = get_trips_meta(df)
    new_score = weighted_reindeer_weariness(df)
    print(f"old score {old_score:20.0f}")
    print(f"new score {new_score:20.0f}")

def modify_trips(df:pd.DataFrame):
    combine(df)
    


if __name__ == "__main__":
    df = pd.read_csv("data/trips_to_giftsr_random.csv", index_col=0)
    modify_trips(df)


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