
from tools import weighted_reindeer_weariness,clear, print_meas, start_meas,weighted_trip_length_tuned
import numpy as np
import pandas as pd
from tools import weight_limit,weighted_reindeer_weariness,end_meas,start_meas,print_meas,weighted_reindeer_weariness_single_trip
import random
from tqdm import tqdm

def opt2(df:pd.DataFrame):
    df_solution = pd.DataFrame()
    # for trip_id in tqdm(df.TripId.unique()):
    for trip_id in tqdm(df.TripId.unique()):
        trip = df[df.TripId == trip_id]
        og_score = weighted_reindeer_weariness_single_trip(trip)
        # trip_rearranged = trip.sort_values("Weight")
        lat = trip.Latitude.to_list()
        lon = trip.Longitude.to_list()
        weight = trip.Weight.to_list()
        g = trip.GiftId.to_list()
        prev_score = weighted_trip_length_tuned(lat,lon,weight)
        for it in range(len(weight)*100):
            rand_idx_1 = random.randint(0, len(g)-1)
            rand_idx_2 = random.randint(0, len(g)-1)
            if rand_idx_1 == rand_idx_2:
                continue
            lat[rand_idx_2], lat[rand_idx_1] = lat[rand_idx_1], lat[rand_idx_2]
            lon[rand_idx_2], lon[rand_idx_1] = lon[rand_idx_1], lon[rand_idx_2]
            weight[rand_idx_2], weight[rand_idx_1] = weight[rand_idx_1], weight[rand_idx_2]
            new_score = weighted_trip_length_tuned(lat,lon,weight)
            if prev_score > new_score:
                prev_score = new_score
                g[rand_idx_2], g[rand_idx_1] = g[rand_idx_1], g[rand_idx_2]
            else:
                lat[rand_idx_2], lat[rand_idx_1] = lat[rand_idx_1], lat[rand_idx_2]
                lon[rand_idx_2], lon[rand_idx_1] = lon[rand_idx_1], lon[rand_idx_2]
                weight[rand_idx_2], weight[rand_idx_1] = weight[rand_idx_1], weight[rand_idx_2]

        trip_rearranged = trip.set_index('GiftId').loc[g].reset_index()
        if og_score>weighted_reindeer_weariness_single_trip(trip_rearranged):
            df_solution = pd.concat([df_solution,trip_rearranged])
        else:
            df_solution = pd.concat([df_solution,trip])
    return df_solution.set_index('GiftId').reset_index()

def combine_trips(df):
    old_score = weighted_reindeer_weariness(df)
    new_score = old_score
    trips_ids = df.TripId.unique().tolist()
    try:
        for i,trip_id in tqdm(enumerate(trips_ids)):
            clear()
            print(f"trip_id {i} - {len(trips_ids)}")
            print(f"old score {old_score:20.0f}")
            print(f"new score {new_score:20.0f}")
            print(f"---------------------------")
            print(f"optimized {old_score-new_score:20.0f}")
            print_meas()
            start_meas()
            old_id = trip_id

            trip1 = df[df.TripId == old_id]

            
            
            lat_1 = trip1['Latitude'].to_list()
            lon_1 = trip1['Longitude'].to_list()
            weight_1 = trip1['Weight'].to_list()
            cost1 = weighted_trip_length_tuned(lat_1,lon_1, weight_1)

            for t in trips_ids:
                
                new_id = t
                if new_id == old_id: 
                    continue
                
                trip2 = df[df.TripId == new_id]
                lat_2  = trip2['Latitude'].to_list()
                lon_2   = trip2['Longitude'].to_list()
                weight_2 = trip2['Weight'].to_list()
                cost2 = weighted_trip_length_tuned(lat_2,lon_2, weight_2)
                new_cost = cost1 + cost2 +10
                if np.sum(weight_1) +np.sum(weight_2) > weight_limit:
                    continue
                comb_lat = [*lat_1,*lat_2]
                comb_lon = [*lon_1,*lon_2]
                comb_weight = [*weight_1,*weight_2]
                new_cost = weighted_trip_length_tuned(comb_lat,comb_lon, comb_weight)
                lon_2

                    
                if new_cost < cost1 + cost2:
                    new_score = new_score + new_cost - cost1 - cost2

                    combined_trip = pd.concat([trip1,trip2], ignore_index=True)
                    df.drop(index=df[df["GiftId"].isin(combined_trip.GiftId)].index, inplace=True)
                    combined_trip["TripId"] = old_id
                    df = pd.concat([df,combined_trip], ignore_index=True)

                    trip1 = combined_trip.copy()
                    lat_1 = comb_lat
                    lon_1 = comb_lon
                    weight_1 = comb_weight
                    cost1 = cost1 + cost2
                    if len(df) <100000:
                        print("error")
                        print(df)
                        raise ValueError
                    try:
                        print(f"drop {new_id}")
                        trips_ids.remove(new_id)
                    except ValueError:
                        print(trips_ids)
                        print(new_id)
                        raise ValueError
                
            trips_ids.remove(old_id)
            end_meas()
        
    except KeyboardInterrupt:
        pass

    new_score = weighted_reindeer_weariness(df)
    print(f"old score {old_score:20.0f}")
    print(f"new score {new_score:20.0f}")
    df.to_csv("data/opt3_combined.csv")

def modify_trips(df:pd.DataFrame):
    combine_trips(df)


if __name__ == "__main__":
    df = pd.read_csv("data/opt3.csv", index_col=0)
    print(len(df))
    modify_trips(df)
    # df = pd.read_csv("data/trips_combined2.csv", index_col=0)
    # print(len(df))
    # modify_trips(df)