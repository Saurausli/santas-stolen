
from tools import weighted_reindeer_weariness,clear, print_meas, start_meas,weighted_trip_length_tuned
import numpy as np
import pandas as pd
from tools import weight_limit,weighted_reindeer_weariness,end_meas,start_meas,print_meas,weighted_reindeer_weariness_single_trip,get_trips_meta,GRAD_RAD,array_haversin,ToHeavy
import random
from tqdm import tqdm
from analyse import check_df
from start_algorithm import get_start_position

def opt2_single_trip(trip:pd.DataFrame):
    lat = trip.Latitude.to_list()
    lon = trip.Longitude.to_list()
    weight = trip.Weight.to_list()
    g = trip.GiftId.to_list()
    prev_score = weighted_trip_length_tuned(lat,lon,weight)

    for it in range(len(weight)*10):
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
    return trip_rearranged

def opt2(df:pd.DataFrame):
    df_solution = pd.DataFrame()
    for trip_id in tqdm(df.TripId.unique()):
        trip = df[df.TripId == trip_id]
        og_score = weighted_reindeer_weariness_single_trip(trip)
        trip_rearranged = opt2_single_trip(trip)

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
        for trip_id in tqdm(trips_ids):
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
                    trips_ids.remove(new_id)

            trips_ids.remove(old_id)
            end_meas()
            clear()

    except KeyboardInterrupt:
        pass

    new_score = weighted_reindeer_weariness(df)
    print(f"old score {old_score:20.0f}")
    print(f"new score {new_score:20.0f}")
    return df# df.to_csv("data/opt3_combined.csv")


def combine_neares(_df: pd.DataFrame):
    df = _df.copy()
    def init_meta(df):
        tr = get_trips_meta(df)
        #tr = tr.set_index("Id")
        # print(tr)
        tr['Latitude'] *= GRAD_RAD
        tr['Longitude'] *= GRAD_RAD
        return tr
    
    tr = init_meta(df)
    for i,t in tqdm(tr.iterrows()):
        if(len(tr)<2):
            break
        tr["Distance"] =  array_haversin(tr['Latitude'].to_list(),tr['Longitude'].to_list(),t['Latitude'],t['Longitude'])
        tr["Weight_Sum"] = tr.Weight + t.Weight
        tr = tr[tr["Weight_Sum"]<weight_limit]
        tr = tr.sort_values("Distance")
        try:
            t_nearest = tr.iloc[1]["Id"]
        except IndexError:
            break
        df1 = df[df["TripId"] == t.Id].copy()
        df2 = df[df["TripId"] == t_nearest].copy()
        
        cost1 = weighted_reindeer_weariness_single_trip(df1)
        cost2 = weighted_reindeer_weariness_single_trip(df2)
        df_comb = pd.concat([df1,df2],ignore_index=False)
        
        trip_new = opt2_single_trip(df_comb)
        try:
            total_cost = weighted_reindeer_weariness_single_trip(trip_new)
            
        except ToHeavy:
            continue

        if total_cost < cost1 + cost2:
            df_comb.TripId = t.Id
            df[df["TripId"] == t_nearest] = t.Id
            df.drop(index=df[df["TripId"] == t.Id].index, inplace=True)
            df = pd.concat([df,df_comb],ignore_index=False)
            tr.drop(index=tr[tr["Id"] == t.Id].index, inplace=True)
            tr.drop(index=tr[tr["Id"] == t_nearest].index, inplace=True)
            # print(weighted_reindeer_weariness(df))
    return df

def modify_trips(df:pd.DataFrame):
    df = combine_trips(df)
    df = opt2(df)
    df.to_csv("optimized.csv",index=False)

if __name__ == "__main__":
    # df = pd.read_csv("data/trips_to_giftsr_random.csv", index_col=0)
    # df = get_start_position(10000)
    # df.to_csv("data/1000.csv",index=False)
    # old_score = weighted_reindeer_weariness(df)
    # check_df(df)
    # while True: 
    #     df = combine_neares(df)
    #     new_score = weighted_reindeer_weariness(df)
    #     check_df(df)
    #     if old_score - new_score < 1e6:
    #         break
    #     old_score = new_score

    # df.to_csv("data/nearest.csv",index=False)
    df = pd.read_csv("data/nearest.csv")
    check_df(df)
    df = opt2(df)
    check_df(df)
    df.to_csv("data/opt2.csv",index=False)