import pandas as pd
from tools import array_haversin,weighted_reindeer_weariness, GRAD_RAD

def distribute_trips_to_gifts(inital_steps:pd.DataFrame,next_steps:pd.DataFrame,gifts:pd.DataFrame):

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

    return inital_steps

def get_start_position(trips= 5000):
    gifts = pd.read_csv('data/gifts.csv')
    gifts['Latitude'] *= GRAD_RAD
    gifts['Longitude'] *= GRAD_RAD
    gifts = gifts.sort_values(by=['Weight'], ascending=True)
    solution = gifts.head(trips).copy()

    gifts.drop(index = solution.index, inplace = True)
    solution['TripId'] = range(0, len(solution))
    
    while len(gifts.index)> 0:
        solution = distribute_trips_to_gifts(solution,gifts,gifts)
    solution['Latitude'] /= GRAD_RAD
    solution['Longitude'] /= GRAD_RAD
    solution = solution[::-1]
    
    print(weighted_reindeer_weariness(solution))
    # solution.to_csv("data/trips_to_giftsr_random.csv")
    return solution

