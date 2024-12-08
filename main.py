


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tools import start_meas,end_meas,print_meas,distribute_gifts_to_trips,distribute_trips_to_gifts,weighted_reindeer_weariness



def first_solution():
    trips= 10000
    weight_difference = 5

    gifts = pd.read_csv('gifts.csv')
    gifts = gifts.sort_values(by=['Weight'], ascending=False)
    solution = gifts.head(trips).copy()

    gifts.drop(index = solution.index, inplace = True)
    # plt.scatter(solution["Longitude"],solution["Latitude"])

    solution['TripId'] = range(0, len(solution))

    while len(gifts.index)> 0:
        # print(f"gifts length {len(gifts.index)}")
        gifts_max = gifts["Weight"].max()
        print(f"{gifts_max:.2f}-{gifts_max-weight_difference:.2f}")

        # next_chunk = gifts[(gifts["Weight"]>gifts_max-weight_difference)].copy() #two

        # next_chunk = gifts.head(trips*2).copy() # three
        next_chunk = gifts.head(int(trips/2)).copy() # four

        solution = distribute_gifts_to_trips(solution,next_chunk,gifts)
        print(f"solution {len(solution.index)}")

    print_meas()
    print(weighted_reindeer_weariness(solution))
    solution.to_csv("data/four_solution.csv")


if __name__ == "__main__":
    trips= 5000
    weight_difference = 5

    gifts = pd.read_csv('data/gifts.csv')
    gifts['Latitude'] *= (3.141592653589793 / 180)
    gifts['Longitude'] *= (3.141592653589793 / 180)
    gifts = gifts.sort_values(by=['Weight'], ascending=True)
    solution = gifts.head(trips).copy()
    # gifts = gifts.sort_values(by=['Weight'], ascending=True)
    gifts.drop(index = solution.index, inplace = True)
    # plt.scatter(solution["Longitude"],solution["Latitude"])

    solution['TripId'] = range(0, len(solution))
    
    while len(gifts.index)> 0:
        # print(f"gifts length {len(gifts.index)}")
        # gifts_max = gifts["Weight"].max()
        # print(f"{gifts_max:.2f}-{gifts_max-weight_difference:.2f}")

        # next_chunk = gifts[(gifts["Weight"]>gifts_max-weight_difference)].copy() #two

        next_chunk = gifts.head(trips*2).copy() # three
        # next_chunk = gifts.head(int(trips/2)).copy() # four

        solution = distribute_trips_to_gifts(solution,next_chunk,gifts)
        print(f"solution {len(solution.index)}")
    solution = solution[::-1]
    print_meas()
    print(weighted_reindeer_weariness(solution))
    solution.to_csv("data/trips_to_giftsr.csv")
# print(solution[solution["TripId"]==0])
# print(weighted_reindeer_weariness(solution))
# all_trips = sample_sub.merge(gifts, on='GiftId')
# # print(all_trips)
# print(weighted_reindeer_weariness(all_trips))