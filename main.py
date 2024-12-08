


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tools import start_meas,end_meas,print_meas,distribute_gifts_to_trips,weighted_reindeer_weariness






if __name__ == "__main__":
    trips= 10000
    weight_difference = 5

    gifts = pd.read_csv('gifts.csv')
    sample_sub = pd.read_csv('sample_submission.csv')
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

        solution = calcutlate_df_loop(solution,next_chunk,gifts)
        print(f"solution {len(solution.index)}")

    print_meas()
    print(weighted_reindeer_weariness(solution))
    solution.to_csv("four_solution.csv")
# print(solution[solution["TripId"]==0])
# print(weighted_reindeer_weariness(solution))
# all_trips = sample_sub.merge(gifts, on='GiftId')
# # print(all_trips)
# print(weighted_reindeer_weariness(all_trips))