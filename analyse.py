import matplotlib.pyplot as plt
from main import weighted_reindeer_weariness
import pandas as pd

sample_sub = pd.read_csv("sample_submission.csv")
gifts = pd.read_csv("gifts.csv")
all_trips = sample_sub.merge(gifts, on='GiftId')
print(f"example:         {weighted_reindeer_weariness(all_trips)}")
print(f"first Solution:  {weighted_reindeer_weariness(pd.read_csv('first_solution.csv'))}")
print(f"second Solution: {weighted_reindeer_weariness(pd.read_csv('second_solution.csv'))}")
print(f"three Solution:  {weighted_reindeer_weariness(pd.read_csv('three_solution.csv'))}")
print(f"four  Solution:  {weighted_reindeer_weariness(pd.read_csv('four_solution.csv'))}")