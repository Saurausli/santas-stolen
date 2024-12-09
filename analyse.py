import matplotlib.pyplot as plt
from main import weighted_reindeer_weariness
import pandas as pd

sample_sub = pd.read_csv("data/sample_submission.csv")
gifts = pd.read_csv("data/gifts.csv")
all_trips = sample_sub.merge(gifts, on='GiftId')
print(f"example:         {weighted_reindeer_weariness(all_trips)}")
print(f"three Solution:  {weighted_reindeer_weariness(pd.read_csv('data/three_solution.csv'))}")
print(f"trips_to_gifts:  {weighted_reindeer_weariness(pd.read_csv('data/trips_to_gifts.csv'))}")
print(f"trips_to_gifts2: {weighted_reindeer_weariness(pd.read_csv('data/trips_to_gifts2.csv'))}")
print(f"trips_to_giftsr: {weighted_reindeer_weariness(pd.read_csv('data/trips_to_giftsr.csv'))}")
print(f"trips_to_giftsr: {weighted_reindeer_weariness(pd.read_csv('data/trips_to_giftsr_random.csv'))}")