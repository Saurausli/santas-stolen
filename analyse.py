import matplotlib.pyplot as plt
from main import weighted_reindeer_weariness
import pandas as pd

# sample_sub = pd.read_csv("data/sample_submission.csv")
# gifts = pd.read_csv("data/gifts.csv")
# all_trips = sample_sub.merge(gifts, on='GiftId')
# print(f"example:         {weighted_reindeer_weariness(all_trips)}")
print(f"trips_to_giftsr: {weighted_reindeer_weariness(pd.read_csv('data/trips_to_giftsr_random.csv'))}")
comb = pd.read_csv('data/trips_combined.csv')
print(len(comb["TripId"].unique()))
print(f"combined:        {weighted_reindeer_weariness(comb)}")