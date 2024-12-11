import matplotlib.pyplot as plt
from tools import weighted_reindeer_weariness
import pandas as pd

# sample_sub = pd.read_csv("data/sample_submission.csv")
# gifts = pd.read_csv("data/gifts.csv")
# all_trips = sample_sub.merge(gifts, on='GiftId')
# print(f"example:         {weighted_reindeer_weariness(all_trips)}")

def check_df(df):
    entries = len(df["GiftId"].unique())
    trips = len(df["TripId"].unique())
    score = weighted_reindeer_weariness(df)
    print(f"-"*40)
    print(f"score:     {score:20.0f}")
    print(f"entries:   {entries:20.0f}")
    print(f"trips:     {trips:20.0f}")
    print(f"-"*40)
# print(f"trips_to_giftsr: {weighted_reindeer_weariness(pd.read_csv('data/trips_to_giftsr_random.csv'))}")
# comb = pd.read_csv('data/trips_combined.csv')
# print(len(comb["TripId"].unique()))


# df = pd.read_csv("data/trips_to_giftsr_random.csv")
# check_df(df)