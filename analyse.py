import matplotlib.pyplot as plt
from main import weighted_reindeer_weariness
import pandas as pd

sample_sub = pd.read_csv("sample_submission.csv")
gifts = pd.read_csv("gifts.csv")
all_trips = sample_sub.merge(gifts, on='GiftId')
first_solution = pd.read_csv("first_solution.csv")
print(len(first_solution["GiftId"].unique()))
print(f"example:      {weighted_reindeer_weariness(all_trips)}")
print(f"own Solution: {weighted_reindeer_weariness(first_solution)}")