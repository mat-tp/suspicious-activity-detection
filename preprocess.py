import pandas as pd
import json

df = pd.read_excel("suspicious-activity-detection/datasetInfo.xlsx")
print(df.head())
print("\n")

# This are Unique activities that I want to detect in my system.
activities = df["Activity"].unique()
print(activities) 

# This could be the main part of the Research: Distortions
# Can we build a traditional model; that can be able to do Video action classification In different distortions?

print("\n DISTORTIONS : ")
distortion = df["Distortion"].unique()

for dis in distortion :
    print(dis)

# print(distortion)

activity_videos = {}

for activity in activities:
    videos = df[df["Activity"] == activity]["Name of Video Series"].tolist()
    activity_videos[activity] = videos

with open("activity_videos.json", "w") as f:
    json.dump(activity_videos, f, indent=4)

print("Processed dataset saved.")