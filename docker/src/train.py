import os
import sys
import pandas as pd

from data_loader import DataLoader

data_path = sys.argv[1]

data_loader = DataLoader(data_path=data_path)


# df = pd.read_csv("data/train/measurement.csv")
# print(df.head())
