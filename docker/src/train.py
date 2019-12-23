import os
import sys
import pandas as pd

from data_loader import DataLoader
from model import SimpleRNNModel

data_path = sys.argv[1]

data_loader = DataLoader(data_path=data_path)
model = SimpleRNNModel(data_loader)

model.train(data_loader.get_train_data(), data_loader.get_valid_data(),
            epochs=10, batch_size=32)

