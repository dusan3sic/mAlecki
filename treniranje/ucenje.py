import numpy as np
import dataset
import koeficijenti

#load
X, y, tX, ty = dataset.load_data("./data.csv")

#process

X, tx = dataset.load_processed("X.csv", "tX.csv")

