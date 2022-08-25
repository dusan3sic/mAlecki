import numpy as np
import dataset

#load
X, y = dataset.load_data("data")

#normalize
for i, x in enumerate(X): X[i] = dataset.normalize(x)

