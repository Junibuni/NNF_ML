import os
import glob
import numpy as np

from processing.ml_lr.linear_regression import create_dataset, LR

cwd = os.getcwd()

#Load
X, y = [], []

file_path = os.path.join(cwd, "processing/data/numpy_data")
file_list = glob.glob(file_path + "/*.npy")

for f in file_list:
    basename = os.path.basename(f)
    rainrate = float(basename.split(".")[0])

    load_arr = np.load(f)

    X.append(rainrate)
    y.append(load_arr)

X_train, y_train = create_dataset(X, y)

model = LR()
model.train(X_train, y_train)

ans = model.predict(13)
print(ans)
import matplotlib.pyplot as plt
plt.imshow(ans, cmap="coolwarm")
plt.colorbar()
plt.show()
