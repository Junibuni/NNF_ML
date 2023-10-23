import os
import glob
import pickle
import numpy as np

from tqdm import tqdm
from processing.ml_lr.linear_regression import create_dataset, LinearRegressionModel
from processing.utils.bfs import remove_small_clusters

cwd = os.getcwd()
model_path = os.path.join(cwd, "processing\data\model.pkl")
#Load
X, y = [], []

file_path = os.path.join(cwd, "processing/data/numpy_data")
file_list = glob.glob(file_path + "/*.npy")

for f in tqdm(file_list):
    basename = os.path.basename(f)
    rainrate = float(basename.split(".")[0])

    load_arr = np.load(f)
    load_arr = remove_small_clusters(load_arr)
    X.append(rainrate)
    y.append(load_arr)

X_train, y_train = create_dataset(X, y)

model = LinearRegressionModel()
model.train(X_train, y_train)
print("훈련세트 점수: {:.2f}".format( model.score(X_train, y_train) ))

pickle.dump(model, open(model_path, "wb"))