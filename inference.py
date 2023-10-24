import pickle
import os
import matplotlib.pyplot as plt
import time

rain_rate_to_predict = 179
cwd = os.getcwd()
model_path = os.path.join(cwd, "processing\data\model.pkl")

model = pickle.load(open(model_path, "rb"))

start = time.time()
ans = model.predict(rain_rate_to_predict)
end = time.time()

"""for i in model.model.coef_:
    print(i, end=" ")"""
print(model.model.intercept_)
print(model.model.n_features_in_)
print(f"time elapsed: {end - start:.2f}")
plt.imshow(ans, cmap="coolwarm", vmin=0, vmax=30)
plt.colorbar()
plt.text(10, -10, f'{rain_rate_to_predict}mm', color='black', fontsize=12, fontweight='bold')
plt.show()


import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
rng = range(5, 800, 5)

for i, array in enumerate(tqdm(rng)): 
    ans = model.predict(array)
    plt.imshow(ans, cmap="coolwarm", vmin=0, vmax=10)  # Customize the colormap as needed
    plt.axis(False)
    plt.text(10, -10, f'{array}mm', color='black', fontsize=12, fontweight='bold')
    plt.savefig(os.path.join(cwd, fr'imgdata\frame_{i:03d}.png'))  # Save each frame as a PNG file
    plt.close()  # Close the plot to free up memory

frames = []
for i, _ in enumerate(tqdm(rng)):
    frame_path = os.path.join(cwd, fr'imgdata\frame_{i:03d}.png')
    frame = Image.open(frame_path)
    if frame is not None:
        frames.append(frame)

if frames:
    frames[0].save('output.gif', save_all=True, append_images=frames[1:], duration=100, loop=0)
else:
    print("No valid frames to create the GIF.")

for i, _ in enumerate(tqdm(rng)):
    frame_path = os.path.join(cwd, fr'imgdata\frame_{i:03d}.png')

    try:
        os.remove(frame_path)
    except PermissionError:
        print(f"Could not delete {frame_path} as it's being used by another process.")