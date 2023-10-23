import pickle
import os
import matplotlib.pyplot as plt

from processing.utils.bfs import remove_small_clusters

cwd = os.getcwd()
model_path = os.path.join(cwd, "processing\data\model.pkl")

model = pickle.load(open(model_path, "rb"))

ans = model.predict(149.999999)
ans = remove_small_clusters(ans)

plt.imshow(ans, cmap="coolwarm", vmin=0)
plt.colorbar()
plt.text(10, -10, f'{10}mm', color='black', fontsize=12, fontweight='bold')
plt.show()

import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
rng = range(5, 500, 5)

for i, array in enumerate(rng): 
    ans = model.predict(array)
    ans = remove_small_clusters(ans)
    plt.imshow(ans, cmap="coolwarm", vmin=0, vmax=10)  # Customize the colormap as needed
    plt.axis(False)
    plt.text(10, -10, f'{array}mm', color='black', fontsize=12, fontweight='bold')

    plt.savefig(fr'D:\WorkSpace\003연구\산림청\산림청AI\cheon_ML\imgdata\frame_{i:03d}.png')  # Save each frame as a PNG file
    plt.close()  # Close the plot to free up memory

frames = []
for i, _ in enumerate(rng):
    frame = Image.open(fr'D:\WorkSpace\003연구\산림청\산림청AI\cheon_ML\imgdata\frame_{i:03d}.png')
    frames.append(frame)
    os.remove(fr'D:\WorkSpace\003연구\산림청\산림청AI\cheon_ML\imgdata\frame_{i:03d}.png')

# Save the frames as an animated GIF
frames[0].save('output.gif', save_all=True, append_images=frames[1:], duration=100, loop=0)



"""x, y, z = [], [], []
print(ans)
for r, row in enumerate(ans):
    for c, column in enumerate(row):
        if ans[r][c] == 0.0:
            continue
        x.append(c)
        y.append(r)
        z.append(ans[r][c])


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

sc = ax.scatter(y, x, z, c=z, cmap='viridis')

cbar = plt.colorbar(sc)
cbar.set_label('Depth')

ax.set_xlabel('Z')
ax.set_ylabel('X')
ax.set_zlabel('Y')

plt.show()"""