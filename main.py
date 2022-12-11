# Import the necessary libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# load the image
img = cv2.imread('color_space_analysis/Lenna_(test_image).png')

# split the image into rgb channels
red, green, blue = cv2.split(img)

# plot each pixel in red, green and blue
ax = plt.figure().add_subplot(projection='3d')
ax.set_xlabel('red', color='red')
ax.set_ylabel('green', color='green')
ax.set_zlabel('blue', color='blue')

ax.set_xticks(np.arange(0, 256, 64))
ax.set_yticks(np.arange(0, 256, 64))
ax.set_zticks(np.arange(0, 256, 64))

C = np.array([red.flatten(), green.flatten(), blue.flatten()]).T
ax.scatter(red.flatten(), green.flatten(), blue.flatten(), c=C/255.0, marker='.')

plt.show()