# Import the necessary libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

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

# linear regression through all pixels with sklearn
rgb_lin_model = linear_model.LinearRegression(fit_intercept=False).fit(np.stack((red.flatten(), green.flatten()), axis=1),blue.flatten())
rgb_coeff = rgb_lin_model.coef_
rgb_intercept = rgb_lin_model.intercept_

# create new unit vectors
e_x1 = np.array([1, -rgb_coeff[1]/rgb_coeff[0], 1/rgb_coeff[0]])
e_x2 = np.array([-rgb_coeff[1]/rgb_coeff[0], 1, 1/rgb_coeff[0]])
e_x3 = np.array([rgb_coeff[0], 1, rgb_coeff[1]])

# create line based on the linear regression
ax.plot(np.linspace(0, 255, 255)*e_x1[0], np.linspace(0, 255, 255)*e_x1[1], np.linspace(0, 255, 255)*e_x1[2], color = 'red')

plt.show()