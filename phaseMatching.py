import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import scipy as sc
import rawpy as rp


def velocityFieldPhase(movedImage: np.ndarray, reference: np.ndarray, kSize) -> np.ndarray:
    height, width = movedImage.shape
    X = np.arange(kSize, height - 3 * kSize, kSize)
    Y = np.arange(kSize, width - 3 * kSize, kSize)
    [YY, XX] = np.meshgrid(Y, X)
    velocity = np.zeros((X.size, Y.size, 4))
    for i in range(X.size):
        startX = X[i]
        for j in range(Y.size):
            startY = Y[j]
            kernel = reference[startX:startX + 2 * kSize + 1, startY:startY + 2 * kSize + 1]
            padding = cv2.copyMakeBorder(kernel, kSize, kSize, kSize, kSize, cv2.BORDER_DEFAULT)
            subImage = movedImage[startX - kSize:startX + 3 * kSize + 1, startY - kSize:startY + 3 * kSize + 1]
            translation, response = cv2.phaseCorrelate(padding, subImage)
            if response >= 0.1:
                y, x = translation
                velocity[i, j, 0] = x
                velocity[i, j, 1] = y
            else:
                velocity[i, j, 0] = -100
                velocity[i, j, 1] = -100
    velocity[..., 2] = XX
    velocity[..., 3] = YY
    return velocity


data = pd.read_csv('gsalesman_sig10.csv', header=None)
kSize = 25
data = data.values.reshape((288, 50, 352)).astype(np.float32)
velocity = velocityFieldPhase(data[:, 18, :], data[:, 17, :], kSize)
validVelocity = velocity[velocity[:, :, 0] != -100, :]
invalidVelocity = velocity[velocity[:, :, 0] == -100, :]
plt.imshow(data[:, 18, :], cmap='gray')
plt.show()
plt.imshow(data[:, 17, :], cmap='gray')
plt.quiver(validVelocity[:, 3] + kSize, validVelocity[:, 2] + kSize, validVelocity[:, 1], validVelocity[:, 0],
           angles='xy', scale=1, color='yellow', units='xy')
plt.quiver(invalidVelocity[:, 3] + kSize, invalidVelocity[:, 2] + kSize, invalidVelocity[:, 1]*0, invalidVelocity[:, 0]*0,
           angles='xy', scale=1, color='red', units='xy')
plt.show()
