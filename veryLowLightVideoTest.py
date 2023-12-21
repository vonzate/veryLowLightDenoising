import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import blockMatching as bm
import canon_utils as cu

from queue import Queue
import rawpy as rp
import scipy.io as so

# frame_num = 3
#
# path = '/mnt/data/submillilux_videos/submillilux_videos_dng/seq17/'
# q = Queue(maxsize=frame_num)
# for i in range(frame_num):
#     filename = path + '0000010' + str(i) + '.dng'
#     image = cu.read_16bit_raw(filename)
#     q.put(image)
fpn = pd.read_csv('/mnt/data/fpn.csv', header=None)
fpn = fpn.values.reshape(640, 4, 1080).astype(np.int64)
fpn = cu.raw_4_to_1(fpn)
raw1 = cu.read_16bit_raw('/mnt/data/submillilux_videos/submillilux_videos_dng/seq17/00000111.dng')
# raw1 -= fpn
# raw1[raw1 <= 0] = 0
plt.imshow(raw1, cmap='gray')
plt.show()
raw2 = cu.read_16bit_raw('/mnt/data/submillilux_videos/submillilux_videos_dng/seq17/00000110.dng')
# raw2 -= fpn
# raw2[raw2 <= 0] = 0
height, width = raw1.shape
kSize = 50
raw1 = raw1.astype(np.float32)
raw2 = raw2.astype(np.float32)
velocity = bm.velocityFieldPhaseV_3(raw1, raw2, kSize).astype(np.float32)
velocity = bm.interpolate(velocity)
velocity[:, :, 0] = cv2.medianBlur(velocity[:, :, 0], ksize=3)
velocity[:, :, 1] = cv2.medianBlur(velocity[:, :, 1], ksize=3)
validVelocity = velocity[velocity[:, :, 0] != -100, :]
invalidVelocity = velocity[velocity[:, :, 0] == -100, :]
# plt.imshow(raw1, cmap='gray')
# X = np.arange(2*kSize, height-2*kSize, 15)
# Y = np.arange(2*kSize, width-2*kSize, 15)
# [XX, YY] = np.meshgrid(X, Y)
# plt.scatter(YY, XX, s=0.5)
# plt.savefig('moved.png', dpi=1000)
# plt.show()
plt.imshow(raw2, cmap='gray')
plt.quiver(validVelocity[:, 3] + kSize, validVelocity[:, 2] + kSize, validVelocity[:, 1], validVelocity[:, 0], angles='xy', scale=1, color='yellow', units='xy')
plt.savefig('vector.png', dpi=1000)
# plt.quiver(invalidVelocity[:, 3] + kSize, invalidVelocity[:, 2] + kSize, invalidVelocity[:, 1] * 0, invalidVelocity[:, 0] * 0, angles='xy', scale=1, color='red', units='xy')
plt.show()
