import numpy as np
import cv2


def complexMultipleConjugate(complex1: np.ndarray, complex2: np.ndarray) -> np.ndarray:
    re1 = complex1[:, :, 0]
    im1 = complex1[:, :, 1]
    re2 = complex2[:, :, 0]
    im2 = -complex2[:, :, 1]
    re = re1 * re2 - im1 * im2
    im = re1 * im2 + re2 * im1
    return np.array([re, im])


def complexModulus(complex1: np.ndarray) -> np.ndarray:
    re = complex1[:, :, 0]
    im = complex1[:, :, 1]
    return np.sqrt(re ** 2 + im ** 2)


def velocityFieldPhase(movedImage: np.ndarray, reference: np.ndarray, kSize: int) -> np.ndarray:
    height, width = movedImage.shape
    X = np.arange(kSize, height - 3 * kSize, 15)
    Y = np.arange(kSize, width - 3 * kSize, 15)
    [YY, XX] = np.meshgrid(Y, X)
    velocity = np.zeros((X.size, Y.size, 4))
    window = cv2.createHanningWindow([2 * kSize + 1, 2 * kSize + 1], cv2.CV_32F)
    for i in range(X.size):
        startX = X[i]
        for j in range(Y.size):
            startY = Y[j]
            kernel = reference[startX:startX + 2 * kSize + 1, startY:startY + 2 * kSize + 1]
            # padding = cv2.copyMakeBorder(kernel, kSize, kSize, kSize, kSize, cv2.BORDER_DEFAULT)
            subImage = movedImage[startX:startX + 2 * kSize + 1, startY:startY + 2 * kSize + 1]
            translation, response = cv2.phaseCorrelate(kernel, subImage, window)
            y, x = translation
            if response >= 0 and (abs(y) <= 2*kSize and abs(x) <= 2*kSize):
                velocity[i, j, 0] = x
                velocity[i, j, 1] = y
            else:
                velocity[i, j, 0] = -100
                velocity[i, j, 1] = -100
    velocity[..., 2] = XX
    velocity[..., 3] = YY
    return velocity


def velocityFieldPhaseV_2(movedImage: np.ndarray, reference: np.ndarray, kSize: int) -> np.ndarray:
    height, width = movedImage.shape
    X = np.arange(kSize, height - 3 * kSize, 15)
    Y = np.arange(kSize, width - 3 * kSize, 15)
    [YY, XX] = np.meshgrid(Y, X)
    velocity = np.zeros((X.size, Y.size, 4))
    window = cv2.createHanningWindow([2 * kSize + 1, 2 * kSize + 1], cv2.CV_32F)
    for i in range(X.size):
        startX = X[i]
        for j in range(Y.size):
            startY = Y[j]
            kernel = reference[startX:startX + 2 * kSize + 1, startY:startY + 2 * kSize + 1]
            # padding = cv2.copyMakeBorder(kernel, kSize, kSize, kSize, kSize, cv2.BORDER_DEFAULT)
            bias = [[-10, -10], [-10, 0], [-10, 10], [0, -10], [0, 10], [10, -10], [10, 0], [10, 10], [0, 0]]
            max_response = 0
            for k in range(9):
                subImage = movedImage[startX+bias[k][0]:startX + bias[k][0] + 2 * kSize + 1, startY + bias[k][1]:startY + bias[k][1] + 2 * kSize + 1]
                translation, response = cv2.phaseCorrelate(kernel, subImage, window)
                y, x = translation
                if response >= max_response and (abs(y) <= 2*kSize and abs(x) <= 2*kSize):
                    velocity[i, j, 0] = x
                    velocity[i, j, 1] = y
                    max_response = response
    velocity[..., 2] = XX
    velocity[..., 3] = YY
    return velocity


def velocityFieldPhaseV_3(movedImage: np.ndarray, reference: np.ndarray, kSize: int) -> np.ndarray:
    height, width = movedImage.shape
    optimalHeight = cv2.getOptimalDFTSize(height)
    optimalWidth = cv2.getOptimalDFTSize(width)
    paddedMovedImage = cv2.copyMakeBorder(movedImage, 0, optimalHeight - height, 0, optimalWidth - width,
                                          cv2.BORDER_DEFAULT)
    paddedReferenceImage = cv2.copyMakeBorder(reference, 0, optimalHeight - height, 0, optimalWidth - width,
                                              cv2.BORDER_DEFAULT)
    windows = cv2.createHanningWindow(paddedReferenceImage.shape, cv2.CV_32F)
    movedFFT = np.fft.fft2(paddedMovedImage)
    referenceFFT = np.fft.fft2(paddedReferenceImage)
    X = np.arange(kSize, height - 3 * kSize, 20)
    Y = np.arange(kSize, width - 3 * kSize, 20)
    [YY, XX] = np.meshgrid(Y, X)
    velocity = np.zeros((X.size, Y.size, 4))
    for i in range(0, X.size):
        startX = X[i]
        for j in range(0, Y.size):
            startY = Y[j]
            Ga = referenceFFT[startX:startX + 2 * kSize + 1, startY:startY + 2 * kSize + 1]
            bias = [[-10, -10], [-10, 0], [-10, 10], [0, -10], [0, 10], [10, -10], [10, 0], [10, 10], [0, 0]]
            max_response = 0
            for k in range(9):
                Gb = movedFFT[startX + bias[k][0]:startX + bias[k][0] + 2 * kSize + 1, startY + bias[k][1]:startY + bias[k][1] + 2 * kSize + 1]
                GaGb_ = Ga * np.conjugate(Gb)
                GaGb_abs = np.absolute(GaGb_)
                R = GaGb_ / GaGb_abs
                r = np.fft.ifft2(R)
                res = np.fft.fftshift(r)
                res = np.real(res)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                y, x = max_loc
                if max_val >= max_response:
                    velocity[i, j, 0] = x - kSize
                    velocity[i, j, 1] = y - kSize
                    max_response = max_val
    velocity[..., 2] = XX
    velocity[..., 3] = YY
    return velocity


def velocityFieldBM4D(movedImage: np.ndarray, reference: np.ndarray, kSize: int) -> np.ndarray:
    height, width = movedImage.shape
    X = np.arange(kSize, height - 3 * kSize, 10)
    Y = np.arange(kSize, width - 3 * kSize, 10)
    [YY, XX] = np.meshgrid(Y, X)
    velocity = np.zeros((X.size, Y.size, 4))
    for i in range(X.size):
        startX = X[i]
        for j in range(Y.size):
            startY = Y[j]
            kernel = reference[startX:startX + 2 * kSize + 1, startY:startY + 2 * kSize + 1]
            subImage = movedImage[startX - kSize:startX + 3 * kSize + 1, startY - kSize:startY + 3 * kSize + 1]
            v, d = blockMatching(kernel, subImage, kSize)
            if d:
                y, x = v
                velocity[i, j, 0] = x
                velocity[i, j, 1] = y
    velocity[..., 2] = XX
    velocity[..., 3] = YY
    return velocity


def blockMatching(kernel: np.ndarray, searchField: np.ndarray, kSize: int) -> tuple:
    height, width = searchField.shape
    minDistance = 1e4
    for i in range(0, height - 2 * kSize):
        for j in range(0, width - 2 * kSize):
            distance = np.linalg.norm(kernel - searchField[i:i + 2 * kSize + 1, j:j + 2 * kSize + 1])
            if distance < minDistance:
                minDistance = distance
                v = [i - kSize, j - kSize]
    return v, minDistance


def interpolate(velocityField: np.ndarray) -> np.ndarray:
    height, width = velocityField.shape[0], velocityField.shape[1]
    for i in range(height):
        for j in range(width):
            if velocityField[i, j, 0] == -100 and velocityField[i, j, 1] == -100:
                # upper left corner
                if i == 0 and j == 0:
                    velocityField[i, j, 0:2] = 1 / 2 * (velocityField[i + 1, j, 0:2] + velocityField[i, j + 1, 0:2])
                # upper right corner
                elif i == 0 and j == width - 1:
                    velocityField[i, j, 0:2] = 1 / 2 * (velocityField[i + 1, j, 0:2] + velocityField[i, j - 1, 0:2])
                # lower right corner
                elif i == height - 1 and j == width - 1:
                    velocityField[i, j, 0:2] = 1 / 2 * (velocityField[i - 1, j, 0:2] + velocityField[i, j - 1, 0:2])
                # lower left corner
                elif i == height - 1 and j == 0:
                    velocityField[i, j, 0:2] = 1 / 2 * (velocityField[i - 1, j, 0:2] + velocityField[i, j + 1, 0:2])
                # the top row
                elif i == 0 and j != 0 and j != width - 1:
                    velocityField[i, j, 0:2] = 1 / 3 * (velocityField[i + 1, j, 0:2] + velocityField[i, j - 1, 0:2] + velocityField[i, j + 1, 0:2])
                # the bottom row
                elif i == height - 1 and j != 0 and j != width - 1:
                    velocityField[i, j, 0:2] = 1 / 3 * (velocityField[i - 1, j, 0:2] + velocityField[i, j - 1, 0:2] + velocityField[i, j + 1, 0:2])
                # the first column
                elif j == 0 and i != 0 and i != height - 1:
                    velocityField[i, j, 0:2] = 1 / 3 * (velocityField[i - 1, j, 0:2] + velocityField[i + 1, j, 0:2] + velocityField[i, j + 1, 0:2])
                # the most right column
                elif j == width - 1 and i != 0 and i != height - 1:
                    velocityField[i, j, 0:2] = 1 / 3 * (velocityField[i - 1, j, 0:2] + velocityField[i + 1, j, 0:2] + velocityField[i, j - 1, 0:2])
                # common cases
                else:
                    velocityField[i, j, 0:2] = 1 / 4 * (velocityField[i - 1, j, 0:2] + velocityField[i + 1, j, 0:2] + velocityField[i, j + 1, 0:2] + velocityField[i, j - 1,  0:2])
    return velocityField


def velocityFieldCorrelate(movedImage: np.ndarray, reference: np.ndarray, kSize: int) -> np.ndarray:
    height, width = movedImage.shape
    X = np.arange(kSize, height - 3 * kSize, 15)
    Y = np.arange(kSize, width - 3 * kSize, 15)
    [YY, XX] = np.meshgrid(Y, X)
    velocityField = np.zeros((X.size, Y.size, 4))
    for i in range(X.size):
        startX = X[i]
        for j in range(Y.size):
            startY = Y[j]
            kernel = reference[startX:startX + 2 * kSize + 1, startY:startY + 2 * kSize + 1]
            subImage = movedImage[startX-kSize:startX + 3 * kSize + 1, startY-kSize:startY + 3 * kSize + 1]
            res = cv2.matchTemplate(subImage, kernel, cv2.TM_CCORR_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            y, x = max_loc
            if max_val >= 0.2:
                velocityField[i, j, 0] = x - kSize
                velocityField[i, j, 1] = y - kSize
            else:
                velocityField[i, j, 0] = -100
                velocityField[i, j, 1] = -100
    velocityField[..., 2] = XX
    velocityField[..., 3] = YY
    return velocityField
