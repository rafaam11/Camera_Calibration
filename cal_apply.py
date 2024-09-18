import numpy as np
import cv2

k1 = -0.353176
k2 = 0.155629
k3 = 0.155629
p1 = 0.000790
p2 = -0.000669
fx = 1033.583486
fy = 1032.753476
cx = 984.598269
cy = 427.667021


dist = np.array([k1, k2, p1, p2, k3])
mtx = np.array([[fx, 0, cx],
      [0, fy, cy],
      [0, 0, 1]])

image = cv2.imread('data/20230223_170757.jpg')

h, w = image.shape[:2]

newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx,
                                                 dist,
                                                 (w, h),
                                                 1, (w, h))

# undistort
dst = cv2.undistort(image, mtx, dist, None, newcameramtx)

x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv2.imshow('result', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()