import cv2
import numpy as np

solution = cv2.imread('data/onlysolution.png',0)
tile = cv2.imread('data/onlytile.png',0)

ret, thresh = cv2.threshold(solution, 180, 255,0)
ret, thresh2 = cv2.threshold(tile, 150, 255,0)
tile_white = cv2.bitwise_not(thresh2)


cv2.imshow("thresh", thresh)
cv2.imshow("thresh2", tile_white)

contours,hierarchy = cv2.findContours(thresh,2,1)
cnt1 = contours[0]
contours,hierarchy = cv2.findContours(tile_white,2,1)
cnt2 = contours[0]

ret = cv2.matchShapes(cnt1,cnt2,1,0.0)
print(ret)
cv2.waitKey(0)