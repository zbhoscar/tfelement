# -*- coding: UTF-8 -*- #
import cv2
import numpy as np

img = cv2.imread("test.png")
res = cv2.resize(img, (1800, 360))

cv2.namedWindow("Image")
cv2.imshow("Image", res)
cv2.waitKey(1000)
print(res.shape, res.dtype)
cv2.destroyAllWindows()

xs = np.array([5, 8, 9, 8, 5])
weight = xs / np.sum(xs)

final = np.zeros((360, 360, 3))
print(final.shape, final.dtype)

for i in range(5):
    print(i)
    omg = res[0:360, i * 360:(i + 1) * 360]
    cv2.namedWindow("Image_%d" % i)
    cv2.imshow("Image_%d" % i, omg)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()
    final = final + omg * weight[i]

# np.ceil(res*0.5).astype(np.uint8)

final = final.astype(np.uint8)
final = cv2.cvtColor(final, cv2.COLOR_BGR2GRAY)
cv2.namedWindow("Final")
cv2.imshow("Final", final)
cv2.waitKey(1000)

cv2.destroyAllWindows()

print('hello, Kitty!')
print('why')
