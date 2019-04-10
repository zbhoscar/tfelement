import numpy as np
import matplotlib.pyplot as plt
import pylab
import imageio
import skimage.io
import numpy as np
import cv2

cap = cv2.VideoCapture('v_TrampolineJumping_g02_c02.avi')
step=0
while(cap.isOpened()):
    ret, frame = cap.read()
    # cv2.imshow('image', frame)
    # k = cv2.waitKey(20)
    # if (k & 0xff == ord('q')):
    #     break
    step += 1
    print(ret, step, cap.isOpened())

cap.release()
cv2.destroyAllWindows()
