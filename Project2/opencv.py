#_AUTHOR:CX

# import re
# import Project1_MNIST.handwriting.cnn as CNN
import numpy as np
import cv2


""" RCNN的目标识别 """

a = cv2.imread("car.jpeg")

print(a.shape)

print(a[0:10,0:10,2])

blue = cv2.split(a)[0]
#merged = cv2.merge([b,g,r])


cv2.namedWindow("img", cv2.WINDOW_NORMAL)
cv2.imshow("1",a)
#cv2.waitKey()
#cv2.destroyAllWindows()

# 使用OpenCV抓取摄像头图像提取蓝色
cap = cv2.VideoCapture(0)
for i in range(0, 19):
    print(cap.get(i))
while (1):
    ret, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([100, 47, 47])
    upper_blue = np.array([124, 255, 255])

    mask = cv2.inRange(hsv, lower_blue, upper_blue)  # 蓝色掩模

    res = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.imshow(u"Capture", frame)
    cv2.imshow(u"mask", mask)
    cv2.imshow(u"res", res)

    key = cv2.waitKey()
    if key & 0xff == ord('q') or key == 27:
        print(frame.shape, ret)
    break
cap.release()
cv2.waitKey()
cv2.destroyAllWindows()
