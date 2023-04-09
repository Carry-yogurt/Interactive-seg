
import cv2
import numpy as np
# Reads in a binary image
list1 =  ['5ba4facefc949c920d7054813a3e846b000969da2ed860148bdfd18456f59bcc.png',"5c6eb9a47852754d4e45eceb9a696c64c7cfe304afc5ea491cdfef11d55c17f3.png","5c235b945b25b9905b9b0429ce59f1db51d0d0c7d48c2c21ab9f3ca54b0715e6.png","5ef4442e5b8b0b4cf824b61be4050dfd793d846e0a6800afa4425a2f66e91456.png"]
for t in list1 :


    image = cv2.imread(t, 0)

    # Create a 5x5 kernel of ones
    kernel = np.ones((3,3),np.uint8)

    # # Dilate the image
    # dilation = cv2.dilate(image, kernel, iterations = 1)
    # Erode the image
    erosion = cv2.erode(image, kernel, iterations =2)

    # closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

    cv2.imwrite("11.png",erosion)
    # cv2.imwrite("closing.png",closing)


    contours, cnt = cv2.findContours(erosion.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h2, w1 = erosion.shape
    image = np.zeros([h2, w1], dtype=erosion.dtype)
    for i in range(cnt.shape[1]):
        M = cv2.moments(contours[i]) # 计算第一条轮廓的各阶矩,字典形式
        if M["m00"] == 0 :
            M["m00"] = 0.000000001
        center_x = int(M["m10"] / M["m00"])
        center_y = int(M["m01"] / M["m00"])
        # cv2.drawContours(image, contours, 0, 255, -1)#绘制轮廓，填充
        cv2.circle(image, (center_x, center_y), 2, 255, -1)#绘制中心点

    cv2.imwrite("1"+t, image)