import cv2
image_path = "./uint/data/GrabCut/data_GT/37073.jpg"
# img = cv2.imread(image_path)
#
# gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#
# sobel_x = cv2.Sobel(gray, cv2.CV_8U, 1, 0)
# sobel_y = cv2.Sobel(gray, cv2.CV_8U, 0, 1)
# sobel = cv2.Sobel(gray, cv2.CV_8U, 1, 1)
#
# scharr_x = cv2.Scharr(gray, cv2.CV_8U, 1, 0)
# scharr_y = cv2.Scharr(gray, cv2.CV_8U, 0, 1)
#
# cv2.imshow("src", img)
# cv2.imshow("Sobel_x", sobel_x)
# cv2.imshow("Sobel_y", sobel_y)
# cv2.imshow("Sobel", sobel)
# cv2.imshow("Scharr_x", scharr_x)
# cv2.imshow("Scharr_y", scharr_y)
#
# cv2.waitKey(0)


#
# import cv2
#
# img = cv2.imread(image_path)
#
# gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#
# laplace = cv2.Laplacian(gray, cv2.CV_8U, ksize=3)
# sobel = cv2.Sobel(gray, cv2.CV_8U, 1, 1)
#
# cv2.imshow("Laplace", laplace)
# cv2.imshow("Sobel", sobel)
#
# cv2.waitKey(0)
#
#
# import cv2
#
# img = cv2.imread(image_path)
#
# gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#
# gauss = cv2.GaussianBlur(gray, (3, 3), 1)
#
# def doCanny(x):
#     position = cv2.getTrackbarPos("CannyBar", "Canny")
#     canny = cv2.Canny(gauss, position, position*2.5)
#     cv2.imshow("Canny", canny)
#
#
# cv2.namedWindow("Canny")
#
# cv2.createTrackbar("CannyBar", "Canny", 1, 100, doCanny)
#
# cv2.waitKey(0)

import timm
import torch

x= torch.randn(3, 3, 320, 480)
model_names = timm.list_models("*efficientnet_b0*",pretrained=True)
print(model_names)
# hrnet_w18  hrnet_w32
model = timm.create_model('gluon_resnet34_v1b', pretrained=True,features_only=True)

result = model(x)


print(result)