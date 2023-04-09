# import argparse
# import os
# import torch.backends.cudnn as cudnn
# import models
# import torchvision.transforms as transforms
# import flow_transforms
# # from scipy.ndimage import imread
#
# from imageio import imread
# from imageio import imsave
# #from scipy.misc import imsave
# from loss import *
# import time
# import random
# from glob import glob
import superpixmodels
import torch
import torchvision.transforms as transforms
from  superpixmodels import flow_transforms
from superpixmodels.train_util import shift9pos
from imageio import imread
from imageio import imsave
import numpy as np
import cv2
from  uint.network.othernetwork.utils.utils import saveTensorToImage
from superpixmodels.Spixel_single_layer import SpixelNet1l_bn

# network_data = torch.load("./pretrained_models/SpixelNet_bsd_ckpt.tar", map_location=torch.device('cpu'))
# model = superpixmodels.__dict__[network_data['arch']](data=network_data)  #

network_data = torch.load("./pretrained_models/SpixelNet_bsd_ckpt.tar", map_location=torch.device('cpu'))
model = SpixelNet1l_bn(network_data)
print(model)
model.eval()

img_path = "00061.jpg"
downsize = 8

input_transform = transforms.Compose([
    flow_transforms.ArrayToTensor(),
    transforms.Normalize(mean=[0, 0, 0], std=[255, 255, 255]),
    transforms.Normalize(mean=[0.411, 0.432, 0.45], std=[1, 1, 1])
])
# may get 4 channel (alpha channel) for some format
img_ = imread(img_path)[:, :, :3]
H, W, _ = img_.shape
H_, W_ = int(np.ceil(H / 16.) * 16), int(np.ceil(W / 16.) * 16)

# get spixel id
n_spixl_h = int(np.floor(H_ / downsize))
n_spixl_w = int(np.floor(W_ / downsize))

spix_values = np.int32(np.arange(0, n_spixl_w * n_spixl_h).reshape((n_spixl_h, n_spixl_w)))
spix_idx_tensor_ = shift9pos(spix_values)

spix_idx_tensor = np.repeat(
    np.repeat(spix_idx_tensor_, downsize, axis=1), downsize, axis=2)

spixeIds = torch.from_numpy(np.tile(spix_idx_tensor, (1, 1, 1, 1))).type(torch.float)  #

n_spixel = int(n_spixl_h * n_spixl_w)

img = cv2.resize(img_, (W_, H_), interpolation=cv2.INTER_CUBIC)
img1 = input_transform(img)
ori_img = input_transform(img_)


output = model(img1.unsqueeze(0))

saveTensorToImage("abc1.jpg",output[0,0])
saveTensorToImage("abc2.jpg",output[0,1])
saveTensorToImage("abc3.jpg",output[0,2])
saveTensorToImage("abc4.jpg",output[0,3])
saveTensorToImage("abc5.jpg",output[0,4])
saveTensorToImage("abc6.jpg",output[0,5])
saveTensorToImage("abc7.jpg",output[0,6])
saveTensorToImage("abc8.jpg",output[0,7])
saveTensorToImage("ab8.jpg",output[0,8])

print(output)

import cv2
import numpy as np

img = cv2.imread(img_path, 0)

gray_lap = cv2.Laplacian(img, cv2.CV_16S, ksize=1)
dst = cv2.convertScaleAbs(gray_lap)

cv2.imshow('laplacian', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()

# #
# # assign the spixel map
# curr_spixl_map = update_spixl_map(spixeIds, output)
# ori_sz_spixel_map = F.interpolate(curr_spixl_map.type(torch.float), size=(H_, W_), mode='nearest').type(torch.int)
#
# mean_values = torch.tensor([0.411, 0.432, 0.45], dtype=img1.unsqueeze(0).dtype).view(3, 1, 1)
# spixel_viz, spixel_label_map = get_spixel_image((ori_img + mean_values).clamp(0, 1), ori_sz_spixel_map.squeeze(),
#                                                 n_spixels=n_spixel, b_enforce_connect=True)
#
# # ************************ Save all result********************************************
# # save img, uncomment it if needed
# # if not os.path.isdir(os.path.join(save_path, 'img')):
# #     os.makedirs(os.path.join(save_path, 'img'))
# # spixl_save_name = os.path.join(save_path, 'img', imgId + '.jpg')
# # img_save = (ori_img + mean_values).clamp(0, 1)
# # imsave(spixl_save_name, img_save.detach().cpu().numpy().transpose(1, 2, 0))
#
#
# # save spixel viz
# if not os.path.isdir(os.path.join(save_path, 'spixel_viz')):
#     os.makedirs(os.path.join(save_path, 'spixel_viz'))
# spixl_save_name = os.path.join(save_path, 'spixel_viz', imgId + '_sPixel.png')
# imsave(spixl_save_name, spixel_viz.transpose(1, 2, 0))
#
# save the unique maps as csv, uncomment it if needed
# if not os.path.isdir(os.path.join(save_path, 'map_csv')):
#     os.makedirs(os.path.join(save_path, 'map_csv'))
# output_path = os.path.join(save_path, 'map_csv', imgId + '.csv')
#   # plus 1 to make it consistent with the toolkit format
# np.savetxt(output_path, (spixel_label_map + 1).astype(int), fmt='%i',delimiter=",")
