



from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt



def dealcrack500Mask(dataset_path,images_dir_name='img', masks_dir_name='masksrc'):
    dataset_path = Path(dataset_path)
    dataset_samples = [x.name for x in sorted(dataset_path.glob('*.*'))]
    data = []
    for mask_path in dataset_samples:
        pathname = mask_path.split(".")[0]
        data.append("JPEGImages/val_data/" +str(mask_path)+" Annotations/val_crop/"+pathname+".png")

    # shuffle(datasets)
    np.savetxt('val1.txt',np.array(data),fmt='%s')
dealcrack500Mask("./uint/data/CRACK500/JPEGImages/val_data")

def dealgaps384Mask(dataset_path,images_dir_name='img', masks_dir_name='masksrc'):
    dataset_path = Path(dataset_path)
    _images_path = dataset_path / images_dir_name
    _insts_path = dataset_path / masks_dir_name
    dataset_samples = [x.name for x in sorted(_images_path.glob('*.*'))]
    _masks_paths = {x.stem: x for x in _insts_path.glob('*.*')}
    datasets = []
    for mask_path in _masks_paths:
        ii = mask_path
        mask_path = str(_masks_paths[mask_path])
        mask_data = cv2.imread(mask_path)
        fore = mask_data[:,:,1]
        ret,markers=cv2.connectedComponents(fore)
        ## 讲改好的图片写到另一个文件中
        cv2.imwrite("uint\\data\\gaps384new\\maskdeal\\"+ii+".png",markers)
        datasets.append(ii)
    # shuffle(datasets)
    np.savetxt('train.txt',np.array(datasets),fmt='%s')



dealgaps384Mask("uint/data/gaps384new")
### 处理掩码 

def dealMask(dataset_path,images_dir_name='data_GT', masks_dir_name='boundary_GT'):
    dataset_path = Path(dataset_path)
    _images_path = dataset_path / images_dir_name
    _insts_path = dataset_path / masks_dir_name
    dataset_samples = [x.name for x in sorted(_images_path.glob('*.*'))]
    _masks_paths = {x.stem: x for x in _insts_path.glob('*.*')}
    for mask_path in _masks_paths:
        ii = mask_path
        mask_path = str(_masks_paths[mask_path])
        mask_data = cv2.imread(mask_path)
        fore = mask_data[:,:,1]
        ret,markers=cv2.connectedComponents(fore)
        ## 讲改好的图片写到另一个文件中
        cv2.imwrite("datasets\\mycdata\\boundary_GT\\"+ii+".png",markers)
        # plt.subplot(131)
        # plt.imshow(mask_data)
        # plt.axis('off')
        # plt.subplot(132)
        # plt.imshow(mask_data)
        # plt.axis('off')
        # plt.subplot(133)
        # plt.imshow(markers)
        # plt.axis('off')
        # plt.show()


    print()
# dealMask("./datasets/mycdata")


# 分割数据集


import shutil

def splitDataSetToTrainandtest(dataset_path,images_dir_name='data_GT', masks_dir_name='boundary_GT'):
    dataset_path = Path(dataset_path)
    _images_path = dataset_path / images_dir_name
    _insts_path = dataset_path / masks_dir_name
    dataset_samples = [x.name for x in sorted(_images_path.glob('*.*'))]
    _masks_paths = {x.stem: x for x in _insts_path.glob('*.*')}

    train_path = dataset_path/ "train"
    test_path = dataset_path/ "test"

    # train_path.mkdir()
    # test_path.mkdir()

    dataset_sieze = len(dataset_samples)

    train_size = int(dataset_sieze *0.8)

    train_samples = dataset_samples[:train_size]
    test_samples = dataset_samples[train_size:]

    # for train_sample in train_samples :
    #     # 拷贝原图
    #     shutil.copyfile(str(_images_path/train_sample),str(train_path/images_dir_name/train_sample))
    #     train_name = train_sample.split(".")[0]
    #     # 拷贝淹摸
    #     shutil.copyfile(str(_insts_path /train_name)+".png", str(train_path / masks_dir_name/train_name)+".png")


    for test_sample in test_samples :
        # 拷贝原图
        shutil.copyfile(str(_images_path/test_sample),str(test_path/images_dir_name/test_sample))
        test_name = test_sample.split(".")[0]
        # 拷贝淹摸
        shutil.copyfile(str(_insts_path /test_name)+".png", str(test_path / masks_dir_name/test_name)+".png")



# splitDataSetToTrainandtest("./datasets/mycdata")



#处理SBD 数据集
## 目的是生成两个文件  一个train 一个val
## 文件中包含  原图路径，lable路径，掩码目标标签 1--。
from scipy.io import loadmat
import os
from random import shuffle

#
# if __name__ == '__main__':
#
#     dataset_path = "uint/data/sbd"
#     images_dir_name = "img"
#     masks_dir_name = "inst"
#     ##遍历mask文件夹
#     dataset_path = Path(dataset_path)
#     _images_path = dataset_path / images_dir_name
#     _insts_path = dataset_path / masks_dir_name
#     dataset_samples = [x.name for x in sorted(_images_path.glob('*.*'))]
#     _masks_paths = {x.stem: x for x in sorted(_insts_path.glob('*.*'))}
#
#     index = 0
#     sbd_datasets = []
#
#     for mask_path in _masks_paths:
#         mask_path_name = _masks_paths[mask_path].name
#         mask_path = str(_masks_paths[mask_path])
#         mask_data = instances_mask = loadmat(str(mask_path))['GTinst'][0][0][0].astype(np.int32)
#         mask_data = np.array(mask_data)
#         max_label = np.max(mask_data)
#         min_label = 0
#         ## 遍历mask中的所有标签 只保留下 像素个数大于10 的标签
#         for mask_label in range(min_label+1 , max_label+1) :
#             current_label_count = np.count_nonzero(mask_data == mask_label)
#             if current_label_count <=10 :
#                 continue
#             else :
#                 sbd_datasets.append(dataset_samples[index]+ " "+  mask_path_name + " " + str(mask_label))
#         index = index +1
#     shuffle(sbd_datasets)
#     train_size = int(len(sbd_datasets) *0.8)
#
#     train_ = sbd_datasets[0:train_size]
#     val_ = sbd_datasets[train_size:]
#
#     np.savetxt('train.txt',np.array(train_),fmt='%s')
#     np.savetxt('val.txt', np.array(val_),fmt='%s')

    # ii = mask_path
    # mask_path = str(_masks_paths[mask_path])
    # mask_data = cv2.imread(mask_path)
    # fore = mask_data[:, :, 1]
    # ret, markers = cv2.connectedComponents(fore)
    # ## 讲改好的图片写到另一个文件中
    # cv2.imwrite("datasets\\mycdata\\boundary_GT\\" + ii + ".png", markers)
    # plt.subplot(131)
    # plt.imshow(mask_data)
    # plt.axis('off')
    # plt.subplot(132)
    # plt.imshow(mask_data)
    # plt.axis('off')
    # plt.subplot(133)
    # plt.imshow(markers)
    # plt.axis('off')
    # plt.show()









def dealgaps384Mask(dataset_path,images_dir_name='img', masks_dir_name='masksrc'):
    dataset_path = Path(dataset_path)
    _images_path = dataset_path / images_dir_name
    _insts_path = dataset_path / masks_dir_name
    dataset_samples = [x.name for x in sorted(_images_path.glob('*.*'))]
    _masks_paths = {x.stem: x for x in _insts_path.glob('*.*')}
    for mask_path in _masks_paths:
        ii = mask_path
        mask_path = str(_masks_paths[mask_path])
        mask_data = cv2.imread(mask_path)
        fore = mask_data[:,:,1]
        ret,markers=cv2.connectedComponents(fore)
        ## 讲改好的图片写到另一个文件中
        cv2.imwrite("uint\\data\\gaps384eval\\maskdeal\\"+ii+".png",markers)
        # plt.subplot(131)
        # plt.imshow(mask_data)
        # plt.axis('off')
        # plt.subplot(132)
        # plt.imshow(mask_data)
        # plt.axis('off')
        # plt.subplot(133)
        # plt.imshow(markers)
        # plt.axis('off')
        # plt.show()
#
dealgaps384Mask("uint/data/gaps384eval")

def  get384trainfile():
    dataset_path = "uint/data/gaps384eval"
    images_dir_name = "img"
    masks_dir_name = "maskdeal"
    ##遍历mask文件夹
    dataset_path = Path(dataset_path)
    _images_path = dataset_path / images_dir_name
    _insts_path = dataset_path / masks_dir_name
    dataset_samples = [x.name for x in sorted(_images_path.glob('*.*'))]
    _masks_paths = {x.stem: x for x in sorted(_insts_path.glob('*.*'))}

    index = 0
    sbd_datasets = []

    for mask_path in _masks_paths:
        mask_path_name = _masks_paths[mask_path].name
        mask_path = str(_masks_paths[mask_path])
        mask_data = cv2.imread(mask_path)[:,:,1]
        mask_data = np.array(mask_data)
        max_label = np.max(mask_data)
        min_label = 0
        ## 遍历mask中的所有标签 只保留下 像素个数大于10 的标签
        for mask_label in range(min_label+1 , max_label+1) :
            current_label_count = np.count_nonzero(mask_data == mask_label)
            if current_label_count <=80 :
                continue
            else :
                sbd_datasets.append(mask_path_name.replace(".png",".jpg")+ " "+  mask_path_name + " " + str(mask_label))
        index = index +1
    shuffle(sbd_datasets)
    train_size = int(len(sbd_datasets) *0.8)

    train_ = sbd_datasets[0:train_size]
    val_ = sbd_datasets[train_size:]

    np.savetxt('train.txt',np.array(train_),fmt='%s')
    np.savetxt('val.txt', np.array(val_),fmt='%s')


get384trainfile()


def generategaps():
    dataset_path = "uint/data/gaps384"
    images_dir_name = "img"
    masks_dir_name = "maskdeal"
    ##遍历mask文件夹
    dataset_path = Path(dataset_path)
    _images_path = dataset_path / images_dir_name
    _insts_path = dataset_path / masks_dir_name
    dataset_samples = [x.name for x in sorted(_images_path.glob('*.*'))]
    _masks_paths = {x.stem: x for x in sorted(_insts_path.glob('*.*'))}

    sbd_datasets = []
    for dataset_sample in dataset_samples:
        ind = dataset_sample.find(".jpg")
        dataset_sample[:ind]
        sbd_datasets.append(dataset_sample[:ind])

    shuffle(sbd_datasets)
    train_size = int(len(sbd_datasets) *0.8)

    train_ = sbd_datasets[0:train_size]
    val_ = sbd_datasets[train_size:]

    np.savetxt('train.txt',np.array(train_),fmt='%s')
    np.savetxt('val.txt', np.array(val_),fmt='%s')

# generategaps()



def generategaps():
    dataset_path = "uint/data/shian"
    images_dir_name = "img"
    masks_dir_name = "masksrc"
    ##遍历mask文件夹
    dataset_path = Path(dataset_path)
    _images_path = dataset_path / images_dir_name
    _insts_path = dataset_path / masks_dir_name
    dataset_samples = [x.name for x in sorted(_images_path.glob('*.*'))]
    _masks_paths = {x.stem: x for x in sorted(_insts_path.glob('*.*'))}

    sbd_datasets = []
    for dataset_sample in _masks_paths:
        ind = dataset_sample.find(".png")
        temp = dataset_sample[:ind]
        sbd_datasets.append(dataset_sample)

    shuffle(sbd_datasets)
    train_size = int(len(sbd_datasets) *0.8)

    train_ = sbd_datasets[0:train_size]
    val_ = sbd_datasets[train_size:]

    np.savetxt('train.txt',np.array(train_),fmt='%s')
    np.savetxt('val.txt', np.array(val_),fmt='%s')


# generategaps()



def generateCAMO():

    dataset_path = "uint/data/CAMO"
    images_dir_name = "Image"
    masks_dir_name = "GT"
    ##遍历mask文件夹
    dataset1 =  "CAMO_TestingDataset"
    dataset2 = "CHAMELEON_TestingDataset"
    dataset3 = "COD10K_CAMO_CombinedTrainingDataset"

    #  处理测试集1
    dataset_path = Path(dataset_path)
    _images_path = dataset_path / dataset1 / images_dir_name
    _insts_path = dataset_path / dataset1 / masks_dir_name

    dataset_samples = [x.name for x in sorted(_images_path.glob('*.*'))]
    _masks_paths = {x.stem: x for x in sorted(_insts_path.glob('*.*'))}

    CAMO_datasets_test = []
    for dataset_sample in dataset_samples:
        ind = dataset_sample.find(".jpg")
        CAMO_datasets_test.append(dataset_sample[:ind]+ " "+dataset1)



      #  处理测试集2
    dataset_path = Path(dataset_path)
    _images_path = dataset_path / dataset2 / images_dir_name
    _insts_path = dataset_path / dataset2 / masks_dir_name

    dataset_samples = [x.name for x in sorted(_images_path.glob('*.*'))]
    _masks_paths = {x.stem: x for x in sorted(_insts_path.glob('*.*'))}

    for dataset_sample in dataset_samples:
        ind = dataset_sample.find(".jpg")

        CAMO_datasets_test.append(dataset_sample[:ind]+" "+dataset2)



    #  处理训练集
    dataset_path = Path(dataset_path)
    _images_path = dataset_path / dataset3 / images_dir_name
    _insts_path = dataset_path / dataset1 / masks_dir_name

    dataset_samples = [x.name for x in sorted(_images_path.glob('*.*'))]
    _masks_paths = {x.stem: x for x in sorted(_insts_path.glob('*.*'))}

    CAMO_datasets_train = []
    for dataset_sample in dataset_samples:
        ind = dataset_sample.find(".jpg")
        CAMO_datasets_train.append(dataset_sample[:ind]+ " "+dataset3)




    np.savetxt('train.txt',np.array(CAMO_datasets_train),fmt='%s')
    np.savetxt('val.txt', np.array(CAMO_datasets_test),fmt='%s')



