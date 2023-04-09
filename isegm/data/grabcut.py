from pathlib import Path

import cv2
import numpy as np

from .base import ISDataset, get_unique_labels

class GrabCutDataset(ISDataset):
    def __init__(self, dataset_path,
                 images_dir_name='data_GT', masks_dir_name='boundary_GT',
                 **kwargs):
        super(GrabCutDataset, self).__init__(**kwargs)
        self.dataset_path = Path(dataset_path)
        self._images_path = self.dataset_path / images_dir_name
        self._insts_path = self.dataset_path / masks_dir_name

        self.dataset_samples = [x.name for x in sorted(self._images_path.glob('*.*'))]
        self._masks_paths = {x.stem: x for x in self._insts_path.glob('*.*')}

    def get_sample(self, index):
        image_name = self.dataset_samples[index]
        image_path = str(self._images_path / image_name)
        mask_path = str(self._masks_paths[image_name.split('.')[0]])

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        instances_mask = cv2.imread(mask_path)[:, :, 0].astype(np.int32)
        instances_mask[instances_mask == 128] = -1
        instances_mask[instances_mask > 128] = 1

        ## 需要删除
        # instances_mask[instances_mask == 1] = 255
        # instances_mask[instances_mask == 0] = 1

        instances_ids = [1]

        instances_info = {
            x: {'ignore': False}
            for x in instances_ids
        }
        # cv2.imwrite("img.jpg",image)
        # cv2.imwrite("instances_mask.jpg", instances_mask)
        # print("instances_info: {}  id :{}  ".format(instances_info,id))
        return {
            'image': image,
            'instances_mask': instances_mask,
            'instances_info': instances_info,
            'image_id': index
        }




##  用来训练的数据加载器
class GrabCutDataset_usedtoTrain(ISDataset):
    def __init__(self, dataset_path,
                 images_dir_name='data_GT', masks_dir_name='boundary_GT',
                 **kwargs):
        super(GrabCutDataset, self).__init__(**kwargs)
        self.dataset_path = Path(dataset_path)
        self._images_path = self.dataset_path / images_dir_name
        self._insts_path = self.dataset_path / masks_dir_name

        self.dataset_samples = [x.name for x in sorted(self._images_path.glob('*.*'))]
        self._masks_paths = {x.stem: x for x in self._insts_path.glob('*.*')}

    def get_sample(self, index):
        image_name = self.dataset_samples[index]
        image_path = str(self._images_path / image_name)
        mask_path = str(self._masks_paths[image_name.split('.')[0]])

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        instances_mask = cv2.imread(mask_path)[:, :, 0].astype(np.int32)
        instances_mask[instances_mask == 128] = -1
        instances_mask[instances_mask > 128] = 1

        ## 需要删除
        # instances_mask[instances_mask == 1] = 255
        # instances_mask[instances_mask == 0] = 1

        instances_ids = [1]

        instances_info = {
            x: {'ignore': False}
            for x in instances_ids
        }
        # cv2.imwrite("img.jpg",image)
        # cv2.imwrite("instances_mask.jpg", instances_mask)
        # print("instances_info: {}  id :{}  ".format(instances_info,id))
        return {
            'image': image,
            'instances_mask': instances_mask,
            'instances_info': instances_info,
            'image_id': index
        }



# 用来训练的数据加载器
class GrabCutDataset_usedToTrain(ISDataset):
    def __init__(self, dataset_path,
                 images_dir_name='data_GT', masks_dir_name='boundary_GT',
                 **kwargs):
        super(GrabCutDataset_usedToTrain, self).__init__(**kwargs)
        self.dataset_path = Path(dataset_path)
        self._images_path = self.dataset_path / images_dir_name
        self._insts_path = self.dataset_path / masks_dir_name

        self.dataset_samples = [x.name for x in sorted(self._images_path.glob('*.*'))]
        self._masks_paths = {x.stem: x for x in self._insts_path.glob('*.*')}

    def get_sample(self, index):
        image_name = self.dataset_samples[index]
        image_path = str(self._images_path / image_name)
        mask_path = str(self._masks_paths[image_name.split('.')[0]])

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        instances_mask = cv2.imread(mask_path)[:, :, 0].astype(np.int32)
        instances_mask[instances_mask <= 128] = 0
        instances_mask[instances_mask > 128] = 1
        instances_ids = get_unique_labels(instances_mask, exclude_zero=True)
        instances_info = {
            x: {'ignore': False}
            for x in instances_ids
        }
        return {
            'image': image,
            'instances_mask': instances_mask,
            'instances_info': instances_info,
            'image_id': index
        }
