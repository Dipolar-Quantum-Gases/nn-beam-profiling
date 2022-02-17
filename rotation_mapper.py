# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 23:17:10 2022

@author: hofer
"""

from detectron2.data import DatasetMapper
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T

import numpy as np
import copy
import torch

class MyDatasetMapper(DatasetMapper):

    def __init__(self, cfg, is_train, augmentations=None):
      super().__init__(cfg, is_train)
      if augmentations is not None:
        self.augmentations = augmentations
    

    def _transform_annotations(self, dataset_dict, transforms, image_shape):
      annos = [
          self.transform_instance_annotations(obj, transforms, image_shape)  ############################################
          for obj in dataset_dict.pop("annotations")
          if obj.get("iscrowd", 0) == 0]
      instances = utils.annotations_to_instances_rotated(annos, image_shape)
      dataset_dict["instances"] = utils.filter_empty_instances(instances)


    def transform_instance_annotations(self, annotation, transforms, image_size, *, keypoint_hflip_indices=None):
      if isinstance(transforms, (tuple, list)):
            transforms = T.TransformList(transforms)
      bbox = np.asarray([annotation['bbox']])
      annotation["bbox"] = transforms.apply_rotated_box(bbox)[0]
      return annotation


    def __call__(self, dataset_dict):
      dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
      image = image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
      utils.check_image_size(dataset_dict, image)

      if "sem_seg_file_name" in dataset_dict:
          sem_seg_gt = utils.read_image(dataset_dict.pop("sem_seg_file_name"), "L").squeeze(2)
      else:
          sem_seg_gt = None
  
      aug_input = T.AugInput(image, sem_seg=sem_seg_gt)
      transforms = self.augmentations(aug_input)
      image, sem_seg_gt = aug_input.image, aug_input.sem_seg

      dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
      image_shape = image.shape[:2]
      self._transform_annotations(dataset_dict, transforms, image_shape)
      return dataset_dict