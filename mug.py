"""
Mask R-CNN
Train on the toy Mug dataset and implement mug detection effect.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 mug.py train --dataset=/path/to/mug/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 mug.py train --dataset=/path/to/mug/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 mug.py train --dataset=/path/to/mug/dataset --weights=imagenet

    # Apply mug detection to an image
    python3 mug.py detect --weights=/path/to/weights/file.h5 --images=<URL or path to directory that saves the images needed to be detected> --result=<URL or path to result directory>

    # Apply mug detection to an image using the last weights you trained
    python3 mug.py detect --weights=last --images=<URL or path to directory that saves the images needed to be detected> --result=<URL or path to result directory>
"""

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw

# MaskRCNN模型所在目录
ROOT_DIR = os.path.abspath("../Mask/Mask_RCNN")

# 导入MaskRCNN
sys.path.append(ROOT_DIR)
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# coco数据集获得的权重文件路径
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################


class MugConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "mug"

    # 处理图片数量
    IMAGES_PER_GPU = 1

    # 类别数目（包含背景）
    NUM_CLASSES = 1 + 1  # Background + mug

    # 每个epoch训练的步数
    STEPS_PER_EPOCH = 100

    # 当小于90%置信度时跳过检测
    DETECTION_MIN_CONFIDENCE = 0.9


############################################################
#  Dataset
############################################################

class MugDataset(utils.Dataset):

    def load_mug(self, dataset_dir, subset):
        """
        载入mug数据集子集
        dataset_dir: 数据集所在目录
        subset: 需要导入的子集: train or val
        """

        # 添加类别（source, class_id, class_name），这里只有一类，即mug
        self.add_class("mug", 1, "mug")

        # Train or validation set?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # 导入via注释得到的json文件
        annotations = json.load(open(os.path.join(dataset_dir, "via_region_data.json")))
        annotations = list(annotations.values())  # 不需要字典的键

        # 跳过没有注释的图片
        annotations = [a for a in annotations if a['regions']]

        # 添加图片
        for a in annotations:
            # 获取每张图片中马克杯轮廓的坐标信息
            polygons = [r['shape_attributes'] for r in a['regions'].values()]

            # 获取图片的尺寸
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "mug",
                image_id=a['filename'],  # 使用文件名作为唯一的id
                path=image_path,
                width=width, height=height,
                polygons=polygons)

    def load_mask(self, image_id):
        """
        为图像生成实例（mug）蒙版
        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """

        # 如果不是mug数据集图像，则委托给父类。
        image_info = self.image_info[image_id]
        if image_info["source"] != "mug":
            return super(self.__class__, self).load_mask(image_id)

        # 将多边形转换为形状的位图蒙版
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # 获取多边形内像素的索引并将它们设置为1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1

        # 返回蒙版，以及每个实例的类ID的数组。
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """
        返回图像的路径
        """
        info = self.image_info[image_id]
        if info["source"] == "mug":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model):
    """
    训练模型
    """

    # 训练集
    dataset_train = MugDataset()
    dataset_train.load_mug(args.dataset, "train")
    dataset_train.prepare()

    # 验证集
    dataset_val = MugDataset()
    dataset_val.load_mug(args.dataset, "val")
    dataset_val.prepare()

    # 基于训练了的COCO数据集权重做我们的训练
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=30,
                layers='heads')

def detect(model, images_dir = None, result_dir = None):

    import shutil

    # 判断result_dir是否存在，不存在则创建，存在则清空
    if os.path.exists(result_dir):
        print("delete directory {}".format(result_dir))
        shutil.rmtree(result_dir)
    os.mkdir(result_dir)
    
    # 读取目录中所有图片
    file_list = os.listdir(images_dir)

    for image_file in file_list:
        image_file = os.path.join(images_dir, image_file)
        print("Running on {}".format(image_file))
        
        # 读取图片
        image = skimage.io.imread(image_file)
        
        # 目标检测
        r = model.detect([image], verbose=1)[0]
        
        # 画边界框
        image_with_box = draw_box(image, r['rois'])
        
        # 保存结果
        file_name = "mug_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        file_name = os.path.join(result_dir, file_name)
        skimage.io.imsave(file_name, image_with_box)
        print("检测完成，结果保存在%s" % file_name)

def draw_box(image, boxes):
    import cv2

    # 获取实例数目
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    for i in range(N):
        y1, x1, y2, x2 = boxes[i]
        box = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        ctr = np.array(box).reshape((-1,1,2)).astype(np.int32)
        cv2.drawContours(image, [ctr], -1, (0, 0, 255), thickness=1)
    
    return image


############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect mug.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'detect'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/mug/dataset/",
                        help='Directory of the Mug dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--images', required=True,
                        metavar="path or URL",
                        help='directory path or url for saving images that need to be detected')
    parser.add_argument('--result', required=True,
                        metavar="path or URL",
                        help='directory path or url for saving detection results')

    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "detect":
        assert args.images and args.result, \
            "Provide --images and --result to apply mug detection"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = MugConfig()
    else:
        class InferenceConfig(MugConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "detect":
        detect(model, images_dir=args.images, result_dir=args.result)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'detect'".format(args.command))


# # 添加类别（source, class_id, class_name）
# self.add_class("object", 1, "mug")
# self.add_class("object", 2, "glass")
# self.add_class("object", 3, "book")

# ...

# self.add_image(
#     "object",
#     image_id=a['filename'],  # 使用文件名作为唯一的id
#     path=image_path,
#     width=width, height=height,
#     polygons=polygons
#     class_ids=class_ids)