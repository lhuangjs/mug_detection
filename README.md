# 使用Mask R-CNN检测马克杯
## 准备
1. 按照下面给出的教程配置环境
文字版教程：[https://github.com/markjay4k/Mask-RCNN-series/blob/master/Mask_RCNN%20Install%20Instructions.ipynb](https://github.com/markjay4k/Mask-RCNN-series/blob/master/Mask_RCNN%20Install%20Instructions.ipynb)
视频版教程：[https://www.youtube.com/watch?v=2TikTv6PWDw](https://www.youtube.com/watch?v=2TikTv6PWDw)

2. 参考官方样例，地址：[https://engineering.matterport.com/splash-of-color-instance-segmentation-with-mask-r-cnn-and-tensorflow-7c761e238b46](https://engineering.matterport.com/splash-of-color-instance-segmentation-with-mask-r-cnn-and-tensorflow-7c761e238b46)


## 开始
1. 克隆本仓库
```
git clone git@github.com:lhuangjs/mug_detection.git
```
2. 修改mug.py，将路径改为Mask R-CMM模型代码所在目录
```
# MaskRCNN模型所在目录
ROOT_DIR = os.path.abspath("../Mask/Mask_RCNN")
```
3. 训练模型
```
python mug.py train --dataset xxx\mug_detection\dataset --weights=coco
```
4. 使用模型
```
python mug.py detect --weights=xxx\Mask\Mask_RCNN\logs\mug20180616T1843\mask_rcnn_mug_0030.h5 --images=xxx\mug_detection\dataset\test --result=xxx\mug_detection\detection_result
```