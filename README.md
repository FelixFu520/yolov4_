# Yolo v4

2020年12月29日
```angular2html
1 Training
1.1 环境搭建
1.2 目录解释
1.3 代码解释
1.3.1 预测步骤
1.3.2 数据准备&训练
1.3.3 评估模型

2 Inference
2.1 导出模型
2.1.1 导出pytorch格式
2.1.2 导出ONNX格式
2.2 Pytorch
2.3 ONNX
2.4 TVM
2.5 TensorRT
```
---

本文包含YOLO的**训练和推理**的实现流程，主要是代码和操作部分，不包含理论部分学习部分。

这一部分理论内容[参考](https://github.com/FelixFu520/README/blob/master/notes/models/yolov4.md)，代码部分[参考](https://github.com/FelixFu520/yolov4_)。

参考镜像：`docker pull fusimeng/yolo_ti:all`

## 1. Training
### 1.1 环境搭建
训练环境如下

| 软件                | 版本         |
| ------------------- | ------------ |
| 系统                | ubuntu 18.04 |
| GPU Driver          | 440.64       |
| CUDA                | 10.2         |
| Python              | 3.6.9        |
| pytorch/torchvision | 1.7.0/0.8.1  |
| onnx                | 1.7          |
### 1.2目录解释

```
|-- img # 测试图片目录
|-- -- street.jpg   # 测试图片

|-- logs    # 存放训练结果（权重）目录

|-- model_data	# 存放模型数据
|-- -- simhei.ttf
|-- -- voc_classes.txt
|-- -- yolo_anchors.txt
|-- -- coco_classes.txt
|-- -- yolo4_weights.pth	# coco + voc 数据集的权重（80类）
|-- -- yolo4_voc_weights.pth	# vo数据集权重（20类）

|-- nets	# 存放网络代码
|-- -- CSPdarknet.py
|-- -- yolo4.py
|-- -- yolo_training.py

|-- utils	# 存放工具单元
|-- -- utils.py
|-- -- dataloader.py

|-- VOCdevkit	# 存放数据集
|-- -- VOC2007
|-- -- -- -- Annotations
|-- -- -- -- ImageSets
|-- -- -- -- JEPGImages

|-- yolo.py     # 预测时会用到
|-- predict.py  # 预测代码
|-- video.py    # 视频预测代码

|-- test.py	    # 查看网络结构

|-- kmeans_for_anchors.py  	# 生成anchors
|-- voc2yolov4.py   # voc的xml形式转换为yolo标注形式
|-- voc_annortation.py  # 生成图片路径和图片中object的位置类别
|-- train.py    # 训练网络
|-- train_with_tensorboard.py	# 使用tensorboard

|-- get_dr_txt.py						# 获取预测标签
|-- get_gt_txt.py						# 获取真实标签
|-- get_map.py							# 计算MAP

|-- ciou_test.py						# 测试CIOU

|-- pytorch2onnx.py					# 模型转换

|-- deploy									# 部署目录
|-- -- models							  # 存放模型
|-- -- pytorch_pc_deploy
|-- -- -- -- predict.py
|-- -- -- -- street.jpg
|-- -- -- -- yolo.py
|-- -- onnx_pc_deploy
|-- -- -- -- onnxruntime_deploy.py
|-- -- -- -- street.jpg
|-- -- tvm_pc_deploy
|-- -- -- -- onnxruntime_deploy.py
|-- -- -- -- deploy_cpp_pc
|-- -- -- -- -- -- CMakeLists.txt
|-- -- -- -- -- -- deploy_so.cpp
|-- -- -- -- -- -- tvm_runtime_pack.cc
|-- -- -- -- deploy_so_python.ipynb
|-- -- -- -- ONNX2TVM.ipynb
|-- -- -- -- Pytorch2TVM.ipynb
|-- -- -- -- street.jpg
|-- -- tvm_rasp_deploy


|-- LICENSE									# 证书
|-- questions.md						# 常见问题
|-- requirements.txt				# 系统环境
```
### 1.3 代码解释
#### 1.3.1 预测步骤
##### 1、使用预训练权重

a、下载完库后解压，在百度网盘下载[yolo4_weights.pth](https://pan.baidu.com/s/1lNTRL-ipME8jnrbjE9pEnA)【 密码: 1gnp】或者yolo4_voc_weights.pth【同上】，放入model_data，运行predict.py，输入`img/street.jpg`可完成预测。

b、利用video.py可进行摄像头检测。

##### 2、使用自己训练的权重

a、按照训练步骤训练。

b、在yolo.py文件里面，在如下部分修改model_path和classes_path使其对应训练好的文件；**model_path对应logs文件夹下面的权值文件，classes_path是model_path对应分的类**。

```
_defaults = {
    "model_path": 'model_data/yolo4_weights.pth',
    "anchors_path": 'model_data/yolo_anchors.txt',
    "classes_path": 'model_data/coco_classes.txt',
    "model_image_size" : (416, 416, 3),
    "confidence": 0.5,
    "cuda": True
}
```

c、运行predict.py，输入`img/street.jpg`可完成预测。

d、利用video.py可进行摄像头检测。
#### 1.3.2 数据准备&训练

**1**、本文使用VOC格式进行训练。

**2**、训练前将标签文件放在VOCdevkit文件夹下的VOC2007文件夹下的Annotation中。

**3**、训练前将图片文件放在VOCdevkit文件夹下的VOC2007文件夹下的JPEGImages中。

**4**、运行根目录下的`kmeans_for_anchors.py`，生成anchors，并存放到`model_data/yolo_anchors.txt`。

**5**、在训练前利用`voc2yolo4.py`文件生成对应的txt。存放所有文件名。

**6**、再运行根目录下的`voc_annotation.py`，运行前需要将classes改成你自己的classes。**注意不要使用中文标签，文件夹中不要有空格！**。

```
classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
```

此时会生成对应的2007_train.txt，每一行对应其**图片位置**及其**真实框的位置**。
`/root/yolov4_/VOCdevkit/VOC2007/JPEGImages/004017.jpg 237,1,450,217,15 10,4,450,375,14`

**7**、**在训练前需要务必在model_data下新建一个txt文档，文档中输入需要分的类，在train.py中将classes_path指向该文件**，示例如下：

```
classes_path = 'model_data/new_classes.txt'    
```

model_data/new_classes.txt文件内容为：

```
cat
dog
...
```

**8**、运行train.py即可开始训练。
#### 1.3.3 评估模型

mAP目标检测精度计算更新，分别运行`get_gt_txt.py`、`get_dr_txt.py`和`get_map.py`文件。

> get_map文件克隆自https://github.com/Cartucho/mAP
>
> 具体mAP计算过程可参考：https://www.bilibili.com/video/BV1zE411u7Vw

## 2. Inferencing
### 2.1 导出模型
#### 2.1.1 导出pytorch格式
见`train.py`文件中的`torch.save`方法

```
torch.save(model.state_dict(), 'logs/Epoch%d-Total_Loss%.4f-Val_Loss%.4f.pth'%((epoch+1),total_loss/(epoch_size+1),val_loss/(epoch_size_val+1)))
```

其加载方式见`yolo.py`文件中的

```
print('Loading weights into state dict...')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
state_dict = torch.load(self.model_path, map_location=device)
self.net.load_state_dict(state_dict)
```

#### 2.1.2 导出onnx格式

`pytorch2onnx.py`

```
import os
from nets.yolo4 import YoloBody
import torch

import numpy as np


# ------------- set the device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('running on device ' + str(device))

# ------------- load the image
example = torch.rand(1, 3, 416, 416).to(device)
print("input size is..", example.shape)

# ------------- load the model
model_path = 'model_data/yolo4_weights.pth'
anchors_path = 'model_data/yolo_anchors.txt'
classes_path = 'model_data/coco_classes.txt'

# get classes
classes_path = os.path.expanduser(classes_path)
with open(classes_path) as f:
    class_names = f.readlines()
class_names = [c.strip() for c in class_names]

# get anchors
anchors_path = os.path.expanduser(anchors_path)
with open(anchors_path) as f:
    anchors = f.readline()
anchors = [float(x) for x in anchors.split(',')]
anchors = np.array(anchors).reshape([-1, 3, 2])[::-1, :, :]

net = YoloBody(len(anchors[0]), len(class_names)).eval()
print('Loading weights into state dict...')
is_cuda = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(is_cuda)
state_dict = torch.load(model_path, map_location=device)
net.load_state_dict(state_dict)

if is_cuda == "cuda":
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    net = net.cuda()

print("model set!")

# -------------- export the model
input_names = ["input_0"]
output_names = ["output_0", "output_1", "output_2"]

output_path = "deploy/yolov4.onnx"
print('exporting model to ONNX...')
torch.onnx.export(net, example, output_path, verbose=True, input_names=input_names, output_names=output_names, opset_version=11)
print('model exported to {:s}'.format(output_path))
```



### 2.2 Pytorch

#### 2.2.1 GPU

使用pytorch框架运行在GPU上，测速。

推理100次，用时`1.992154836654663`s

详细代码，参考`deploy/pytorch_pc_deploy/`文件夹下的代码。

#### 2.2.2 CPU

使用pytorch框架运行在CPU上，测速。

推理100次，用时`34.87498211860657`s

详细代码，参考`deploy/pytorch_pc_deploy/`文件夹下的代码。

### 2.3 ONNX

#### 2.3.1 ONNXRuntime

**ONNX_CPU_PC_Python_ONNXRuntime**

| 软件                | 版本         |
| ------------------- | ------------ |
| 系统                | ubuntu 18.04 |
| GPU Driver          | 440.64       |
| CUDA                | 10.2         |
| Python              | 3.6.9        |
| pytorch/torchvision | 1.7.0/0.8.1  |
| onnx                | 1.7.0        |
| onnxruntime         | 1.6.0        |

使用pytorch框架运行在CPU上，测速。

推理100次，用时`39.027772188186646`s

详细代码，参考`deploy/onnx_pc_deploy/`文件夹下的代码。

#### **2.3.2 [cONNXr](https://github.com/alrevuelta/cONNXr)**

`TODO`

#### **2.3.3 [onnx_runtime_cpp](https://github.com/xmba15/onnx_runtime_cpp)**

`TODO`

### 2.2 TVM

#### 2.2.1 PC

| 软件                | 版本         |
| ------------------- | ------------ |
| 系统                | ubuntu 18.04 |
| GPU Driver          | 440.64       |
| CUDA                | 10.2         |
| Python              | 3.6.9        |
| pytorch/torchvision | 1.7.0/0.8.1  |
| onnx                | 1.7.0        |
| tvm                 | 0.8          |

使用tvm框架运行在CPU上，测速。

推理100次，用时`39ms`(cpp)，`Time elapsed is 0m 38s`(python)

详细代码，参考`deploy/tvm_pc_deploy/`文件夹下的代码。

#### 2.2.2 Rasp





### 2.3 TensorRT

`TODO`

### 2.4 MNN

`TODO`

### 2.5 NCNN

`TODO`

###  