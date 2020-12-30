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

output_path = "deploy/models/yolov4.onnx"
print('exporting model to ONNX...')
torch.onnx.export(net, example, output_path, verbose=True, input_names=input_names, output_names=output_names, opset_version=11)
print('model exported to {:s}'.format(output_path))