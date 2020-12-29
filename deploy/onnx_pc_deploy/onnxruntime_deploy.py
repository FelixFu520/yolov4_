import onnxruntime as ort

from PIL import Image
import matplotlib.pyplot as plt

import numpy as np
import os


# ------------ load image
img = Image.open("street.jpg")
img = img.resize((416, 416))
img = np.array(img, dtype = np.float32)
img /= 255.0
img = img.transpose((2, 0, 1))
img = np.expand_dims(img, axis=0)
# img = img.astype(np.float32)

# ------------ load models
sess = ort.InferenceSession("../yolov4.onnx")

# ------------ check input & output
input_name = sess.get_inputs()[0].name
print("Input name  :", input_name)
input_shape = sess.get_inputs()[0].shape
print("Input shape :", input_shape)
input_type = sess.get_inputs()[0].type
print("Input type  :", input_type)

output_name0 = sess.get_outputs()[0].name
output_name1 = sess.get_outputs()[1].name
output_name2 = sess.get_outputs()[2].name
print("Output name0  :", output_name0)
print("Output name1  :", output_name1)
print("Output name2  :", output_name2)

print("Output shape0 :", sess.get_outputs()[0].shape)
print("Output shape1 :", sess.get_outputs()[1].shape)
print("Output shape2 :", sess.get_outputs()[2].shape)
output_type = sess.get_outputs()[0].type
print("Output type  :", output_type)

output_name = "output_0"
# ------------ run
result = sess.run([output_name0, output_name1, output_name2], {input_name: img})
print(type(result))
print(np.array(result[0]).shape)
print(np.array(result[1]).shape)
print(np.array(result[2]).shape)
print(result[0])

import time
# start = time.time()
# for _ in range(100):
#     result = sess.run([output_name], {input_name: img})
# end = time.time()
# print("CPU infer time:", str(end - start))


