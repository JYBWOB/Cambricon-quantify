import os
os.environ["GLOG_minloglevel"] = '5'
os.environ['TFU_ENABLE']='1'
os.environ['TFU_NET_FILTER']='0'

import caffe
import numpy as np
from PIL import Image

import cPickle
import time

from pprint import pprint

prototxt_path = './models/vgg64_full.prototxt'
caffemodel_path = './models/vgg64_full.caffemodel'

batch_size = 50
num_eval_steps = 1

# caffe.set_mode_mfus()
#caffe.set_mode_mlu()
caffe.set_mode_cpu()
caffe.set_core_number(16)
caffe.set_batch_size(batch_size)

caffe.set_rt_core("MLU270")

model = caffe.Net(prototxt_path, caffemodel_path, caffe.TEST)

input_name = model.blobs.keys()[0]
model.blobs[input_name].reshape(*(batch_size, 3, 32, 32))

total_num = num_eval_steps * batch_size


# forward
input_data = np.load("./data/x_test.npy")
input_data = input_data[:batch_size, :]

model.blobs[input_name].data[...] = input_data.transpose((0, 3, 1, 2))

model.forward()

layer_names = list(model.blobs)
for name in layer_names:
    if "tmp" in name:
        layer_names.remove(name)

for index, layer in zip(range(len(model.layers)), model.layers):
    if layer.type == "Convolution" or layer.type == "InnerProduct":
        layer_name = layer_names[index]
        data_name = layer_names[index - 1]

        data = model.blobs[data_name].data
        data = data
        np.save("./layerdata/{0}.npy".format(layer_name), data)
        print("save {0}.npy".format(layer_name))
# for item in model.blobs.keys():
#     print(item)
