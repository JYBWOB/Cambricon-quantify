import os
os.environ["GLOG_minloglevel"] = '5'
os.environ['TFU_ENABLE']='1' 
os.environ['TFU_NET_FILTER']='0' 
os.environ['CNRT_PRINT_INFO']='false' 
os.environ['CNRT_GET_HARDWARE_TIME']='false'
os.environ['CNML_PRINT_INFO']='false'  

import caffe
import numpy as np

from pprint import pprint

prototxt_path = './models/lenet_cam.prototxt'
caffemodel_path = './models/lenet_cam.caffemodel'

caffe.set_mode_mfus()
#caffe.set_mode_mlu()
caffe.set_core_number(16)
caffe.set_simple_flag(1)

caffe.set_rt_core("MLU270")
 # import model 
model = caffe.Net(prototxt_path, caffemodel_path, caffe.TEST)

input_data = np.ones((1, 1, 28, 28))

# set input layer data
model.blobs['input_1'].data[...] = input_data

# forward
output = model.forward()
# predict resule
layer_name = list(model.blobs)[-1]
output = output[layer_name]

print(output)
