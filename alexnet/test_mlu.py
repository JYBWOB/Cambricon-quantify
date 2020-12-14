import os
os.environ["GLOG_minloglevel"] = '5'
os.environ['TFU_ENABLE']='1' 
os.environ['TFU_NET_FILTER']='0' 
os.environ['CNRT_PRINT_INFO']='false' 
os.environ['CNRT_GET_HARDWARE_TIME']='false'
os.environ['CNML_PRINT_INFO']='false'  

import caffe
import numpy as np
from PIL import Image

import cPickle

from pprint import pprint

def arg_topK(matrix, K, axis=1):
    full_sort = np.argsort(matrix, axis=axis)
    return full_sort.take(np.arange(-1, -K-1, -1), axis=axis)

prototxt_path = './models/alexnet_cam.prototxt'
caffemodel_path = './models/alexnet_cam.caffemodel'
img_lst_path = './data/image.lst'
label_path = './data/label.txt'

input_shape = [0, 3, 224, 224]

batch_size = 1


# caffe.set_mode_mfus()
caffe.set_mode_mlu()
caffe.set_core_number(16)
caffe.set_batch_size(batch_size)
caffe.set_simple_flag(1)

caffe.set_rt_core("MLU270")
 # import model 
model = caffe.Net(prototxt_path, caffemodel_path, caffe.TEST)

# input data & labels
    
# input_data = np.ones((1, 3, 224, 224))
im = Image.open("./data/imgs/ILSVRC2012_val_00000001.JPEG")
# im = im.resize((256,256))
im = im.resize((224,224))
im = im.convert('RGB')
in_ = np.array(im)
# RGB 2 BGR
in_ = in_[..., ::-1]
in_ = in_ - np.array((103.94, 116.78, 123.68))   

in_=in_[np.newaxis,:]
in_ = in_.transpose((0, 3, 1, 2))
input_data = in_

# set input layer data
model.blobs['input_1'].data[...] = input_data

# forward
output = model.forward()
# predict resule
# layer_name = list(model.blobs)[-1]
# output = output[layer_name]
output = model.blobs['Q_conv2d_1'].data

print(output.shape)
pprint(output)    


# print('predicts:', type(predicts), predicts.shape)
# print(predicts)
# print('label   :', type(labels), labels.shape)
# print(labels)
# print('==   :', predicts == labels)
