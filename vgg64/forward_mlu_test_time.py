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

from pprint import pprint

with open('top1_log.csv', 'w') as f:
    f.write('batch,hit,miss,rate\n')

with open('top5_log.csv', 'w') as f:
    f.write('batch,hit,miss,rate\n')

def arg_topK(matrix, K, axis=1):
    full_sort = np.argsort(matrix, axis=axis)
    return full_sort.take(np.arange(-1, -K-1, -1), axis=axis)

prototxt_path = './models/vgg7_64_cam.prototxt'
caffemodel_path = './models/vgg7_64_cam.caffemodel'

batch_size = 50

# input data
x_test = np.load("./data/x_test.npy")
x_test = x_test.transpose((0, 3, 1, 2))

# labels
y_test = np.load("./data/y_test.npy")
y_test = np.where(y_test==1)
y_test = y_test[1]


# imnames = imnames[:50]
# labels = labels[:50]


caffe.set_mode_mfus()
#caffe.set_mode_mlu()
caffe.set_core_number(16)
caffe.set_batch_size(batch_size)
caffe.set_simple_flag(1)

caffe.set_rt_core("MLU270")

 # import model 
model = caffe.Net(prototxt_path, caffemodel_path, caffe.TEST)

total_num = len(y_test)
top1_total_hit_num = 0
top5_total_hit_num = 0

import time
start_time = time.time()

cur_index = 0
while(cur_index < total_num):
    print(cur_index, total_num)
    # input data & labels
    
    if cur_index + batch_size < total_num:
        test_data = x_test[cur_index: cur_index + batch_size]
        test_label = y_test[cur_index: cur_index + batch_size].reshape(batch_size, 1)
    else:
        test_data = x_test[cur_index:]
        test_label = y_test[cur_index:].reshape(len(test_data), 1)
    test_len = len(test_data)
    
    # set input layer data
    model.blobs['input_1'].data[...] = test_data
    
    # forward
    output = model.forward()
    # predict resule
    layer_name = list(model.blobs)[-1]
    output = output[layer_name]
    # pprint(output)
    

    # print('predicts:', type(predicts), predicts.shape)
    # print(predicts)
    # print('label   :', type(labels), labels.shape)
    # print(labels)
    # print('==   :', predicts == labels)
    
    
    cur_index += batch_size
    
    # top1
    top1_predicts = arg_topK(output, 1, axis=1)
    matrix = np.where(top1_predicts == test_label, 1, 0)
    matrix = np.max(matrix, axis=1)
    top1_hit_num = np.sum(matrix)

    # top5
    top5_predicts = arg_topK(output, 5, axis=1)
    matrix = np.where(top5_predicts == test_label, 1, 0)
    matrix = np.max(matrix, axis=1)
    top5_hit_num = np.sum(matrix)

end_time = time.time()
print('total time(s): ', end_time - start_time)