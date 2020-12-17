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

with open('top1_log_mlu.csv', 'w') as f:
    f.write('batch,hit,miss,rate\n')

with open('top5_log_mlu.csv', 'w') as f:
    f.write('batch,hit,miss,rate\n')

def arg_topK(matrix, K, axis=1):
    full_sort = np.argsort(matrix, axis=axis)
    return full_sort.take(np.arange(-1, -K-1, -1), axis=axis)

prototxt_path = './models/alexnet_qtz.prototxt'
caffemodel_path = './models/alexnet_full.caffemodel'

batch_size = 64

caffe.set_mode_mfus()
#caffe.set_mode_mlu()
caffe.set_core_number(16)
caffe.set_batch_size(batch_size)
caffe.set_simple_flag(1)

caffe.set_rt_core("MLU270")
 # import model 
model = caffe.Net(prototxt_path, caffemodel_path, caffe.TEST)

num_eval_steps = 782

total_num = num_eval_steps * batch_size
top1_total_hit_num = 0
top5_total_hit_num = 0


for i in range(num_eval_steps):
    input_data = np.load("../valdata/data/data-batch-%d.npy"%(i))
    label = np.load("../valdata/label/label-batch-%d.npy"%(i))

    label.resize((batch_size, 1))
    
    model.blobs['input_1'].data[...] = input_data.transpose((0, 3, 1, 2))

    model.forward()
    
    blobs = list(model.blobs)
    layer_name = blobs[-1]
    output = model.blobs[layer_name].data
    
    # top1
    top1_predicts = arg_topK(output, 1, axis=1)
    # print(top1_predicts)
    matrix = np.where(top1_predicts == label, 1, 0)
    matrix = np.max(matrix, axis=1)
    top1_hit_num = np.sum(matrix)
    
    top1_total_hit_num += top1_hit_num
    info = 'batch %d, top1: (hit: %d, miss: %d, hit_rate: %.2f' % (i, top1_hit_num, batch_size - top1_hit_num, 100.0 * top1_hit_num / batch_size)
    print(info + '%)')
    
    with open('top1_log_mlu.csv', 'a') as f:
        f.write('%d, %d,%d,%.2f\n' %  (i, top1_hit_num, batch_size - top1_hit_num, 100.0 * top1_hit_num / batch_size))
    
    # top5
    top5_predicts = arg_topK(output, 5, axis=1)
    matrix = np.where(top5_predicts == label, 1, 0)
    matrix = np.max(matrix, axis=1)
    top5_hit_num = np.sum(matrix)
    
    top5_total_hit_num += top5_hit_num
    info = 'batch %d, top5: (hit: %d, miss: %d, hit_rate: %.2f' % (i, top5_hit_num, batch_size - top5_hit_num, 100.0 * top5_hit_num / batch_size)
    print(info + '%)')
    
    with open('top5_log_mlu.csv', 'a') as f:
        f.write('%d, %d,%d,%.2f\n' % (i, top5_hit_num, batch_size - top5_hit_num, 100.0 * top5_hit_num / batch_size))
    

print('all the batchs finished')
print('final result')
info = 'top1:: hit: %d, miss: %d, hit_rate: %.2f' % (top1_total_hit_num, total_num - top1_total_hit_num, 100.0 * top1_total_hit_num / total_num)
print(info + '%')

info = 'top5:: hit: %d, miss: %d, hit_rate: %.2f' % (top5_total_hit_num, total_num - top5_total_hit_num, 100.0 * top5_total_hit_num / total_num)
print(info + '%')
with open('final_mlu.csv', 'w') as f:
    f.write('type,hit,miss,rate\n')
    f.write('top1,%d,%d,%.2f\n' % (top1_total_hit_num, total_num - top1_total_hit_num, 100.0 * top1_total_hit_num / total_num))
    f.write('top5,%d,%d,%.2f\n' % (top5_total_hit_num, total_num - top5_total_hit_num, 100.0 * top5_total_hit_num / total_num))
