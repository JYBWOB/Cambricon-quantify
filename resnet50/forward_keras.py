# import os
# os.environ["GLOG_minloglevel"] = '5'
# os.environ['TFU_ENABLE']='1' 
# os.environ['TFU_NET_FILTER']='0' 

import keras
import numpy as np
from PIL import Image

from pprint import pprint

model_path = './models/resnet50.h5'
img_lst_path = './data/image.lst'
label_path = './data/label.txt'

with open('top1_log_keras.csv', 'w') as f:
    f.write('batch,hit,miss,rate\n')

with open('top5_log_keras.csv', 'w') as f:
    f.write('batch,hit,miss,rate\n')

def arg_topK(matrix, K, axis=1):
    full_sort = np.argsort(matrix, axis=axis)
    return full_sort.take(np.arange(-1, -K-1, -1), axis=axis)

input_shape = [0, 224, 224, 3]

batch_size = 50

# input data
with open(img_lst_path) as f:
    imnames = f.readlines()
imnames = [x.strip() for x in imnames]


# labels
with open(label_path) as f:
    labels = f.readlines()
labels = [int(x.strip().split(' ')[-1]) for x in labels]

# imnames = imnames[:10]
# labels = labels[:10]


 # import model 
model = keras.models.load_model(model_path)

total_num = len(imnames)
top1_total_hit_num = 0
top5_total_hit_num = 0

cur_index = 0
while(cur_index < total_num):
    # input data & labels
    
    if cur_index + batch_size < total_num:
        test_lst = imnames[cur_index: cur_index + batch_size]
        test_label = np.array(labels[cur_index: cur_index + batch_size]).reshape(batch_size, 1)
    else:
        test_lst = imnames[cur_index:]
        test_label = np.array(labels[cur_index:]).reshape(len(test_lst), 1)
    test_len = len(test_lst)
        
    input_data = np.zeros(input_shape)

    for img_name in test_lst:
        
        im = Image.open(img_name)
        im = im.resize((256,256))
        im = im.convert('RGB')
        in_ = np.array(im)
        in_ = in_[16:16+224,16:16+224]
        # RGB 2 BGR
        in_ = in_[..., ::-1]
        
        in_ = in_ - np.array((103.939, 116.779, 123.68))   
        
        in_=in_[np.newaxis,:]
        # in_ = in_.transpose((0, 3, 1, 2))
        input_data = np.concatenate((input_data, in_), axis = 0)
    
    
    # forward
    output = model.predict(input_data)

    # print('predicts:', type(predicts), predicts.shape)
    # print(predicts)
    # print('label   :', type(labels), labels.shape)
    # print(labels)
    # print('==   :', predicts == labels)
    
    
    cur_index += batch_size
    
    # top1
    top1_predicts = arg_topK(output, 1, axis=1)
    # print(top1_predicts)
    matrix = np.where(top1_predicts == test_label, 1, 0)
    matrix = np.max(matrix, axis=1)
    top1_hit_num = np.sum(matrix)
    
    top1_total_hit_num += top1_hit_num
    info = 'batch %d, top1: (hit: %d, miss: %d, hit_rate: %.2f' % (cur_index / test_len, top1_hit_num, test_len - top1_hit_num, 100.0 * top1_hit_num / test_len)
    print(info + '%)')
    
    with open('top1_log_keras.csv', 'a') as f:
        f.write('%d, %d,%d,%.2f\n' %  (cur_index / test_len, top1_hit_num, test_len - top1_hit_num, 100.0 * top1_hit_num / test_len))
    
    # top5
    top5_predicts = arg_topK(output, 5, axis=1)
    matrix = np.where(top5_predicts == test_label, 1, 0)
    matrix = np.max(matrix, axis=1)
    top5_hit_num = np.sum(matrix)
    
    top5_total_hit_num += top5_hit_num
    info = 'batch %d, top5: (hit: %d, miss: %d, hit_rate: %.2f' % (cur_index / test_len, top5_hit_num, test_len - top5_hit_num, 100.0 * top5_hit_num / test_len)
    print(info + '%)')
    
    with open('top5_log_keras.csv', 'a') as f:
        f.write('%d, %d,%d,%.2f\n' % (cur_index / test_len, top5_hit_num, test_len - top5_hit_num, 100.0 * top5_hit_num / test_len))
    

print('all the batchs finished')
print('final result')
info = 'top1:: hit: %d, miss: %d, hit_rate: %.2f' % (top1_total_hit_num, total_num - top1_total_hit_num, 100.0 * top1_total_hit_num / total_num)
print(info + '%')

info = 'top5:: hit: %d, miss: %d, hit_rate: %.2f' % (top5_total_hit_num, total_num - top5_total_hit_num, 100.0 * top5_total_hit_num / total_num)
print(info + '%')
with open('final_keras.csv', 'w') as f:
    f.write('type,hit,miss,rate\n')
    f.write('top1,%d,%d,%.2f\n' % (top1_total_hit_num, total_num - top1_total_hit_num, 100.0 * top1_total_hit_num / total_num))
    f.write('top5,%d,%d,%.2f\n' % (top5_total_hit_num, total_num - top5_total_hit_num, 100.0 * top5_total_hit_num / total_num))
