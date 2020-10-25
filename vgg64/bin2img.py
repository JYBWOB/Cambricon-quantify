from pprint import pprint
import cPickle
from PIL import Image
import numpy as np

prefix = './data/cifar/'

batch_size = 10000

def unpickler(file):
    with open(file, 'rb') as fo:
        d = cPickle.load(fo)
    return d  

def pickler(data, file):
    with open(file, 'wb') as fo:
        cPickle.dump(data, fo)

def print_progress(total_num, num):
    progress = 100 * num / total_num
    block = progress // 5
    block_str = ''
    for i in range(20):
        block_str = block_str + '>' if i <= block else block_str + '='
    print('[{0}][{1}%]'.format(block_str, progress))

d = unpickler(prefix + 'test_batch')

data = d['data']
labels = d['labels']

total_num = len(data)
# total_num = 50
# bin file to img

with open(prefix+'image.lst', 'w') as f:
    for num in range(total_num):
        cur_data = data[num].reshape((3, 32, 32))
        cur_data = cur_data.transpose((1, 2, 0))
        image = Image.fromarray(np.uint8(cur_data))
        filename = prefix + "imgs/{0}.jpg".format(num)
        image.save(filename)
        
        # write image.lst
        f.write(filename + '\n')
        
        if (100 * num) % total_num == 0:
            print_progress(total_num, num)

print('img & imglst saved successfully')

# d = {'labels': labels}
# pickler(d, './data/labels')
# print('labels saved successfully')
