#!/usr/bin/env python2

'''
    This program is used for cambricon mlu100
'''
from __future__ import print_function
import os
os.environ["GLOG_minloglevel"] = '5'
os.environ['TFU_ENABLE']='1' 
os.environ['TFU_NET_FILTER']='0'
os.environ['CNRT_PRINT_INFO']='false' 
os.environ['CNRT_GET_HARDWARE_TIME']='false'
os.environ['CNML_PRINT_INFO']='false'  
import caffe
import math
import shutil
import stat
import subprocess
import sys
import numpy as np
import collections
import copy
import time
import traceback
import datetime
import cv2
from PIL import Image
import scipy.misc
import scipy.io
import scipy


if __name__ == '__main__':
    if len(sys.argv)!=5:
        print("Usage:{} prototxt caffemodel batch_size datasource".format(sys.argv[0]))
        print(sys.argv)
        sys.exit(1)
    
    prototxt=sys.argv[1]
    print(prototxt)

    caffemodel=sys.argv[2]
    

    batch_size = int(sys.argv[3])
    
    datasource = sys.argv[4]
    datasource = datasource.strip()

    # caffe.set_mode_mlu()
    # caffe.set_mode_mfus()
    caffe.set_mode_cpu()

    caffe.set_rt_core("MLU270")
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    input_name = net.blobs.keys()[0]
    
    input_data = np.load(datasource)
    input_data = input_data[:batch_size]
    input_data = input_data.transpose((0, 3, 1, 2))

    net.blobs[input_name].reshape(*input_data.shape)
    #tmp=input_data.flatten()
    #print(np.min(tmp),np.max(tmp))

    net.blobs[input_name].data[...]=input_data
    for i in range(4):
        start_time=time.time()
        output = net.forward()
        end_time=time.time()
        print("%.4f"%(end_time-start_time), end='\t')
    print("done!")
