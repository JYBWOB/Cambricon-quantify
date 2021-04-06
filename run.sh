#!/bin/bash
batch_size='10'
valdata='./valdata/data/data-batch-0.npy'
cifardata='./cifarnet/data/x_test.npy'
ministdata='./lenet/data/x_test.npy'
python forward_cpu.py ./alexnet/models/alexnet.prototxt ./alexnet/models/alexnet_cam.caffemodel $batch_size $valdata
python forward_cpu.py ./resnet18/models/resnet18.prototxt ./resnet18/models/resnet18.caffemodel $batch_size $valdata
python forward_cpu.py ./resnet50/models/resnet50.prototxt ./resnet50/models/resnet50.caffemodel $batch_size $valdata
python forward_cpu.py ./cifarnet/models/cifar.prototxt ./cifarnet/models/cifar_cam.caffemodel $batch_size $cifardata
python forward_cpu.py ./vgg64/models/vgg64_full.prototxt ./vgg64/models/vgg64_full.caffemodel $batch_size $cifardata
python forward_cpu.py ./lenet/models/lenet.prototxt ./lenet/models/lenet_cam.caffemodel $batch_size $ministdata
python forward_mlu.py ./alexnet/models/alexnet_cam.prototxt ./alexnet/models/alexnet_cam.caffemodel $batch_size $valdata
python forward_mlu.py ./resnet18/models/resnet18_cam.prototxt ./resnet18/models/resnet18_cam.caffemodel $batch_size $valdata
python forward_mlu.py ./resnet50/models/resnet50_qtz_50.prototxt ./resnet50/models/resnet50.caffemodel $batch_size $valdata
python forward_mlu.py ./cifarnet/models/cifar_cam.prototxt ./cifarnet/models/cifar_cam.caffemodel $batch_size $cifardata
python forward_mlu.py ./vgg64/models/vgg7_64_cam.prototxt ./vgg64/models/vgg7_64_cam.caffemodel $batch_size $cifardata
python forward_mlu.py ./lenet/models/lenet_cam.prototxt ./lenet/models/lenet_cam.caffemodel $batch_size $ministdata
