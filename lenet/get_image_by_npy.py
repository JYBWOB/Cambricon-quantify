import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import os

from tqdm import tqdm
import json

netname = 'lenet'
<<<<<<< HEAD
bin_num = 2048  # 2048
=======
bin_num = 1024   # 2048
>>>>>>> 3299b84d532aea57a000ffd89ebd26ed70461725
MIN_BINS = 128   # 128


MAX_INT = 127
eps = 1e-3


def get_histogram(data, bin_num=2048):
    max_val = np.max(data)
    width = max_val / bin_num
    x = [i * width + width / 2 for i in range(bin_num)]

    bins = [i * width for i in range(bin_num + 1)]
    y, bins = np.histogram(data, bins=bins)
    # y = y / float(np.sum(y))
    return x, y

<<<<<<< HEAD
import numpy as np
import copy

def compute_kl_divergence(P,Q):
    length=len(P)
    sum=0.0
    for i in range(length):
        if P[i]!=0:
            if Q[i]==0:
                sum+=1.0
            else:
                sum+=P[i]*np.log((P[i]+1e-5)/(Q[i]+1e-5))
    return sum


def threshold_distribution(distribution,target_bin):
    KL_list = []
    
    target_threshold = target_bin
    min_kl_divergence = 10000000000000
    length = len(distribution)

    for threshold in tqdm(range(target_bin,length)):
        #t_distribution=np.empty((threshold,))
        t_distribution=copy.deepcopy(distribution[0:threshold])
        t_distribution[threshold - 1] += np.sum(distribution[threshold:])

        #get P
        num_per_bin = float(threshold) / float(target_bin)

        quantize_distribution = np.zeros((target_bin,))

        for i in range(target_bin):
            start = i * num_per_bin
            end = start + num_per_bin

            left_upper = int(np.ceil(start))
            if left_upper > start:
                left_scale = left_upper - start
                quantize_distribution[i] += left_scale * distribution[left_upper - 1]
            right_lower = int(np.floor(end))

            if right_lower < end:
                right_scale = end - right_lower
                quantize_distribution[i] += right_scale * distribution[right_lower]

            for j in range(left_upper,right_lower):
                quantize_distribution[i] += distribution[j]

        # get Q
        expand_distribution=np.zeros_like(t_distribution)

        for i in range(target_bin):
            start = i * num_per_bin
            end = start + num_per_bin

            count = 0

            left_upper = int(np.ceil(start))
            left_scale = 0.0
            if left_upper > start:
                left_scale = left_upper - start
                if t_distribution[left_upper - 1] != 0:
                    count += left_scale

            right_lower = int(np.floor(end))
            right_scale = 0.0
            if right_lower < end:
                right_scale = end - right_lower
                if t_distribution[right_lower] != 0:
                    count += right_scale

            for j in range(left_upper,right_lower):
                if t_distribution[j] != 0:
                    count+=1

            expand_value = quantize_distribution[i] / count

            if left_upper > start:
                if t_distribution[left_upper - 1] != 0:
                    expand_distribution[left_upper - 1] += expand_value * left_scale
            if right_lower < end:
                if t_distribution[right_lower] != 0:
                    expand_distribution[right_lower] += expand_value * right_scale
            for j in range(left_upper,right_lower):
                if t_distribution[j] != 0:
                    expand_distribution[j] += expand_value
        kl_divergence = compute_kl_divergence(t_distribution, expand_distribution)
        KL_list.append(kl_divergence)
        if kl_divergence < min_kl_divergence:
            min_kl_divergence = kl_divergence
            target_threshold = threshold

    return target_threshold, KL_list
=======
# def get_threshold(data, x, y, bin_num=2048, min_bins=128):
#     divergences = []
#     for i in tqdm(range(min_bins, bin_num)):
#         # P
#         reference_distribution_P = y.copy()
#         reference_distribution_P = reference_distribution_P[:i]
#         reference_distribution_P[i - 1] += np.sum(reference_distribution_P[i:])
        
#         # Q
#         threshold = x[i]
#         scale = MAX_INT / threshold
#         temp_data = data[np.where(data < threshold)]
#         temp_data = temp_data * scale
#         temp_data = np.round(temp_data)
#         _, candidate_distribution_Q = get_histogram(temp_data, i)

#         reference_distribution_P = reference_distribution_P / float(np.sum(reference_distribution_P))
#         candidate_distribution_Q = candidate_distribution_Q / float(np.sum(candidate_distribution_Q))
#         divergences.append(KL_divergence(reference_distribution_P, candidate_distribution_Q, eps))
#     index = divergences.index(min(divergences))
#     # print(divergences)
#     return x[index + min_bins]

def get_threshold(data, x, y, bin_num=2048, min_bins=128):
    divergences = []
    for i in tqdm(range(min_bins, bin_num)):
        # P
        reference_distribution_P = y.copy()
        
        # Q
        threshold = x[i]
        scale = MAX_INT / threshold
        temp_data = data[np.where(data < threshold)]
        temp_data = temp_data * scale
        temp_data = np.round(temp_data)
        _, candidate_distribution_Q = get_histogram(temp_data, bin_num)

        reference_distribution_P = reference_distribution_P / float(np.sum(reference_distribution_P))
        candidate_distribution_Q = candidate_distribution_Q / float(np.sum(candidate_distribution_Q))
        divergences.append(KL_divergence(reference_distribution_P, candidate_distribution_Q, eps))
    index = divergences.index(min(divergences))
    # print(divergences)
    return x[index + min_bins]


def KL_divergence(p, q, eps=1e-3):
    if len(p) != len(q):
        print("length of p {0}, length of q {1}".format(len(p), len(q)))
        return None
    value = 0
    for x, y in zip(p, q):
        value = value + x * np.log((x + eps) / (y + eps))
    return value
>>>>>>> 3299b84d532aea57a000ffd89ebd26ed70461725

def save_dict(filename, dic):
    '''save dict into json file'''
    with open(filename,'w') as json_file:
        json.dump(dic, json_file, ensure_ascii=False)
        
d = {}
d['first'] = 'conv2d_1'
        
for filename in os.listdir("./layerdata/"):
    if "npy" in filename:
    # if filename == "dense_1.npy":
        layer_name = filename[:-4]
        print(layer_name)

        data = np.load("./layerdata/{0}".format(filename))
        data = data.flatten()
        data = np.abs(data)

        # x, y = np.unique(data, return_counts=True)

        print("\tgenerate histogram data")
        x, y = get_histogram(data, bin_num)
        y = y / float(np.sum(y))
        print("\tget threshold")
<<<<<<< HEAD
        threshold_at_num, KL_list = threshold_distribution(y, MIN_BINS)
        threshold = x[threshold_at_num]
=======
        threshold = get_threshold(data, x, y, bin_num, MIN_BINS)
>>>>>>> 3299b84d532aea57a000ffd89ebd26ed70461725


        print("\tdraw figure...")
        # plt.yscale('symlog', linthreshx=0.0000002)
<<<<<<< HEAD
        plt.title('{0}: {1}'.format(netname, layer_name), fontsize=10)
        plt.xlabel('Input data', fontsize=10)
        plt.ylabel('Normalized number of counts', fontsize=10)
        # plt.ylabel('number of counts')
        
        y_tick = [10**i for i in range(-9, 1)]
        plt.yticks(y_tick)
        
        plt.vlines(threshold, 0, np.max(y))
        plt.text(threshold, np.max(y), "%.2f" % (threshold), fontsize=15)
=======
        plt.title('{0}: {1}'.format(netname, layer_name))
        plt.xlabel('Input data')
        plt.ylabel('Normalized number of counts')
        # plt.ylabel('number of counts')
        plt.vlines(threshold, 0, np.max(y))
        plt.text(threshold, np.max(y), "%.2f"%format(threshold), fontsize=15)
>>>>>>> 3299b84d532aea57a000ffd89ebd26ed70461725
        # plt.scatter(x, y, marker='o')
        plt.semilogy(x, y, '.', marker='D')
        plt.savefig("./layerdata/" + netname + "_" + layer_name + ".png")
        print("\t./layerdata/" +  netname + "_" + layer_name + '.png saved')
        plt.cla()
        
        plt.title('{0}: {1}'.format(netname, layer_name), fontsize=15)
        plt.xlabel('Input data', fontsize=15)
        plt.ylabel('KL-divergence', fontsize=15)
        # plt.ylabel('number of counts')
        
        y_tick = [10**i for i in range(-9, 1)]
        plt.yticks(y_tick)
        # plt.scatter(x[MIN_BINS:], KL_list, marker='o')
        plt.semilogy(x[MIN_BINS:], KL_list, '.', marker='D')
        plt.savefig("./layerdata/" + netname + "_" + layer_name + "_KL" + ".png")
        print("\t./layerdata/" +  netname + "_" + layer_name + '.png saved')
        plt.cla()
        
        scale = 1.0 * MAX_INT / threshold
        d[layer_name] = scale
save_dict('./layerdata/result.json', d)
