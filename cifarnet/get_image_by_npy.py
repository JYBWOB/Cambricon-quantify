import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import os

from tqdm import tqdm
import json

netname = 'cifarnet'
bin_num = 1024   # 2048
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
        print("\tget threshold")
        threshold = get_threshold(data, x, y, bin_num, MIN_BINS)

        y = y / float(np.sum(y))

        print("\tdraw figure...")
        # plt.yscale('symlog', linthreshx=0.0000002)
        plt.title('{0}: {1}'.format(netname, layer_name))
        plt.xlabel('Input data')
        plt.ylabel('Normalized number of counts')
        # plt.ylabel('number of counts')
        plt.vlines(threshold, 0, np.max(y))
        plt.text(threshold, np.max(y), "%.2f"%format(threshold), fontsize=15)
        # plt.scatter(x, y, marker='o')
        plt.semilogy(x, y, '.', marker='D')
        plt.savefig("./layerdata/" + layer_name + ".png")
        print("\t./layerdata/" + layer_name + '.png saved')
        plt.cla()
        
        scale = 1.0 * MAX_INT / threshold
        d[layer_name] = scale
save_dict('./layerdata/result.json', d)