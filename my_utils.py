import numpy as np
import matplotlib.pyplot as plt
import math
from tqdm import tqdm
from random import randrange
import torch

def NormalizeData(data): # (within [-1, 1] box)
    return 2 *(data - torch.min(data)) / (torch.max(data) - torch.min(data)) - 1

def get_pdist_row_col(idx):
    col = int((1 + math.sqrt(1 +8*idx)) // 2)
    row = int(idx - (col-1)*(col)/2)
    return (row, col)

def get_cdist_row_col(idx, nr_cols):
    row = torch.div(idx, nr_cols, rounding_mode='floor')
    return row, idx-row*nr_cols

def plot_box(n_data, dumbell_dist, dumbells, title="Basic Plot"):
    left=[]
    right=[]
    for i in tqdm(range(len(dumbells))):
        rand_left = randrange(len(dumbells[i][0]))
        rand_right = randrange(len(dumbells[i][1]))
        left.append(dumbells[i][0][rand_left])
        right.append(dumbells[i][1][rand_right])

    pq_dist = torch.norm(n_data[left] - n_data[right], p=2, dim =1)

    diff = abs( dumbell_dist- pq_dist)/pq_dist

    diff = np.array(diff)
    fig1, ax1 = plt.subplots()
    ax1.set_title(title + f'med={np.median(diff)}, avg={np.mean(diff)}')
    ax1.boxplot(diff)
    ax1.boxplot(diff)

    
if __name__ == "__main__":
    pass
