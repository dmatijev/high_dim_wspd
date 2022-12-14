import os
import pandas as pd
import numpy as np
from tqdm import tqdm
#from scipy.spatial.distance import pdist
import torch
import torch.nn as nn
from my_utils import get_cdist_row_col
 


"""
def get_and_add_box(dumbell_indices,boxes,data, i):
    l,r = dumbell_indices[i]
    # loop loop loop loop
    lbox = np.array([[min(x),max(x)] for x in data[l].T])
    rbox = np.array([[min(x),max(x)] for x in data[r].T])
    boxes.append([lbox,rbox])
    #print_status(len(boxes), len(dumbell_indices))

"""

def get_and_add_box(dumbell_indices,boxes,data, i):
    l,r = dumbell_indices[i]
    # loop loop loop loop
    lbox = torch.tensor([[torch.min(x),torch.max(x)] for x in data[l].T])
    rbox = torch.tensor([[torch.min(x),torch.max(x)] for x in data[r].T])
    boxes.append([lbox,rbox])
    #print_status(len(boxes), len(dumbell_indices))
    
# runs concurrently
def get_and_add_sep(separations, boxes, i):
    l,r = boxes[i]
    # loop loop
    lc = (l[:,0] + l[:,1])/2
    lr = np.linalg.norm(l[:,0] - lc) 
    # loop loop
    rc = (r[:,0] + r[:,1])/2
    rr = np.linalg.norm(r[:,0] - rc)
    if lr == 0 and rr == 0:
        sep = -1
    else:
        sep = get_sep(lc,lr,rc,rr)
        separations.append(sep)
    

# get separation given centers and radii
def get_sep(lc,lr,rc,rr):
    r = max(lr,rr)
    if r == 0: return 0
    # loop
    #return max(0, np.linalg.norm(lc - rc)/r - 2)
    return np.linalg.norm(lc - rc)/r - 2


def info_to_dict(info):
    return { 
            's' : info[0],
            'lr' : info[1],
            'rr' : info[2]
          }

def compute_centroid(points):
    return torch.mean(points, 0)

 
def compute_centers(boxes, lcenters, rcenters, i):
    l, r = boxes[i]
    lcenters[i] = (l[:,0] + l[:,1])/2
    rcenters[i] = (r[:,0] + r[:,1])/2

def compute_wspd_centers(dumbells, data, i):
    l,r = dumbells[i]
    lbox = torch.tensor([[torch.min(x),torch.max(x)] for x in data[l].T])
    rbox = torch.tensor([[torch.min(x),torch.max(x)] for x in data[r].T])
    lcenter = (lbox[:,0] + lbox[:,1])/2
    rcenter = (rbox[:,0] + rbox[:,1])/2
    return (lcenter, rcenter)
    
def refine_dumbell(dumbells, data, i, new_dumbells):
    l, r = dumbells[i]
    left_dumbell_pts = X[l]
    right_dumbell_pts = X[r]
    dumbell_dist_mtx = torch.cdist(left_dumbell_pts, right_dumbell_pts, p=2)
    
def compute_avg_dist(dumbells, X, i):
    l, r = dumbells[i]
    left_dumbell_pts = X[l]
    right_dumbell_pts = X[r]
    dumbell_dist_mtx = torch.cdist(left_dumbell_pts, right_dumbell_pts, p=2)
    dumbell_dist = dumbell_dist_mtx.mean()
    
    return dumbell_dist


def compute_analytics_rand_pr(dumbells, lcenters, rcenters, X, verbose = False):
    dist_X_wspd_cen = []
    dist_X_wspd_avg = []
    for i in tqdm(range(len(lcenters))):
        lc = lcenters[i]        
        rc = rcenters[i]        
        lc_X_wspd, rc_X_wspd = compute_wspd_centers(dumbells, X, i)
        
        dist_X_wspd_cen.append(torch.norm(lc_X_wspd - rc_X_wspd))
        dist_X_wspd_avg.append(compute_avg_dist(dumbells, X, i))
           
    return dist_X_wspd_cen, dist_X_wspd_avg


def compute_max_min_dumbell_dist(left_dumbell_pts, right_dumbell_pts): 
    
    dumbell_dist_mtx = torch.cdist(left_dumbell_pts, right_dumbell_pts, p=2)
    dist_max, max_idx = dumbell_dist_mtx.max(), dumbell_dist_mtx.argmax()
    dist_min, min_idx = dumbell_dist_mtx.min(), dumbell_dist_mtx.argmin()
    nr_cols = dumbell_dist_mtx.shape[1]
    
    return (dist_max, dist_min, get_cdist_row_col(max_idx, nr_cols), get_cdist_row_col(min_idx, nr_cols))

def split_set(dumbell_pts, pt1, pt2, new_l1, new_l2):
    for idx in range(len(dumbell_pts)): 
        d1 = torch.dist(dumbell_pts[idx], pt1)
        d2 = torch.dist(dumbell_pts[idx], pt2)
        if d1 < d2:
            new_l1.append(dumbell_pts[idx])
        else:
            new_l2.append(dumbell_pts[idx])

def refine_dumbell(left_dumbell_pts, right_dumbell_pts, max_pair, min_pair, new_dumbells): 
    
 
    l_p1, l_p2 = left_dumbell_pts[max_pair[0]], left_dumbell_pts[min_pair[0]]
    r_p1, r_p2 = right_dumbell_pts[max_pair[1]], right_dumbell_pts[min_pair[1]]
    
    if torch.dist(l_p1, l_p2, p=2) > torch.dist(r_p1, r_p2, p=2): # split left dumbell
        new_l1 = []
        new_l2 = []
            
        split_set(left_dumbell_pts, l_p1, l_p2, new_l1, new_l2)   
        new_dumbells.append([torch.stack(new_l1), right_dumbell_pts])
        new_dumbells.append([torch.stack(new_l2), right_dumbell_pts])
    else: #split right dumbell
        new_l1 = []
        new_l2 = []
        
        split_set(right_dumbell_pts, r_p1, r_p2, new_l1, new_l2)
        new_dumbells.append([left_dumbell_pts, torch.stack(new_l1)])
        new_dumbells.append([left_dumbell_pts, torch.stack(new_l2)])                            
       
    return len(new_dumbells)


def analyse_dumbells(dumbells, X, S=2, refine = False, refine_depth = 1, verbose=False):

    X = X.cpu()
    input_dim = X.shape[1]
    
    wspd_dist = 0
    wspd_dist_X = 0
    wspd_dist_X_wspd = 0
    wspd_dist_X_avg = 0

    dumbell_sets = [(X[left_pts], X[right_pts]) for (left_pts,right_pts) in dumbells]
   
     
    
    
    for repeat in range(refine_depth):
        for_delete = []
        new_dumbells = []

        for i, (left_dumbell_pts, right_dumbell_pts) in tqdm(enumerate(dumbell_sets)):
           
            (d_max, d_min, max_pair, min_pair) = compute_max_min_dumbell_dist(left_dumbell_pts, right_dumbell_pts)

            if refine:
                trashold = (1+4/S)
            else:
                trashold = float('inf') # don't refine dumbells 

            if d_max/d_min >= trashold:
                refine_dumbell(left_dumbell_pts, right_dumbell_pts, max_pair, min_pair, new_dumbells)
                for_delete.append(i)
    
        if verbose:
            print(f"depth = {repeat}, deleted dumbells: {len(for_delete)}, added dumbells: {len(new_dumbells)}")
        # delete dumbells from dumbell_sets
        for_delete.sort(reverse = True)
        for i in for_delete:
            dumbell_sets.pop(i)
        
        dumbell_sets.extend(new_dumbells)
        
        if len(for_delete) == 0:
            break

    dist_X_wspd_diff_avg = []
    for i in tqdm(range(len(dumbell_sets))):
        (d_max, d_min, max_pair, min_pair) = compute_max_min_dumbell_dist(dumbell_sets[i][0], dumbell_sets[i][1])
        dist_X_wspd_diff_avg.append(d_max/d_min)

    return dist_X_wspd_diff_avg

 

"""
def compute_analytics(dumbells, lcenters, rcenters, X, encoder, decoder, device, ac_fn, verbose=False):
    X = X.cpu()
    pX = encoder(X)
    #rX = decoder(ac_fn(pX))
    
    wspd_dist = 0
    wspd_dist_X = 0
    wspd_dist_X_wspd = 0
    wspd_dist_X_avg = 0
    
    if verbose:
        print("Computing wspd distances....")
    
    with torch.no_grad():
        lcenters_X = decoder(lcenters)
        rcenters_X = decoder(rcenters)
    
    
    # allocate memory for centers 
  
    #lcenters_X_wspd = torch.empty(lcenters_X.shape)
    #rcenters_X_wspd = torch.empty(lcenters_X.shape)
    #boxes = []
    #print("computing boxes...") 
  
    #[get_and_add_box(dumbells,boxes, X ,i) for i in tqdm(range(len(dumbells)))]
    #print("Computing centers...")
    #[compute_centers(boxes, lcenters_X_wspd, rcenters_X_wspd, i) for i in tqdm(range(len(boxes)))]
   
    dist_X_wspd_cen = []
    dist_X_wspd_avg = []
    print("Compute wspd distances (from centers to centers)...")
    for i in tqdm(range(len(lcenters))):
        lc = lcenters[i]        
        rc = rcenters[i]        
        lc_X = lcenters_X[i]
        rc_X = rcenters_X[i]
        
   
        
        lc_X_wspd, rc_X_wspd = compute_wspd_centers(dumbells, X, i)
        #wspd_dist_X_avg += compute_avg_dist(dumbells, X, i)
        
        dist_X_wspd_cen.append(torch.norm(lc_X_wspd - rc_X_wspd))
        dist_X_wspd_avg.append(compute_avg_dist(dumbells, X, i))
        
        
        nr_l = len(dumbells[i][0])
        nr_r = len(dumbells[i][1])

        
        wspd_dist += (nr_l*nr_r)*torch.norm(lc - rc)
        wspd_dist_X += (nr_l*nr_r)*torch.norm(lc_X - rc_X)
        #wspd_dist_X_wspd += (nr_l*nr_r)*torch.norm(lc_X_wspd - rc_X_wspd)
        

    if verbose:
        print("Computing distances...")


    dist_X = torch.sum(torch.pdist(X))
    dist = torch.sum(torch.pdist(pX))
    #dist_rX = torch.sum(torch.pdist(rX))

    print("dist_pX =      ", dist)
    print("wspd_dist_pX = ", wspd_dist)
    print("dist_X =      ", dist_X)
    print("wspd_dist_X = ", wspd_dist_X)
    #print("wspd_dist_X_WSPD = ", wspd_dist_X_wspd)

    print("relative error in projected space: ", abs(dist-wspd_dist)/dist)
    print("relative error in original  space (reconstructed centers): ", abs(dist_X-wspd_dist_X)/dist_X)
    #print("relative error in original  space (wspd centers): ", abs(dist_X-wspd_dist_X_wspd)/dist_X)


    
    return dist_X_wspd_cen, dist_X_wspd_avg

"""

def run_wspd(pX, verbose = False, S = 2):
    """
    * Compute WSPD of pX with separation s
    * for every dumbell (A,B) in projected space, compute the separation constant s
    * Identify the separation s' of (A,B) in original space 
    * define Loss:  \sum (s' - s)^2, where sum is over all dumbells. 
    """
    # save transformed data to hdd such that WSPD can read it
    pX_np = pX.cpu().numpy() 
    #X_np = X.cpu().numpy()
    
    df = pd.DataFrame(pX_np) #convert to a dataframe
    file_name = 'temp.tsv'
    with open(file_name, "w") as text_file:
        text_file.write(f'{pX.shape[0]}\t{pX.shape[1]}\n')
    df.to_csv(file_name,index=False, header = False, sep = "\t", mode='a') 
    
    # run wspd on transformed data
    import time

    start = time.time()
    

    #out = os.system('./wsp ' + 'temp.tsv' + ' ' + str(S)
    #                   + ' 1>/dev/null 2>/dev/null')
    out = os.system('./wsp ' + 'temp.tsv' + ' ' + str(S))
    assert out == 0, "Fail to run WSD computation"
    end = time.time()
    if verbose: 
        print(f'wspd done in {end - start} secs')
        
    # read wspd result
    dumbell_indices = []
    #low_d_info = []
    #separations = []
    left_centers = []
    right_centers = []
    # dumbell_indices: s lc lr rc rr | l1 l2 ... | r1 r2 ...
    with open(file_name +  '.wsp_out.txt') as f:
        for line in f:
            lc, rc,l,r = line.split('|')
            #low_d_info.append(info_to_dict([float(x) for x in i.split()]))            
            lc = np.array([float(x) for x in lc.split()])
            rc = np.array([float(x) for x in rc.split()])            
            left_centers.append(lc)
            right_centers.append(rc)
            dumbell_indices.append([[int(x) for x in l.split()],[int(y) for y in r.split()]])
    if verbose: 
        print('wspd contains ', len(dumbell_indices), ' dumbells')
    
    return torch.tensor(np.array(left_centers), dtype=torch.float32), torch.tensor(np.array(right_centers), dtype = torch.float32), dumbell_indices

    
   
"""
    # {{{  compute separations for WS dumbells in original X space
    boxes = []
    [get_and_add_box(dumbell_indices,boxes,X_np ,i) for i in range(len(dumbell_indices))]

    separations_in_X = []
    [get_and_add_sep(separations_in_X, boxes, i) for i in range(len(boxes))]
    if verbose:
        print('separations and radii computed for all non-singletones dumbells in original space')
        
    #}}}
    
    #return MSELOSS(separations, separations_in_X)

"""