# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 09:28:08 2019

@author: domagoj
"""

import  numpy as np
import argparse
import sys


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nr_points', type=int, required=True)
    parser.add_argument('--dim', type=int, required=True)   
    parser.add_argument('--dist', type=str, required=True) # normal, gumbel, laplace, logistic
    parser.add_argument('--loc', type=float, required=False)
    parser.add_argument('--scale', type=float, required=False)

    return parser
def generate_data(distribution, nr_points, dim, loc =0.0, scale=1.0):
    np.random.seed(42)
    if distribution == 'normal':
        pt = np.random.normal(size = (nr_points, dim), loc = loc, scale = scale)
    elif distribution == 'gumbel':
        pt = np.random.gumbel(size = (nr_points, dim), loc = loc, scale = scale)
    elif distribution == 'laplace':
        pt = np.random.laplace(size = (nr_points, dim), loc = loc, scale = scale)
    elif distribution == 'logistic':
        pt = np.random.logistic(size = (nr_points, dim), loc = loc, scale = scale)
    elif distribution == 'uniform':
        pt = np.random.uniform(low=loc, high=scale, size=(nr_points, dim))
    else:
        print("Proper distribution should be selected : uniform, normal, gumbel, laplace, logistic")
        sys.exit()
        
    return pt

if __name__ == "__main__":
    args = get_parser().parse_args()
    
    nr_points = args.nr_points
    dim = args.dim
    distribution = args.dist
    
    loc  = args.mean if args.loc else 0.0
    scale = args.st_dev if args.scale else 1.0
    
    pt = generate(distribution, nr_points, dim, loc, scale)

    
    
    print("{}\t{}".format(nr_points, dim))
    for r in range(nr_points):
        for c in range(dim):
            if c < dim-1:
                print("{0:.4f}".format(pt[r][c]), end='\t') 	
            elif r < nr_points-1:
                print("{0:.4f}".format(pt[r][c]))
            else:
                print("{0:.4f}".format(pt[r][c]), end='')


