#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 11:49:00 2023

@author: jopo
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.sparse.linalg import eigsh
import math 
import scipy.sparse as sp
from main import hoti_hamiltonian_rect as hamiltonian
from main import gen_equipoly as equipoly
from PIL import Image, ImageDraw

# Bi2Te3
A=4.003; m0=-0.296; m2=177.355; B=0.9
 
# Global constants
MU = m2*abs(m0)/(A**2)
BETA = B*abs(m0)/(A**2)
R0 = A/abs(m0)
V = 100


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    no_vert = 3
    side_len = 5
    dn = 0.2
    theta = 0
    
    vert = equipoly(no_vert, side_len, dn, theta)
    nx, ny, h = hamiltonian(vert, dn)
    print(f"nx: {nx} \nny: {ny}")
    
    # Eigen-problem solver
    current_time = time.strftime("%H:%M:%S", time.localtime())
    t1 = time.time()
    print(f"Beginning eigsh -> current time: {current_time}")
    eigenvalues, eigenvectors = eigsh(h, sigma=0.0000001, which='LM')
    t2 = time.time()
    print(f"\tFinished -> time elapsed: {t2-t1} s")
    
    
    # Normalise probability density
    wfs = np.transpose(eigenvectors)
    norm = np.sum(abs(wfs), axis=1, keepdims=1)
    wfs_normalised = wfs/norm
    no_eigen = np.shape(wfs_normalised)[0]
    p = np.sum(np.reshape(abs(wfs_normalised)**2, (no_eigen, 4, nx*ny)),axis=1)

    
    # Save
    path = "/Users/jopo/Documents/23/work/ti/work_data/hoti_equilat_polygon"
    for i in np.arange(no_eigen):    
        np.savetxt(path+f"/e{no_vert}_len{side_len}_{i}.csv", np.reshape(p[i], (ny, nx)), delimiter=",")
    
    # Plot
    fig, ax = plt.subplots()
    im = ax.imshow(np.reshape(p[0], (nx, ny)))
    fig.colorbar(im, ax=ax, label='Colorbar')
    plt.show()