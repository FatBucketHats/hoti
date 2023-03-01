#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 09:39:26 2023

@author: jopo
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import time
import matplotlib.pyplot as plt
import math
from PIL import Image, ImageDraw
from main import hoti_hamiltonian_rect_n as hamiltonian
from main import gen_equipoly as equipoly

# Bi2Te3
A=2; m0=-1; m2=0.19; B=3
 
# Global constants
MU = m2*abs(m0)/(A**2)
BETA = B*abs(m0)/(A**2)
R0 = A/abs(m0)
V = 100

# path = "../data/"
path = "/Users/jopo/Documents/23/work/ti/work_data/hoti_equilat_polygon"

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    no_vert = 4
    side_len = 10
    dn = 0.2
    theta = 0
    
    vert = equipoly(no_vert, side_len, dn, theta)
    # nx, ny, h = hamiltonian(vert, dn)
    ###########################################################
    # Start time
    current_time = time.strftime("%H:%M:%S", time.localtime())
    print(f"Beginning hoti_hamiltonian_square -> current time: {current_time}")
    t1 = time.time()
    
    ## Setup mask
    # Round vert to integer
    vert = vert/dn
    vert = vert.astype('int')
    vert = vert.ravel().tolist()
    
    # Calculate nx, ny and create mask
    nx = math.ceil(max(vert[::2]))+1
    ny = math.ceil(max(vert[1::2]))+1
    n = max(ny, nx)
    nx = n 
    ny = n
    img = Image.new('L', (nx, ny), 0)
    ImageDraw.Draw(img).polygon(vert, outline=1, fill=1)
    mask = np.array(img)
    
    # Show mask
    fig_temp, ax_temp = plt.subplots()
    im = ax_temp.imshow(mask)
    fig_temp.colorbar(im, ax=ax_temp, label='Colorbar')
    plt.show()
    
    # Flatten mask for processing
    mask = mask.flatten()
    
    ## Setup hamiltonian
    # Differential operators, d1 and d2
    diagx = np.ones(nx)
    diags1x = np.array([diagx, -diagx]) / dn
    d1x = sp.spdiags(diags1x, (1, -1))
    diags2x = np.array([diagx, -2 * diagx, diagx]) / dn ** 2
    d2x = sp.spdiags(diags2x, (1, 0, -1))

    diagy = np.ones(ny)
    diags1y = np.array([diagy, -diagy]) / dn
    d1y = sp.spdiags(diags1y, (1, -1))
    diags2y = np.array([diagy, -2 * diagy, diagy]) / dn ** 2
    d2y = sp.spdiags(diags2y, (1, 0, -1))

    # Hamiltonian
    h = (sp.kron(sp.diags([1, -1, 1, -1]), -(sp.identity(nx * ny) + MU*sp.kronsum(d2y, d2x)))
          + sp.kron(sp.diags([[1, 0, 0], [0, 0, 1]], [1, -1]), sp.kronsum(-d1y, -1j*d1x))
          + sp.kron(sp.diags([[1, 0, 0], [0, 0, 1]], [-1, 1]), sp.kronsum(d1y, 1j*d1x))
          + sp.kron(sp.diags([[1], [0, -1, 0], [0, 1, 0], [-1]], [-3, -1, 1, 3]), -1j * BETA * sp.kronsum(d2y, -d2x)))
    
    # Apply bc's
    mask = np.tile(mask, 4)
    h = h.multiply(mask)
    h = h.multiply(np.reshape(mask,(4*n**2,1)))
    diag = 1 - mask # Flip 0's and 1's in mask
    v = diag*V 
    h.setdiag(h.diagonal() + v)
    
    # End
    t2 = time.time()
    print(f"\tFinished -> time elapsed: {t2-t1} s")
    ###########################################################
    
    print(f"nx: {nx} \nny: {ny}")
    
    # Eigen-problem solver
    current_time = time.strftime("%H:%M:%S", time.localtime())
    t1 = time.time()
    print(f"Beginning eigsh -> current time: {current_time}")
    #eigenvalues, eigenvectors = eigsh(h, sigma=1e-15, which='LM')
    eigenvalues, eigenvectors = eigsh(h, sigma = 0, which='LM', k=10)
    t2 = time.time()
    print(f"\tFinished -> time elapsed: {t2-t1} s")
    
    
    # Normalise probability density
    wfs = np.transpose(eigenvectors)
    norm = np.sum(abs(wfs), axis=1, keepdims=1)
    wfs_normalised = wfs/norm
    no_eigen = np.shape(wfs_normalised)[0]
    p = np.sum(np.reshape(abs(wfs_normalised)**2, (no_eigen, 4, nx*ny)),axis=1)

    print(*eigenvalues, sep='\n')
    
    # Save
    path = "/Users/jopo/Documents/23/work/ti/work_data/hoti_equilat_polygon"
    for i in np.arange(no_eigen):    
        np.savetxt(path+f"/e{no_vert}_len{side_len}_{i}.csv", np.reshape(p[i], (ny, nx)), delimiter=",")
    
    # Plot
    fig, ax = plt.subplots()
    im = ax.imshow(np.reshape(p[0], (ny, nx)))
    fig.colorbar(im, ax=ax, label='Colorbar')
    plt.show()