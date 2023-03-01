#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 09:39:26 2023

@author: jopo
"""
import numpy as np
import matplotlib.pyplot as plt
from hotipolys import h_circle as hamiltonian
from hotipolys import eigen_solve


# path = "../data/"
# path = "/Users/jopo/Documents/23/work/ti/work_data/hoti_equilat_polygon"

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Params
    r = 50
    dn = 0.2
    mode = "LM"
    buffer = 0
    
    # Function calls
    n, h, mask = hamiltonian(r, dn, show = True)
    evalues, evectors = eigen_solve(h, mode)
    
    
    # Modulus squared
    wfs = np.transpose(evectors)
    no_eigen = np.shape(wfs)[0]
    p = np.sum(np.reshape(abs(wfs)**2, (no_eigen, 4, n, n)),axis=1)
    
    # Sort
    sortinds = np.argsort(abs(evalues))
    evalues = evalues[sortinds]
    evectors = evectors[sortinds]
    
    ################# Outputs #################
    print(f"n: {n}")
    print(*evalues, sep='\n')
    
    # Saves
    path = "/Users/jopo/Documents/23/work/ti/work_data/hoti_equilat_polygon"
    for i in np.arange(no_eigen):    
        np.savetxt(path+f"/circle_r{r}_{i}.csv", np.reshape(p[i], (n, n)), delimiter=",")
    
    # Plot
    fig, ax = plt.subplots()
    p = np.sum(p[:4], axis=0)
    im = ax.imshow(p)
    #im = ax.imshow(p[0])
    fig.colorbar(im, ax=ax, label='Colorbar')
    plt.show()