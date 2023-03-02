#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 01:05:54 2023

@author: jopo
"""
import numpy as np
import matplotlib.pyplot as plt

n=81
i=3

no_vert = 3
side_len = 20

path = "/Users/jopo/Documents/23/work/ti/work_data/hoti_equilat_polygon"
#path = "../data"

#typ = f"/e{no_vert}_len{side_len}_{i}.csv"
typ = f"/{n}x{n}square_{i}.csv"


p = np.zeros((n,n))
# for i in np.arange(4):
#     p += np.loadtxt(f"{n}x{n}square_{i}.csv", delimiter=",")

for i in np.arange(4):
    p += np.loadtxt(path+typ, delimiter=",")

# p = np.loadtxt(paths, delimiter=",")
fig, ax = plt.subplots()
im = ax.imshow(p)
fig.colorbar(im, ax=ax, label='Colorbar')
plt.show()
