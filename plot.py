#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 01:05:54 2023

@author: jopo
"""
import numpy as np
import matplotlib.pyplot as plt

n=30
i=1

no_vert = 3
side_len = 10


path = "/Users/jopo/Documents/23/work/ti/work_data/hoti_equilat_polygon"
paths = path+f"/e{no_vert}_len{side_len}_{i}.csv"

fig, ax = plt.subplots()
# p = np.loadtxt(f"{n}x{n}square_{i}.csv", delimiter=",")
p = np.loadtxt(paths, delimiter=",")
im = ax.imshow(p)
fig.colorbar(im, ax=ax, label='Colorbar')
plt.show()
