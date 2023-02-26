#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 01:05:54 2023

@author: jopo
"""
import numpy as np
import matplotlib.pyplot as plt

n=30
i=4
fig, ax = plt.subplots()
p = np.loadtxt(f"{n}x{n}square_{i}.csv", delimiter=",")
im = ax.imshow(p)
fig.colorbar(im, ax=ax, label='Colorbar')
plt.show()
