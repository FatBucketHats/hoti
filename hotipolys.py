#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Numerically solve the 2D schrodinger equation for a Hamiltonian describing 
a higher order topologically insulating (hoti) geometry.

This is done by discretizing the 2D domain, within which our hoti geometry is 
defined using p coordinates. These coordinates describe the verticies of the 
polygon and are connected sequentially. Within the corresponding Hamiltonian
the bc's are applied by disconnecting points outside the boundary and setting 
the potential there to v, avoiding the lowest energy states residing outside 
the hoti. Unless specified, abs(m0) and the compton length will be used to 
non-dimensionalise our implimentaions. These give the energy and length scales, 
respectively. If this script is run, it will test our implimentation for the 
square (simplest case).
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import time
import matplotlib.pyplot as plt
from matplotlib.path import Path

# Global params for model
A=2; m0=-1; m2=0.19; B=3
 
#  constants
MU = m2*abs(m0)/(A**2)
BETA = B*abs(m0)/(A**2)
R0 = A/abs(m0)
V = 1e9

# Paths for personal and cluster
path = "/Users/jopo/Documents/23/work/ti/work_data/hoti_equilat_polygon"
#path = "../data" 

def h_square(n, dn):
    """Generate dimensionless hoti Hamiltonian for square 
    
    Simplest implementation. Effectively this hamiltonian is written in the 
    basis s_kron_y_kron_x with s, y and x labeling the spinor and 
    spatial components.
    
    :param n: dimension of nxn domain
    :param dn: discretization param
    :returns: 
    """
    current_time = time.strftime("%H:%M:%S", time.localtime())
    print(f"Beginning hoti_hamiltonian_square -> current time: {current_time}")
    t1 = time.time()
    
    # Differential operators, d1 and d2
    diag = np.ones(n)
    diags1 = np.array([diag, -diag]) / (2*dn)
    d1 = sp.spdiags(diags1, (1, -1))
    diags2 = np.array([diag, -2 * diag, diag]) / (dn ** 2)
    d2 = sp.spdiags(diags2, (1, 0, -1))

    # Hamiltonian
    h = (sp.kron(sp.diags([1, -1, 1, -1]), -(sp.identity(n ** 2) + MU*sp.kronsum(d2, d2)))
         + sp.kron(sp.diags([[1, 0, 0], [0, 0, 1]], [1, -1]),  sp.kronsum(-1j*d1, -d1))
         + sp.kron(sp.diags([[1, 0, 0], [0, 0, 1]], [-1, 1]),  sp.kronsum(-1j*d1, d1))
         + sp.kron(sp.diags([[1], [0, -1, 0], [0, 1, 0], [-1]], [-3, -1, 1, 3]), -1j * BETA * sp.kronsum(-d2, d2)))

    # End
    t2 = time.time()
    print(f"\tFinished -> time elapsed: {t2-t1} s")
    return h

def h_test(n, dn):
    """Generate dimensionless hoti Hamiltonian for square 
    
    Simplest implementation. Effectively this hamiltonian is written in the 
    basis s_kron_y_kron_x with s, y and x labeling the spinor and 
    spatial components.
    
    :param n: dimension of nxn domain
    :param dn: discretization param
    :returns: 
    """
    current_time = time.strftime("%H:%M:%S", time.localtime())
    print(f"Beginning hoti_hamiltonian_square -> current time: {current_time}")
    t1 = time.time()
    
    # Differential operators, d1 and d2
    diag = np.ones(n)
    diags1 = np.array([diag, -diag]) / (2*dn)
    d1 = sp.spdiags(diags1, (1, -1))
    diags2 = np.array([diag, -2 * diag, diag]) / (dn ** 2)
    d2 = sp.spdiags(diags2, (1, 0, -1))

    # Hamiltonian
    h = (sp.kron(sp.diags([1, -1, 1, -1]), -(sp.identity(n ** 2) + MU*sp.kronsum(d2, d2)))
         + sp.kron(sp.diags([[1, 0, 0], [0, 0, 1]], [1, -1]),  sp.kronsum(-1j*d1, -d1))
         + sp.kron(sp.diags([[1, 0, 0], [0, 0, 1]], [-1, 1]),  sp.kronsum(-1j*d1, d1)))

    # End
    t2 = time.time()
    print(f"\tFinished -> time elapsed: {t2-t1} s")
    return h

def h_square_units(n, dn):
    """Generate unscaled hoti Hamiltonian for square (NOT WORKING)
    
    Simplest implementation. Effectively this hamiltonian is written in the 
    basis s_kron_y_kron_x with s, y and x labeling the spinor and 
    spatial components.
    :param n: dimension of nxn domain
    :param dn: discretization param
    :returns: 
    """
    current_time = time.strftime("%H:%M:%S", time.localtime())
    print(f"Beginning h_square -> current time: {current_time}")
    t1 = time.time()

    # Differential operators, d1 and d2
    diag = np.ones(n)
    diags1 = np.array([diag, -diag]) / (2*dn)
    d1 = sp.spdiags(diags1, (1, -1))
    diags2 = np.array([diag, -2 * diag, diag]) / dn ** 2
    d2 = sp.spdiags(diags2, (1, 0, -1))

    # Hamiltonian
    h = (sp.kron(sp.diags([1, -1, 1, -1]), (m0*sp.identity(n ** 2) - m2*sp.kronsum(d2, d2)))
         + sp.kron(sp.diags([[1, 0, 0], [0, 0, 1]], [1, -1]), A*sp.kronsum(-d1, -1j*d1))
         + sp.kron(sp.diags([[1, 0, 0], [0, 0, 1]], [-1, 1]),  A*sp.kronsum(d1, -1j*d1))
         + sp.kron(sp.diags([[1], [0, -1, 0], [0, 1, 0], [-1]], [-3, -1, 1, 3]), -1j * B * sp.kronsum(d2, -d2)))

    # End
    t2 = time.time()
    print(f"\tFinished -> time elapsed: {t2-t1} s")
    return h

def h_rect(nx, ny, dn):
    """Generate dimensionless hoti Hamiltonian for rectangle in deiscretization 
    grid with dim nx x ny
    
    Simplest implementation. Effectively this hamiltonian is written in the 
    basis s_kron_y_kron_x with s, y and x labeling the spinor and 
    spatial components.
    :param n: dimension of nxn domain
    :param dn: discretization param
    :returns: 
    """

    # Differential operators, d1 and d2
    diagx = np.ones(nx)
    diags1x = np.array([diagx, -diagx]) / (2*dn)
    d1x = sp.spdiags(diags1x, (1, -1))
    diags2x = np.array([diagx, -2 * diagx, diagx]) / dn ** 2
    d2x = sp.spdiags(diags2x, (1, 0, -1))
    
    diagy = np.ones(ny)
    diags1y = np.array([diagy, -diagy]) / (2*dn)
    d1y = sp.spdiags(diags1y, (1, -1))
    diags2y = np.array([diagy, -2 * diagy, diagy]) / dn ** 2
    d2y = sp.spdiags(diags2y, (1, 0, -1))
    
    # Hamiltonian
    h = (sp.kron(sp.diags([1, -1, 1, -1]), -(sp.identity(nx * ny) + MU*sp.kronsum(d2x, d2y)))
         + sp.kron(sp.diags([[1, 0, 0], [0, 0, 1]], [1, -1]), sp.kronsum(-1j*d1x, -d1y))
         + sp.kron(sp.diags([[1, 0, 0], [0, 0, 1]], [-1, 1]), sp.kronsum(-1j*d1x, d1y))
         + sp.kron(sp.diags([[1], [0, -1, 0], [0, 1, 0], [-1]], [-3, -1, 1, 3]), -1j * BETA * sp.kronsum(-d2x, d2y)))
    
    return h

def h_poly(vert, dn, equal_dims = True, show = False, buffer = 0):
    """Generate unscaled hoti Hamiltonian for square  rectangle that fits polygon.
       Create mask and apply to h (apply bc's).
       :param vert:  [x1,y1,x2,y2,...]
       :param equal_dims: if True discretization grid dim is n x n, otherwise
       it is nx x ny
       :param dn: discretization param
       """
    # Start time
    current_time = time.strftime("%H:%M:%S", time.localtime())
    print(f"Beginning h_poly -> current time: {current_time}")
    t1 = time.time()
    
    ## Setup mask
    # Convert verticies to indicies on discretization grid
    vert /= dn
    vert = vert.astype(int)
    
    # Calculate nx, ny and create mask
    if equal_dims:
        nx = int(max(vert[:,0]))+1 + 2*buffer
        ny = int(max(vert[:,1]))+1 + 2*buffer
        nx = ny = max(ny, nx)
    else: 
        nx = int(max(vert[:,0]))+1
        ny = int(max(vert[:,1]))+1 
    
    # Shift by buffer
    vert += buffer
    
    # Create mask
    path = Path(vert)
    x, y = np.meshgrid(np.arange(nx), np.arange(ny))
    x, y = x.flatten(), y.flatten()
    points = np.vstack((x,y)).T
    mask = path.contains_points(points, radius=dn/3)
    mask = mask.reshape((ny,nx)).astype(int)
    
    # Show mask
    if show:
        fig_temp, ax_temp = plt.subplots()
        im = ax_temp.imshow(mask)
        fig_temp.colorbar(im, ax=ax_temp)
    
    # Flatten mask for processing
    mask_tmp=mask
    mask = mask.flatten()
    
    # Hamiltonian
    h = h_rect(nx, ny, dn)
    
    # Apply bc's
    mask = np.tile(mask, 4)
    h = h.multiply(mask)
    h = h.multiply(np.reshape(mask,(4*nx*ny,1)))
    diag = 1 - mask # Flip 0's and 1's in mask
    v = diag*V 
    h.setdiag(h.diagonal() + v)
    
    # End
    t2 = time.time()
    print(f"\tFinished -> time elapsed: {t2-t1} s")
    return nx, ny, h, mask_tmp

def h_circle(r, dn, show = False, buffer = 0):
    """Generate unscaled hoti Hamiltonian for disk
       Create mask and apply to h (apply bc's).
       :param vert:  [x1,y1,x2,y2,...]
       :param equal_dims: if True discretization grid dim is n x n, otherwise
       it is nx x ny
       :param dn: discretization param
       """
    # Start time
    current_time = time.strftime("%H:%M:%S", time.localtime())
    print(f"Beginning h_circle -> current time: {current_time}")
    t1 = time.time()
    
    ### Setup mask
    # Convert r to index on discretization grid
    r /= dn
    path = Path.circle(center = (r,r), radius = r)
    n = 2*r + 1
    n = int(n)
    show = True
    # Create mask
    
    x, y = np.meshgrid(np.arange(n), np.arange(n))
    x, y = x.flatten(), y.flatten()
    points = np.vstack((x,y)).T
    mask = path.contains_points(points, radius=dn/3)
    mask = np.reshape(mask,(n,n)).astype(int)
    
    # Show mask
    if show:
        fig_temp, ax_temp = plt.subplots()
        im = ax_temp.imshow(mask)
        fig_temp.colorbar(im, ax=ax_temp)
    
    # Flatten mask for processing
    mask_tmp=mask
    mask = mask.flatten()
    
    # Hamiltonian
    h = h_test(n, dn)
    
    # Apply bc's
    mask = np.tile(mask, 4)
    h = h.multiply(mask)
    h = h.multiply(np.reshape(mask,(4*n*n,1)))
    diag = 1 - mask # Flip 0's and 1's in mask
    v = diag*V 
    h.setdiag(h.diagonal() + v)
    
    # End
    t2 = time.time()
    print(f"\tFinished -> time elapsed: {t2-t1} s")
    return n, h, mask_tmp

def gen_equipoly(no_vert, side_len, dn, alpha=0):
    """Generate coordinates of verticies describing equilateral polygon
    :param no_vert: number of verticies
    :param side_len: side length
    :param alpha: angle of rotation (deg), zero defines polygon with one edge 
    aligned with x axis. 
    :returns: np.array([x1, y1, x2, y2...])
    """
    
    # Exception
    if no_vert < 3:
        raise Exception("A polygon must have at least 3 verticies.")
    
    # Auxillary calcs
    deg_to_rad = np.pi / 180
    alpha = alpha * deg_to_rad
    theta = deg_to_rad * 360 / no_vert
    
    # Initialise returned variable, [[x1, y1], [x2, y2], ...]
    vert = [[0,0]]
    
    # Coords
    for i in np.arange(no_vert - 1):
        x = (vert[i][0] 
             + side_len * np.cos(theta * i + alpha))
        y = (vert[i][1]
             + side_len * np.sin(theta * i + alpha))
        vert += [[x, y]]
    
    # Convert to numpy array
    vert = np.array(vert)
    
    # Shifts
    x_shift = min(vert[:,0])
    y_shift = min(vert[:,1])
    vert[:,0] -= x_shift
    vert[:,0] -= y_shift
    
    return vert

def eigen_solve(h, which="SM", k=8):
    current_time = time.strftime("%H:%M:%S", time.localtime())
    t1 = time.time()
    print(f"Beginning eigsh -> current time: {current_time}")
    if which=="SM":
        eigenvalues, eigenvectors = eigsh(h, which='SM', k=k)
    elif which=="LM":
        eigenvalues, eigenvectors = eigsh(h, sigma=0, which='LM', k=k)
    else:
        raise Exception("mode must be LM or SM")
    t2 = time.time()
    print(f"\tFinished -> time elapsed: {t2-t1} s")
    
    return eigenvalues, eigenvectors    

# # Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Params
    L = 10
    dn = 0.2
    n = int(L/dn)
    mode = "LM"
    
    # Setup Hamiltonian and solve
    h = h_square(n, dn)
    evalues, evectors = eigen_solve(h, mode)

    # Normalise probability density
    wfs = np.transpose(evectors)
    no_eigen = np.shape(wfs)[0]
    p = np.sum(np.reshape(abs(wfs)**2, (no_eigen, 4, n, n)),axis=1)
    
    # Output
    print(*evalues, sep='\n')
    for i in np.arange(no_eigen):    
        np.savetxt(path + f"/{n}x{n}square_{i}.csv", p[i], delimiter=",")
    
    # Plot
    fig, ax = plt.subplots()
    # im = ax.imshow(np.sum(p, axis=0))
    im = ax.imshow(p[0])
    fig.colorbar(im, ax=ax)
    plt.show()