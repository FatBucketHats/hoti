# -*- coding: utf-8 -*-
"""Numerically solve the 2D schrodinger equation for a Hamiltonian describing a higher order topologically
insulating (hoti) geometry.

This is done by discretizing the 2D domain, within which our hoti geometry is defined using p coordinates. These
coordinates describe the verticies of the polygon and are connected sequentially. Within the corresponding Hamiltonian
the bc's are applied by disconnecting points outside the boundary and setting the potential there to v, avoiding
the lowest energy states residing outside the hoti. This problem is scaled using abs(m0) and the compton length for our
energies and lengths, respectively.
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import time
import matplotlib.pyplot as plt


def hoti_hamiltonian_square(n, dn):
    """Generate hoti Hamiltonian for square (simplest implementation). Effectively this hamiltonian is written in the
    basis s_kron_y_kron_x with s, y and x labeling the spinor and spatial components.
    :param n: dimension of nxn domain
    :param dn: discretization param
    :returns: 
    """
    current_time = time.strftime("%H:%M:%S", time.localtime())
    print(f"Beginning hoti_hamiltonian_square -> current time: {current_time}")
    t1 = time.time()
    # Constants
    mu = 1
    beta = 1
    # Differential operators, d1 and d2
    diag = np.ones(n)
    diags1 = np.array([diag, -diag]) / dn
    d1 = sp.spdiags(diags1, (1, -1))
    diags2 = np.array([diag, -2 * diag, diag]) / dn ** 2
    d2 = sp.spdiags(diags2, (1, 0, -1))

    # Hamiltonian
    h = (sp.kron(sp.diags([1, -1, 1, -1]), -(sp.identity(n ** 2) + sp.kronsum(d2, d2)))
         + sp.kron(sp.diags([[1, 0, 0], [0, 0, 1]], [1, -1]), -1j * sp.kronsum(-d1, d1))
         + sp.kron(sp.diags([[1, 0, 0], [0, 0, 1]], [-1, 1]), -1j * sp.kronsum(d1, d1))
         + sp.kron(sp.diags([[1], [0, -1, 0], [0, 1, 0], [-1]], [-3, -1, 1, 3]), -1j * beta * sp.kronsum(d1, -d1)))

    # End
    t2 = time.time()
    print(f"\tFinished -> time elapsed: {t2-t1} s")
    return h


def gen_equipoly(no_vert, side_len, alpha=0):
    """Generate coordinates of verticies describing equilateral polygon
    :param no_vert: number of verticies
    :param side_len: side length
    :param alpha: angle of rotation, zero defines polygon with one edge aligned with x axis
    """
    # Exception
    if no_vert < 3:
        raise Exception("A polygon must have at least 3 verticies.")

    # Initialise returned variable, [x1, y1, x2, y2...]
    vert = np.zeros(2 * no_vert)

    # Generate coords
    theta = 360 / no_vert
    for i in np.arange(no_vert - 1):
        vert[2 + 2 * i] += (vert[2 * i]
                            + side_len * np.cos(theta * i + alpha))  # X
        vert[3 + 2 * i] += (vert[1 + 2 * i]
                            + side_len * np.sin(theta * i + alpha))  # y

    # Shift into +ve quadrant
    x_shift = min(vert[::2])
    y_shift = min(vert[1::2])
    vert[::2] = vert[::2] - x_shift
    vert[1::2] = vert[1::2] - y_shift

    return vert


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    n = 100
    h = hoti_hamiltonian_square(n, 0.1)
    
    # Eigen-problem solver
    current_time = time.strftime("%H:%M:%S", time.localtime())
    t1 = time.time()
    print(f"Beginning eigsh -> current time: {current_time}")
    eigenvalues, eigenvectors = eigsh(h, which='SM')
    t2 = time.time()
    print(f"\tFinished -> time elapsed: {t2-t1} s")
    
    
    # Normalise probability density
    wfs = np.transpose(eigenvectors)
    norm = np.sum(wfs, axis=1)
    wfs_normalised = np.divide(wfs, norm)
    no_eigen = np.shape(wfs_normalised)[0]
    p = np.sum(np.reshape(wfs_normalised, (no_eigen, n**2, 4)),axis=2)
    
    # Normalise probability density
    # wf = eigenvectors[:,1].flatten() 
    # norm = np.sum(abs(wf)**2)
    # p = np.sum(np.reshape(abs(wf)**2, (4, n**2)) / norm, axis=0)
    # p = np.reshape(p, (n,n))
    
    # Save
    for i in np.arange(no_eigen):    
        np.savetxt(f"{n}x{n}square_{i}.csv", np.reshape(p[i], (n, n)), delimiter=",")
    
    # Plot
    fig, ax = plt.subplots()
    im = ax.imshow(np.reshape(p[0], (n, n)))
    fig.colorbar(im, ax=ax, label='Interactive colorbar')
    plt.show()
    
    
