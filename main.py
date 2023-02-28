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
import math
from PIL import Image, ImageDraw

# Bi2Te3
A=2; m0=-1; m2=0.19; B=3
 
# Global constants
MU = m2*abs(m0)/(A**2)
BETA = B*abs(m0)/(A**2)
R0 = A/abs(m0)
V = 100
# path = "../data/"
path = "/Users/jopo/Documents/23/work/ti/work_data/hoti_equilat_polygon"
def row_divide(b, x):
    b.data /= np.take(x, b.indices)
    return b


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

    # Differential operators, d1 and d2
    diag = np.ones(n)
    diags1 = np.array([diag, -diag]) / dn
    d1 = sp.spdiags(diags1, (1, -1))
    diags2 = np.array([diag, -2 * diag, diag]) / dn ** 2
    d2 = sp.spdiags(diags2, (1, 0, -1))

    # Hamiltonian
    h = (sp.kron(sp.diags([1, -1, 1, -1]), -(sp.identity(n ** 2) + MU*sp.kronsum(d2, d2)))
         + sp.kron(sp.diags([[1, 0, 0], [0, 0, 1]], [1, -1]),  sp.kronsum(-d1, -1j*d1))
         + sp.kron(sp.diags([[1, 0, 0], [0, 0, 1]], [-1, 1]),  sp.kronsum(d1, -1j*d1))
         + sp.kron(sp.diags([[1], [0, -1, 0], [0, 1, 0], [-1]], [-3, -1, 1, 3]), -1j * BETA * sp.kronsum(d2, -d2)))

    # End
    t2 = time.time()
    print(f"\tFinished -> time elapsed: {t2-t1} s")
    return h

def hoti_hamiltonian_square_units(n, dn):
    """Generate hoti Hamiltonian for square (simplest implementation). Effectively this hamiltonian is written in the
    basis s_kron_y_kron_x with s, y and x labeling the spinor and spatial components.
    :param n: dimension of nxn domain
    :param dn: discretization param
    :returns: 
    """
    current_time = time.strftime("%H:%M:%S", time.localtime())
    print(f"Beginning hoti_hamiltonian_square -> current time: {current_time}")
    t1 = time.time()

    # Differential operators, d1 and d2
    diag = np.ones(n)
    diags1 = np.array([diag, -diag]) / dn
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


def gen_equipoly(no_vert, side_len, dn, alpha=0):
    """Generate coordinates of verticies describing equilateral polygon
    :param no_vert: number of verticies
    :param side_len: side length
    :param alpha: angle of rotation, zero defines polygon with one edge aligned with x axis
    """
    # Exception
    if no_vert < 3:
        raise Exception("A polygon must have at least 3 verticies.")
    
    alpha = alpha*np.pi/180
    # Initialise returned variable, [x1, y1, x2, y2...]
    vert = np.zeros(2 * no_vert)

    # Generate coords
    theta = (np.pi/180)*360 / no_vert
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

def hoti_hamiltonian_rect(vert, dn):
    """Create h describing the smallest rectangle that fits polygon.
       Create mask and apply to h (apply bc's).
       :param verticies:  [x1,y1,x2,y2,...]
       :param dn: discretization param
       """
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
    nx = math.ceil(max(vert[::2]))
    ny = math.ceil(max(vert[1::2]))
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
    h = h.multiply(np.transpose(mask))
    diag = 1 - mask # Flip 0's and 1's in mask
    v = diag*V 
    h.setdiag(h.diagonal() + v)
    
    # End
    t2 = time.time()
    print(f"\tFinished -> time elapsed: {t2-t1} s")
    
    return nx, ny, h

def hoti_hamiltonian_rect_n(vert, dn):
    """Create h describing the smallest rectangle that fits polygon.
       Create mask and apply to h (apply bc's).
       :param verticies:  [x1,y1,x2,y2,...]
       :param dn: discretization param
       """
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
    h = h.multiply(np.transpose(mask))
    diag = 1 - mask # Flip 0's and 1's in mask
    v = diag*V 
    h.setdiag(h.diagonal() + v)
    
    # End
    t2 = time.time()
    print(f"\tFinished -> time elapsed: {t2-t1} s")
    
    return nx, ny, h


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    L = 10
    dn = 0.1
    
    n = int(L/dn)
    h = hoti_hamiltonian_square(n, dn)
    
    # Eigen-problem solver
    current_time = time.strftime("%H:%M:%S", time.localtime())
    t1 = time.time()
    print(f"Beginning eigsh -> current time: {current_time}")
    eigenvalues, eigenvectors = eigsh(h, sigma=0, which='LM')
    t2 = time.time()
    print(f"\tFinished -> time elapsed: {t2-t1} s")
    
    
    # Normalise probability density
    wfs = np.transpose(eigenvectors)
    norm = np.sum(abs(wfs), axis=1, keepdims=1)
    wfs_normalised = wfs/norm
    no_eigen = np.shape(wfs_normalised)[0]
    p = np.sum(np.reshape(abs(wfs_normalised)**2, (no_eigen, 4, n**2)),axis=1)
    
    print(*eigenvalues, sep='\n')
    
    # Save
    for i in np.arange(no_eigen):    
        np.savetxt(path + f"/{n}x{n}square_{i}.csv", np.reshape(p[i], (n, n)), delimiter=",")
    
    # Plot
    fig, ax = plt.subplots()
    im = ax.imshow(np.reshape(p[0], (n, n))/max(p[0]))
    fig.colorbar(im, ax=ax, label='Colorbar')
    plt.show()
    
    
