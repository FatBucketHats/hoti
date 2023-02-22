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
from PIL import Image, ImageDraw

def hoti_hamiltonian_square(n, dn):
    """Generate hoti Hamiltonian for square (simplest implementation). Effectively this hamiltonian is written in the
    basis s_kron_y_kron_x with s, y and x labeling the spinor and spatial components.
    :param n: dimension of nxn domain
    :param dn: discretization param
    """
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

    return h

def hoti_hamiltonian_rect(nx, ny, dn):
    """Generate hoti Hamiltonian for square (simplest implementation). Effectively this hamiltonian is written in the
    basis s_kron_y_kron_x with s, y and x labeling the spinor and spatial components.
    :param n: dimension of nxn domain
    :param dn: discretization param
    """
    # Constants
    mu = 1
    beta = 1
    # Differential operators, d1 and d2
    diagx = np.ones(nx)
    diags1x= np.array([diagx, -diagx]) / dn
    d1x = sp.spdiags(diags1x, (1, -1))
    diags2x = np.array([diagx, -2 * diagx, diagx]) / dn ** 2
    d2x = sp.spdiags(diags2x, (1, 0, -1))

    diagy = np.ones(n)
    diags1y = np.array([diagy, -diagy]) / dn
    d1y = sp.spdiags(diags1y, (1, -1))
    diags2y = np.array([diagy, -2 * diagy, diagy]) / dn ** 2
    d2y = sp.spdiags(diags2y, (1, 0, -1))

    # Hamiltonian
    h = (sp.kron(sp.diags([1, -1, 1, -1]), -(sp.identity(nx*ny) + sp.kronsum(d2y, d2x)))
         + sp.kron(sp.diags([[1, 0, 0], [0, 0, 1]], [1, -1]), -1j * sp.kronsum(-d1y, d1x))
         + sp.kron(sp.diags([[1, 0, 0], [0, 0, 1]], [-1, 1]), -1j * sp.kronsum(d1y, d1x))
         + sp.kron(sp.diags([[1], [0, -1, 0], [0, 1, 0], [-1]], [-3, -1, 1, 3]), -1j * beta * sp.kronsum(d1y, -d1x)))

    return h

def hoti_hamiltonian_equilat_polygon(verticies, dn, v):
    """Create h describing the smallest square (might change so that it implements smallest rect) that fits polygon.
       Create mask and apply to h (apply bc's).
       :param verticies:  [x1,y1,x2,y2,...]
       :param dn: discretization param
       """
    # Along the x and y directions calculate the largest distance
    x = verticies[::2]
    y = verticies[1::2]
    n = max(max(x) - min(x), max(y) - min(y))
    # Create h for square
    h = hoti_hamiltonian_square(n, dn)
    # Create mask
    img = Image.new('L', (n, n), 0)
    ImageDraw.Draw(img).polygon(verticies, outline=1, fill=1)
    mask = np.array(img).flatten()
    zero = np.zeros(n**2)
    for i in np.arange(0, n**2):
        if not mask[i]:
            h[i,:] = zero
            h[i,i::n**2] = v
    return h

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    h = hoti_hamiltonian_square(100, 0.1)

    np.savetxt("foo.csv", a, delimiter=",")