import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from PIL import Image, ImageDraw

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
    diags1x = np.array([diagx, -diagx]) / dn
    d1x = sp.spdiags(diags1x, (1, -1))
    diags2x = np.array([diagx, -2 * diagx, diagx]) / dn ** 2
    d2x = sp.spdiags(diags2x, (1, 0, -1))

    diagy = np.ones(n)
    diags1y = np.array([diagy, -diagy]) / dn
    d1y = sp.spdiags(diags1y, (1, -1))
    diags2y = np.array([diagy, -2 * diagy, diagy]) / dn ** 2
    d2y = sp.spdiags(diags2y, (1, 0, -1))

    # Hamiltonian
    h = (sp.kron(sp.diags([1, -1, 1, -1]), -(sp.identity(nx * ny) + sp.kronsum(d2y, d2x)))
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
    zero = np.zeros(n ** 2)
    for i in np.arange(0, n ** 2):
        if not mask[i]:
            h[i, :] = zero
            h[i, i::n ** 2] = v
    return h