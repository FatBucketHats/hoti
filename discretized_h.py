# -*- coding: utf-8 -*-
"""Generate a discretized Hamiltonian for a TI disk in an applied magnetic field

This is done for the upper block of an effective Hamiltonian
describing the low energy properties of a TI disk. The Hamiltonian also contains terms
which describe a magnetic field which
"""

import numpy as np
import scipy.sparse as sp


def ti_hamiltonian(m0, m2, A,
                   r,
                   j, lb, mag_on=True
                   ):
    '''
    Function to create nxn matrix describing the HOTI Hamiltonian, where n is the number of elements in the
    discretized domain.

    Args:
        m0: material param
        m2: material param
        A: material param
        j: quantum number which takes half-integer values and is the eigenvalue of the projection along z of
           the total angular momentum J
        lb: magnetic length in units of compton length
        mag_on: controls whether mag field is switched on or off
        r: discretized 1D domain in units of compton length

    Returns:
        object:
    '''

    # Auxillary calcs
    n = len(r)  # Number of discretization points
    r0 = abs(A / m0)

    # Convert from compton length to SI units
    dr = (r0
          * (r[1] - r[0]))  # Discretization parameter
    lb = lb * r0

    # Set mag field on/off
    if mag_on:
        mag_control = 1
    else:
        mag_control = 0

    # Generate discretized first derivative nxn matrix
    d2_ = (np.diag(-np.ones(n) * 2)
           + np.diag(np.ones(n - 1), k=1)
           + np.diag(np.ones(n - 1), k=-1))
    d2_mat = d2_ / (dr ** 2)

    # Generate m tilda nxn matrix
    m_til_mat = (np.diag(m0 * np.ones(n))
                 - m2 * d2_mat)

    # Generate Wc,+ and Wc,- nxn matricies
    sub_mat = np.diag((j - mag_control * (r ** 2 / (2 * lb ** 2))))  # Substitution of j -> j - (r**2) / (2*lb**2)
    inv_r_mat = np.diag(1 / r)
    wcp_mat = m2 * inv_r_mat * (sub_mat ** 2 + sub_mat)
    wcn_mat = m2 * inv_r_mat * (sub_mat ** 2 - sub_mat)

    # H00 & H11 using the labels 0(1) to represent the block components of the Hamiltonian
    H00 = m_til_mat + wcn_mat
    H11 = -(m_til_mat + wcp_mat)
    # H01 & H10
    d1_mat = np.diag(np.ones(n - 1) / 2 * dr, k=1) + np.diag(np.ones(n - 1) / 2 * dr, k=-1)
    H01 = -A * 1j * (d1_mat + (inv_r_mat * sub_mat))
    H10 = -A * 1j * (d1_mat - (inv_r_mat * sub_mat))

    # Concatenate matricies to obtain h
    h_eff = np.block([
        [H00, H01],
        [H10, H11]
    ])

    return h_eff

def ti_hamiltonian(m0, m2, A,
                   r,
                   j, lb, mag_on=True
                   ):
    '''
    Function to create nxn matrix describing the TI Hamiltonian, where n is the number of elements in the
    discretized domain.

    Args:
        m0: material param
        m2: material param
        A: material param
        j: quantum number which takes half-integer values and is the eigenvalue of the projection along z of
           the total angular momentum J
        lb: magnetic length in units of compton length
        mag_on: controls whether mag field is switched on or off
        r: discretized 1D domain in units of compton length

    Returns:
        object:
    '''

    # Auxillary calcs
    n = len(r)  # Number of discretization points
    r0 = abs(A / m0)

    # Convert from compton length to SI units
    dr = (r0
          * (r[1] - r[0]))  # Discretization parameter
    lb = lb * r0

    # Set mag field on/off
    if mag_on:
        mag_control = 1
    else:
        mag_control = 0

    # Generate discretized first derivative nxn matrix
    d2_ = (np.diag(-np.ones(n) * 2)
           + np.diag(np.ones(n - 1), k=1)
           + np.diag(np.ones(n - 1), k=-1))
    d2_mat = d2_ / (dr ** 2)

    # Generate m tilda nxn matrix
    m_til_mat = (np.diag(m0 * np.ones(n))
                 - m2 * d2_mat)

    # Generate Wc,+ and Wc,- nxn matricies
    sub_mat = np.diag((j - mag_control * (r ** 2 / (2 * lb ** 2))))  # Substitution of j -> j - (r**2) / (2*lb**2)
    inv_r_mat = np.diag(1 / r)
    wcp_mat = m2 * inv_r_mat * (sub_mat ** 2 + sub_mat)
    wcn_mat = m2 * inv_r_mat * (sub_mat ** 2 - sub_mat)

    # H00 & H11 using the labels 0(1) to represent the block components of the Hamiltonian
    H00 = m_til_mat + wcn_mat
    H11 = -(m_til_mat + wcp_mat)
    # H01 & H10
    d1_mat = np.diag(np.ones(n - 1) / 2 * dr, k=1) + np.diag(np.ones(n - 1) / 2 * dr, k=-1)
    H01 = -A * 1j * (d1_mat + (inv_r_mat * sub_mat))
    H10 = -A * 1j * (d1_mat - (inv_r_mat * sub_mat))

    # Concatenate matricies to obtain h
    h_eff = np.block([
        [H00, H01],
        [H10, H11]
    ])

    return h_eff



def eig_prob_sol(h, sort=1):
    '''
    Returns eigenvalue and vectors for given material parameters and
    magnetic field strength

    Args:
        h: nxn effective Hamiltonian
        sort: sorting method
            1:
            2:
    '''

    # Diagonalise h to find eigenvalues and eigenfunctions
    eigen_val, eigen_vec = np.linalg.eigh(h, UPLO='U')

    # Sorting methods
    if sort == 1:  # Method 1 -> Sort by absolute value of eigenenergies
        sort_i = np.argsort(abs(eigen_val))
        eigen_vec = eigen_vec[:, sort_i]
        eigen_val = eigen_val[sort_i]

    return eigen_val, eigen_vec

if __name__ == '__main__':
    # -------------------------------------------------------------------
    # material param for Bi2Te3
    A = 4.003
    m0 = -0.296
    m2 = 177.355

    # other param (distances in units of compton length)
    j = 1 / 2  # total AM quantum number
    R = 100  # Geometry R
    dr = 0.05  # Discretization param
    lb = 10  # magnetic length
    magControl = 1  # control mag field 1 - on, 0 - off
    # -------------------------------------------------------------------

    # Discretization param
    ri = dr
    rf = R - dr
    n = int(round((rf - ri) / dr)) + 1  # number of points
    r = np.linspace(ri, rf, n)

    h = ti_hamiltonian(m0, m2, A, r, j, lb)
