#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:copyright:
    Christian Boehm (christian.boehm@erdw.ethz.ch),
    Naiara Korta Martiartu (naiara.korta@erdw.ethz.ch),
    Andreas Fichtner (andreas.fichtner@erdw.ethz.ch),
    2018-21
:license:
    BSD 3-Clause ("BSD New" or "BSD Simplified")
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp

def compute_ray_path(source, receiver, g):

    assert g.dim == 2
    assert len(source) == 2
    assert len(receiver) == 2

    delta = receiver - source
    if (delta[0] > 0):
        six = int(np.floor((source[0] - g.origin[0]) / g.spacing[0]) + 1)
        rix = int(np.ceil((receiver[0] - g.origin[0]) / g.spacing[0]) - 1)
        nx=max(0,rix-six+1)
    else:
        six = int(np.ceil((source[0] - g.origin[0]) / g.spacing[0]) - 1)
        rix = int(np.floor((receiver[0] - g.origin[0]) / g.spacing[0]) + 1)
        nx=max(0,six-rix+1)

    if (delta[1] > 0):
        siy = int(np.floor((source[1] - g.origin[1]) / g.spacing[1]) + 1)
        riy = int(np.ceil((receiver[1] - g.origin[1]) / g.spacing[1]) - 1)
        ny=max(0,riy-siy+1)
    else:
        siy = int(np.ceil((source[1] - g.origin[1]) / g.spacing[1]) - 1)
        riy = int(np.floor((receiver[1] - g.origin[1]) / g.spacing[1])+1)
        ny=max(0,siy-riy+1)

    numpoints = nx+ny+2

    unsortedpoints = np.zeros((numpoints,2))
    unsortedpoints[0,:] = source

    if (nx == 1):
        unsortedpoints[1,0] = g.origin[0] + six * g.spacing[0]
        unsortedpoints[1,1] = source[1] + (unsortedpoints[1,0] - source[0]) / delta[0] * delta[1]

    elif (nx > 1):
        order = int(np.sign(rix-six))
        unsortedpoints[1:nx+1,0] = g.origin[0] + np.arange(six, six + order * nx, order) * g.spacing[0]
        unsortedpoints[1:nx+1,1] = source[1] + (unsortedpoints[1:nx+1,0] - source[0]) / delta[0] * delta[1]

    if (ny == 1):
        unsortedpoints[1+nx,1] = g.origin[1] + siy * g.spacing[1]
        unsortedpoints[1+nx,0] = source[0] + (unsortedpoints[1+nx,1] - source[1]) / delta[1] * delta[0]

    elif (ny > 1):
        order = np.sign(riy-siy)
        unsortedpoints[1+nx:1+nx+ny,1] = g.origin[1] + np.arange(siy, siy + order * ny, order) * g.spacing[1]
        unsortedpoints[1+nx:1+nx+ny,0] = source[0] + (unsortedpoints[1+nx:1+nx+ny,1] - source[1]) / delta[1] * delta[0]

    unsortedpoints[numpoints-1,:] = receiver
    ix = np.lexsort((unsortedpoints[:, 1], unsortedpoints[:, 0]))
    sortedpoints = unsortedpoints[ix]

    midpoints = (sortedpoints[0:-1,:] + sortedpoints[1:,:] ) / 2
    ix = np.clip(np.floor((midpoints[:,0] - g.origin[0]) / g.spacing[0]), 0, g.npoints[0]-1)
    iy = np.clip(np.floor((midpoints[:,1] - g.origin[1]) / g.spacing[1]), 0, g.npoints[1]-1)
    indices = g.npoints[1] * ix + iy
    if max(indices) >= g.npoints.prod():
        print('indices',indices)
        print('source',source)
        print('receiver',receiver)
        print('sortedpoints',sortedpoints)
        print('midpoints',midpoints)
        print('ix',ix)
        print('iy',iy)
    vals = np.linalg.norm(sortedpoints[1:,:] -  sortedpoints[0:-1,:],axis = 1)

    return indices, vals


def plot_ray_density(A, g):
    ray_density = np.sum(A, axis=0)
    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(111)
    ax.set_title('ray density',pad=15)
    plt.imshow(ray_density.reshape(g.npoints).T, cmap='gray', extent=[g.origin[0],g.origin[0]+ g.npoints[0]*g.spacing[0],g.origin[1],g.origin[1]+g.npoints[1]*g.spacing[1]])
    ax.set_aspect('equal')
    plt.xlabel('$x$ [m]')
    plt.ylabel('$y$ [m]')
    plt.colorbar()
    plt.savefig('raydensity.pdf',format='pdf')
    plt.show()


def get_all_to_all_locations(src_locations, rec_locations):

    nsrc = src_locations.shape[1]
    nrec = rec_locations.shape[1]
    ndata = nsrc * nrec
    sources = np.zeros((2,ndata))
    receivers = np.zeros((2,ndata))
    for i in range(0,nsrc):
        sources[:,i*nrec:(i+1)*nrec] = np.ones((2,nrec)) * src_locations[:,i].reshape((2,1))
        receivers[:,i*nrec:(i+1)*nrec] = rec_locations

    return sources, receivers


def plot_rays(sources, receivers, g, only_locations=False):
    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(111)
    if not only_locations:
        for i in range(0,sources.shape[1]):
            plt.plot(np.array([sources[0,i], receivers[0,i]]),
                     np.array([sources[1,i], receivers[1,i]]),
                     'k')

    plt.scatter(sources[0,:], sources[1,:], marker='*', color=[0.5,0.5,0.5], s=200)
    plt.scatter(receivers[0,:], receivers[1,:], marker='v', color=[0.5,0.5,0.5], s=200)

    ax.set_aspect('equal')
    plt.xlabel('$x$ [m]')
    plt.ylabel('$y$ [m]')
    plt.title('ray coverage',pad=20)
    plt.show()


def create_forward_operator(sources, receivers, g):

    # determine size of the sparse matrix
    # The number of columns is equal to the number of pixels in the 2D grid
    # The number of rows is equal to the number of ray paths.
    ncol = g.npoints.prod()
    nrow = sources.shape[1]

    # There are several ways to create a sparse matrix in python.
    # Here, we use three vectors that contain the following information
    # - all_data: the actual values of the nonzero entries
    # - all_indices: the column indices of the nonzero entries
    # - all_indptr: the start idx in the above two vectors for each row

    # pre-allocate space for the nonzero entries of the matrix
    # This improves the efficiency when concatenating the indices and values in memory
    # Here, we assume that any ray will intersect at most 2 times the average number of
    # pixels in x- and y-direction
    all_indices = np.zeros((nrow * 2 * int(g.npoints.mean()),), dtype=int)
    all_data = np.zeros((nrow * 2* int(g.npoints.mean()),))
    all_indptr = np.zeros((nrow+1,), dtype=int)

    # Now loop through all source-receiver pairs and create the ray paths
    for i in range(0,nrow):
        [indices, vals]=compute_ray_path(sources[:,i], receivers[:,i], g)
        all_indices[all_indptr[i]:all_indptr[i]+len(indices)] = indices
        all_data[all_indptr[i]:all_indptr[i]+len(indices)] = vals
        all_indptr[i+1] = all_indptr[i] + len(indices)

    assert(max(all_indices) < ncol)
    assert(min(all_indices) >= 0)

    # After gathering data for all rays, we can now initialize the sparse matrix
    return sp.csr_matrix((all_data[:all_indptr[nrow]],
                          all_indices[:all_indptr[nrow]],
                          all_indptr),
                         shape=(nrow,ncol))


