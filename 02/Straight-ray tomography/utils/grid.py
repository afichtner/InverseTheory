#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:copyright:
    Christian Boehm (christian.boehm@erdw.ethz.ch),
    Naiara Korta Martiartu (naiara.korta@erdw.ethz.ch),
    Andreas Fichtner (andreas.fichtner@erdw.ethz.ch)
    2018-21
:license:
    BSD 3-Clause ("BSD New" or "BSD Simplified")
"""

from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter
from scipy.sparse import diags
from scipy.sparse import lil_matrix

def plot_model(m, g, title=None, caxis=None, savename=None):
    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(111)
    if not title == None:
        ax.set_title(title,pad=20)
        
    plt.imshow(np.reshape(m,g.npoints).T, cmap='gray', extent=[g.origin[0], g.origin[0]+ g.npoints[0]*g.spacing[0], g.origin[1], g.origin[1]+g.npoints[1]*g.spacing[1]])
    plt.xlabel('$x$ [m]')
    plt.ylabel('$y$ [m]')
    plt.colorbar(shrink=0.7) 
    if caxis is not None:
        plt.clim(caxis)
    if savename is not None:
        plt.savefig(savename)
    plt.tight_layout()   
    plt.show()

    

class grid(object):
    """
    Simple class to store phantom data on rectilinear meshes in 2D and 3D

    This class stores values for different tissue parameters, can interpolate parameter
    fields onto arbirtary point clouds and contains utilities to smooth, plot and output
    phantoms to an HDF5 file.
    """


    def __init__(self, dim, origin, spacing, npoints):
        """
        Sets variables that define the rectilinear mesh.

        :param dim: Dimension (must be `2` or `3`).
        :param origin: dim-dimensional vector specifying the origin of the rectilinear domain.
        :param spacing: dim-dimensional vector specifying the grid spacing in each dimension.
        :param npoints: dim-dimensional vector specifying the number of gris points in each dimension.
        """

        assert (dim == 2 or dim == 3)
        assert len(origin) == dim
        assert len(spacing) == dim
        assert len(npoints) == dim

        self.dim = dim
        self.origin = origin.copy()
        self.spacing = spacing.copy()
        self.npoints = npoints.astype(int).copy()
        self.names = []
        self.data = {}

    def points(self):
        """
        Return a vector of all grid points of the mesh

        """

        p = np.zeros([np.prod(self.npoints),self.dim])

        x = np.linspace(self.origin[0],
                        self.origin[0] + self.spacing[0] * self.npoints[0],
                        num=self.npoints[0],endpoint=False)
        y = np.linspace(self.origin[1],
                        self.origin[1] + self.spacing[1] * self.npoints[1],
                        num=self.npoints[1],endpoint=False)

        if (self.dim == 3):
            z = np.linspace(self.origin[2],
                            self.origin[2] + self.spacing[2] * self.npoints[2],
                            num=self.npoints[2],endpoint=False)

        if (self.dim == 2):
            xv, yv = np.meshgrid(x, y, indexing='ij')
        else:
            xv, yv, zv = np.meshgrid(x, y, z, indexing='ij')

        p[:,0] = xv.flatten()
        p[:,1] = yv.flatten()
        if (self.dim == 3):
            p[:,2] = zv.flatten()

        return p.copy()

    def attach_field(self, name, values, verbose=True):
        """
        Attach a parameter field to the mesh.

        If a field with the same name already exists, the values will be overwritten.
        :param name: Name of the parameter field.
        :param values: Parameter values. Must have the same dimension as the number of grid points in the mesh.
        :param verbose: Toggle verbosity.
        """

        assert(np.array_equal(np.asarray(values.reshape(self.npoints).shape),self.npoints))
        if name in self.data:
            print('Overwriting field', name)
        else:
            print('Adding new field', name)
        self.data[name] = values.reshape(self.npoints)

    def plot(self):
        """
        Plot all attached parameter fields.

        This works only in 2D.

        """

        if (self.dim == 2):
            for field in self.data:
                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.set_title(field)
                plt.imshow(self.data[field])
                ax.set_aspect('equal')
                plt.colorbar()
                plt.show()
        else:
            print('Plotting only implemented for 2D phantoms.')


    def extract_slice(self, axis, slice_idx):
        """
        Exract a 2D slice from a 3D grid

        :param axis: Constant axis (`0`, `1`, or `2`)
        :param slice_idx: Index of the slice to be extracted.
        """

        assert(self.dim == 3)
        assert(slice_idx >=0 and slice_idx < self.npoints[axis])

        if (axis == 0):
            idx = [1,2]
        elif (axis == 1):
            idx = [0,2]
        else:
            idx = [0,1]

        origin = np.array([self.origin[idx[0]], self.origin[idx[1]]])
        spacing = np.array([self.spacing[idx[0]], self.spacing[idx[1]]])
        npoints = np.array([self.npoints[idx[0]], self.npoints[idx[1]]])

        g = grid(2, origin, spacing, npoints)

        for field in self.data:
            if (axis == 0):
                g.attach_field(field, self.data[field][slice_idx,:,:])
            elif (axis == 1):
                g.attach_field(field, self.data[field][:,slice_idx,:])
            else:
                g.attach_field(field, self.data[field][:,:,slice_idx])

        return g


    def interpolate(self, points, smoothing=0):
        """
        Interpolate all datafields onto an arbitrary point cloud

        :param points: Vector of point locations.
        :param smoothing: Optional Gaussian smoothing.
        """

        idata = {}

        x = np.linspace(self.origin[0],
                            self.origin[0] + self.spacing[0] * self.npoints[0],
                            num=self.npoints[0],endpoint=False)
        y = np.linspace(self.origin[1],
                            self.origin[1] + self.spacing[1] * self.npoints[1],
                            num=self.npoints[1],endpoint=False)

        if (self.dim == 2):

            for field in self.data:
                if (smoothing > 0):
                    interpolator = RectBivariateSpline(x,y,gaussian_filter(self.data[field], smoothing))
                else:
                    interpolator = RectBivariateSpline(x,y,self.data[field])

                idata[field] = interpolator(*points.transpose(), grid=False)
        else:
            z = np.linspace(self.origin[2],
                            self.origin[2] + self.spacing[2] * self.npoints[2],
                            num=self.npoints[2],endpoint=False)
            for field in self.data:
                if (smoothing > 0):
                    interpolator = RegularGridInterpolator((x,y,z),gaussian_filter(self.data[field], smoothing))
                else:
                    interpolator = RegularGridInterpolator((x,y,z),self.data[field])

                idata[field] = interpolator(points)

        return idata


    def get_laplacian(self):

        nx = self.npoints[0]
        ny = self.npoints[1]
        N  = nx*ny
        main_diag = -4.0 * np.ones(N)
        side_diag = np.ones(N-1)
        side_diag[np.arange(1,N)%4==0] = 0
        up_down_diag = np.ones(N-3)
        diagonals = [main_diag,side_diag,side_diag,up_down_diag,up_down_diag]

        return diags(diagonals, [0, -1, 1,nx,-nx], format="csc")


    def get_gaussian_prior(self, smoothing_length):

        nx = self.npoints[0]
        ny = self.npoints[1]
        N = nx * ny

        s = 4 * int(np.ceil(smoothing_length))
        dist_x = np.exp(-0.5 * ((np.arange(-s, s+1) ** 2) * self.spacing[0]**2 / (smoothing_length**2)))
        dist_y = np.exp(-0.5 * ((np.arange(-s, s+1) ** 2) * self.spacing[1]**2 / (smoothing_length**2)))

        K = dist_x.reshape((len(dist_x),1)).dot(dist_y.reshape((1,len(dist_x))))

        C = lil_matrix((N, N))
        for idx in range(0,nx):
            for idy in range(0,ny):
                i = ny * idx + idy
                scale = K[max(s-idx,0):min(2*s+1,nx+s-idx),max(s-idy,0):min(2*s+1,ny+s-idy)].sum()
                for ii in range(max(0,idx-s), min(idx+s+1,nx)):
                    ix = ii-idx+s
                    for jj in range(max(0,idy-s), min(idy+s+1,ny)):
                        iy = jj-idy+s
                        j = ny * ii + jj
                        C[i,j] = K[ix,iy] / scale


        return C.tocsc()
