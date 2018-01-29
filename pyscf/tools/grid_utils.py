import numpy as np

from pyscf import lib


class grid(object):
    """Calculate a set of grid points around a molecule.

    Args:
        coords (numpy.ndarray): Coordinates of atoms in the molecule.
        nx (int): Number of grid point divisions in x direction.
           Note this is function of the molecule's size; a larger molecule
           will have a coarser representation than a smaller one for the
           same value.
        ny (int): Number of grid point divisions in y direction.
        nz (int): Number of grid point divisions in z direction.
        pad (float): Amount of padding (in Angstrom) in all dimensions that will be applied in the automatic construction of the rectangular grid volume based on the geometry of the system.
 
    """
    def __init__(self, coords, nx, ny, nz, pad):
        self.pad = pad / lib.param.BOHR
        self.nx = nx
        self.ny = ny
        self.nz = nz

        self.coord = coords
        self.box = (np.max(self.coord,axis=0) + self.pad) - (np.min(self.coord,axis=0) - self.pad)
        self.boxorig = np.min(self.coord,axis=0) - self.pad

        self.xs = np.arange(nx) * (self.box[0]/(nx - 1))
        self.ys = np.arange(ny) * (self.box[1]/(ny - 1))
        self.zs = np.arange(nz) * (self.box[2]/(nz - 1))

        self.coords = lib.cartesian_prod([self.xs,self.ys,self.zs])
        self.coords = np.asarray(self.coords, order='C') - (-self.boxorig)
