import numpy as np

from pyscf import lib


class grid(object):
    """Calculate a set of grid points around a molecule.

    Args:
        atom_coords (numpy.ndarray): Coordinates of atoms in the molecule.
        nx (int): Number of grid point divisions in x direction.
           Note this is function of the molecule's size; a larger molecule
           will have a coarser representation than a smaller one for the
           same value.
        ny (int): Number of grid point divisions in y direction.
        nz (int): Number of grid point divisions in z direction.
        pad (float): Amount of padding (in Angstrom) in all dimensions that will be applied in the automatic construction of the rectangular grid volume based on the geometry of the system.
        gridspacing (float):  Distance, in Angstroms, between points in the grid, in all dimensions. This will override the nx,ny,nz the parameters. Note the following values:
               value/Angstroms	points/Bohr     Gaussian grid term
               0.1763          	3               	Coarse
               0.0882          	6               	Medium
               0.0441          	12              	Fine
 
    """
    def __init__(self, atom_coords, nx, ny, nz, pad, gridspacing):
        self.pad = pad / lib.param.BOHR
        self.nx = nx
        self.ny = ny
        self.nz = nz

        self.atom_coords = atom_coords

	self.boxmax = np.max(self.atom_coords,axis=0) + self.pad
	self.boxmin = np.min(self.atom_coords,axis=0) - self.pad

	self.box = self.boxmax - self.boxmin
	self.boxorig = self.boxmin

        if gridspacing:
           self.gridspacing = gridspacing / lib.param.BOHR

           self.nx = np.ceil( np.abs(self.box[0] / self.gridspacing) ).astype('int') + 1
           self.ny = np.ceil( np.abs(self.box[1] / self.gridspacing) ).astype('int') + 1
           self.nz = np.ceil( np.abs(self.box[2] / self.gridspacing) ).astype('int') + 1
           # +1 due to fencepost error.
           # Note, the effective gridspacing has changed due to use of ceil().

        self.xs = np.arange(self.nx) * (self.box[0]/(self.nx - 1))
        self.ys = np.arange(self.ny) * (self.box[1]/(self.ny - 1))
        self.zs = np.arange(self.nz) * (self.box[2]/(self.nz - 1))

        self.coords = lib.cartesian_prod([self.xs, self.ys, self.zs])
        self.coords = np.asarray(self.coords, order='C') - (-self.boxorig)
