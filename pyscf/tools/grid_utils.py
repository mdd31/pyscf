#!/usr/bin/env python
#
# Authors: Mark J. Williamson <mw529@cam.ac.uk>
#          Mark D. Driver <mdd31@cam.ac.uk>
#


import numpy as np

from pyscf import lib

LOGGER = lib.logger.new_logger(verbose=0)


class Grid(object):
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
    def __init__(self, mol, nx, ny, nz, pad, gridspacing):
        self.pad = pad / lib.param.BOHR
        self.nx = nx
        self.ny = ny
        self.nz = nz

        self.mol = mol

        atom_coords = self.mol.atom_coords()

        self.boxmax = np.max(atom_coords, axis=0) + self.pad
        self.boxmin = np.min(atom_coords, axis=0) - self.pad

        self.box = self.boxmax - self.boxmin
        self.boxorig = self.boxmin

        if gridspacing:
            self.gridspacing = gridspacing / lib.param.BOHR

            self.nx = (self.box[0] // self.gridspacing).astype('int') + 1
            self.ny = (self.box[1] // self.gridspacing).astype('int') + 1
            self.nz = (self.box[2] // self.gridspacing).astype('int') + 1
            # +1 due to fencepost error.

            self.xs = np.arange(self.nx) * self.gridspacing
            self.ys = np.arange(self.ny) * self.gridspacing
            self.zs = np.arange(self.nz) * self.gridspacing

        else:

            # .../(nx-1) to get symmetric mesh
            # see also the discussion on https://github.com/sunqm/pyscf/issues/154
            self.xs = np.arange(self.nx) * (self.box[0]/(self.nx - 1))
            self.ys = np.arange(self.ny) * (self.box[1]/(self.ny - 1))
            self.zs = np.arange(self.nz) * (self.box[2]/(self.nz - 1))

        self.coords = lib.cartesian_prod([self.xs, self.ys, self.zs])
        self.coords = np.asarray(self.coords, order='C') - (-self.boxorig)

    def create_grid_info_lines(self):
        """This generates the lines containing information about the grid
        directions and grid origin, and number of atoms.
        """
        grid_line_fortran_format = '{:5d}{:12.6f}{:12.6f}{:12.6f}'
        grid_info_lines = []
        grid_info_lines.append(grid_line_fortran_format.format(self.mol.natom,
                                                               *self.boxorig.tolist()))
        grid_info_lines.append(grid_line_fortran_format.format(self.nx, self.xs[1], 0, 0))
        grid_info_lines.append(grid_line_fortran_format.format(self.ny, 0, self.ys[1], 0))
        grid_info_lines.append(grid_line_fortran_format.format(self.nz, 0, 0, self.zs[1]))
        return grid_info_lines

    def create_atom_cube_lines(self):
        """This generates the atom lines for any cube file.
        """
        atom_fortran_format = '{:5d}{:12.6f}{:12.6f}{:12.6f}{:12.6f}'
        atom_lines = []
        atom_charges = self.mol.atom_charges()
        atom_coords = self.mol.atom_coords()
        for atom_index in self.mol.natom:
            atom_lines.append(atom_fortran_format.format(atom_charges[atom_index],
                                                         atom_charges[atom_index],
                                                         atom_coords[atom_index][0],
                                                         atom_coords[atom_index][1],
                                                         atom_coords[atom_index][2]))
        return atom_lines
