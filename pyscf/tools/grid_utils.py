#!/usr/bin/env python
#
# Authors: Mark J. Williamson <mw529@cam.ac.uk>
#          Mark D. Driver <mdd31@cam.ac.uk>
#

'''
Gaussian cube file format.  Reference:
http://paulbourke.net/dataformats/cube/
http://gaussian.com/cubegen/

The formatted cube file has the following format

Comment line
Comment line
N_atom Ox Oy Oz         # number of atoms, followed by the coordinates of the origin
N1 vx1 vy1 vz1          # number of grids along each axis, followed by the step size in x/y/z direction.
N2 vx2 vy2 vz2          # ...
N3 vx3 vy3 vz3          # ...
Atom1 Z1 x y z          # Atomic number, charge, and coordinates of the atom
...                     # ...
AtomN ZN x y z          # ...
Data on grids           # (N1*N2) lines of records, each line has N3 elements

An unformatted cube file differs in format. In this the Data has only one
entry per line:
    x y z Value
The unformatted format is required for isomep surface output.

'''

import time
import numpy as np
import pyscf
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

        Returns:
            Grid information lines as list of formatted Strings.
        """
        grid_line_fortran_format = '{:5d}{:12.6f}{:12.6f}{:12.6f}'
        grid_info_lines = []
        grid_info_lines.append(grid_line_fortran_format.format(self.mol.natm,
                                                               *self.boxorig.tolist()))
        grid_info_lines.append(grid_line_fortran_format.format(self.nx, self.xs[1], 0, 0))
        grid_info_lines.append(grid_line_fortran_format.format(self.ny, 0, self.ys[1], 0))
        grid_info_lines.append(grid_line_fortran_format.format(self.nz, 0, 0, self.zs[1]))
        return grid_info_lines

    def create_atom_cube_lines(self):
        """This generates the atom lines for any cube file.

        Returns:
            Atom lines for a cube file as list of formatted Strings
        """
        atom_fortran_format = '{:5d}{:12.6f}{:12.6f}{:12.6f}{:12.6f}'
        atom_lines = []
        atom_charges = self.mol.atom_charges()
        atom_coords = self.mol.atom_coords()
        for atom_index in range(self.mol.natm):
            atom_lines.append(atom_fortran_format.format(atom_charges[atom_index],
                                                         atom_charges[atom_index],
                                                         atom_coords[atom_index][0],
                                                         atom_coords[atom_index][1],
                                                         atom_coords[atom_index][2]))
        return atom_lines


def write_cube_header_lines(cube_information):
    """This writes the header comment lines for the cube file.

    Returns:
        Header comment lines: First line contains information of contents.
                              Second contains Pyscf information.
    """
    header_lines = []
    header_lines.append(cube_information)
    header_lines.append("PySCF Version: {:s}  Date: {:s}".format(pyscf.__version__, time.ctime()))
    return header_lines


def write_formatted_cube_data_lines(grid, property_values_array):
    """This generates the Data lines for a formatted cube file.

    Args:
        grid: grid object
        property_values_array: array containing the values, of shape (NX, NY, NZ).

    Returns:
        Data lines as list of formatted strings
    """
    data_lines = []
    value_fortran_format = "{:13.5E}"
    for ix in range(grid.nx):
        for iy in range(grid.ny):
            for iz in range(0, grid.nz, 6):
                remainder = (grid.nz-iz)
                if remainder > 6:
                    line_format = value_fortran_format * 6
                    data_lines.append(line_format.format(*property_values_array[ix, iy,
                                                                                iz:iz+6].tolist()))
                else:
                    line_format = value_fortran_format * remainder
                    data_lines.append(line_format.format(*property_values_array[ix, iy,
                                                                                iz:iz+remainder].tolist()))
                    break
    return data_lines

def write_unformatted_cube_data_lines(property_value_array_with_coords):
    """This writes the property lines for the unformatted cube file.
    Data format is:
        x y z value

    Args:
        property_value_array_with_coords: (N,4) array. N data values. rows are (x, y, z, value).

    Returns:
        Data lines as list of formatted strings.
    """
    data_lines = []
    line_fortran_format = "{:13.6f}{:13.6f}{:13.6f}{:13.6f}"
    for i in range(len(property_value_array_with_coords)):
        data_lines.append(line_fortran_format.format(*property_value_array_with_coords[i]))
    return data_lines


def write_formatted_cube_file(filename, cube_information, grid, property_values_array):
    """This writes the information to a formatted cube file.

    Args:
        filename: file to write to.
        cube_information: description of data in cube file.
        grid: grid object for molecule, containing the grid coordinates
        property_values_array: array containing the property value data, in array
                               of shape (NX,NY,NZ).
    """
    with open(filename, 'w') as outfile:
        header_lines = write_cube_header_lines(cube_information)
        outfile.write("\n".join(header_lines))
        outfile.write("\n")
        grid_info_lines = grid.create_grid_info_lines()
        outfile.write("\n".join(grid_info_lines))
        outfile.write("\n")
        atom_info_lines = grid.create_atom_cube_lines()
        outfile.write("\n".join(atom_info_lines))
        outfile.write("\n")
        cube_data_lines = write_formatted_cube_data_lines(grid, property_values_array)
        outfile.write("\n".join(cube_data_lines))
        outfile.write("\n")

def write_unformatted_cube_file(filename, cube_information, grid, property_value_array_with_coords):
    """This writes the information to an unformatted cube file.

    Args:
        filename: file to write to.
        cube_information: description of data in cube file.
        grid: grid object for molecule, containing the grid coordinates
        property_value_array_with_coords: (N,4) array containing the data points.
                                          N data values. Rows are of format (x, y, z, value).
    """
    with open(filename, 'w') as outfile:
        header_lines = write_cube_header_lines(cube_information)
        outfile.write("\n".join(header_lines))
        outfile.write("\n")
        grid_info_lines = grid.create_grid_info_lines()
        outfile.write("\n".join(grid_info_lines))
        outfile.write("\n")
        atom_info_lines = grid.create_atom_cube_lines()
        outfile.write("\n".join(atom_info_lines))
        outfile.write("\n")
        cube_data_lines = write_unformatted_cube_data_lines(property_value_array_with_coords)
        outfile.write("\n".join(cube_data_lines))
        outfile.write("\n")
