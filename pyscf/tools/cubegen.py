#!/usr/bin/env python
#
# Authors: Qiming Sun <osirpt.sun@gmail.com>
#          Mark J. Williamson <mw529@cam.ac.uk>
#

'''
Gaussian cube file format.  Reference:
http://paulbourke.net/dataformats/cube/
http://gaussian.com/cubegen/

The output cube file has the following format

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
'''

import numpy
import time
import pyscf
from pyscf import lib
from pyscf.dft import numint, gen_grid
from pyscf.tools import grid_utils

LOGGER = lib.logger.Logger(verbose=7)

def density(mol, outfile, dm, nx=80, ny=80, nz=80, pad=4.0, gridspacing=None):
    """Calculates electron density.

    Args:
        mol (Mole): Molecule to calculate the electron density for.
        outfile (str): Name of Cube file to be written.
        dm (str): Density matrix of molecule.
        nx (int): Number of grid point divisions in x direction.
           Note this is function of the molecule's size; a larger molecule
           will have a coarser representation than a smaller one for the
           same value.
        ny (int): Number of grid point divisions in y direction.
        nz (int): Number of grid point divisions in z direction.
        pad (float): Amount of padding (in Angstrom) in all dimensions that will be applied in the automatic construction of the rectangular grid volume based on the geometry of the system.
        gridspacing (float):  Distance, in Angstroms, between points in the grid, in all dimensions. This will override the nx,ny,nz the parameters. Note the following values:
               value/Angstroms  points/Bohr     Gaussian grid term
               0.1763           3                       Coarse
               0.0882           6                       Medium
               0.0441           12                      Fine

    """
    grid = grid_utils.Grid(mol, nx, ny, nz, pad, gridspacing)
    
    ngrids = grid.coords.shape[0]
    blksize = min(8000, ngrids)
    rho = numpy.empty(ngrids)
    ao = None
    for ip0, ip1 in gen_grid.prange(0, ngrids, blksize):
        ao = numint.eval_ao(mol, grid.coords[ip0:ip1], out=ao)
        rho[ip0:ip1] = numint.eval_rho(mol, ao, dm)
    rho = rho.reshape(grid.nx, grid.ny, grid.nz)

    grid_utils.write_formatted_cube_file(outfile,
                                         'Electron density in real space (e/Bohr^3)',
                                         grid, rho)

def mep(mol, outfile, dm, nx=80, ny=80, nz=80, pad=4.0, gridspacing=None):
    """Calculates the molecular electrostatic potential (MEP).

    Args:
        mol (Mole): Molecule to calculate the MEP for.
        outfile (str): Name of Cube file to be written.
        dm (str): Density matrix of molecule.
        nx (int): Number of grid point divisions in x direction.
           Note this is function of the molecule's size; a larger molecule
           will have a coarser representation than a smaller one for the
           same value.
        ny (int): Number of grid point divisions in y direction.
        nz (int): Number of grid point divisions in z direction.
        pad (float): Amount of padding (in Angstrom) in all dimensions that will be applied in the automatic construction of the rectangular grid volume based on the geometry of the system.


    """
    grid = grid_utils.Grid(mol, nx, ny, nz, pad, gridspacing)

    mep_values = mep_for_coords(mol, dm, grid.coords)
    mep_values = mep_values.reshape(grid.nx, grid.ny, grid.nz)

    grid_utils.write_formatted_cube_file(outfile,
                                         'Molecular electrostatic potential in real space',
                                         grid, mep_values)

def mep_for_coords(mol, dm, coords):
    """This calculates the MEP value at the given coords.
    """
    # Nuclear potential at given points
    Vnuc = 0
    for i in range(mol.natm):
       r = mol.atom_coord(i)
       Z = mol.atom_charge(i)
       rp = r - coords
       Vnuc += Z / numpy.einsum('xi,xi->x', rp, rp)**.5

    # Potential of electron density
    Vele = []
    for p in coords:
        mol.set_rinv_orig_(p)
        Vele.append(numpy.einsum('ij,ij', mol.intor('cint1e_rinv_sph'), dm))

    # MEP at each point
    mep_values = Vnuc - Vele

    mep_values = numpy.asarray(mep_values)
    return mep_values

def isomep(mol, outfile, dm, electronic_iso=0.002, iso_tol=0.00003, nx=80, ny=80, nz=80, pad=4.0, gridspacing=None):
    """Calculates MEP on a specific electron density surface.

    Args:
        mol (Mole): Molecule to calculate the electron density for.
        outfile (str): Name of Cube file to be written.
        dm (str): Density matrix of molecule.
        nx (int): Number of grid point divisions in x direction.
           Note this is function of the molecule's size; a larger molecule
           will have a coarser representation than a smaller one for the
           same value.
        ny (int): Number of grid point divisions in y direction.
        nz (int): Number of grid point divisions in z direction.
        pad (float): Amount of padding (in Angstrom) in all dimensions that will be applied in the automatic construction of the rectangular grid volume based on the geometry of the system.
        gridspacing (float):  Distance, in Angstroms, between points in the grid, in all dimensions. This will override the nx,ny,nz the parameters. Note the following values:
               value/Angstroms  points/Bohr     Gaussian grid term
               0.1763           3                       Coarse
               0.0882           6                       Medium
               0.0441           12                      Fine

    """
    grid = grid_utils.Grid(mol, nx, ny, nz, pad, gridspacing)
    LOGGER.debug("grid coords shape: %s", grid.coords.shape)
    LOGGER.debug("grid coords first element shape: %s", grid.coords[0].shape)
    ngrids = grid.coords.shape[0]
    blksize = min(8000, ngrids)
    rho = numpy.empty(ngrids)
    ao = None
    for ip0, ip1 in gen_grid.prange(0, ngrids, blksize):
        ao = numint.eval_ao(mol, grid.coords[ip0:ip1], out=ao)
        rho[ip0:ip1] = numint.eval_rho(mol, ao, dm)

    # Number of voxels *within* the defined electronic_iso surface
    # Used for volume
    inner_voxel_count = 0

    LOGGER.note("Total number of density voxels: %d", len(rho))

    is_surface_voxel = numpy.logical_and(rho > (electronic_iso - iso_tol),
                                         rho < (electronic_iso + iso_tol))
    LOGGER.debug("Surface voxel count from logical_and: %d", numpy.count_nonzero(is_surface_voxel))
    
    surface_voxel_grid_indices = numpy.nonzero(is_surface_voxel)[0]
    LOGGER.debug2("Surface voxel indices: %s", surface_voxel_grid_indices)
    LOGGER.debug2("Surface voxel indices shape: %s", surface_voxel_grid_indices.shape)
    
    # Number of voxels at the defined electronic_iso surface
    # Used for area
    # Voxels at surface
    #(num > (ISO - TOL) ) && (num < (ISO + TOL) )
    surface_voxel_count = surface_voxel_grid_indices.shape[0]
    surface_voxel_coords = grid.coords[surface_voxel_grid_indices[0]]
    for i in range(1,surface_voxel_grid_indices.shape[0]):
        surface_voxel_coord = grid.coords[surface_voxel_grid_indices[i]]
        LOGGER.debug4("surface voxel coord shape: %s", surface_voxel_coord.shape)
        surface_voxel_coords = numpy.append(surface_voxel_coords, surface_voxel_coord, axis=0)
    LOGGER.debug("surface_voxel_coords shape: %s", surface_voxel_coords.shape)
    surface_voxel_coords = surface_voxel_coords.reshape((surface_voxel_grid_indices.shape[0], 3))
    LOGGER.debug("surface_voxel_coords shape: %s", surface_voxel_coords.shape)
    LOGGER.debug3("First coord from grid: %s", grid.coords[surface_voxel_grid_indices[0]])
    LOGGER.debug3("First coord from surf voxel array: %s",surface_voxel_coords[0])
    
    is_in_surface = numpy.greater(rho, electronic_iso)
    LOGGER.debug("Voxel count from numpy.greater: %d", numpy.count_nonzero(is_in_surface))
    # Number of voxels *within* the defined electronic_iso surface
    # Used for volume
    inner_voxel_count = numpy.count_nonzero(is_in_surface)
    # This time, actually change
    for index, voxel in enumerate(rho):
       # Volume calculation
       if ( voxel  > electronic_iso ):
          rho[index] = voxel
       else:
          rho[index] = 0.0
    LOGGER.debug("rho shape: %s", rho.shape)
    rho = rho.reshape(grid.nx, grid.ny, grid.nz)

    LOGGER.debug("rho shape: %s", rho.shape)
    LOGGER.note("surface voxels_found: %d", surface_voxel_count)
    voxel_area = gridspacing * gridspacing
    LOGGER.info("Each voxel area /  A^2: %f", voxel_area)
    LOGGER.info("inner surface area / A^2: %f", surface_voxel_count * voxel_area)
    
    LOGGER.info("inner voxel count: %d", inner_voxel_count)
    voxel_volume = gridspacing * gridspacing * gridspacing
    LOGGER.info("Each voxel volume /  A^3: %f", voxel_volume)
    inner_volume = inner_voxel_count * voxel_volume
    LOGGER.info("Total inner volume / A^3: %f", inner_volume)
    
    
    mep_values = mep_for_coords(mol, dm, surface_voxel_coords)
    LOGGER.debug("MEP values shape: %s", mep_values.shape)
    mep_values = mep_values.reshape((mep_values.shape[0], 1))
    LOGGER.debug("MEP values shape: %s", mep_values.shape)
    #Add the potentials to the coordinates: each row describes one point.
    coords_with_mep_values = numpy.append(surface_voxel_coords, mep_values, axis=1)
    
    cube_information = "Molecular electrostatic potential in real space on {:.5f} isodensity surface. Volume: {:.6f}".format(electronic_iso, inner_volume)
    
    grid_utils.write_unformatted_cube_file(outfile, cube_information, grid, coords_with_mep_values)



if __name__ == '__main__':
    from pyscf import gto, scf
    from pyscf.tools import cubegen
    mol = gto.M(atom='O 0.00000000,  0.000000,  0.000000; H 0.761561, 0.478993, 0.00000000,; H -0.761561, 0.478993, 0.00000000,', basis='6-31g*')
    mf = scf.RHF(mol)
    mf.scf()
    cubegen.density(mol, 'h2o_den.cube', mf.make_rdm1())
    cubegen.mep(mol, 'h2o_pot.cube', mf.make_rdm1())

