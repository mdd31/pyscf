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
    grid = grid_utils.grid(mol.atom_coords(), nx, ny, nz, pad, gridspacing)

    ngrids = grid.coords.shape[0]
    blksize = min(8000, ngrids)
    rho = numpy.empty(ngrids)
    ao = None
    for ip0, ip1 in gen_grid.prange(0, ngrids, blksize):
        ao = numint.eval_ao(mol, grid.coords[ip0:ip1], out=ao)
        rho[ip0:ip1] = numint.eval_rho(mol, ao, dm)
    rho = rho.reshape(grid.nx, grid.ny, grid.nz)

    with open(outfile, 'w') as f:
        f.write('Electron density in real space (e/Bohr^3)\n')
        f.write('PySCF Version: %s  Date: %s\n' % (pyscf.__version__, time.ctime()))
        f.write('%5d' % mol.natm)
        f.write('%12.6f%12.6f%12.6f\n' % tuple(grid.boxorig.tolist()))
        f.write('%5d%12.6f%12.6f%12.6f\n' % (grid.nx, grid.xs[1], 0, 0))
        f.write('%5d%12.6f%12.6f%12.6f\n' % (grid.ny, 0, grid.ys[1], 0))
        f.write('%5d%12.6f%12.6f%12.6f\n' % (grid.nz, 0, 0, grid.zs[1]))
        for ia in range(mol.natm):
            chg = mol.atom_charge(ia)
            f.write('%5d%12.6f'% (chg, chg))
            f.write('%12.6f%12.6f%12.6f\n' % tuple(mol.atom_coords()[ia]))

        for ix in range(grid.nx):
            for iy in range(grid.ny):
                for iz in range(0,grid.nz,6):
                    remainder  = (grid.nz-iz)
                    if (remainder > 6 ):
                        fmt = '%13.5E' * 6 + '\n'
                        f.write(fmt % tuple(rho[ix,iy,iz:iz+6].tolist()))
                    else:
                        fmt = '%13.5E' * remainder + '\n'
                        f.write(fmt % tuple(rho[ix,iy,iz:iz+remainder].tolist()))
                        break

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
    grid = grid_utils.grid(mol.atom_coords(), nx, ny, nz, pad, gridspacing)

    # Nuclear potential at given points
    Vnuc = 0
    for i in range(mol.natm):
       r = mol.atom_coord(i)
       Z = mol.atom_charge(i)
       rp = r - grid.coords
       Vnuc += Z / numpy.einsum('xi,xi->x', rp, rp)**.5

    # Potential of electron density
    Vele = []
    for p in grid.coords:
        mol.set_rinv_orig_(p)
        Vele.append(numpy.einsum('ij,ij', mol.intor('cint1e_rinv_sph'), dm))

    # MEP at each point
    MEP = Vnuc - Vele

    MEP = numpy.asarray(MEP)
    MEP = MEP.reshape(grid.nx, grid.ny, grid.nz)

    with open(outfile, 'w') as f:
        f.write('Molecular electrostatic potential in real space\n')
        f.write('PySCF Version: %s  Date: %s\n' % (pyscf.__version__, time.ctime()))
        f.write('%5d' % mol.natm)
        f.write('%12.6f%12.6f%12.6f\n' % tuple(grid.boxorig.tolist()))
        f.write('%5d%12.6f%12.6f%12.6f\n' % (grid.nx, grid.xs[1], 0, 0))
        f.write('%5d%12.6f%12.6f%12.6f\n' % (grid.ny, 0, grid.ys[1], 0))
        f.write('%5d%12.6f%12.6f%12.6f\n' % (grid.nz, 0, 0, grid.zs[1]))
        for ia in range(mol.natm):
            chg = mol.atom_charge(ia)
            f.write('%5d%12.6f'% (chg, chg))
            f.write('%12.6f%12.6f%12.6f\n' % tuple(mol.atom_coords()[ia]))

        for ix in range(grid.nx):
            for iy in range(grid.ny):
                for iz in range(0,grid.nz,6):
                    remainder  = (grid.nz-iz)
                    if (remainder > 6 ):
                        fmt = '%13.5E' * 6 + '\n'
                        f.write(fmt % tuple(MEP[ix,iy,iz:iz+6].tolist()))
                    else:
                        fmt = '%13.5E' * remainder + '\n'
                        f.write(fmt % tuple(MEP[ix,iy,iz:iz+remainder].tolist()))
                        break

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
    grid = grid_utils.grid(mol.atom_coords(), nx, ny, nz, pad, gridspacing)

    ngrids = grid.coords.shape[0]
    blksize = min(8000, ngrids)
    rho = numpy.empty(ngrids)
    ao = None
    for ip0, ip1 in gen_grid.prange(0, ngrids, blksize):
        ao = numint.eval_ao(mol, grid.coords[ip0:ip1], out=ao)
        rho[ip0:ip1] = numint.eval_rho(mol, ao, dm)

    # Number of voxels at the defined electronic_iso surface
    # Used for area
    surface_voxel_count = 0

    # Number of voxels *within* the defined electronic_iso surface
    # Used for volume
    inner_voxel_count = 0

    print "Total number of density voxels: " + str(len(rho))

    # Just count
    for index, voxel in enumerate(rho):
       # Voxels at surface
       #(num > (ISO - TOL) ) && (num < (ISO + TOL) )
       if ( (voxel > (electronic_iso - iso_tol)) and (voxel < (electronic_iso + iso_tol))  ):
          surface_voxel_count += 1
          #print index,voxel

    # This time, actually change
    for index, voxel in enumerate(rho):
       # Volume calculation
       if ( voxel  > electronic_iso ):
          inner_voxel_count += 1
          rho[index] = voxel

       else:
          rho[index] = 0.0

    rho = rho.reshape(grid.nx, grid.ny, grid.nz)

    print "surface voxels_found: " + str(surface_voxel_count)
    voxel_area = gridspacing * gridspacing
    print "Each voxel area /  A^2: " + str(voxel_area)
    print "inner surface area / A^2: " + str(surface_voxel_count * voxel_area)

    print "inner voxel count: " + str(inner_voxel_count)
    voxel_volume = gridspacing * gridspacing * gridspacing
    print "Each voxel volume /  A^3: " + str(voxel_volume)
    print "Total inner volume / A^3: " + str(inner_voxel_count * voxel_volume)

    with open(outfile, 'w') as f:
        f.write('Electron density in real space (e/Bohr^3)\n')
        f.write('PySCF Version: %s  Date: %s\n' % (pyscf.__version__, time.ctime()))
        f.write('%5d' % mol.natm)
        f.write('%12.6f%12.6f%12.6f\n' % tuple(grid.boxorig.tolist()))
        f.write('%5d%12.6f%12.6f%12.6f\n' % (grid.nx, grid.xs[1], 0, 0))
        f.write('%5d%12.6f%12.6f%12.6f\n' % (grid.ny, 0, grid.ys[1], 0))
        f.write('%5d%12.6f%12.6f%12.6f\n' % (grid.nz, 0, 0, grid.zs[1]))
        for ia in range(mol.natm):
            chg = mol.atom_charge(ia)
            f.write('%5d%12.6f'% (chg, chg))
            f.write('%12.6f%12.6f%12.6f\n' % tuple(mol.atom_coords()[ia]))

        for ix in range(grid.nx):
            for iy in range(grid.ny):
                for iz in range(0,grid.nz,6):
                    remainder  = (grid.nz-iz)
                    if (remainder > 6 ):
                        fmt = '%13.5E' * 6 + '\n'
                        f.write(fmt % tuple(rho[ix,iy,iz:iz+6].tolist()))
                    else:
                        fmt = '%13.5E' * remainder + '\n'
                        f.write(fmt % tuple(rho[ix,iy,iz:iz+remainder].tolist()))
                        break







if __name__ == '__main__':
    from pyscf import gto, scf
    from pyscf.tools import cubegen
    mol = gto.M(atom='O 0.00000000,  0.000000,  0.000000; H 0.761561, 0.478993, 0.00000000,; H -0.761561, 0.478993, 0.00000000,', basis='6-31g*')
    mf = scf.RHF(mol)
    mf.scf()
    cubegen.density(mol, 'h2o_den.cube', mf.make_rdm1())
    cubegen.mep(mol, 'h2o_pot.cube', mf.make_rdm1())

