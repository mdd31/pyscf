#!/usr/bin/env python
#
# Authors: Qiming Sun <osirpt.sun@gmail.com>
#          Mark J. Williamson <mw529@cam.ac.uk>
#

import numpy
import time
import pyscf
from pyscf import lib
from pyscf.dft import numint, gen_grid
from pyscf.tools import grid_utils

'''
Gaussian cube file format
'''
def density(mol, outfile, dm, nx=80, ny=80, nz=80, pad=2.0):
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


    """
    grid = grid_utils.grid(mol.atom_coords(), nx, ny, nz, pad)

    ngrids = grid.coords.shape[0]
    blksize = min(8000, ngrids)
    rho = numpy.empty(ngrids)
    ao = None
    for ip0, ip1 in gen_grid.prange(0, ngrids, blksize):
        ao = numint.eval_ao(mol, grid.coords[ip0:ip1], out=ao)
        rho[ip0:ip1] = numint.eval_rho(mol, ao, dm)
    rho = rho.reshape(nx,ny,nz)

    with open(outfile, 'w') as f:
        f.write('Electron density in real space (e/Bohr^3)\n')
        f.write('PySCF Version: %s  Date: %s\n' % (pyscf.__version__, time.ctime()))
        f.write('%5d' % mol.natm)
        f.write('%12.6f%12.6f%12.6f\n' % tuple(grid.boxorig.tolist()))
        f.write('%5d%12.6f%12.6f%12.6f\n' % (nx, grid.xs[1], 0, 0))
        f.write('%5d%12.6f%12.6f%12.6f\n' % (ny, 0, grid.ys[1], 0))
        f.write('%5d%12.6f%12.6f%12.6f\n' % (nz, 0, 0, grid.zs[1]))
        for ia in range(mol.natm):
            chg = mol.atom_charge(ia)
            f.write('%5d%12.6f'% (chg, chg))
            f.write('%12.6f%12.6f%12.6f\n' % tuple(mol.atom_coords()[ia]))

        for ix in range(nx):
            for iy in range(ny):
                for iz in range(0,nz,6):
                    remainder  = (nz-iz)
                    if (remainder > 6 ):
                        fmt = '%13.5E' * 6 + '\n'
                        f.write(fmt % tuple(rho[ix,iy,iz:iz+6].tolist()))
                    else:
                        fmt = '%13.5E' * remainder + '\n'
                        f.write(fmt % tuple(rho[ix,iy,iz:iz+remainder].tolist()))
                        break

def mep(mol, outfile, dm, nx=80, ny=80, nz=80, pad=2.0):
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
    grid = grid_utils.grid(mol.atom_coords(), nx, ny, nz, pad)

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
    MEP = MEP.reshape(nx,ny,nz)

    with open(outfile, 'w') as f:
        f.write('Molecular electrostatic potential in real space\n')
        f.write('PySCF Version: %s  Date: %s\n' % (pyscf.__version__, time.ctime()))
        f.write('%5d' % mol.natm)
        f.write('%12.6f%12.6f%12.6f\n' % tuple(grid.boxorig.tolist()))
        f.write('%5d%12.6f%12.6f%12.6f\n' % (nx, grid.xs[1], 0, 0))
        f.write('%5d%12.6f%12.6f%12.6f\n' % (ny, 0, grid.ys[1], 0))
        f.write('%5d%12.6f%12.6f%12.6f\n' % (nz, 0, 0, grid.zs[1]))
        for ia in range(mol.natm):
            chg = mol.atom_charge(ia)
            f.write('%5d%12.6f'% (chg, chg))
            f.write('%12.6f%12.6f%12.6f\n' % tuple(mol.atom_coords()[ia]))

        for ix in range(nx):
            for iy in range(ny):
                for iz in range(0,nz,6):
                    remainder  = (nz-iz)
                    if (remainder > 6 ):
                        fmt = '%13.5E' * 6 + '\n'
                        f.write(fmt % tuple(MEP[ix,iy,iz:iz+6].tolist()))
                    else:
                        fmt = '%13.5E' * remainder + '\n'
                        f.write(fmt % tuple(MEP[ix,iy,iz:iz+remainder].tolist()))
                        break



if __name__ == '__main__':
    from pyscf import gto, scf
    from pyscf.tools import cubegen
    mol = gto.M(atom='O 0.00000000,  0.000000,  0.000000; H 0.761561, 0.478993, 0.00000000,; H -0.761561, 0.478993, 0.00000000,', basis='6-31g*')
    mf = scf.RHF(mol)
    mf.scf()
    cubegen.density(mol, 'h2o_den.cube', mf.make_rdm1())
    cubegen.mep(mol, 'h2o_pot.cube', mf.make_rdm1())

