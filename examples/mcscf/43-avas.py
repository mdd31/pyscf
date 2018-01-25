#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Section IV.A of AVAS paper  arXiv:1701.07862 [physics.chem-ph]
'''

from functools import reduce
import numpy
from pyscf import gto
from pyscf import scf
from pyscf import mcscf
from pyscf import fci

#=============example: FERROCENE====================

mol = gto.Mole()
mol.atom = '''
Fe     0.000000    0.000000    0.000000 
C     -0.713500   -0.982049   -1.648000 
C      0.713500   -0.982049   -1.648000 
C      1.154467    0.375109   -1.648000 
C      0.000000    1.213879   -1.648000 
C     -1.154467    0.375109   -1.648000 
H     -1.347694   -1.854942   -1.638208 
H      1.347694   -1.854942   -1.638208 
H      2.180615    0.708525   -1.638208 
H      0.000000    2.292835   -1.638208 
H     -2.180615    0.708525   -1.638208 
C     -0.713500   -0.982049    1.648000 
C     -1.154467    0.375109    1.648000 
C     -0.000000    1.213879    1.648000 
C      1.154467    0.375109    1.648000 
C      0.713500   -0.982049    1.648000 
H     -1.347694   -1.854942    1.638208 
H     -2.180615    0.708525    1.638208 
H      0.000000    2.292835    1.638208 
H      2.180615    0.708525    1.638208 
H      1.347694   -1.854942    1.638208  
'''
mol.basis = 'cc-pvtz-dk'
mol.spin = 0
mol.verbose = 4
mol.build()

mf = scf.sfx2c1e(scf.ROHF(mol))
mf.kernel()

#
# The active space can be generated by pyscf.mcscf.avas.avas function
#
from pyscf.mcscf import avas
# See also 43-dmet_cas.py and function gto.mole.search_ao_label for the rules
# of "ao_labels" in the following
ao_labels = ['Fe 3d', 'C 2pz']
norb, ne_act, orbs = avas.avas(mf, ao_labels, canonicalize=False)


#==================================================================================
#             run CASSCF with this active space for 7 singlet and 6 triplet states
#==================================================================================

weights = numpy.ones(13)/13
solver1 = fci.addons.fix_spin(fci.direct_spin1.FCI(mol), ss=2)
solver1.spin = 2
solver1.nroots = 6
solver2 = fci.addons.fix_spin(fci.direct_spin1.FCI(mol), ss=0)
solver2.spin = 0
solver2.nroots = 7

mycas = mcscf.CASSCF(mf, norb, ne_act)
mycas.chkfile ='fecp2_3dpz.chk'
mcscf.state_average_mix_(mycas, [solver1, solver2], weights)
mycas.verbose = 6
mycas.kernel(orbs)


