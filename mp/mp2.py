#!/usr/bin/env python
# $Id$
# -*- coding: utf-8

import time
import tempfile
import numpy
import h5py

from pyscf import lib
from pyscf import ao2mo
import pyscf.lib.logger as log


'''
spin-adapted MP2
t2[i,j,b,a] = (ia|jb) / D_ij^ab
'''

# the MO integral for MP2 is (ov|ov). The most efficient integral
# transformation is
# (ij|kl) => (ij|ol) => (ol|ij) => (ol|oj) => (ol|ov) => (ov|ov)
#   or    => (ij|ol) => (oj|ol) => (oj|ov) => (ov|ov)

def kernel(mp, mo_coeff, mo_energy, nocc, verbose=None):
    ovov = mp.ao2mo(mo_coeff, nocc)

    nvir = len(mo_energy) - nocc
    eia = mo_energy[:nocc,None] - mo_energy[None,nocc:]
    t2 = numpy.empty((nocc,nocc,nvir,nvir))
    emp2 = 0
    for i in range(nocc):
        djba = (eia.reshape(-1,1) + eia[i].reshape(1,-1)).ravel()
        gi = numpy.array(ovov[i*nvir:(i+1)*nvir]).reshape(nvir,nocc,nvir)
        gi = gi.transpose(1,2,0)
        t2[i] = (gi.ravel()/djba).reshape(nocc,nvir,nvir)
        # 2*ijab-ijba
        theta = gi*2 - gi.transpose(0,2,1)
        emp2 += numpy.einsum('jab,jab', t2[i], theta)

    return emp2, t2

def make_rdm1(mp, mo_coeff, mo_energy, nocc, verbose=None):
    ovov = mp.ao2mo(mo_coeff, nocc)
    nvir = len(mo_energy) - nocc
    dm1occ = numpy.zeros((nocc,nocc))
    dm1vir = numpy.zeros((nvir,nvir))
    eia = mo_energy[:nocc,None] - mo_energy[None,nocc:]
    emp2 = 0
    for i in range(nocc):
        djba = (eia.reshape(-1,1) + eia[i].reshape(1,-1)).ravel()
        gi = numpy.array(ovov[i*nvir:(i+1)*nvir]).reshape(nvir,nocc,nvir)
        gi = gi.transpose(1,2,0)
        t2i = (gi.ravel()/djba).reshape(nocc,nvir,nvir)
        # 2*ijab-ijba
        theta = gi*2 - gi.transpose(0,2,1)
        emp2 += numpy.einsum('jab,jab', t2i, theta)

        dm1vir += numpy.einsum('jca,jcb->ab', t2i, t2i) * 2 \
                - numpy.einsum('jca,jbc->ab', t2i, t2i)
        dm1occ += numpy.einsum('iab,jab->ij', t2i, t2i) * 2 \
                - numpy.einsum('iab,jba->ij', t2i, t2i)

    rdm1 = numpy.zeros((nmo,nmo))
# *2 for beta electron
    rdm1[:nocc,:nocc] =-dm1occ * 2
    rdm1[nocc:,nocc:] = dm1vir * 2
    rdm1 = reduce(numpy.dot, (mo_coeff, rdm1, mo_coeff.T))
    return emp2, rdm1


class MP2(object):
    def __init__(self, mf):
        self.mol = mf.mol
        self._scf = mf
        self.verbose = self.mol.verbose
        self.stdout = self.mol.stdout
        self.max_memory = mf.max_memory

        self.emp2 = None
        self.t2 = None

    def run(self, mo=None, mo_energy=None, nocc=None):
        if mo is None:
            mo = self._scf.mo_coeff
        if mo_energy is None:
            mo_energy = self._scf.mo_energy
        if nocc is None:
            nocc = self.mol.nelectron / 2

        self.emp2, self.t2 = \
                kernel(self, mo, mo_energy, nocc, verbose=self.verbose)
        log.log(self, 'RMP2 energy = %.15g', self.emp2)
        return self.emp2, self.t2

    # return eri_ovov array[nocc*nvir,nocc*nvir]
    def ao2mo(self, mo, nocc):
        log = lib.logger.Logger(self.stdout, self.verbose)
        time0 = (time.clock(), time.time())
        log.debug('transform (ia|jb)')
        nmo = mo.shape[1]
        nvir = nmo - nocc
        co = mo[:,:nocc]
        cv = mo[:,nocc:]
        if nocc*nvir*nmo**2/2*8/1e6 < self.max_memory*.9 \
           and self._scf._eri is not None:
            eri = ao2mo.incore.general(self._scf._eri, (co,cv,co,cv))
        else:
            erifile = tempfile.NamedTemporaryFile()
            ao2mo.direct.general(self.mol, (co,cv,co,cv), erifile.name, \
                                 self.max_memory, verbose=self.verbose)
            feri = h5py.File(erifile.name, 'r')
            eri = feri['eri_mo']
        time1 = log.timer('Integral transformation', *time0)
        return eri


if __name__ == '__main__':
    from pyscf import scf
    from pyscf import gto
    mol = gto.Mole()
    mol.verbose = 5
    mol.output = 'out_h2o'
    mol.atom = [
        [8 , (0. , 0.     , 0.)],
        [1 , (0. , -0.757 , 0.587)],
        [1 , (0. , 0.757  , 0.587)]]

    mol.basis = {'H': 'cc-pvdz',
                 'O': 'cc-pvdz',}
    mol.build()
    mf = scf.RHF(mol)
    print(mf.scf())

    nocc = mol.nelectron/2
    nmo = mf.mo_energy.size
    nvir = nmo - nocc

    co = mf.mo_coeff[:,:nocc]
    cv = mf.mo_coeff[:,nocc:]
    g = ao2mo.incore.general(mf._eri, (co,cv,co,cv)).ravel()
    eia = mf.mo_energy[:nocc,None] - mf.mo_energy[nocc:]
    t2ref0 = g/(eia.reshape(-1,1)+eia.reshape(-1)).ravel()
    t2ref0 = t2ref0.reshape(nocc,nvir,nocc,nvir).transpose(0,2,3,1)

    pt = MP2(mf)
    emp2, t2 = pt.run()
    print(emp2 - -0.204019967288338)
    print('incore', numpy.allclose(t2, t2ref0))
    pt.max_memory = 1
    print('direct', numpy.allclose(pt.run()[1], t2ref0))

    t2s = numpy.zeros((nocc*2,nocc*2,nvir*2,nvir*2))
    t2s[ ::2, ::2, ::2, ::2] = t2ref0 - t2ref0.transpose(0,1,3,2)
    t2s[1::2,1::2,1::2,1::2] = t2ref0 - t2ref0.transpose(0,1,3,2)
    t2s[ ::2,1::2,1::2, ::2] = t2ref0
    t2s[1::2, ::2, ::2,1::2] = t2ref0
    t2s[ ::2,1::2, ::2,1::2] = -t2ref0.transpose(0,1,3,2)
    t2s[1::2, ::2,1::2, ::2] = -t2ref0.transpose(0,1,3,2)
    dm1occ =-.5 * numpy.einsum('ikab,jkab->ij', t2s, t2s)
    dm1vir = .5 * numpy.einsum('ijac,ijbc->ab', t2s, t2s)
    dm1ref = numpy.zeros((nmo,nmo))
    dm1ref[:nocc,:nocc] = dm1occ[ ::2, ::2]+dm1occ[1::2,1::2]
    dm1ref[nocc:,nocc:] = dm1vir[ ::2, ::2]+dm1vir[1::2,1::2]
    dm1ref = reduce(numpy.dot, (mf.mo_coeff, dm1ref, mf.mo_coeff.T))
    rdm1 = make_rdm1(pt, mf.mo_coeff, mf.mo_energy, nocc)[1]
    print(numpy.allclose(rdm1, dm1ref))