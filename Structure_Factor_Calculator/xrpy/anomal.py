# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 18:00:01 2015

@author: bren
"""
from __future__ import print_function
import numpy as np
import Structure_Factor_Calculator.xrpy as Xrpy
import h5py
import os
pathStr = os.path.dirname(os.path.abspath(__file__))+os.sep+'Anomalous.h5'
h5A = h5py.File(pathStr,'r', driver='core',backing_store=False)

pathStr = os.path.dirname(os.path.abspath(__file__))+os.sep+'Henke.h5'
h5H = h5py.File(pathStr,'r', driver='core',backing_store=False)

def anomal(zed, energy):
    """    
    Switchyard routine for cromer.py and henke.py
    based on value of xrpy.const.Anomal
    
    Parameters
    __________
    zed : integer or string
        atomic number or name up to Uranium
    energy : float
        single or array of energies in [eV]
           
    Returns
    _______
    fp : float
        single or array of floats.  Cross-section in [electrons/atom]
    Even if henke is called fp (not f1) is returned.

    fpp : float
        single or array of floats.  Cross-section in [electrons/atom]   
    """
    #print(Xrpy.const.Anomal)
    if Xrpy.const.Anomal == 0:
        fp,fpp = Xrpy.cromer(zed, energy)
    else:
        f1,fpp = Xrpy.henke(zed, energy)
        if f1.all() != 0.:
            fp = f1 - float(Xrpy.element(zed)[0])
        else:
            fp = np.zeros(np.size(f1))
    
    return fp, fpp

def henke(zed, energy):
    """    
    Calculate f1 and fpp parts of index of refraction vs. energy
    where f1 = fp + zed
    based on tables of Henke & Gullickson '93
        
    Parameters
    __________
    zed : integer or string
        atomic number or name up to Uranium
    energy : float
        single or array of energies in [eV]
           
    Returns
    _______
    f1 : float
        single or array of floats.  = fp + zed [electrons/atom]
    fpp : float
        single or array of floats.  Cross-section in [electrons/atom]   
    """
    if np.size(energy) > 1:
        lErg = len(energy)
    else:
        lErg = 1
    f1 = np.zeros(lErg)
    fpp = np.zeros(lErg)
    
    zed=Xrpy.element(zed)[0]
    if zed < 1 or zed > 92:
        print(' Atomic number out of range')
        return f1, fpp

    if type(energy) is int:
        energy= float(energy)
    if type(energy) is list:
        energy = np.array(energy).astype('float')

    hErg = h5H[str(zed)+'erg'][:]
    if lErg > 1:
        if min(energy) < 30. or max(energy) > 30000.:
            print(' Energy out of range')
            return f1, fpp
        for jdx, erdx in enumerate(energy):
            for idx, erg in enumerate(hErg):
                if erg > erdx:
                    break
            f1[jdx], fpp[jdx] = doHenkCalc(zed, erdx, idx-1, idx)            
    else:
        if energy < 30. or energy > 30000.:
            print(' Energy out of range')
            return f1, fpp
        for idx, erg in enumerate(hErg):
            if erg > energy:
                break
        f1[0], fpp[0] = doHenkCalc(zed, energy, idx-1, idx)            

    return f1, fpp

def doHenkCalc(zed, energy, idLo, idHi):
    hfp =  h5H[str(zed)+'fp'][:]
    hfpp = h5H[str(zed)+'fpp'][:]
    hErg = h5H[str(zed)+'erg'][:]
    fract = (energy - hErg[idLo])/(hErg[idHi]- hErg[idLo])
    f1 = hfp[idLo] + fract * (hfp[idHi]- hfp[idLo])
    fpp = hfpp[idLo] + fract * (hfpp[idHi]- hfpp[idLo])

    return f1, fpp

def rayComp(zed, energy):
    """    
    Calculate elastic and Compton absorption cross-sections
    
    Based on McMasters parameterization.
        
    Parameters
    __________
    zed : integer or string
        atomic number or name up to Uranium
    energy : float
        single or array of energies in [eV]
    
    Returns
    _______
    ray : float
        single or array of floats.  Cross-section in [electrons/atom]
    comp : float
        single or array of floats.  Cross-section in [electrons/atom]
    
    """
    zed=Xrpy.element(zed)[0]

    if zed < 1 or zed > 92:
        print('Atomic number out of range')
        return 0., 0.

    isArray = True
    if type(energy) is int:
        energy= float(energy)
        isArray = False
    elif type(energy) is float:
        isArray = False
    elif type(energy) is list:
        energy = np.array(energy).astype('float')

    if isArray:
        lErg = len(energy)         
        if min(energy) < 1. or max(energy) > 1.e5:
            print(' Energy out of range')
            return 0., 0.

    else:
        lErg = 1
        if energy < 1. or energy > 1.e5:
            print(' Energy out of range')
            return 0., 0.

    ray = np.zeros(lErg)
    comp = np.zeros(lErg)
    clRC= h5A[str(zed)+'rayComp'][:]
    clRay = clRC[0:4]
    clComp = clRC[4:8]
    p1= np.log(energy/1000.)
    p2= p1**2
    p3= p1**3
        
    ray = ( Xrpy.const.barns2Elec* energy * 
            np.exp( clRay[0] + clRay[1]* p1 + clRay[2]* p2 + clRay[3]* p3))
    comp = ( Xrpy.const.barns2Elec* energy* 
            np.exp( clComp[0] + clComp[1]* p1 + clComp[2]* p2 + clComp[3]* p3))
        
    return ray, comp

def cromer(zed, energy):
    """    
    Calculate real (fp) and imaginary (fpp) parts of index of refraction vs. energy.
    
    From D.T. Cromer and D.A. Liberman, Acta Cryst. A 37, 267 (1981).
        
    Parameters
    __________
    zed : integer or string
        atomic number or name up to Uranium
    energy : float
        single or array of energies in [eV]
           
    Returns
    _______
    fp : float
        single or array of floats.  Cross-section in [electrons/atom]
    fpp : float
        single or array of floats.  Cross-section in [electrons/atom]   
    """

    if np.size(energy) > 1:
        lErg = len(energy)
    else:
        lErg = 1
    
    fp = np.zeros(lErg)
    fpp = np.zeros(lErg)
    
    zed=Xrpy.element(zed)[0]    
    if zed < 1 or zed > 92:
        print(' Atomic number out of range')
        return fp, fpp

    if type(energy) is int:
        energy= float(energy)
    if type(energy) is list:
        energy = np.array(energy).astype('float')

    if lErg > 1:
        if min(energy) < 30. or max(energy) > 1.e5:
            print(' Energy out of range')
            return fp, fpp
    else:
        if energy < 30. or energy > 1.e5:
            print(' Energy out of range')
            return fp, fpp

    if zed < 3:
        fp, fpp = cromerMcm(zed, lErg, energy)
        #print(' McMaster')
        return fp, fpp


    Norb = h5A['Norb'][zed-1]
    relCor = h5A['Relcor'][zed-1]
    KPCor = h5A['KPcor'][zed-1]
    """
    without the [:] you just get a pointer to the data
    with the [:] you get the actual data.
    """
    clXsect = h5A[str(zed)+'clXsect'][:]
    clErg = h5A[str(zed)+'clErg'][:]
    bindErg = h5A[str(zed)+'BindErg'][:]
    Nparms = h5A[str(zed)+'Nparms'][:]
    funType = h5A[str(zed)+'FunType'][:]
    
    # start doing stuff
    bindErgAu = bindErg/ Xrpy.const.kevPerRyd
    
    """
    xscEdgeAu is the non-zero values of the last row of the 11x24 matrix
    Note that clXsect[zed] is only Norb long to begin with.
    """
    xscEdgeAu = clXsect[-1:,].flatten()/ Xrpy.const.au    
    xscInt = clXsect[5:10,:].transpose()/ Xrpy.const.au
    ergInt = clErg[5:10,:].transpose()
    ergInt = ergInt[:,::-1]
    xscBarnsMat = np.zeros((Norb,lErg))
    
    sIdx = np.argsort(clErg,0)
    ergSort = np.sort(clErg,0)
    ergSort[np.isnan(ergSort)] = 1.
    ergSort = ergSort.conj().T
    xscSort = np.zeros((Norb,11))
    for i in range(0,Norb):
        xscSort[i,:] = clXsect[sIdx[:,i],i]
        
    sIdxInt = np.argsort(ergInt,1)
    sIdxInt = np.fliplr(sIdxInt)
    fred = np.zeros((Norb,5))
    for i in range(0,Norb):
        for j in range(0,5):
            fred[i,j] = Norb* sIdxInt[i,j] + i
    mort= fred.flatten().astype(np.int64)
    tim = xscInt.reshape(5*Norb,1,order='F').copy()
    xscIntSort = tim[mort].reshape(5,Norb).copy()
    
    ergSortLog = np.log(ergSort)
    logIdx = np.nonzero(xscSort)
    xscSortLog = np.zeros((Norb,11))
    xscSortLog[logIdx] = np.log(xscSort[logIdx])
    nrgMat = np.tile(np.array(energy),(Norb,1))/1000. # change to keV
    logNrgMat = np.log(nrgMat)
    nrgAuMat = nrgMat/ Xrpy.const.kevPerRyd

    bindErgMat = np.tile(np.array([bindErg]).transpose(),(1,lErg))
    bindErgAuMat = bindErgMat/ Xrpy.const.kevPerRyd
    delBindErgMat = (bindErgMat - nrgMat)
    idxDelNeg = delBindErgMat <= 0
    
    xscRowSum = (xscSortLog != 0.).sum(1)
    xscIdxMat = np.tile(np.array([xscRowSum]).transpose(),(1,lErg))
    
    cmpMat1 = (np.tile(np.array([ergSortLog.flatten()]).transpose(),(1,lErg)) - np.tile(np.array([np.log(energy/1000.)]),((Norb*11),1)))
    cmpMat = cmpMat1.reshape(11,(Norb*lErg),order='F').copy()

    idAknint = (cmpMat < 0.).sum(0)-1
    idxAknint = idAknint.reshape(Norb,lErg,order='F').copy()
    xscNon0 = Nparms-xscIdxMat[:,0].conj().T+1
    xscN0Mat = np.tile(np.array([xscNon0]).transpose(),(1,lErg))

    idxAkmax = np.maximum(idxAknint, xscN0Mat)
    idxAknint = np.minimum(idxAkmax,(xscN0Mat+xscIdxMat-3))    
    idxAknint = idxAknint + np.tile(np.array(
                 [11* np.arange(0,Norb)]).transpose(),(1,lErg))
    idxAkninV = idxAknint.reshape(1,Norb*lErg,order='F').copy()
    
    ergSortLogV = ergSortLog.flatten()
    xscSortLogV = xscSortLog.flatten()

    nrg1 = ergSortLogV[idxAkninV-1].reshape(Norb,lErg,order='F').copy()
    nrg2 = ergSortLogV[idxAkninV].reshape(Norb,lErg,order='F').copy()
    nrg3 = ergSortLogV[idxAkninV+1].reshape(Norb,lErg,order='F').copy()
    
    xscT1 = xscSortLogV[idxAkninV-1].reshape(Norb,lErg,order='F').copy()
    xscT2 = xscSortLogV[idxAkninV].reshape(Norb,lErg,order='F').copy()
    xscT3 = xscSortLogV[idxAkninV+1].reshape(Norb,lErg,order='F').copy()
    
    nrg4 = nrg1 - logNrgMat
    nrg5 = nrg2 - logNrgMat
    nrg6 = nrg3 - logNrgMat
    
    xscBarnsMat = ((nrg6 * (xscT1*nrg5 - xscT2*nrg4) / (nrg2- nrg1) -
                nrg5 * (xscT1*nrg6 - xscT3*nrg4) / (nrg3- nrg1)) /
                (nrg3- nrg2))
    xscBarnsMat = np.exp(xscBarnsMat)/ Xrpy.const.au

    varErgMat = - delBindErgMat / Xrpy.const.kevPerRyd
    idxVarErgMat = varErgMat <= 0
    varErgMat[idxVarErgMat] = 1.  # don't want log of negative number or 0

    fppOrb = (Xrpy.const.invFineStruct* idxDelNeg* xscBarnsMat* nrgMat
                     / (4.*np.pi*Xrpy.const.kevPerRyd))
                     
    fpCorr = -0.5 * (xscBarnsMat* nrgAuMat
              * np.log( (nrgAuMat + bindErgAuMat)/ varErgMat) )
    fpCorr[idxVarErgMat] = np.nan
    fpCorr = np.nan_to_num(fpCorr)

    funTypeMat = np.tile(np.array([funType]).transpose(),(1,lErg))
    idxFt0 = funTypeMat==0    
    idxFt1 = funTypeMat==1
    idxFt2 = funTypeMat==2
    
    delErgMat= idxFt0* delBindErgMat
    idxPos = delErgMat > 0
    idxNeg = delErgMat <= 0
    """
    Anything with Mat in the name is [Norb x lErg]
    All the idxFtx are [Norb x lErg] to match delBindErgMat, etc.
    All the Gauss versions are [5*Norb x lErg]
    """    
    bindErgAuMat = np.tile(np.array([bindErgAu]).transpose(),(1,lErg))    
    bErgAuGauss  = np.repeat(bindErgAuMat,5,axis=0)    
    xscIntGauss = np.tile(np.array([xscIntSort.flatten()]).transpose(),(1,lErg))
    xscBarnsGauss = np.repeat(xscBarnsMat,5,axis=0)
    xscEdgeAuMat = np.tile(np.array([xscEdgeAu]).transpose(),(1,lErg))
    """
    xscEdgeAuMat is the 11th row of cross-sections turned into Norb x lErg
    All the other cross-sections have only 10 values (max)
    """
    xscEdgeGauss = np.repeat(xscEdgeAuMat,5,axis=0)
    nrgAuGauss = np.repeat(nrgAuMat,5,axis=0)
    
    """    
    These next two are the remnants of the lgndr() function,
    Starting as column vectors and turning into Norb repetitions
    of the 5xlErg arrays.  Each matrix is [5*Norb x lErg]
    """
    bb = np.tile(np.array([[0.11846344252810], [0.23931433524968], 
                         [0.28444444444444], [0.23931433524968], 
                         [0.11846344252810]]),(Norb,lErg))
    cc = np.tile(np.array([[0.95308992296933], [0.76923465505284], 
                  [0.50000000000000], [0.23076534494716], 
                  [0.04691007703067]]),(Norb,lErg))

    fpOrb = np.zeros((Norb,lErg))
    """
    if first element of idxFt1 is true do the calc
    it is only the first row that can be 1
    """
    if idxFt1[0,0]:
        orbTemp = bb[0:5,:] *(0.5 * bErgAuGauss[0:5,:]**3 * xscIntGauss[0:5,:] /
             (np.sqrt(cc[0:5,:])* (nrgAuGauss[0:5,:]**2 * cc[0:5,:]**2 -
             bErgAuGauss[0:5,:]**2 * cc[0:5,:])))
        fpOrb[0,:] = orbTemp.sum(axis=0)

    idxFt2Gauss = np.repeat(idxFt2,5,axis=0)
    vSmall = 1.e-15
    idx1 = np.abs(xscIntGauss-xscBarnsGauss) < vSmall 

    dd1 = idx1* (-2.* idxFt2Gauss* xscIntGauss* bErgAuGauss
          / cc**3)
    idx2 = np.abs(xscIntGauss-xscBarnsGauss) >= vSmall
    denomGauss = cc**3 * nrgAuGauss**2 - bErgAuGauss**2 / cc
    denomGauss = denomGauss* idxFt2Gauss* idx2
    
    idx3 = idx2* np.abs(denomGauss)< vSmall
    dd3 = idx3* -2.* (xscIntGauss* bErgAuGauss/ cc**3 )

    idxDenom = denomGauss==0
    denomGauss[idxDenom]=1.
    dd4 = 2. *((xscIntGauss/cc)* (bErgAuGauss/cc)**3 
           - bErgAuGauss* xscBarnsGauss* nrgAuGauss**2)/ denomGauss
    dd4[idxDenom] = np.nan
    dd4 = idx2* idxFt2Gauss* dd4
    dd = bb* (dd1+dd3+ dd4)
    dd = np.nan_to_num(dd)
    fpOrbforSum = dd.reshape(5,Norb*lErg,order='F').copy()
    fpOrbSummed = fpOrbforSum.sum(axis=0)
    fpOrbSummed = np.nan_to_num(fpOrbSummed)
    fpOrb = fpOrb+ fpOrbSummed.reshape(Norb,lErg,order='F').copy()
    
    idxFt0Gauss = np.repeat(idxFt0,5,axis=0)    
    fred = idxFt0Gauss* bb* (bErgAuGauss**3 * 
         (xscIntGauss - xscEdgeGauss* cc**2) 
         / (cc**2 * (cc**2 * nrgAuGauss**2 - bErgAuGauss**2)))
    fpOrb0forSum = fred.reshape(5,Norb*lErg,order='F').copy()
    fpOrb0Summed = fpOrb0forSum.sum(axis=0)
    fpOrb0Summed = np.nan_to_num(fpOrb0Summed)
    fpOrb0 = fpOrb0Summed.reshape(Norb,lErg,order='F').copy()
    dp = idxFt0Gauss* (nrgAuGauss**2 * cc**2 - bErgAuGauss**2)
    idxDp = dp == 0
    dp[idxDp]=1.  # get rid of divide by zero warnings.

    dd = idxFt0Gauss* (xscIntGauss* bErgAuGauss / cc**2)
    idxFt0GNZ = np.abs(dp) > vSmall
    fpOrb1forSum = idxFt0GNZ* ((xscIntGauss* bErgAuGauss**3 / cc**2
                   - bErgAuGauss* xscBarnsGauss* nrgAuGauss**2) / dp)
    # return ones back to nans
    fpOrb1forSum[idxDp] = np.nan
    fpOrb1forSum = np.nan_to_num(fpOrb1forSum)
    fpOrb1forSum = bb*( fpOrb1forSum+ dd - idxFt0GNZ* dd)
    fpOrb1forSum = fpOrb1forSum.reshape(5,Norb*lErg,order='F').copy()
    fpOrb1Summed = fpOrb1forSum.sum(axis=0)
    fpOrb1 = fpOrb1Summed.reshape(Norb,lErg,order='F').copy()
    fpOrb1 = fpOrb1 - idxPos*fpOrb1 + idxPos* fpOrb0
    
    fpOrb = fpOrb + fpOrb1
    
    # correction for index0 only
    
    fpCorrOld = idxFt0* fpCorr
    fpCorr1 = idxFt0* 0.5* (xscEdgeAuMat* bindErgAuMat**2 
                * np.log(-np.abs(nrgAuMat- bindErgAuMat)
                /(-bindErgAuMat-nrgAuMat)) / nrgAuMat)
    fpCorr1 = fpCorr1 - idxNeg*fpCorr1 + idxNeg* fpCorrOld
    fpCorr = fpCorr - idxFt0* fpCorr + idxFt0* fpCorr1
    fpOrb = fpOrb+ fpCorr
    fp = fpOrb.sum(axis=0)
    """
    Note: the Jensen correction to f' was subsequently shown to be incorrect
    (see L. Kissel and R.H. Pratt, Acta Cryst. A46, 170 (1990))
    and that the relativistic correction that Ludwig used is also
    wrong.  This section retained as comments for historical reasons.
    
		jensen_cor = -0.5*float(zed)
    1			*(d_energy_au/INV_FINE_STRUCT**2)**2

    Subtract relcor ala ludwig and change back to real*4

		fp = d_sum_fp+jensen_cor-relcor(zed)

    Kissel and Pratt give better corrections.  The relativistic correction
    that Ludwig used is (5/3)(E_tot/mc^2).  Kissel and Pratt say that this
    should be simply (E_tot/mc^2), but their correction (KPCOR) apparently
    takes this into account.  So we can use the old RELCOR and simply add
    the (energy independent) KPCOR term:
    """
    fp = Xrpy.const.finePi* fp + KPCor - relCor
    fpp = fppOrb.sum(axis=0)
    return fp, fpp

def cromerMcm(zed, lErg, energy):
    """
    McMaster parameterization of fp and fpp for H and He
    
    Helper routine for cromer. Should never be called independantly.
        
    Parameters
    __________
    zed : integer
        atomic number or name up to Uranium.  cromer does int buffering
    lErg : integer
        length of energy array since called from cromer
    energy : float
        single or array of energies in [eV]
           
    Returns
    _______
    fp : float
        single or array of floats.  Cross-section in [electrons/atom]
    fpp : float
        single or array of floats.  Cross-section in [electrons/atom]
    """
    fp = np.zeros(lErg)
    fpp = np.zeros(lErg)
    p1 = np.log( energy/1000.)
    p2 = p1**2
    p3 = p1**3
        
    if zed == 1:  # hydrogen
        fpp = (Xrpy.const.barns2Elec* energy * 
        np.exp( 2.44964 - 3.34953* p1 - 0.047137* p2 + 0.0070996* p3))
    elif zed == 2: # helium
        fpp = (Xrpy.const.barns2Elec* energy * 
        np.exp( 6.06488 - 3.2905* p1 - 0.107256* p2 + 0.0144465* p3))

    return fp, fpp
