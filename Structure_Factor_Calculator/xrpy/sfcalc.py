# -*- coding: utf-8 -*-
"""
Created on Tue Feb 24 12:29:38 2015

@author: bren
"""
from __future__ import print_function
import numpy as np
import Structure_Factor_Calculator.xrpy
import h5py
import os
pathStr = os.path.dirname(os.path.abspath(__file__))+os.sep+'Anomalous.h5'
h5A = h5py.File(pathStr,'r', driver='core',backing_store=False)

def sfCoef( zed, kval):
    """    
    Calculate atomic form factor and Compton scattering    
    Elastic from Cromer and Mann, Acta Cryst. A24, 321 (1968)   
    Compton from Balyuzi, Acta Cryst. A31 600 (1975).
    
    Parameters
    __________
    zed : integer or string
        atomic number or name up to Uranium
    kval : float
        kval expected as 4piSin(th)/lambda
        and converted to sin(th)/lambda
    
    Returns
    _______
    scatF : float
        single-valued or array of floats
    comp : float
        single or array of floats
    
    """
    zed=xrpy.element(zed)[0]    
    
    if zed < 1 or zed > 92:
        print(' Atomic number out of range')
        return 0., 0.

    isArray = True
    if type(kval) is int:
        kval= float(kval)
        isArray = False
    elif type(kval) is float:
        isArray = False
    elif type(kval) is list:
        kval = np.array(kval).astype('float')

    if isArray:
        lKval = len(kval)         
        if min(kval) < 0. or max(kval) > 75.:
            print(' k must be positive and less than 75')
            return 0., 0.

    else:
        lKval = 1
        if kval < 0. or kval > 75.:
            print(' k must be positive and less than 75')
            return 0., 0.
 
    sfCoef = h5A[str(zed)+'sfCoef'][:]
    sfFind = sfCoef[0:9]  # 9 total, 2x4 + 1
    sfComp = sfCoef[9:19]  # 10 total, 2x5

    # 0.0063326=(1/4pi)^2 to convert from k to sin(th)/lambda
    ksqr = -xrpy.const.inv4PiSqr* kval**2  # ksqr neg so coef in exp is positive
    scatF = np.ones(lKval)* sfFind[8]
    comp = np.ones(lKval)* float(zed)

    fInd = sfFind[0:8].reshape(2,4,order='F')  # only first 8 reshaped.
    fIndA = fInd[0,:]
    fIndB = fInd[1,:]
    for i, x in enumerate(fIndA):
        scatF= scatF + x* np.exp(fIndB[i]* ksqr)

    compR = sfComp.reshape(2,5,order='F')
    compA = compR[0,:]
    compB = compR[1,:]
    for i, x in enumerate(compA):
        comp= comp -(x * np.exp(compB[i]* ksqr))
        
    return scatF, comp

def sfWaas( zed, kval):
    """
    calculate atomic form factor
    From D. Waasmaier and A. Kirfel, Acta Cryst A51, 416 (1995)    
    Note that hydrogen is H(minus) ion divided by 2.
  
    Parameters
    __________
    zed : integer or string
        atomic number or name up to Uranium
    kval : float
        kval expected as 4piSin(theta)/lambda
        and converted to sin(th)/lambda
    
    Returns
    _______
    scatF : float
        single-valued or array of floats

    """
    zed=xrpy.element(zed)[0]
    if zed < 1 or zed > 92:
        print(' Atomic number out of range')
        return 0.

    isArray = True
    if type(kval) is int:
        kval= float(kval)
        isArray = False
    elif type(kval) is float:
        isArray = False
    elif type(kval) is list:
        kval = np.array(kval).astype('float')

    if isArray:
        lKval = len(kval)         
        if min(kval) < 0. or max(kval) > 75.:
            print(' k must be positive and less than 75')
            return 0.

    else:
        lKval = 1
        if kval < 0. or kval > 75.:
            print(' k must be positive and less than 75')
            return 0.

    waas = h5A[str(zed)+'waas'][:]
    waasA = waas[0:5]
    waasB = waas[5:10]
    waasC = waas[10]
        
    scatF = np.ones(lKval)*waasC
        
    # 0.0063326=(1/4pi)^2 to convert from k to sin(th)/lambda
    ksqr = -xrpy.const.inv4PiSqr* kval**2  # set ksqr neg so waasB is positive
    for i, x in enumerate(waasA):
        scatF= scatF + (x* np.exp(waasB[i]*ksqr))
        
    return scatF
