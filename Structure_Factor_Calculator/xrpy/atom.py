# -*- coding: utf-8 -*-
"""
Created on Tue Feb 24 12:30:27 2015

@author: bren
"""
from __future__ import print_function
import numpy as np
import Structure_Factor_Calculator.xrpy as Xrpy
import h5py
import os
import re

pathStr = os.path.dirname(os.path.abspath(__file__))+os.sep+'Anomalous.h5'
h5A = h5py.File(pathStr,'r', driver='core',backing_store=False)

def atomData(zed):
    """
    Return Atomic mass, density, #Orbits and edge energies (in keV)

    Parameters
    __________
    zed : integer or string
        atomic number or name up to Uranium

    Returns
    _______
    AMU : float
        Atomic mass [gms]
    rho : float
        density [gms/cc]
    Norb : integer
        number of atomic shells in the atom
    edges : float
        array of binding energies [keV]!!
        
    """
    zed = Xrpy.element(zed)[0]    
    AMU=rho=Norb=edges=0.
    if zed <1 or zed > 92:
        print(' Atomic number out of range')
    else:
        AMU = h5A['AMU'][zed-1]
        rho = h5A['Rho'][zed-1]
        Norb = int(h5A['Norb'][zed-1])
        edges = h5A[str(zed)+'BindErg'][:]
    
    return AMU, rho, Norb, edges

def element(zed):
    """
    Given a name, or number returns an atomic #, symbol and full name
    
    up to Uranium.  If invalid returns 0

    Parameters
    __________
    zed : integer or string
        atomic number or name up to Uranium
    
    Returns
    _______
    zed : integer
        atomic number 1-92
    symbol : string
        symbolic name for element
    elName : string
        full name of element
    """
    elemSyms= ['H','He','Li','Be','B','C','N','O','F','Ne',
               'Na','Mg','Al','Si','P','S','Cl','Ar','K','Ca',
               'Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn',
               'Ga','Ge','As','Se','Br','Kr','Rb','Sr','Y','Zr',
               'Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd','In','Sn',
               'Sb','Te','I','Xe','Cs','Ba','La','Ce','Pr','Nd',
               'Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb',
               'Lu','Hf','Ta','W','Re','Os','Ir','Pt','Au','Hg',
               'Tl','Pb','Bi','Po','At','Rn','Fr','Ra','Ac','Th',
               'Pa','U','Np','Pu']

    elemNames= ['Hydrogen','Helium','Lithium','Beryllium','Boron',
                'Carbon','Nitrogen','Oxygen','Fluorine','Neon',
                'Sodium','Magnesium','Aluminium','Silicon','Phosphorus',
			    'Sulfur','Chlorine','Argon','Potassium','Calcium',
                'Scandium','Titanium','Vanadium','Chromium','Manganese',
                'Iron','Cobalt','Nickel','Copper','Zinc','Gallium',
                'Germanium','Arsenic','Selenium','Bromine','Krypton',
			    'Rubidium','Strontium','Yttrium','Zirconium','Niobium',
                'Molybdenum','Technetium','Ruthenium','Rhodium',
			    'Palladium','Silver','Cadmium','Indium','Tin','Antimony',
			    'Tellurium','Iodine','Xenon','Cesium','Barium','Lanthanum',
			    'Cerium','Praseodymium','Neodymium','Promethium','Samarium',
                'Europium','Gadolinium','Terbium','Dysprosium','Holmium',
                'Erbium','Thulium','Ytterbium','Lutetium','Hafnium',
			    'Tantalum','Tungsten','Rhenium','Osmium','Iridium','Platinum',
			    'Gold','Mercury','Thallium','Lead','Bismuth','Polonium',
			    'Astatine','Radon','Francium','Radium','Actinium','Thorium',
			    'Protactinium','Uranium','Neptunium','Plutonium']
    symbol = ' '
    elName = ' '
    name= str(zed)
    zed = 0

    if name.startswith('-'):
        print(' Atomic number must be positive')
        return 0, symbol, elName
    
    dotLen = name.find('.')  # if a float, convert to int
    if dotLen >= 0:  # -1 if not present
        #print(' Integers please!')
        return 0, symbol, elName

    if name.isdigit():
        zed= int(name)
        if zed < 1 or zed > 92:
            print(' Atomic number out of range')
            return 0, symbol, elName
        else:
            symbol = elemSyms[zed-1]
            elName = elemNames[zed-1]

    elif name.isalpha():
        nameCap = name.title()
        if len(nameCap) <= 2:
            if nameCap in elemSyms:
                zed=elemSyms.index(nameCap)
                elName= elemNames[zed]
                symbol= nameCap
                zed= zed+1
            else:
                zed=0
                #print(' Element not recognized')
        else:
            if nameCap in elemNames:
                zed= elemNames.index(nameCap)
                symbol= elemSyms[zed]
                elName= nameCap
                zed= zed+1
            else:
                zed = 0
        
    """
    try:
        basestring  # attempt to evaluate basestring
        def isstr(s):
            return isinstance(s, basestring)
    except NameError:
        def isstr(s):
            return isinstance(s, str)
    """        

    return zed, symbol, elName

def absEdges(zed):
    """
    Display binding and fluorescent energies for an element
        
    Parameters
    __________
    zed : integer or string
        atomic number or name up to Uranium
    
    """
    abNames= ['K ','L1','L2','L3','M1','M2','M3','M4','M5','N1',
              'N2','N3','N4','N5']
    flNames= ['Ka2','Ka1','Li','Le','Kb3','Lb4','Kb1','Lb3',
              'Lb1','La2','La1','Lg3','Lg1','Lb2']

    zed, symbol, elemName = Xrpy.element(zed)
    if zed == 0:
        return
    print(' Element %d: %s' % ( zed, elemName))
    amu, rho, nedges, kedges = Xrpy.atomData(zed)
    edges= 1000. * kedges
    if nedges > 14:
        nedges = 14

    result= []
    if nedges > 2:
        result.append([0, edges[0]- edges[2]])
    if nedges > 3:
        result.append([1, edges[0]- edges[3]])
    if nedges > 4:
        result.append([2, edges[3]- edges[4]])
        result.append([3, edges[2]- edges[4]])
    if nedges > 6:
        result.append([4, edges[0]- edges[5]])
        result.append([5, edges[1]- edges[5]])
    if nedges > 8:
        result.append([6, edges[0]- edges[6]])
        result.append([7, edges[1]- edges[6]])
    if nedges > 10:
        result.append([8, edges[2]- edges[7]])
        result.append([9, edges[3]- edges[7]])
    if nedges > 11:
        result.append([10, edges[3]- edges[8]])
    if nedges > 12:
        result.append([11, edges[1]- edges[11]])
    if nedges > 13:
        result.append([12, edges[2]- edges[12]])
        result.append([13, edges[3]- edges[13]])
    arresult = np.array(result)
    idx= np.argsort(arresult[:,1])
    arresult= arresult[idx,:]
    arresult= np.flipud(arresult)
    for i, x in enumerate(arresult):
        print(' %d %s %7.1f %s %7.1f' % (i, abNames[i], edges[i], 
               flNames[int(arresult[i,0])], arresult[i,1]))
    
    idx = arresult.shape[0]
# if there are more edges than fluorescent lines, print those out
    if idx < nedges:
        for i in range(idx, nedges):
            print(' %d %s %7.1f' % (i, abNames[i], edges[i]))
    
    return

def parseFormula(eStr):
    """    
    Convert a formula into arrays of elements and fractions
    
    Parameters
    ____________
    eStr : string containing the formula
           Fractional components are allowed but only if
           preceded by a '.'  If '0.xx' the parser will
           gack.
    So: H2O is fine, PbZr.35Ti.65O3 is fine, 
    but In0.5Ga0.5As will break.

    Returns
    __________
    elem : array of elements (string)

    eNum : array of proportions (float)
    
    """
    formList = re.findall( r'([A-Z][a-z]*)(\.*\d*)', eStr)    
    formList = [x if(x[1] != '') else (x[0],'1') for x in formList]
    #print(formList)
    if len(formList) > 1:
        elem,uNum = map(list, zip(*formList))
        eNum = [float(x) for x in uNum]
    else:
        elem = ' '
        eNum = 0
    return elem, eNum
