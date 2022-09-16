# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 15:54:22 2015

@author: bren
"""
import numpy as np
import Structure_Factor_Calculator.xrpy

def muRef(zed, energy, thick):
    """    
    Calculate absorption mu and Refractive Index
        
    Parameters
    __________
    zed : integer or string
        atomic number or name up to Uranium
    energy : float
        single or array of energies in [eV]
    thick : float
        single valued thickness [microns]
    
    Returns
    _______
    mu : float
        single or array of floats.  Absorption in [1/microns]
    tran : float
        single or array of floats.  Transmitted Intensity
    refract : complex
        Refractive index alpha - i*beta [dimensionless]
    """
    zed = xrpy.element(zed)[0]  # Sanitize zed
    
    if type(energy) is int:
        energy= float(energy)
    elif type(energy) is list:
        energy = np.array(energy).astype('float')

    fp, fpp = xrpy.anomal(zed,energy)
    ray, comp = xrpy.rayComp(zed,energy)
    AMU, rho = xrpy.atomData(zed)[0:2]
    mu = (xrpy.const.fpp2mu * rho/AMU) * (fpp + ray + comp)/ energy
    tran = np.exp(-mu* thick)
    refrConst = xrpy.const.fp2refr * rho / (energy**2 * AMU)
    alpha= 1- (refrConst * (zed + fp))
    #Both fpp and compton are "absorptive", i.e. imaginary terms.
    beta= refrConst * (fpp+ comp)
    refrac= alpha - beta * 1j

    return mu, tran, refrac
    
def muRefMatl(elem, eNum, energy, thick, rhoIn):
    """    
    Calculate absorption mu for a material
        
    Parameters
    __________
    elem : array of elements (string)

    eNum : array of proportions (float)
    
    energy : float
        single or array of energies in [eV]
    thick : float
        single valued thickness [microns]
    rhoIn : float
        single valued density [gms/cm^3]
    
    Returns
    _______
    mu : float
        single or array of floats.  Absorption in [1/microns]
    tran : float
        single or array of floats.  Transmitted Intensity
    rhoAve : float
        sum of fractional densities.
    amuAve :  float
        sum of fractional AMUs
    refract : Complex
        Refractive Index: alpha - i*beta [dimensionless]
    """
    mu = 0.0
    tran = 0.0
    rhoAve = 0.0
    amuAve = 0.0
    eNumSum = np.sum(eNum)
    eNumNorm = eNum/eNumSum
    #print(eNumNorm)
    mus = 0.
    rhoAve = 0.
    amuAve = 0.
    beta = 0.
    alpha = 0.
    
    if type(energy) is int:
        energy= float(energy)
    elif type(energy) is list:
        energy = np.array(energy).astype('float')

    for sym, eFrac in zip(elem,eNumNorm):
        zed = xrpy.element(sym)[0]  # Sanitize zed
        if zed > 0:
            fp, fpp = xrpy.anomal(zed,energy)
            ray, comp = xrpy.rayComp(zed,energy)
            AMU, rho = xrpy.atomData(zed)[0:2]
            mus = mus + eFrac* (fpp + ray + comp)
            beta = beta + eFrac* (fpp + comp)
            alpha = alpha + eFrac*(zed + fp)
            rhoAve = rhoAve + eFrac* rho
            amuAve = amuAve + eFrac* AMU 
        else:
            return mu, tran, rhoAve, amuAve
    if rhoIn > 0.0:
        rhoAve = rhoIn
    mu = (xrpy.const.fpp2mu * mus* rhoAve)/(amuAve * energy)
    tran = np.exp(-mu* thick)
    refrConst = xrpy.const.fp2refr * rhoAve / (energy**2 * amuAve)
    alpha= 1- (refrConst * alpha)
    #Both fpp and compton are "absorptive", i.e. imaginary terms.
    refract = alpha - refrConst * beta * 1j

    return mu, tran, rhoAve, amuAve, refract
    """
    Notes on various constants used in these calculations:
    
    f'' [electrons/atom] to mu in [1/microns]
    mu= (4piNe^2/mwc)*f'' (James, p. 138)
       = 2*rsubelectron*hc*Avogadro*rho*f''/ atomic_mass*energy
    Thus 4.208e3 = 2*rsube*hc*Av*10^-16[cm^2/AA^2]*10^-4[cm/micron]
         4.208e3 [AA/electron][ev-AA][atoms/mole][cm^2/AA^2][cm/micron]
         4.208e3 [ev-atoms-cm^3/electron-mole-micron]
    where rsube= 2.8179e-5 [A/electron]
        hc= 12398.4244 [eV-A]
        Av= 6.022e23 [atoms/mole]
        rho in [gms/cm^3]
        amu in [gms/mole]
        energy in [eV]
        fpp in [electrons/atom]
    mu= (4208.031548* rho*( fpp+ ray+ comp))/( amu* energy) [1/microns]

    How do we get to the constant 415.181?  Let's look back at the notes in    
    Notebook IV, p 111:  alpha = 2*pi*e^2*N/(m*omega^2) =
    (2*pi) *(h/2*pi*erg)^2 * c^2 * e^2/(m*c^2) * N
    = (hc)^2/(2*pi) * r_e (the electron radius)
    now N [atoms/cc] = (Avogadro * rho)/(amu *10^24) to get [atoms/AA^3]
    so dimensionally we have:
    alpha= (hc^2/2pi [eV-AA]^2 * r_e [AA] * N_a [atoms/AA^3] * rho/amu)/erg^2 [eV^2]
    and alpha becomes dimensionless.
    constant= (fhc^2*f_e_rad*6.022e23)/(2*pi*1e24) = 415.17
    I get  alpha= 0.0586; beta= 0.0018 for Si at 91.6 eV

    Eric's equation is:
    nindex = 1- (1/2*pi)*N*r_0*lambda^2(f1+if2)
    lambda= 12398.42/91.6
    N_atm= 0.6022*2.32/28.085
    [fp,fpp]=henke(14,91.6)
    nindex= 1- N_atm*f_e_rad*lambda^2*(fp+1i*fpp)/(2*pi)
    and he gets alpha= .0012; beta= .0018 for Si at 91.6 eV
    """