#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 09:42:15 2019

@author: jamespittard
"""
import numpy as np

class Diff_Environment(object):   
    """Diffraction environment is an object containing required general information.  
    This object must be passed into structure factor calculations, as well as the initialisation
    of crystal objects."""
    
    def __init__(self,temperature_K,crystal_system,unit_cell_params,angle_plane_wavelength):
        
        self.temp_K = temperature_K
        
        self.angle_rad = angle_plane_wavelength[0]
        self.angle_deg = self.angle_rad*(180/np.pi)
        self.hkl = angle_plane_wavelength[1]
        self.wavelength_A = angle_plane_wavelength[2]
        
        self.d_A = Diff_Environment.d_hkl(unit_cell_params,self.hkl,crystal_system)

    def update(self,temperature_K,crystal_system,unit_cell_params,angle_plane_wavelength):

        self.temp_K = temperature_K
        
        self.angle_rad = angle_plane_wavelength[0]
        self.angle_deg = self.angle_rad*(180/np.pi)
        self.hkl = angle_plane_wavelength[1]
        self.wavelength_A = angle_plane_wavelength[2]
        
        self.d_A = Diff_Environment.d_hkl(unit_cell_params,self.hkl,crystal_system)
        
    def information(self):
        """When called, this will print out attributes of the object."""
        
        print("-"*20)
        print("\nTemperature = ",self.temp_K,"K")
        print("Diffracting Angle = ",self.angle_rad,"rad or ",self.angle_deg, "deg")
        print("Plane = ",self.hkl)
        print("Wavelength = ",self.wavelength_A,"Angstrom")
        print("Plane spacing = ",self.d_A,"Angstrom\n")
        print("-"*20)   
    
    
    def d_hkl(unit_cell_params,plane,crystal_system):
        """Calculates plane spacing for all crystal systems."""
        
        a_A = unit_cell_params[0]
        b_A = unit_cell_params[1]
        c_A = unit_cell_params[2]
        
        alpha_rad = unit_cell_params[3]
        beta_rad = unit_cell_params[4]
        gamma_rad = unit_cell_params[5]
        
        h = plane[0]
        k = plane[1]
        l = plane[2]
        
        if crystal_system == "Cubic":
            recip_d_squared_invA2 = (h**2 + k**2 + l**2)*(1/a_A**2)
            
        if crystal_system == "Tetragonal":
            recip_d_squared_invA2 = ((h**2 + k**2) + (l**2)*((a_A/c_A)**2))*(1/a_A**2)
            
        if crystal_system == "Orthorhombic":
            recip_d_squared_invA2 = ((h/a_A)**2 + (k/b_A)**2 + (l/c_A)**2)
            
        if crystal_system == "Hexagonal": # need to add other equations for these
            recip_d_squared_invA2 = ((4/3)*(h**2 + k**2 + h*k) + (l**2)*((a_A/c_A)**2))*(1/a_A**2)
        
        if crystal_system == "Rhombohedral":
            numerator = (h**2 + k**2 + l**2)*(np.sin(alpha_rad)**2)
            numerator += 2*(h*k+k*l+h*l)*(np.cos(alpha_rad)**2-np.cos(alpha_rad))
            
            denominator = (a_A**2)*(1-3*(np.cos(alpha_rad)**2)+2*(np.cos(alpha_rad)**3))
            
            recip_d_squared_invA2 = numerator/denominator
            
        if crystal_system == "Monoclinic":
            
            recip_d_squared_invA2 = (((h/a_A)**2)+((k*np.sin(beta_rad)/b_A)**2) + ((l/c_A)**2) - 2*h*l*np.cos(beta_rad)/(a_A*c_A))*(1/np.sin(beta_rad))**2
            
        if crystal_system == "Triclinic":
            numerator = (h*np.sin(alpha_rad)/a_A)**2 + (k*np.sin(beta_rad)/b_A)**2 + (l*np.sin(gamma_rad)/c_A)**2
            numerator += 2*k*l*(np.cos(beta_rad)*np.cos(gamma_rad) - np.cos(alpha_rad))/(b_A*c_A) + 2*h*l*(np.cos(gamma_rad)*np.cos(alpha_rad) - np.cos(beta_rad))/(a_A*c_A) + 2*h*k*(np.cos(alpha_rad)*np.cos(beta_rad) - np.cos(gamma_rad))/(a_A*b_A)
                        
            denominator = 1 - np.cos(alpha_rad)**2 - np.cos(beta_rad)**2 - np.cos(gamma_rad)**2
            denominator += 2*np.cos(alpha_rad)*np.cos(beta_rad)*np.cos(gamma_rad)
            
            recip_d_squared_invA2 = numerator/denominator
        
        hkl_mag = []
        for count in range (0,3):            
            hkl_mag.append(abs(plane[count]))  
                 
        if sum(hkl_mag)> 1e-4:
            d_A = np.sqrt(1/recip_d_squared_invA2)
        else:
            d_A = None
        
        return d_A

    def cell_volume(unit_cell_params):
        a_A = unit_cell_params[0];
        b_A = unit_cell_params[1]
        c_A = unit_cell_params[2]
        
        ca = np.cos(unit_cell_params[3])
        cb = np.cos(unit_cell_params[4])
        cg = np.cos(unit_cell_params[5])

        volume = a_A*b_A*c_A * np.sqrt(1 + 2*ca*cb*cg - ca**2 - cb**2 - cg**2)
        return volume
