#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 09:43:39 2019

@author: jamespittard
"""
import numpy as np
from Structure_Factor_Calculator.tools import Tools

import sys
sys.path.append("Structure_Factor_Calculator")

import Structure_Factor_Calculator.xrpy as Xrpy

class Atom(object):
    
    """Atom objects represent individual atoms within the crystal. These are constructed 
    within, and form part of, the crystal object. """

    data_table = Tools.data_import()

    def __init__(self,element,coord,environment_obj,latt_vec_A,beta_matrix=None,M_and_TD=None):       

        self.element = element
        self.coordinates = coord
        self.cartesian_coordinates = Tools.cartesian_converter(coord,latt_vec_A)
        self.Environment = environment_obj
        self.form_factor_coefficients = Atom.data_selecter(Atom.data_table,self.element)
        self.form_factor = self.scattering_factor()[0]
        self.f0 = self.scattering_factor()[2]
        self.form_factor_fwd = self.scattering_factor()[3]
        if (beta_matrix is None) and (M_and_TD is None):
            raise ValueError("Either a beta matrix, or the atomic mass and Debye temperature, are required.")
        elif (beta_matrix is not None) and (M_and_TD is not None):
            raise ValueError("Input either a beta matrix, or the atomic mass and Debye temperature, but not both.")
        else:
            self.beta_matrix = beta_matrix
            self.M_and_TD = M_and_TD

    def update(self,coord,environment_obj,latt_vec_A,beta_matrix=None,M_and_TD=None):
        self.coordinates = coord
        self.cartesian_coordinates = Tools.cartesian_converter(coord,latt_vec_A)
        self.Environment = environment_obj
        self.form_factor = self.scattering_factor()[0]
        self.f0 = self.scattering_factor()[2]
        self.form_factor_fwd = self.scattering_factor()[3]
        if (beta_matrix is None) and (M_and_TD is None):
            raise ValueError("Either a beta matrix, or the atomic mass and Debye temperature, are required.")
        elif (beta_matrix is not None) and (M_and_TD is not None):
            raise ValueError("Input either a beta matrix, or the atomic mass and Debye temperature, but not both.")
        else:
            self.beta_matrix = beta_matrix
            self.M_and_TD = M_and_TD
        
    def information(self):
        """When called, this will print out attributes of the object."""
        
        print("-"*20)
        print("\nElement = ",self.element)
        print("Coordinates = ",self.coordinates)
        print("Scattering Factor = ",self.form_factor)
        print("Normal component of scattering factor for diffraction = ",self.f0)
        print("Scattering factor in forward direction = ",self.form_factor_fwd)
        if (self.M_and_TD is None):
            print("Beta Matrix = ", self.beta_matrix)
        elif (self.beta_matrix is None):
            print("Atomic mass in amu = ",self.M_and_TD[0])
            print("Debye temperature = ",self.M_and_TD[1])
        print("-"*20)
        
    def data_selecter(data_table,element=None):
        """Selects the appropriate row of coefficients from an excel file containing
        data from Waasmaier and Kirfel"""
    
        form_factor_table_atom_column = list(data_table[:,[0]])
        form_factor_coefficients = 0
        
        if element!= None:
            form_factor_coefficients = Tools.row_selecter(form_factor_table_atom_column,element,data_table)

        return form_factor_coefficients, form_factor_table_atom_column

        
    def scattering_factor(self):
        """Determines corrected scattering factor, does not include Debye Waller/
        thermal elipsoid correction"""
               
        theta_rad = self.Environment.angle_rad
        wavelength_A = self.Environment.wavelength_A

        coefficients = self.form_factor_coefficients[0]
        c = coefficients[-1]
        s_invA = np.sin(theta_rad)/wavelength_A
        photon_energy_eV = Tools.energy_calc_eV(wavelength_A)

        element_list = []
        element_list_fwd = []
        f = 0
        index = 1
                    
        while index < 10:
            
            a = float(coefficients[index+1])

            b = float(coefficients[index+2])
        
            sum_part = a*np.exp(-b*s_invA**2)
            sum_part_fwd = a
            element_list.append(sum_part)
            element_list_fwd.append(sum_part_fwd)
            index+=2

        f0 = sum(element_list) + c
        f0_fwd = sum(element_list_fwd) + c

        Z = int(coefficients[1])

        anom_disp_corrections = Xrpy.cromer(Z,photon_energy_eV)
        Rayleigh_Compton_corrections = Xrpy.rayComp(Z,photon_energy_eV)
        
        f_prime = anom_disp_corrections[0]
        f_2prime = anom_disp_corrections[1]
        
        Rayleigh = Rayleigh_Compton_corrections[0]
        Compton = Rayleigh_Compton_corrections[1]
        
        i = 1j
        
        imaginary = f_2prime + Rayleigh + Compton 
        
        f = f0 + f_prime + i*imaginary
        f_fwd = f0_fwd + f_prime + i*imaginary
        
        return f,coefficients,f0,f_fwd