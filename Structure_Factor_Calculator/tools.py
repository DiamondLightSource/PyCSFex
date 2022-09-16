#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 09:31:39 2019

@author: jamespittard
Modifications by A. Baron
"""

import numpy as np
import pandas as pd
import Structure_Factor_Calculator.physical_constants as pc

from Structure_Factor_Calculator.checks import Check

class Tools:
    """Simple class containing a variety of general functions which are used throughout 
    the script.""" 
    
    def row_selecter(atom_column,atom,full_array):
        """Scans 'atom_column' for 'atom' and then extracts the row of 
        coefficients relating to the atoms from 'full_array'."""
        
        row_index = atom_column.index(atom)
    
        row = full_array[row_index]
        
        Check.length(row,13)
    
        return row
    
    def data_import():
        """Imports table of coefficients from an Excel file. """
        
        form_factor_table = np.array(pd.read_excel("../Form_factor_coefficients.xlsx", sep= " "))
        form_factor_table = np.delete(form_factor_table,[0],axis=0)
        
        
        return form_factor_table
    
    def unit_cell_check(point):
        """Insures atom coordinates are within the unit cell and shifts them if not. """
        
        def zero_to_one(x):
             
            while x<0:
                x+=1
                                
            while x>1:
                x+=-1

            return x
        
        for index in range (0,3):
            point[index] = zero_to_one(point[index])
                
        point = [point[0],point[1],point[2]]
                
        return point

    def cross3x3(v1,v2):
        """manual cross product because numpy is SLOW for 3-vectors"""
        r0 = v1[1]*v2[2] - v1[2]*v2[1]
        r1 = v1[2]*v2[0] - v1[0]*v2[2]
        r2 = v1[0]*v2[1] - v1[1]*v2[0]
        cp = [r0,r1,r2]
        return cp
    
    def AllRecipVectors(lattice_vectors):
        """Calculates all 3 reciprocal vectors.  Avoids np.cross function"""
        a = lattice_vectors[0] ;        b = lattice_vectors[1];        c = lattice_vectors[2]
        numerator1  = (Tools.cross3x3(b,c))
        numerator2  = (Tools.cross3x3(c,a))
        numerator3  = (Tools.cross3x3(a,b))
        denominator = np.dot(a,numerator1)
        recip1 = numerator1/denominator ;         recip2 = numerator2/denominator ;        recip3 = numerator3/denominator
        return recip1,recip2,recip3
    
    def cartesian_converter(atom_lattice_coord, LatticeVectors):
        """Converts lattice coordinates to cartesian."""
           
        a_vec = LatticeVectors[0]
        b_vec = LatticeVectors[1]
        c_vec = LatticeVectors[2]
                    
        u = atom_lattice_coord[0]
        v = atom_lattice_coord[1]
        w = atom_lattice_coord[2]
            
        Cartesian = u*a_vec + v*b_vec + w*c_vec
        
        return Cartesian
    
    def lattice_converter(cartesian,lattice_unit_cell_params,lattice_vectors):
        """Converts cartesian coordinates to lattice """
        
        a = lattice_unit_cell_params[0]
        b = lattice_unit_cell_params[1]
        c = lattice_unit_cell_params[2]
        
        a_vec = lattice_vectors[0]
        b_vec = lattice_vectors[1]
        c_vec = lattice_vectors[2]
                            
        alpha = lattice_unit_cell_params[3]
        beta = lattice_unit_cell_params[4]
        gamma= lattice_unit_cell_params[5]
        
        sg = np.sin(gamma); cg = np.cos(gamma)
        ca = np.cos(alpha); cb = np.cos(beta)
        
        V = np.dot(a_vec,(Tools.cross3x3(b_vec,c_vec)))
                                      
        conversion_matrix = np.array(([1/a,-cg/(a*sg),b*c*(ca*cg-cb)/(V*sg)], [0,1/(b*sg),a*c*(cb*cg-ca)/(V*sg)],[0,0,(a*b*sg)/V]))
        
        lattice = np.dot(conversion_matrix,cartesian)
        
        return lattice
    
    def energy_calc_eV(wavelength_A):
        """Calculates photon energy in electron volts."""
        
        energy_eV = (pc.h_Js*pc.c_m_per_s*((wavelength_A*pc.electron_charge_C)**(-1)))*1e10
    
        return energy_eV

    def set_to_zero(matrix, no_of_columns,no_of_rows,cut_off):
        """This sets values close to zero (determined by the 'cut_off' value) to zero."""
        
        for count in range (0,no_of_columns):
            for count2 in range (0,no_of_rows):                
                
                if abs(matrix[count][count2]) < cut_off:
                    matrix[count][count2] = 0
        
        return matrix
    
    def min_energy_calc(d_A):
        """Calculates minimum photon energy to fulfill Bragg condition,
        d refers to plane spacing."""
                       
        max_wavelength_A = 2*d_A
            
        min_energy_eV = Tools.energy_calc_eV(max_wavelength_A)
        
        return min_energy_eV
