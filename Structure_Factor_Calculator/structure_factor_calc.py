#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 09:50:02 2019

@author: jamespittard
"""
import numpy as np
import math
import Structure_Factor_Calculator.physical_constants as pc
from Structure_Factor_Calculator.tools import Tools

class Structure_Factor:
    
    def debye_waller(crystal,scat_vec_invA,scat_vec_sqr_invA2,atom,test = None):
        """Calculate anisotropic DW correction for thermal ellipsoids """

        latt_vectors_A = crystal.lattice_vectors_A
                
        v_Direct = Tools.lattice_converter(scat_vec_invA,crystal.unit_cell_params,latt_vectors_A)              
        v_Direct_t = np.transpose(v_Direct)
        G_A2 = crystal.G_mtrx_A2
        Gt_A2 = np.transpose(G_A2)
        beta = atom.beta_matrix
        numerator = np.dot(v_Direct_t,np.dot(Gt_A2,np.dot(beta,np.dot(G_A2,v_Direct))))
        denominator = 2*(np.pi**2)*np.dot(v_Direct_t,np.dot(G_A2,v_Direct))
        mean_sqr_dis_A2 = numerator/denominator
        
        DW = np.exp(-2*(np.pi**2)*scat_vec_sqr_invA2*mean_sqr_dis_A2)
        return DW

    def isotropic_debye_waller(M_amu,T_Debye_K,scat_vec_sqr_invA2,Temp_K):

        """
        Estimates the isotropic Debye-Waller factor, exp[-W], for an atom of mass M_amu
            W = integral(E,0,Ed) of (Er/E) * g(E)* Coth(E /(2 kB T)) / 2
        where g(E) = DOS = 3 E^2/Ed^3 for a debye model giving
            W = integral(E,0,Ed) of 3*Er*E Coth(E /(2 kB T)) / 2 / Ed^3
        This is a simple integral of a slowly varying function -> just do the sum.
        In fact, 10 divisions gets you most of the way, and 100 divisions is more than enough for ppm level
        This formula is from V. F. Sears and S. A. Shelley, "Debye-Waller Factor for Elemental Crystals," Acta Cryst. A47, 441-446 (1991).
        NOTE: Sears & Shelley define q = 4*pi*sin(theta)/lambda but we have defined q = 2*sin(theta)/lambda - hence the extra factor 4*pi**2 for the recoil Er_meV.
        """
        # recoil energy
        sf = 1000. * pc.hbar_Js**2 * 1e20 / pc.kg_per_amu / pc.electron_charge_C
        Er_meV = sf * (4*math.pi**2*scat_vec_sqr_invA2) / M_amu / 2.
        Edebye_meV = 1000. * T_Debye_K * pc.kB_eV_per_K
        kT_meV = 1000. * Temp_K * pc.kB_eV_per_K
        ne = 100  # 10 points is mostly enough
        de = Edebye_meV / float(ne)

        sum = 0.
        for ie in range(ne) :
            e_meV = de * float(ie) + de/2.
            sum += e_meV / math.tanh(e_meV/kT_meV/2.)
        W = sum * de * 3. * Er_meV / 2. / Edebye_meV**3
        DW = math.exp(-W)

        return DW,W,Er_meV,Edebye_meV

    def atom_scat_phase_DW(crystal,Miller,scat_vec_invA,scat_vec_sqr_invA2,atom,DW_Previous,DW_Refresh,test=None):
        """Makes final DW correction then multiplies atomic scattering factor by the phase information"""
        
        u = atom.coordinates[0]
        v = atom.coordinates[1]
        w = atom.coordinates[2]

        h = Miller[0]
        k = Miller[1]
        l = Miller[2]

        f = atom.form_factor
        f_fwd = atom.form_factor_fwd
        
        i = 1j
        
        exponent = 2*np.pi*i*(h*u + k*v + l*w)
        DW = 1
        
        hkl_mag = []
        for count in range (0,3):            
            hkl_mag.append(abs(Miller[count]))
        
        if (sum(hkl_mag) > 1e-4):
            if (atom.beta_matrix is not None):
                DW = Structure_Factor.debye_waller(crystal,scat_vec_invA,scat_vec_sqr_invA2,atom,test)
            else:
                if (DW_Refresh):
                    M_amu = atom.M_and_TD[0]
                    T_Debye_K = atom.M_and_TD[1]
                    Temp_K = crystal.temperature_K
                    DW = Structure_Factor.isotropic_debye_waller(M_amu,T_Debye_K,scat_vec_sqr_invA2,Temp_K)[0]
                else:
                    DW = DW_Previous

        eetemp = np.exp(exponent)
        sum_component = f*eetemp*DW # for structure factor F
        sum_component_bar = f*DW/eetemp # for structure factor FBAR
        sum_component_fwd = f_fwd
        return sum_component, sum_component_bar, sum_component_fwd, DW
    
    def F_hkl(crystal,diff_environment,test=None): 
        """Sums the scattering effects of individual atoms to give structure factor"""
        atom_list = crystal.atoms
        value_list = []
        value_list_bar = []
        value_list_fwd = []

        no_of_atoms = len(atom_list)

        h = diff_environment.hkl[0]
        k = diff_environment.hkl[1]
        l = diff_environment.hkl[2]  
        Miller = (h,k,l)

        [a_star_invA,b_star_invA,c_star_invA] = Tools.AllRecipVectors(crystal.lattice_vectors_A)
        scat_vec_invA = h*a_star_invA + k*b_star_invA + l*c_star_invA # scattering vector in Cartesian coordinates
        scat_vec_sqr_invA2 = np.dot(scat_vec_invA,scat_vec_invA)

        DW_Previous = -1
        
        for index in range (0,no_of_atoms): 
            #Multiplying phase and corrected f together for each atom and appending
            atom = atom_list[index]

            # If the thermal motion is isotropic, we save time by preventing repeated calculations of the Debye-Waller factor as below.
            if (atom.beta_matrix is not None): # anisotropic Debye-Waller factor
                DW_Refresh = True # always perform a fresh Debye-Waller calculation, since different atoms have different thermal ellipsoids.
            elif (atom.M_and_TD is not None): # isotropic Debye-Waller factor
                if (index == 0):
                    M_and_TD_Current = atom.M_and_TD
                    DW_Refresh = True # perform a fresh Debye-Waller calculation
                else:
                    M_and_TD_Previous = M_and_TD_Current
                    M_and_TD_Current = atom.M_and_TD
                    if (M_and_TD_Previous is None): # for the unlikely case where the previous atom's thermal motion is treated anisotropically while the current atom's is treated isotropically.
                        DW_Refresh = True # perform a fresh Debye-Waller calculation
                    else: # the previous atom and the current atom both have isotropic thermal motion
                        MDiffAbs = abs(M_and_TD_Current[0] - M_and_TD_Previous[0])
                        TDDiffAbs = abs(M_and_TD_Current[1] - M_and_TD_Previous[1])
                        if (MDiffAbs < 1e-8) and (TDDiffAbs < 1e-8): # the current atom and the previous atom have the same properties
                           DW_Refresh = False # no need for a fresh Debye-Waller calculation
                        else:
                           DW_Refresh = True

            form_list = Structure_Factor.atom_scat_phase_DW(crystal,Miller,scat_vec_invA,scat_vec_sqr_invA2,atom,DW_Previous,DW_Refresh)
            f_and_phase = form_list[0]
            fbar_and_phase = form_list[1]
            ffwd_and_phase = form_list[2]
            debye_waller_factor = form_list[3]

            value_list.append(f_and_phase)
            value_list_bar.append(fbar_and_phase)
            value_list_fwd.append(ffwd_and_phase)

            DW_Previous = debye_waller_factor
                        
        F_hkl = sum(value_list)
        FBAR_hkl = sum(value_list_bar)
        FFWD = sum(value_list_fwd)
        return F_hkl, FBAR_hkl, FFWD

    def SF_output(SF,plane):
        """Prints structure factor and intensity"""
        print("-"*20)
        print("\nThe structure factor for these conditions from the",plane,"plane is:\n")
        print(SF)
        print("\nThis produces an intensity of:\n")
        print(np.real(np.conjugate(SF)*SF),"\n")
        print("-"*20)
