# Developed with Python 3.10.5, numpy 1.22.4
# CSRR, SPring-8 (JASRI)
# Ver 1.0, Jun 2022

# Authors: Taishun Manjo & Alfred Q. R. Baron
# Contact: manjo.taishun@spring8.or.jp, baron@spring8.or.jp

# This code was created by T.Manjo in discussion with A. Baron based on the code developed by John P. Sutter et al
# This code creates the basic information of Germanium

import numpy as np
import matplotlib.pyplot as plt
from .general_crystal import GeneralCrystal
from Structure_Factor_Calculator.checks import Check
from Structure_Factor_Calculator.atom import Atom
from Structure_Factor_Calculator.diffraction_environment import Diff_Environment
from Structure_Factor_Calculator.tools import Tools

class Ge_isodwf(GeneralCrystal):

    description = "Ge, isotropic thermal atomic motion using Debye model (Trange 3-1085K, Debye T 290K)"

    def __init__(self,temperature_K,hkl,energy_eV):

        self.TminK = 3
        self.TmaxK = 1085

        self.temperature_K = temperature_K

        self.crystal_system = "Cubic"

        #all atom in unit cell
        self.element_list = ["Ge", "Ge", "Ge", "Ge", "Ge", "Ge", "Ge", "Ge"]

        self.environment = None

        self.atoms = None

        #B. W. Batterman et al.; Physical Review 127, 3 (1962)
        self.Ge_T_Debye_K = 290

        self.set_temp_miller_energy(temperature_K,hkl,energy_eV)

    def set_temp_miller_energy(self,temperature_K,hkl,energy_eV):
        if (temperature_K < self.TminK):
           raise ValueError("The temperature is below the valid range of ",self.TminK," to ",self.TmaxK," K.")
        elif (temperature_K > self.TmaxK):
           raise ValueError("The temperature is above the valid range of ",self.TminK," to ",self.TmaxK," K.")
        else:
           self.temperature_K = temperature_K

        self.unit_cell_params = self.lattice_unit_cell_params(temperature_K)
        self.refatom_coords = self.refatom_coordinates()
        super().__init__(self.crystal_system,self.unit_cell_params,hkl,energy_eV)
        if (self.environment is None):
            self.environment = Diff_Environment(self.temperature_K,self.crystal_system,self.unit_cell_params,self.angle_plane_wavelength)
        else:
            self.environment.update(self.temperature_K,self.crystal_system,self.unit_cell_params,self.angle_plane_wavelength)
        if (self.atoms is None):
            self.atoms = self.atoms_init_and_update(Initial=True)
        else:
            self.atoms = self.atoms_init_and_update(Initial=False)

    def set_miller_energy(self,hkl,energy_eV):
        self.angle_plane_wavelength = self.angle_finder(self.unit_cell_params,self.crystal_system,hkl,energy_eV)
        self.environment.update(self.temperature_K,self.crystal_system,self.unit_cell_params,self.angle_plane_wavelength)
        for atom in self.atoms:
            atom.Environment = self.environment  
            atom.form_factor = atom.scattering_factor()[0]
            atom.f0 = atom.scattering_factor()[2]
            atom.form_factor_fwd = atom.scattering_factor()[3]  

    def information(self):        
        """When called, this will print out attributes of the object."""
                
        print("-"*20)
        print("\nCrystal = Ge")
        print("Structure = ",self.crystal_system)
        #print("Handedness of screw axis = Left")
        print("Model for thermal motion: isotropic Debye")
        print("Debye temperature of Ge atoms = ",self.Ge_T_Debye_K,"K")
        #print("Debye temperature of O atoms = ",self.O_T_Debye_K,"K")
        print("a = ",self.a_A,"Angstroms")
        print("b = ",self.b_A,"Angstroms")
        print("c = ",self.c_A,"Angstroms")
        print("Alpha = ",self.alpha_rad*(180/np.pi),"deg")
        print("Beta = ",self.beta_rad*(180/np.pi),"deg")
        print("Gamma = ",round(self.gamma_rad*(180/np.pi),6),"deg")
        print("Lattice Vectors (Angstrom) = ",self.lattice_vectors_A)
        print("G Matrix (Angstrom sqr) = ",self.G_mtrx_A2,"\n")
        print("Ge reference atom coordinates = ",self.refatom_coords[0])
        #print("O reference atom coordinates = ",self.refatom_coords[1])
        print("Bragg angle = ",self.angle_plane_wavelength[0]*180/np.pi,"deg")
        print("Miller indices of diffracting plane = ",self.angle_plane_wavelength[1])
        print("Photon wavelength = ",self.angle_plane_wavelength[2],"Angstroms")
        print("Spacing of diffracting planes = ",self.environment.d_A,"Angstroms")
        for atom in self.atoms:
            atom.information()

        print("-"*20)   

    def lattice_unit_cell_params(self,temp_K):
        """Generates lattice parameters for Ge"""
             
        alpha_rad = beta_rad = gamma_rad = np.pi/2
        
        a_lattice_A = self.calc_lattice_para_Ge(temp_K)
        b_lattice_A = a_lattice_A
        c_lattice_A = a_lattice_A

        return a_lattice_A, b_lattice_A, c_lattice_A, alpha_rad, beta_rad, gamma_rad
    
    # This function is added by T.Manjo
    # Manjo start
    def calc_lattice_para_Ge(self, temp_K):
        #data(<70) from Collela's data
        #data(300<) from Singh; Acta Cryst. A 24, 469 (1968)
        #Manjo fit the all data with middlemann's function (eq.(5) in T. Middelmann, et al.; PRB 92, 174113 (2015))
        
        a_i = [3.8323969E-05, -1.1478089E-06, 1.8409071E-05]    #m/K
        theta_i = [3.6889770E+02, 1.0543187E+02, 2.5643340E+03] #K
        l_0 = 5.6523642  #Angstrom

        a_T = l_0
        for i in range(3):
            a_T = a_T + ((a_i[i] * theta_i[i])/(-1.0 + np.exp(theta_i[i] / temp_K))) # meter
        
        return a_T
    # Manjo_end
    
    def refatom_coordinates(self):
        """Calculates temperature-dependent coordinates of the reference Ge atom
        from temperature fits."""            
        
        Ge_refatom_pos_1 = np.array(["Ge", 0.125, 0.125, 0.125])
        Ge_refatom_pos_2 = np.array(["Ge", 0.875, 0.375, 0.375])

        refatom_coords = np.array([Ge_refatom_pos_1, Ge_refatom_pos_2])
        
        Check.length(Ge_refatom_pos_1,4)
        Check.length(Ge_refatom_pos_2,4)
        
        return refatom_coords

    def atoms_init_and_update(self,Initial):        
        """Determines coordinates of each atom, sets its mass and Debye temperature, and initializes an Atom object for it."""        

        element_list_arr = np.array(self.element_list)
        element_list_arr = element_list_arr.reshape(len(self.element_list),1)
            
        Ge_M_amu = 72.630
        Ge_T_Debye_K = self.Ge_T_Debye_K

        #This part is added by T. Manjo
        # Manjo_start
        """calculate atom coordination of Ge (Diamond structure, Space Group:Fd-3m(No.227 Origin Choice 2))"""
        #initialize
        coord_list = []
        M_and_TD_Atom = []
        
        #create refatom position as np.array type
        coords_ori = np.empty((np.shape(self.refatom_coords)[0], 3), dtype=float)
        for i in range(np.shape(self.refatom_coords)[0]):
            for j in range(3):
                coords_ori[i][j] = float(self.refatom_coords[i][j+1])

        #Shift vector from International Table Vol.A (No.227, Origin Choice 2)
        Shift_vector = [np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.5, 0.5]), np.array([0.5, 0.0, 0.5]), np.array([0.5, 0.5, 0.0])]
        
        #calculate 8 atoms position in unit cell.
        for i in range(np.shape(coords_ori)[0]):  #loop refatom pos
            for j in range(4):  #loop shift vector
                Ge_coord_c = coords_ori[i] + Shift_vector[j]
                Ge_coord_c = np.where(Ge_coord_c >= 1.0, Ge_coord_c - 1.0, Ge_coord_c)  #0 <= atom coordination < 1.0 
                Ge_coord_c = np.where(Ge_coord_c < 0.0, Ge_coord_c + 1.0, Ge_coord_c)
                coord_list.append(Ge_coord_c)
                M_and_TD_Atom.append([Ge_M_amu,Ge_T_Debye_K])
        # Manjo_end
        
        atom_list = []
        
        for index in range (0,len(self.element_list)):

            if (Initial): # we create new atom objects from scratch

                atom = Atom(self.element_list[index],coord_list[index],self.environment,
                        self.lattice_vectors_A,M_and_TD=M_and_TD_Atom[index])

                atom.coordinate = Tools.unit_cell_check(atom.coordinates)
                
                atom_list.append(atom)

            else: # we update the already existing atom objects

                self.atoms[index].update(coord_list[index],self.environment,self.lattice_vectors_A,M_and_TD=M_and_TD_Atom[index]) 

                self.atoms[index].coordinate = Tools.unit_cell_check(self.atoms[index].coordinates)
                
                atom_list.append(self.atoms[index])
       
        return atom_list
