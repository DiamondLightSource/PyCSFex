import numpy as np
import matplotlib.pyplot as plt
from .general_crystal import GeneralCrystal
from Structure_Factor_Calculator.checks import Check
from Structure_Factor_Calculator.atom import Atom
from Structure_Factor_Calculator.diffraction_environment import Diff_Environment
from Structure_Factor_Calculator.tools import Tools

class AlphaQuartz_Dextro_zp_isodwf(GeneralCrystal):

    description = "dextro alpha quartz (left-handed screw), z+ (reverse) setting, isotropic thermal atomic motion using Debye model"

    def __init__(self,temperature_K,hkl,energy_eV):

        self.TminK = 20
        self.TmaxK = 838

        self.temperature_K = temperature_K

        self.crystal_system = "Hexagonal"

        self.element_list = ["Si","Si","Si","O","O","O","O","O","O"]

        self.environment = None

        self.atoms = None

        self.Si_T_Debye_K = 470
        self.O_T_Debye_K = 470

        self.fit_coefficients = { 'a_A': ( 4.90137, 0.1929, 1.8038, 0.49873 ), 'c_A': ( 5.39806, 0.10774, 1.72837, 0.56369 ),
                                  'Si_u': ( 0.46796, 0.14030, 3.34189, 0.30790 ), 'O_xyz': ( -0.05454, 2.09511, 3.50034, 0.28351 ) 
                                }

        self.Tc_K = 846 # temperature in K of phase transition from alpha (low) quartz to beta (high) quartz
        self.ValidTemps_K = range(5, self.Tc_K)

        self.O_Reference_Line_Start = (0.414147,0.263209,0.122435)
        self.O_Reference_Line_Vector = (0.023053,-0.143842,0.118118)

        self.set_temp_miller_energy(temperature_K,hkl,energy_eV)

        self.fit_Measured_temps = { 'ac_T_K': ( 5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,22,24,26,28,30,35,40,50,57.5,65,75,85,100,125,150,175,200,225,250,273,283,293,298,398,498,597,697,773,813,838 ),
                               'AtomPos_T_K': ( 13,78,94,115,150,190,240,296,298,298,398,498,597,697,773,813,838 ) }

        self.fit_Measured_data = { 'a_A': ( 4.90127, 4.90127, 4.90127, 4.90127, 4.90127, 4.90127, 4.90127, 4.90127, 4.90127, 4.90127, 4.90127, 4.90127, 4.90127, 4.90128, 4.90128, 4.90128, 4.90129, 4.90129, 4.9013, 4.90132, 4.90133, 4.90138, 4.90144, 4.9016, 4.90175, 4.90192, 4.90218, 4.90248, 4.90297, 4.90394, 4.90504, 4.90628, 4.90762, 4.90906, 4.91058, 4.91205, 4.91271, 4.91338, 4.9137, 4.9209, 4.9297, 4.9384, 4.9509, 4.9628, 4.9728, 4.9841 ),
                                   'c_A': ( 5.398, 5.398, 5.398, 5.398, 5.398, 5.398, 5.398, 5.398, 5.398, 5.398, 5.398, 5.398, 5.398, 5.398, 5.398, 5.398, 5.398, 5.39801, 5.39801, 5.39801, 5.39802, 5.39803, 5.39806, 5.39812, 5.39817, 5.39824, 5.39835, 5.39848, 5.3987, 5.39916, 5.39973, 5.40039, 5.40112, 5.40193, 5.40282, 5.40368, 5.40408, 5.40448, 5.4047, 5.4091, 5.4151, 5.4213, 5.4285, 5.436, 5.4425, 5.45 ),
                                   'Si_u': ( 0.468, 0.4682, 0.46808, 0.46819, 0.46848, 0.46886, 0.46937, 0.47, 0.46981, 0.4697, 0.4709, 0.4723, 0.474, 0.4764, 0.4791, 0.4816, 0.4855 ),
                                   'O_x': ( 0.4124, 0.4125, 0.41303, 0.4131, 0.41328, 0.41341, 0.41356, 0.4131, 0.41372, 0.4133, 0.4138, 0.4141, 0.4149, 0.4157, 0.4159, 0.4173, 0.4174 ),
                                   'O_y': ( 0.2712, 0.2707, 0.27068, 0.27037, 0.2699, 0.26932, 0.2685, 0.2677, 0.26769, 0.2672, 0.2654, 0.263, 0.26, 0.2555, 0.2503, 0.2474, 0.2397 ),
                                   'O_z': ( 0.1163, 0.1163, 0.11651, 0.11671, 0.11705, 0.11754, 0.11819, 0.1189, 0.1188, 0.1188, 0.1206, 0.1224, 0.1246, 0.1281, 0.1321, 0.1363, 0.1422 )
                                 }

        self.fit_Measured_errors = { 'a_A': ( 5E-4, 5E-4, 5E-4, 5E-4, 5E-4, 5E-4, 5E-4, 5E-4, 5E-4, 5E-4, 5E-4, 5E-4, 5E-4, 5E-4, 5E-4, 5E-4, 5E-4, 5E-4, 5E-4, 5E-4, 5E-4, 5E-4, 5E-4, 5E-4, 5E-4, 5E-4, 5E-4, 5E-4, 5E-4, 5E-4, 5E-4, 5E-4, 5E-4, 5E-4, 5E-4, 5E-4, 5E-4, 5E-4, 8E-4, 4E-4, 7E-4, 7E-4, 8E-4, 9E-4, 9E-4, 0.0012 ),
                                     'c_A': ( 5E-4, 5E-4, 5E-4, 5E-4, 5E-4, 5E-4, 5E-4, 5E-4, 5E-4, 5E-4, 5E-4, 5E-4, 5E-4, 5E-4, 5E-4, 5E-4, 5E-4, 5E-4, 5E-4, 5E-4, 5E-4, 5E-4, 5E-4, 5E-4, 5E-4, 5E-4, 5E-4, 5E-4, 5E-4, 5E-4, 5E-4, 5E-4, 5E-4, 5E-4, 5E-4, 5E-4, 5E-4, 5E-4, 0.0012, 5E-4, 0.001, 0.001, 0.0011, 0.0013, 0.0012, 0.0017 ),
                                     'Si_u': ( 4.00E-04, 4.00E-04, 6.00E-05, 7.00E-05, 6.00E-05, 6.00E-05, 7.00E-05, 2.00E-04, 7.00E-05, 1.00E-04, 1.00E-04, 1.00E-04, 1.00E-04, 1.00E-04, 2.00E-04, 2.00E-04, 2.00E-04 ),
                                     'O_x': ( 2.00E-04, 2.00E-04, 1.40E-04, 1.50E-04, 1.40E-04, 1.40E-04, 1.60E-04, 2.00E-04, 1.70E-04, 3.00E-04, 4.00E-04, 4.00E-04, 4.00E-04, 4.00E-04, 5.00E-04, 5.00E-04, 5.00E-04 ),
                                     'O_y': ( 1.00E-04, 1.00E-04, 1.40E-04, 1.50E-04, 1.40E-04, 1.40E-04, 1.60E-04, 2.00E-04, 1.70E-04, 3.00E-04, 3.00E-04, 3.00E-04, 3.00E-04, 4.00E-04, 5.00E-04, 5.00E-04, 5.00E-04 ),
                                     'O_z': ( 1.00E-04, 1.00E-04, 9.00E-05, 1.00E-04, 9.00E-05, 9.00E-05, 1.00E-04, 1.00E-04, 1.00E-04, 2.00E-04, 2.00E-04, 2.00E-04, 2.00E-04, 3.00E-04, 3.00E-04, 3.00E-04, 4.00E-04 )
                                   }

    def plot_fit_data(self):

        fig_acPos = plt.figure(1)

        Lattice_a_plot = fig_acPos.add_subplot(3,2,1)
        a_Fit_A = [ self.fitting_equation(T_K, *self.fit_coefficients['a_A']) for T_K in self.ValidTemps_K ]
        Lattice_a_plot.plot( self.ValidTemps_K, a_Fit_A )
        Lattice_a_plot.errorbar( self.fit_Measured_temps['ac_T_K'], self.fit_Measured_data['a_A'], yerr = self.fit_Measured_errors['a_A'], fmt = 'rx' )
        Lattice_a_plot.set(xlabel = 'Temp (K)',ylabel = 'a (A)')

        Lattice_c_plot = fig_acPos.add_subplot(3,2,2)
        c_Fit_A = [ self.fitting_equation(T_K, *self.fit_coefficients['c_A']) for T_K in self.ValidTemps_K ]
        Lattice_c_plot.plot( self.ValidTemps_K, c_Fit_A )
        Lattice_c_plot.errorbar( self.fit_Measured_temps['ac_T_K'], self.fit_Measured_data['c_A'], yerr = self.fit_Measured_errors['c_A'], fmt = 'rx' )
        Lattice_c_plot.set(xlabel = 'Temp (K)',ylabel = 'c (A)') 
       
        Si_u_plot = fig_acPos.add_subplot(3,2,3)
        Si_u_Fit = [ self.fitting_equation(T_K, *self.fit_coefficients['Si_u']) for T_K in self.ValidTemps_K ]
        Si_u_plot.plot( self.ValidTemps_K, Si_u_Fit )
        Si_u_plot.errorbar( self.fit_Measured_temps['AtomPos_T_K'], self.fit_Measured_data['Si_u'], yerr = self.fit_Measured_errors['Si_u'], fmt = 'rx')        
        Si_u_plot.set(xlabel = 'Temp (K)',ylabel = 'Si u')

        O_LineShift_Fit = [ self.fitting_equation(T_K, *self.fit_coefficients['O_xyz']) for T_K in self.ValidTemps_K ]
        O_x_Fit = [ self.O_Reference_Line_Start[0] + O_LineShift_Fit[iT]*self.O_Reference_Line_Vector[0] for iT in range(0,len(O_LineShift_Fit)) ]
        O_y_Fit = [ self.O_Reference_Line_Start[1] + O_LineShift_Fit[iT]*self.O_Reference_Line_Vector[1] for iT in range(0,len(O_LineShift_Fit)) ]
        O_z_Fit = [ self.O_Reference_Line_Start[2] + O_LineShift_Fit[iT]*self.O_Reference_Line_Vector[2] for iT in range(0,len(O_LineShift_Fit)) ]

        O_x_plot = fig_acPos.add_subplot(3,2,4)
        O_x_plot.plot( self.ValidTemps_K, O_x_Fit )
        O_x_plot.errorbar( self.fit_Measured_temps['AtomPos_T_K'], self.fit_Measured_data['O_x'], yerr = self.fit_Measured_errors['O_x'], fmt = 'rx')                
        O_x_plot.set(xlabel = 'Temp (K)',ylabel = 'O x')

        O_y_plot = fig_acPos.add_subplot(3,2,5)
        O_y_plot.plot( self.ValidTemps_K, O_y_Fit )
        O_y_plot.errorbar( self.fit_Measured_temps['AtomPos_T_K'], self.fit_Measured_data['O_y'], yerr = self.fit_Measured_errors['O_y'], fmt = 'rx')                        
        O_y_plot.set(xlabel = 'Temp (K)',ylabel = 'O y')

        O_z_plot = fig_acPos.add_subplot(3,2,6)
        O_z_plot.plot( self.ValidTemps_K, O_z_Fit )
        O_z_plot.errorbar( self.fit_Measured_temps['AtomPos_T_K'], self.fit_Measured_data['O_z'], yerr = self.fit_Measured_errors['O_z'], fmt = 'rx')                        
        O_z_plot.set(xlabel = 'Temp (K)',ylabel = 'O z')
        
        plt.savefig('AlphaQuartz_Dextro_zp_isodwf_Lattice&AtomPos.png')
        plt.close(1)

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
        print("\nCrystal = Alpha Quartz")
        print("Structure = ",self.crystal_system)
        print("Handedness of screw axis = Left")
        print("Model for thermal motion: isotropic Debye")
        print("Debye temperature of Si atoms = ",self.Si_T_Debye_K,"K")
        print("Debye temperature of O atoms = ",self.O_T_Debye_K,"K")
        print("a = ",self.a_A,"Angstroms")
        print("b = ",self.b_A,"Angstroms")
        print("c = ",self.c_A,"Angstroms")
        print("Alpha = ",self.alpha_rad*(180/np.pi),"deg")
        print("Beta = ",self.beta_rad*(180/np.pi),"deg")
        print("Gamma = ",round(self.gamma_rad*(180/np.pi),6),"deg")
        print("Lattice Vectors (Angstrom) = ",self.lattice_vectors_A)
        print("G Matrix (Angstrom sqr) = ",self.G_mtrx_A2,"\n")
        print("Si reference atom coordinates = ",self.refatom_coords[0])
        print("O reference atom coordinates = ",self.refatom_coords[1])
        print("Bragg angle = ",self.angle_plane_wavelength[0]*180/np.pi,"deg")
        print("Miller indices of diffracting plane = ",self.angle_plane_wavelength[1])
        print("Photon wavelength = ",self.angle_plane_wavelength[2],"Angstroms")
        print("Spacing of diffracting planes = ",self.environment.d_A,"Angstroms")
        for atom in self.atoms:
            atom.information()

        print("-"*20)   

    """The following functions are not used in all crystals but are useful when working with alpha quartz."""

    def fitting_equation(self, temp_K, f0, P, Q, n):
        """Equation used for the temperature dependence of lattice parameters, 
	atomic positions and thermal ellipsoid beta matrix elements in alpha quartz."""
        
        data_fit = f0 + P*np.exp(-Q/(np.log(self.Tc_K/(self.Tc_K-temp_K)))**n)
        
        return data_fit

    def screw_matrix(self,atom_coord):
        """Performs screw transformation to calculate atom coordinates. """
        
        m = -1
        
        rotation = self.three_fold_matrix()
        translation = np.array([0,0,m/3])
        
        new_atom_coord = np.dot(rotation,atom_coord,) + translation
        
        Check.length(new_atom_coord,3)
        
        return new_atom_coord
    
    def two_fold_matrix(self):
        """Performs 180deg rotation used for coordinate and beta matrix calculations."""
    
        M = np.array(([1,-1,0],[0,-1,0],[0,0,-1]))
    
        return M
    
    
    def three_fold_matrix(self):
        """Performs 120deg rotation used for coordinate and beta matrix calculations."""
        
        M = np.array(([0,-1,0],[1,-1,0],[0,0,1]))
        
        return M

    """End of functions that are specific to alpha quartz."""

    def lattice_unit_cell_params(self,temp_K):
        """Generates lattice parameters for alpha quartz"""
             
        alpha_rad = beta_rad = np.pi/2
        gamma_rad = 2*np.pi/3
            
        a_lattice_A = self.fitting_equation(temp_K, *self.fit_coefficients['a_A'])
        b_lattice_A = a_lattice_A
                    
        c_lattice_A = self.fitting_equation(temp_K, *self.fit_coefficients['c_A'])
                
        return a_lattice_A, b_lattice_A, c_lattice_A, alpha_rad, beta_rad, gamma_rad

    def refatom_coordinates(self):
        """Calculates temperature-dependent coordinates of the reference Si atom
        and the reference O atom from temperature fits."""            

        Si_u = self.fitting_equation(self.temperature_K, *self.fit_coefficients['Si_u'])
        
        Si_refatom_pos = np.array(["Si",Si_u,0,0])

        t_multiplier = self.fitting_equation(self.temperature_K, *self.fit_coefficients['O_xyz'])
            
        O_refatom_pos = np.array(self.O_Reference_Line_Start) + t_multiplier*(np.array(self.O_Reference_Line_Vector))
                    
        O_refatom_pos = np.array(("O",O_refatom_pos[0],O_refatom_pos[1],O_refatom_pos[2]))
            
        refatom_coords = np.array([Si_refatom_pos, O_refatom_pos])
            
        Check.length(O_refatom_pos,4)
        Check.length(Si_refatom_pos,4)
        
        return refatom_coords

    def atoms_init_and_update(self,Initial):        
        """Determines coordinates of each atom, sets its mass and Debye temperature, and initializes an Atom object for it."""        

        element_list_arr = np.array(self.element_list)
        element_list_arr = element_list_arr.reshape(len(self.element_list),1)
            
        Si_u = float(self.refatom_coords[0][1])
            
        #Coordinates calculated by applying matrix transformations
        Si1_coord = np.array([Si_u,0,0])
        Si2_coord = self.screw_matrix(Si1_coord)
        Si3_coord = self.screw_matrix(Si2_coord) 
            
        O_u = float(self.refatom_coords[1][1])
        O_v = float(self.refatom_coords[1][2])
        O_w = float(self.refatom_coords[1][3])
            
        O1_coord = np.array([O_u,O_v,O_w])
        O2_coord = self.screw_matrix(O1_coord)
        O3_coord = self.screw_matrix(O2_coord)
            
        M = self.two_fold_matrix()
        O4_coord = np.dot(M,O1_coord)
        O5_coord = np.dot(M,O2_coord)
        O6_coord = np.dot(M,O3_coord)
            
        Si_M_amu = 28.085
        Si_T_Debye_K = self.Si_T_Debye_K
        O_M_amu = 15.999
        O_T_Debye_K = self.O_T_Debye_K

        M_and_TD_Atom = [[Si_M_amu,Si_T_Debye_K],[Si_M_amu,Si_T_Debye_K],[Si_M_amu,Si_T_Debye_K],[O_M_amu,O_T_Debye_K],[O_M_amu,O_T_Debye_K],[O_M_amu,O_T_Debye_K],[O_M_amu,O_T_Debye_K],[O_M_amu,O_T_Debye_K],[O_M_amu,O_T_Debye_K]]
            
        coord_list = [Si1_coord,Si2_coord,Si3_coord,O1_coord,O2_coord,O3_coord,
                      O4_coord,O5_coord,O6_coord]
                
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