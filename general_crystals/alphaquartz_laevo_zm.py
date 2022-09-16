import numpy as np
import matplotlib.pyplot as plt
from .general_crystal import GeneralCrystal
from Structure_Factor_Calculator.checks import Check
from Structure_Factor_Calculator.atom import Atom
from Structure_Factor_Calculator.diffraction_environment import Diff_Environment
from Structure_Factor_Calculator.tools import Tools

class AlphaQuartz_Laevo_zm(GeneralCrystal):

    description = "laevo alpha quartz (right-handed screw), z- (reverse) setting, anisotropic thermal ellipsoids"

    def __init__(self,temperature_K,hkl,energy_eV):

        self.TminK = 20
        self.TmaxK = 838

        self.temperature_K = temperature_K

        self.crystal_system = "Hexagonal"

        self.element_list = ["Si","Si","Si","O","O","O","O","O","O"]

        self.environment = None

        self.atoms = None

        self.fit_coefficients = { 'a_A': ( 4.90137, 0.1929, 1.8038, 0.49873 ), 'c_A': ( 5.39806, 0.10774, 1.72837, 0.56369 ),
                                  'Si_u': ( 0.46796, 0.14030, 3.34189, 0.30790 ), 'O_xyz': ( -0.05454, 2.09511, 3.50034, 0.28351 ),
                                  'Si_b11': ( 33.80581, 316.85367, 1.15728, 0.63590 ), 'Si_b22': ( 35.46433, 234.81469, 1.24881, 0.70758 ),
                                  'Si_b33': ( 16.72506, 166.37647, 1.07478, 0.61059 ), 'Si_b23': ( -1.98183,  22.36979, 3.52304, 0.65136 ),
                                  'O_b11': ( 80.23419, 619.20673, 1.00988, 0.71444 ), 'O_b22': ( 55.09116, 910.22226, 1.66614, 0.50506 ),
                                  'O_b33': ( 29.84124, 650.47191, 1.75296, 0.45406 ), 'O_b12': ( 43.81216, 409.36732, 1.13880, 0.65112 ),
                                  'O_b13': ( -5.35134, -52.36748, 0.43392, 0.84086 ), 'O_b23': ( -11.15409,-718.15015, 2.39901, 0.33189 )  
                                }

        self.Tc_K = 846 # temperature in K of phase transition from alpha (low) quartz to beta (high) quartz
        self.ValidTemps_K = range(5, self.Tc_K)

        self.O_Reference_Line_Start = (0.414147,0.263209,0.122435)
        self.O_Reference_Line_Vector = (0.023053,-0.143842,0.118118)

        self.set_temp_miller_energy(temperature_K,hkl,energy_eV)

        self.fit_Measured_temps = { 'ac_T_K': ( 5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,22,24,26,28,30,35,40,50,57.5,65,75,85,100,125,150,175,200,225,250,273,283,293,298,398,498,597,697,773,813,838 ),
                               'AtomPos_T_K': ( 13,78,94,115,150,190,240,296,298,298,398,498,597,697,773,813,838 ),
                               'ThermEll_T_K': ( 13,78,94,115,150,190,240,296,298,298,398,498,597,697,773,813,838 ),
                               'ThermEll_Si_b23_T_K': ( 13,78,94,115,150,190,240,298,298,398,498,597,697,773,813,838 ) }

        self.fit_Measured_data = { 'a_A': ( 4.90127, 4.90127, 4.90127, 4.90127, 4.90127, 4.90127, 4.90127, 4.90127, 4.90127, 4.90127, 4.90127, 4.90127, 4.90127, 4.90128, 4.90128, 4.90128, 4.90129, 4.90129, 4.9013, 4.90132, 4.90133, 4.90138, 4.90144, 4.9016, 4.90175, 4.90192, 4.90218, 4.90248, 4.90297, 4.90394, 4.90504, 4.90628, 4.90762, 4.90906, 4.91058, 4.91205, 4.91271, 4.91338, 4.9137, 4.9209, 4.9297, 4.9384, 4.9509, 4.9628, 4.9728, 4.9841 ),
                                   'c_A': ( 5.398, 5.398, 5.398, 5.398, 5.398, 5.398, 5.398, 5.398, 5.398, 5.398, 5.398, 5.398, 5.398, 5.398, 5.398, 5.398, 5.398, 5.39801, 5.39801, 5.39801, 5.39802, 5.39803, 5.39806, 5.39812, 5.39817, 5.39824, 5.39835, 5.39848, 5.3987, 5.39916, 5.39973, 5.40039, 5.40112, 5.40193, 5.40282, 5.40368, 5.40408, 5.40448, 5.4047, 5.4091, 5.4151, 5.4213, 5.4285, 5.436, 5.4425, 5.45 ),
                                   'Si_u': ( 1-0.468, 1-0.4682, 1-0.46808, 1-0.46819, 1-0.46848, 1-0.46886, 1-0.46937, 1-0.47, 1-0.46981, 1-0.4697, 1-0.4709, 1-0.4723, 1-0.474, 1-0.4764, 1-0.4791, 1-0.4816, 1-0.4855 ),
                                   'O_x': ( 1-0.4124, 1-0.4125, 1-0.41303, 1-0.4131, 1-0.41328, 1-0.41341, 1-0.41356, 1-0.4131, 1-0.41372, 1-0.4133, 1-0.4138, 1-0.4141, 1-0.4149, 1-0.4157, 1-0.4159, 1-0.4173, 1-0.4174 ),
                                   'O_y': ( 1-0.2712, 1-0.2707, 1-0.27068, 1-0.27037, 1-0.2699, 1-0.26932, 1-0.2685, 1-0.2677, 1-0.26769, 1-0.2672, 1-0.2654, 1-0.263, 1-0.26, 1-0.2555, 1-0.2503, 1-0.2474, 1-0.2397 ),
                                   'O_z': ( 1-0.1163, 1-0.1163, 1-0.11651, 1-0.11671, 1-0.11705, 1-0.11754, 1-0.11819, 1-0.1189, 1-0.1188, 1-0.1188, 1-0.1206, 1-0.1224, 1-0.1246, 1-0.1281, 1-0.1321, 1-0.1363, 1-0.1422 ),
                                   'Si_b11': ( 33, 30, 39.31398604, 43.35310012, 49.23399589, 56.07757655, 66.37837599, 68, 75.86227509, 80, 105, 124, 148, 176, 198, 219, 237 ),
                                   'Si_b22': ( 41, 34, 33.29095197, 36.56549354, 40.04364999, 44.16245794, 52.62233096, 54, 59.07665675, 61, 81, 98, 117, 135, 152, 172, 192 ),
                                   'Si_b33': ( 15, 16, 22.35188951, 24.17786841, 27.48873083, 31.33453734, 36.65671286, 44, 41.48896875, 45, 57, 67, 82, 95, 107, 118, 123 ),
                                   'Si_b23': ( -4, -4, -0.344497804, -0.516640721, -0.516406694, -1.204162558, -1.374860453, -1.373126121, -3, -2, -3, -1, -1, 2, 4, 3 ),
                                   'O_b11': ( 79, 84, 83.55590905, 92.83694168, 106.2360222, 122.3212635, 147.0586718, 195, 168.2921735, 179, 232, 282, 334, 394, 451, 487, 515 ),
                                   'O_b22': ( 57, 59, 59.79230189, 66.23390296, 75.82035368, 88.32491589, 107.5373361, 120, 120.551259, 130, 175, 208, 260, 316, 364, 417, 481 ),
                                   'O_b33': ( 30, 30, 37.32088219, 41.51549954, 48.74848818, 56.30741915, 66.07676838, 72, 76.35591968, 85, 109, 131, 162, 196, 233, 262, 302 ),
                                   'O_b12': ( 42, 48, 46.65113664, 52.22077969, 60.28429275, 69.30445133, 83.95554463, 104, 95.69982403, 102, 134, 162, 195, 228, 262, 286, 312 ),
                                   'O_b13': ( -4, -6, -11.71292533, -13.34655195, -16.00860753, -19.35261255, -22.42741114, -23, -25.91775553, -26, -32, -39, -43, -50, -51, -48, -48 ),
                                   'O_b23': ( -11, -13, -17.65551245, -19.97677453, -23.32436903, -27.69573884, -34.02779621, -45, -39.30573521, -41, -54, -68, -86, -113, -133, -150, -182 )
                                 }

        self.fit_Measured_errors = { 'a_A': ( 5E-4, 5E-4, 5E-4, 5E-4, 5E-4, 5E-4, 5E-4, 5E-4, 5E-4, 5E-4, 5E-4, 5E-4, 5E-4, 5E-4, 5E-4, 5E-4, 5E-4, 5E-4, 5E-4, 5E-4, 5E-4, 5E-4, 5E-4, 5E-4, 5E-4, 5E-4, 5E-4, 5E-4, 5E-4, 5E-4, 5E-4, 5E-4, 5E-4, 5E-4, 5E-4, 5E-4, 5E-4, 5E-4, 8E-4, 4E-4, 7E-4, 7E-4, 8E-4, 9E-4, 9E-4, 0.0012 ),
                                     'c_A': ( 5E-4, 5E-4, 5E-4, 5E-4, 5E-4, 5E-4, 5E-4, 5E-4, 5E-4, 5E-4, 5E-4, 5E-4, 5E-4, 5E-4, 5E-4, 5E-4, 5E-4, 5E-4, 5E-4, 5E-4, 5E-4, 5E-4, 5E-4, 5E-4, 5E-4, 5E-4, 5E-4, 5E-4, 5E-4, 5E-4, 5E-4, 5E-4, 5E-4, 5E-4, 5E-4, 5E-4, 5E-4, 5E-4, 0.0012, 5E-4, 0.001, 0.001, 0.0011, 0.0013, 0.0012, 0.0017 ),
                                     'Si_u': ( 4.00E-04, 4.00E-04, 6.00E-05, 7.00E-05, 6.00E-05, 6.00E-05, 7.00E-05, 2.00E-04, 7.00E-05, 1.00E-04, 1.00E-04, 1.00E-04, 1.00E-04, 1.00E-04, 2.00E-04, 2.00E-04, 2.00E-04 ),
                                     'O_x': ( 2.00E-04, 2.00E-04, 1.40E-04, 1.50E-04, 1.40E-04, 1.40E-04, 1.60E-04, 2.00E-04, 1.70E-04, 3.00E-04, 4.00E-04, 4.00E-04, 4.00E-04, 4.00E-04, 5.00E-04, 5.00E-04, 5.00E-04 ),
                                     'O_y': ( 1.00E-04, 1.00E-04, 1.40E-04, 1.50E-04, 1.40E-04, 1.40E-04, 1.60E-04, 2.00E-04, 1.70E-04, 3.00E-04, 3.00E-04, 3.00E-04, 3.00E-04, 4.00E-04, 5.00E-04, 5.00E-04, 5.00E-04 ),
                                     'O_z': ( 1.00E-04, 1.00E-04, 9.00E-05, 1.00E-04, 9.00E-05, 9.00E-05, 1.00E-04, 1.00E-04, 1.00E-04, 2.00E-04, 2.00E-04, 2.00E-04, 2.00E-04, 3.00E-04, 3.00E-04, 3.00E-04, 4.00E-04 ),
                                     'Si_b11': ( 3, 3, 1.095097104, 1.204252781, 1.203497677, 1.093130147, 1.310099526, 3, 1.30797026, 2, 2, 2, 2, 3, 3, 3, 4 ),
                                     'Si_b22': ( 5, 6, 1.095097104, 1.204252781, 1.203497677, 1.093130147, 1.310099526, 6, 1.30797026, 3, 3, 2, 2, 3, 4, 4, 4 ),
                                     'Si_b33': ( 3, 3, 0.745062984, 0.744976338, 0.744768569, 0.676771865, 0.811587739, 3, 0.810859324, 2, 2, 1, 2, 2, 2, 2, 2 ),
                                     'Si_b23': ( 2, 2, 0.688995608, 0.688854294, 0.688542259, 0.688092891, 0.687430226, 0.858203826, 2, 2, 1, 2, 2, 2, 2, 2 ),
                                     'O_b11': ( 3, 3, 2.409213629, 2.627460613, 2.516404235, 2.623512353, 3.056898894, 4, 3.26992565, 5, 6, 5, 6, 9, 11, 11, 13 ),
                                     'O_b22': ( 3, 3, 2.190194208, 2.299028037, 2.188177595, 2.295573309, 2.620199052, 3, 2.833935564, 5, 6, 5, 6, 7, 10, 10, 12 ),
                                     'O_b33': ( 1, 1, 1.151460975, 1.28677731, 1.218712204, 1.218189357, 1.420278543, 1, 1.486575427, 3, 3, 3, 3, 4, 5, 5, 5 ),
                                     'O_b12': ( 3, 3, 1.971174788, 2.189550511, 2.078768715, 2.076947279, 2.511024092, 3, 2.61594052, 4, 5, 4, 5, 6, 8, 8, 10 ),
                                     'O_b13': ( 2, 2, 1.377991216, 1.549922162, 1.463152301, 1.462197392, 1.718575566, 2, 1.802228034, 3, 4, 3, 4, 5, 6, 6, 7 ),
                                     'O_b23': ( 1, 2, 1.119617863, 1.205495015, 1.204948954, 1.204162558, 1.460789231, 2, 1.544766886, 3, 3, 3, 4, 5, 6, 6, 7 )
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
        Si_u_Fit = [ 1 - self.fitting_equation(T_K, *self.fit_coefficients['Si_u']) for T_K in self.ValidTemps_K ]
        Si_u_plot.plot( self.ValidTemps_K, Si_u_Fit )
        Si_u_plot.errorbar( self.fit_Measured_temps['AtomPos_T_K'], self.fit_Measured_data['Si_u'], yerr = self.fit_Measured_errors['Si_u'], fmt = 'rx')        
        Si_u_plot.set(xlabel = 'Temp (K)',ylabel = 'Si u')

        O_LineShift_Fit = [ self.fitting_equation(T_K, *self.fit_coefficients['O_xyz']) for T_K in self.ValidTemps_K ]
        O_x_Fit = [ 1 - (self.O_Reference_Line_Start[0] + O_LineShift_Fit[iT]*self.O_Reference_Line_Vector[0]) for iT in range(0,len(O_LineShift_Fit)) ]
        O_y_Fit = [ 1 - (self.O_Reference_Line_Start[1] + O_LineShift_Fit[iT]*self.O_Reference_Line_Vector[1]) for iT in range(0,len(O_LineShift_Fit)) ]
        O_z_Fit = [ 1 - (self.O_Reference_Line_Start[2] + O_LineShift_Fit[iT]*self.O_Reference_Line_Vector[2]) for iT in range(0,len(O_LineShift_Fit)) ]

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
        
        plt.savefig('AlphaQuartz_Laevo_zm_Lattice&AtomPos.png')
        plt.close(1)

        fig_SiThermEllBeta = plt.figure(2)

        Si_b11_plot = fig_SiThermEllBeta.add_subplot(2,2,1)
        Si_b11_Fit = [ self.fitting_equation(T_K, *self.fit_coefficients['Si_b11']) for T_K in self.ValidTemps_K ]
        Si_b11_plot.plot( self.ValidTemps_K, Si_b11_Fit )
        Si_b11_plot.errorbar( self.fit_Measured_temps['ThermEll_T_K'], self.fit_Measured_data['Si_b11'], yerr=self.fit_Measured_errors['Si_b11'], fmt = 'rx' )
        Si_b11_plot.set(xlabel = 'Temp (K)',ylabel = 'Si b11')
       
        Si_b22_plot = fig_SiThermEllBeta.add_subplot(2,2,2)
        Si_b22_Fit = [ self.fitting_equation(T_K, *self.fit_coefficients['Si_b22']) for T_K in self.ValidTemps_K ]
        Si_b22_plot.plot( self.ValidTemps_K, Si_b22_Fit )
        Si_b22_plot.errorbar( self.fit_Measured_temps['ThermEll_T_K'], self.fit_Measured_data['Si_b22'], yerr=self.fit_Measured_errors['Si_b22'], fmt = 'rx' )        
        Si_b22_plot.set(xlabel = 'Temp (K)',ylabel = 'Si b22')

        Si_b33_plot = fig_SiThermEllBeta.add_subplot(2,2,3)
        Si_b33_Fit = [ self.fitting_equation(T_K, *self.fit_coefficients['Si_b33']) for T_K in self.ValidTemps_K ]
        Si_b33_plot.plot( self.ValidTemps_K, Si_b33_Fit )
        Si_b33_plot.errorbar( self.fit_Measured_temps['ThermEll_T_K'], self.fit_Measured_data['Si_b33'], yerr=self.fit_Measured_errors['Si_b33'], fmt = 'rx' )   
        Si_b33_plot.set(xlabel = 'Temp (K)',ylabel = 'Si b33')

        Si_b23_plot = fig_SiThermEllBeta.add_subplot(2,2,4)
        Si_b23_Fit = [ self.fitting_equation(T_K, *self.fit_coefficients['Si_b23']) for T_K in self.ValidTemps_K ]
        Si_b23_plot.plot( self.ValidTemps_K, Si_b23_Fit )
        Si_b23_plot.errorbar( self.fit_Measured_temps['ThermEll_Si_b23_T_K'], self.fit_Measured_data['Si_b23'], yerr=self.fit_Measured_errors['Si_b23'], fmt = 'rx' )             
        Si_b23_plot.set(xlabel = 'Temp (K)',ylabel = 'Si b23')

        plt.savefig('AlphaQuartz_Laevo_zm_SiThermEll.png')
        plt.close(2)

        fig_OThermEllBeta = plt.figure(3)

        O_b11_plot = fig_OThermEllBeta.add_subplot(3,2,1)
        O_b11_Fit = [ self.fitting_equation(T_K, *self.fit_coefficients['O_b11']) for T_K in self.ValidTemps_K ]
        O_b11_plot.plot( self.ValidTemps_K, O_b11_Fit )
        O_b11_plot.errorbar( self.fit_Measured_temps['ThermEll_T_K'], self.fit_Measured_data['O_b11'], yerr=self.fit_Measured_errors['O_b11'], fmt = 'rx' ) 
        O_b11_plot.set(xlabel = 'Temp (K)',ylabel = 'O b11')
        
        O_b22_plot = fig_OThermEllBeta.add_subplot(3,2,2)
        O_b22_Fit = [ self.fitting_equation(T_K, *self.fit_coefficients['O_b22']) for T_K in self.ValidTemps_K ]
        O_b22_plot.plot( self.ValidTemps_K, O_b22_Fit )
        O_b22_plot.errorbar( self.fit_Measured_temps['ThermEll_T_K'], self.fit_Measured_data['O_b22'], yerr=self.fit_Measured_errors['O_b22'], fmt = 'rx' )   
        O_b22_plot.set(xlabel = 'Temp (K)',ylabel = 'O b22')

        O_b33_plot = fig_OThermEllBeta.add_subplot(3,2,3)
        O_b33_Fit = [ self.fitting_equation(T_K, *self.fit_coefficients['O_b33']) for T_K in self.ValidTemps_K ]
        O_b33_plot.plot( self.ValidTemps_K, O_b33_Fit )
        O_b33_plot.errorbar( self.fit_Measured_temps['ThermEll_T_K'], self.fit_Measured_data['O_b33'], yerr=self.fit_Measured_errors['O_b33'], fmt = 'rx' )   
        O_b33_plot.set(xlabel = 'Temp (K)',ylabel = 'O b33')

        O_b12_plot = fig_OThermEllBeta.add_subplot(3,2,4)
        O_b12_Fit = [ self.fitting_equation(T_K, *self.fit_coefficients['O_b12']) for T_K in self.ValidTemps_K ]
        O_b12_plot.plot( self.ValidTemps_K, O_b12_Fit )
        O_b12_plot.errorbar( self.fit_Measured_temps['ThermEll_T_K'], self.fit_Measured_data['O_b12'], yerr=self.fit_Measured_errors['O_b12'], fmt = 'rx' )   
        O_b12_plot.set(xlabel = 'Temp (K)',ylabel = 'O b12')

        O_b13_plot = fig_OThermEllBeta.add_subplot(3,2,5)
        O_b13_Fit = [ self.fitting_equation(T_K, *self.fit_coefficients['O_b13']) for T_K in self.ValidTemps_K ]
        O_b13_plot.plot( self.ValidTemps_K, O_b13_Fit )
        O_b13_plot.errorbar( self.fit_Measured_temps['ThermEll_T_K'], self.fit_Measured_data['O_b13'], yerr=self.fit_Measured_errors['O_b13'], fmt = 'rx' )   
        O_b13_plot.set(xlabel = 'Temp (K)',ylabel = 'O b13')

        O_b23_plot = fig_OThermEllBeta.add_subplot(3,2,6)
        O_b23_Fit = [ self.fitting_equation(T_K, *self.fit_coefficients['O_b23']) for T_K in self.ValidTemps_K ]
        O_b23_plot.plot( self.ValidTemps_K, O_b23_Fit )
        O_b23_plot.errorbar( self.fit_Measured_temps['ThermEll_T_K'], self.fit_Measured_data['O_b23'], yerr=self.fit_Measured_errors['O_b23'], fmt = 'rx' )   
        O_b23_plot.set(xlabel = 'Temp (K)',ylabel = 'O b23')
                                            
        plt.savefig('AlphaQuartz_Laevo_zm_OThermEll.png')      
        plt.close(3)

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
        print("Handedness of screw axis = Right")
        print("Model for thermal motion: anisotropic thermal ellipsoids")
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
        
        m = +1
        
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

    def beta_matrix_gen(self, element, beta_list = None):
        """Generates full 3x3 matrix from list of beta values."""
        
        B_11 = beta_list[0]
        B_22 = beta_list[1]
        B_33 = beta_list[2]
            
        if element =="Si":
                
            if len(beta_list) == 4:
                
                B_23 = beta_list[3]
                B_13 = 0.5*B_23
                B_12 = 0.5*B_22
                
            else:                    
                print("Error: Length of temperature factor list is incorrect.")
                Beta = None
                        
        if element == "O":
                
            if len(beta_list) == 6:

                B_12 = beta_list[3]
                B_13 = beta_list[4]
                B_23 = beta_list[5]
                
            else:
                print("Error: Length of temperature factor list is incorrect.")
                Beta = None
                
        Beta = np.array(([B_11,B_12,B_13],[B_12,B_22,B_23],[B_13,B_23,B_33]))
        
        Check.length(Beta,9)
                
        return Beta

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

        Si_u = 1 - self.fitting_equation(self.temperature_K, *self.fit_coefficients['Si_u'])
        
        Si_refatom_pos = np.array(["Si",Si_u,0,0])

        t_multiplier = self.fitting_equation(self.temperature_K, *self.fit_coefficients['O_xyz'])
            
        O_refatom_pos = np.array((1,1,1)) - (np.array(self.O_Reference_Line_Start) + t_multiplier*(np.array(self.O_Reference_Line_Vector)))
                    
        O_refatom_pos = np.array(("O",O_refatom_pos[0],O_refatom_pos[1],O_refatom_pos[2]))
            
        refatom_coords = np.array([Si_refatom_pos, O_refatom_pos])
            
        Check.length(O_refatom_pos,4)
        Check.length(Si_refatom_pos,4)
        
        return refatom_coords

    def atoms_init_and_update(self,Initial):        
        """Determines coordinates and beta matrix of each atom and initializes an Atom object for it."""        

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
            
        #Below is the calculation of beta matrices for atoms, to be used in DW correction
        #for left-handed screw in right-handed coordinate system
        Si_b_11 = (self.fitting_equation(self.temperature_K, *self.fit_coefficients['Si_b11']))*1e-4
        Si_b_22 = (self.fitting_equation(self.temperature_K, *self.fit_coefficients['Si_b22']))*1e-4
        Si_b_33 = (self.fitting_equation(self.temperature_K, *self.fit_coefficients['Si_b33']))*1e-4
        Si_b_23 = (self.fitting_equation(self.temperature_K, *self.fit_coefficients['Si_b23']))*1e-4
            
        beta_list_Si = [Si_b_11,Si_b_22,Si_b_33,Si_b_23]            
            
        beta_mtrx_Si1 = self.beta_matrix_gen("Si",beta_list_Si)
            
        M = self.three_fold_matrix()
        beta_mtrx_Si2 =  np.dot(M,(np.dot(beta_mtrx_Si1,np.transpose(M))))
        beta_mtrx_Si3 =  np.dot(M,(np.dot(beta_mtrx_Si2,np.transpose(M))))
            
        O_b_11 = (self.fitting_equation(self.temperature_K, *self.fit_coefficients['O_b11']))*1e-4
        O_b_22 = (self.fitting_equation(self.temperature_K, *self.fit_coefficients['O_b22']))*1e-4
        O_b_33 = (self.fitting_equation(self.temperature_K, *self.fit_coefficients['O_b33']))*1e-4
        O_b_12 = (self.fitting_equation(self.temperature_K, *self.fit_coefficients['O_b12']))*1e-4
        O_b_13 = (self.fitting_equation(self.temperature_K, *self.fit_coefficients['O_b13']))*1e-4
        O_b_23 = (self.fitting_equation(self.temperature_K, *self.fit_coefficients['O_b23']))*1e-4                                   
            
        beta_list_O = [O_b_11,O_b_22,O_b_33,O_b_12,O_b_13,O_b_23]            
            
        beta_mtrx_O1 = self.beta_matrix_gen("O",beta_list_O)  
            
        beta_mtrx_O2 =  np.dot(M,(np.dot(beta_mtrx_O1,np.transpose(M))))
        beta_mtrx_O3 =  np.dot(M,(np.dot(beta_mtrx_O2,np.transpose(M))))
            
        M = self.two_fold_matrix()
        beta_mtrx_O4 =  np.dot(M,(np.dot(beta_mtrx_O1,np.transpose(M))))
        beta_mtrx_O5 =  np.dot(M,(np.dot(beta_mtrx_O2,np.transpose(M))))
        beta_mtrx_O6 =  np.dot(M,(np.dot(beta_mtrx_O3,np.transpose(M))))
            
        beta_mtrx_list = [beta_mtrx_Si1,beta_mtrx_Si2,beta_mtrx_Si3,beta_mtrx_O1,beta_mtrx_O2,
                          beta_mtrx_O3,beta_mtrx_O4,beta_mtrx_O5,beta_mtrx_O6]
            
        coord_list = [Si1_coord,Si2_coord,Si3_coord,O1_coord,O2_coord,O3_coord,
                      O4_coord,O5_coord,O6_coord]
                
        atom_list = []
        
        for index in range (0,len(self.element_list)):

            if (Initial): # we create new atom objects from scratch

                atom = Atom(self.element_list[index],coord_list[index],self.environment,
                        self.lattice_vectors_A,beta_matrix=beta_mtrx_list[index])

                atom.coordinate = Tools.unit_cell_check(atom.coordinates)
                
                atom_list.append(atom)

            else: # we update the already existing atom objects

                self.atoms[index].update(coord_list[index],self.environment,self.lattice_vectors_A,beta_matrix=beta_mtrx_list[index]) 

                self.atoms[index].coordinate = Tools.unit_cell_check(self.atoms[index].coordinates)
                
                atom_list.append(self.atoms[index])
       
        return atom_list