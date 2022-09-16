import numpy as np
import sys
sys.path.append("..")

from general_crystals import CrystalFactory
from general_crystals.Ge_Origin2_isodwf import Ge_isodwf
from Structure_Factor_Calculator.structure_factor_calc import Structure_Factor

print("sys.path = ",sys.path)
descriptions = CrystalFactory.get_descriptions()

# User parameters

hkl1 = [6,2,0]
tempmin_K = 3
tempmax_K = 1085
energy_eV = 7051.47 # this is the energy of the Fe Kb1,3 emission line.
OutFile1 = 'Ge_Origin2_SF_Out_hkl1=620_E=7051p47eV.txt'

# Crystal object initialization

GeIso = Ge_isodwf(tempmin_K,hkl1,energy_eV)

# Calculation and output of structure factors versus temperature

for tempdK in range(tempmin_K,tempmax_K+1):

    GeIso.set_temp_miller_energy(tempdK,hkl1,energy_eV)
    ThBragg_deg = GeIso.environment.angle_deg
    SF = Structure_Factor.F_hkl(GeIso,GeIso.environment)

    SF_H = SF[0][0]
    SF_H_Mag = np.abs(SF_H)**2
    SF_H_Angrad = np.angle(SF_H)

    SF_Hbar = SF[1][0]
    SF_Hbar_Mag = np.abs(SF_Hbar)**2
    SF_Hbar_Angrad = np.angle(SF_Hbar)

    SF_0 = SF[2][0]
    SF_0_Mag = np.abs(SF_0)**2
    SF_0_Angrad = np.angle(SF_0)
    
    with open(OutFile1,'a') as out:
        out.write( '{0:d} {1:5f} {2:5f} {3:5f} {4:5f} {5:5f} {6:5f} {7:5f}'.format( tempdK, ThBragg_deg, SF_H_Mag, SF_H_Angrad, SF_Hbar_Mag, SF_Hbar_Angrad, SF_0_Mag, SF_0_Angrad ) + '\n' )