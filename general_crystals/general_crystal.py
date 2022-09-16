from abc import ABC, abstractmethod

import pkgutil
import importlib
import os

import numpy as np
from Structure_Factor_Calculator.tools import Tools
from Structure_Factor_Calculator.diffraction_environment import Diff_Environment

class GeneralCrystal(ABC):
    """A general crystal object constructs a basic crystal with a simple
    set of attributes and methods that are useful regardless of the material
    that will ultimately be used. Information for specific crystal materials
    (e.g. alpha quartz dextro/laevo, sapphire, silicon) will be added in 
    separate modules to produce child classes of GeneralCrystal."""

    @property
    @abstractmethod
    def description(self):
        pass

    def __init__(self,crystal_system,unit_cell_params,hkl,energy_eV):

        self.crystal_system = crystal_system # this is the crystal system (cubic, tetragonal, orthorhombic, hexagonal, rhombohedral, monoclinic, triclinic)
        self.unit_cell_params = unit_cell_params # this is the set of lattice parameters (a,b,c,alpha,beta,gamma)

        self.a_A = unit_cell_params[0]
        self.b_A = unit_cell_params[1]
        self.c_A = unit_cell_params[2]

        self.alpha_rad = unit_cell_params[3]
        self.beta_rad = unit_cell_params[4]
        self.gamma_rad = unit_cell_params[5]
        self.alpha_deg = self.alpha_rad*(180/np.pi)
        self.beta_deg = self.beta_rad*(180/np.pi)
        self.gamma_deg = round(self.gamma_rad*(180/np.pi),6)

        self.lattice_vectors_A = self.latt_vec_A()

        self.G_mtrx_A2 = self.G_matrix_A2()

        self.angle_plane_wavelength = self.angle_finder(self.unit_cell_params,self.crystal_system,hkl,energy_eV)

    def angle_finder(self,lattice_unit_cell_params,crystal_system,hkl_,energy_eV):
        wavelength_A = Tools.energy_calc_eV(energy_eV)
        d_A = Diff_Environment.d_hkl(lattice_unit_cell_params,hkl_,crystal_system)
        bragg_condition = 0.5*wavelength_A/d_A
        theta_rad = np.arcsin(bragg_condition)
        return theta_rad,hkl_,wavelength_A

    def latt_vec_A(self):
        """lattice vector function, returns a1,a2,a3 in cartesian vector form"""

        a_vec_A = np.array([self.a_A,0,0])
        b_vec_A = np.array([self.b_A*np.cos(self.gamma_rad),self.b_A*np.sin(self.gamma_rad),self.b_A*np.cos(self.alpha_rad)])
        c_vec_A = np.array([self.c_A*np.cos(self.beta_rad),self.c_A*np.cos(self.beta_rad),self.c_A*np.sin(self.beta_rad)])
                
        latt_vec_arr = np.array((a_vec_A,b_vec_A,c_vec_A))
        
        latt_vec_arr = Tools.set_to_zero(latt_vec_arr,3,3,1e-8)

        return latt_vec_arr
    

    def G_matrix_A2(self):
        """Generates G_matrix from lattice vectors. Required for Debye Waller correction """
        
        a_vec_A = self.lattice_vectors_A[0]
        b_vec_A = self.lattice_vectors_A[1]
        c_vec_A = self.lattice_vectors_A[2]
        
        G_A2 = np.array([[np.dot(a_vec_A,a_vec_A),np.dot(a_vec_A,b_vec_A),np.dot(a_vec_A,c_vec_A)],
                       [np.dot(b_vec_A,a_vec_A),np.dot(b_vec_A,b_vec_A),np.dot(b_vec_A,c_vec_A)],
                       [np.dot(c_vec_A,a_vec_A),np.dot(c_vec_A,b_vec_A),np.dot(c_vec_A,c_vec_A)]])
    
        G_A2 = Tools.set_to_zero(G_A2,3,3,1e-8)
    
        return G_A2

    @abstractmethod
    def information(self):
        pass

    @abstractmethod
    def lattice_unit_cell_params(self):
        pass

    @abstractmethod
    def refatom_coordinates(self):
        pass

    @abstractmethod
    def atoms_init_and_update(self):
        pass
_registry = {}

class CrystalFactory:

    @classmethod
    def get_descriptions(cls):
        if not _registry:
            cls._fill_registry()

        return list(_registry.keys())

    @classmethod
    def get_crystal(cls, description, *args):
        if not _registry:
            cls._fill_registry()

        if description in _registry:
            return _registry[description](*args)
        else:
            return None

    @classmethod
    def _fill_registry(cls):
        pkg_dir = os.path.dirname(__file__)
        for (module_loader, description, ispkg) in pkgutil.iter_modules([pkg_dir]):
            print(description)
            importlib.import_module('.' + description, __package__)

        
        for cls in GeneralCrystal.__subclasses__():
            _registry[cls.description] = cls
