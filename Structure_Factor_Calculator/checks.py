#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 11:10:35 2019

@author: jamespittard
"""
import numpy as np

class Check:
    """Class that contains various checks throughout the code for inputs that may not
    otherwise produce an error and so may cause incorrect values. Especially
    important if the code is being edited"""
                
    def length(in_,length):
        """Simply checks the length of a list or size of an array """
        if type(in_) == list: 
            if len(in_) != length:
                raise TypeError("Length of ",in_," variable is incorrect.")
        
        if type(in_) == np.ndarray:
            if in_.size != length:
                raise TypeError("Size of ",in_," array is incorrect.")
            
    def coefficient(extracted_value, value):
        """Compares two values """
        
        if extracted_value != value:
            raise ValueError("Coefficients are not correctly read.")
    
    def row(row,expected_first_element):
        """Insures correct row is used when reading data tables"""
        
        if row[0] != expected_first_element:
            raise ValueError("Incorrect row has been read.")
            
        