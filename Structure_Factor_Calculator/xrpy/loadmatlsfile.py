# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 14:09:46 2015

@author: bren
"""

from __future__ import print_function
import os
import xlrd

pathStr = os.path.dirname(os.path.abspath(__file__))+os.sep+'materials.xls'

def loadMatlsFile():
    book = xlrd.open_workbook(pathStr)
    firstSheet = book.sheet_by_index(0)
    flatList = []
    for row in range(firstSheet.nrows):
        for col in range(firstSheet.ncols):
            flatList.append(firstSheet.cell(row,col).value)
    matList = zip(*[iter(flatList)]*3)
    name, density, formula = map(list,zip(*matList))
    density[0] = 0.0
    #uDen.type()
    name = [x.encode('UTF8') for x in name]
    name[0] = ' '
    #density = [float(x) for x in uDen]
    formula = [x.encode('UTF8') for x in formula]
    formula[0] = ' '
    return name

def loadMaterial(item):
    book = xlrd.open_workbook(pathStr)
    firstSheet = book.sheet_by_index(0)
    matList = []
    for col in range(firstSheet.ncols):
        matList.append(firstSheet.cell(item,col).value)
    return matList
    