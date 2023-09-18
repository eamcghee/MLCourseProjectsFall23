#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 11:30:41 2023

@author: em
"""

import sys
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import datetime
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import export_graphviz
from sklearn.inspection import permutation_importance
import pydot
import matplotlib.pyplot as plt
# %matplotlib inline

#error: @rpath/libtiff.5.dylib not loaded


print(f"python version = {sys.version}")
print(f"numpy version = {np.__version__}")
print(f"scikit-learn version = {sklearn.__version__}")  

# I have created a csv file from a spectrogram image with pixel RGB values in each cell


file_path = '/Users/em/Documents/Courses 2022-2024/ATS 780A7_Machine_Learning/DR01_0102_0407pixel_values_1.xlsx'
with open(file_path, 'r') as file:
    content = file.read()
