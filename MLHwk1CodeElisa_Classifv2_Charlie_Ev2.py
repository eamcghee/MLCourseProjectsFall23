#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 15:43:08 2023

@author: em
"""

# //////////////////////////////////////
# Setup
# //////////////////////////////////////

# # Requires Python 3.7 and Scikit-Learn ≥ 1.0.1:

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.tree import export_graphviz
#from graphviz import Source # To plot trees
from sklearn.inspection import permutation_importance

import graphviz
from graphviz import Source

from pathlib import Path
from packaging import version

assert sys.version_info >= (3, 7)
# assert version.parse(sklearn.__version__) >= version.parse("1.0.1")


print(f"python version = {sys.version}")
print(f"scikit-learn version = {sklearn.__version__}")


# let's define the default font sizes to make the figures prettier


plt.rc('font', size=14)
plt.rc('axes', labelsize=14, titlesize=14)
plt.rc('legend', fontsize=14)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)

print(f"python version = {sys.version}")
print(f"numpy version = {np.__version__}")
print(f"scikit-learn version = {sklearn.__version__}")

# Create the images/classification folder
#Define the save_fig() function

Path = '/Users/em/Documents/Courses 2022-2024/ATS 780A7_Machine_Learning'
#IMAGES_PATH = Path() / "images" / "classification"
#IMAGES_PATH.mkdir(parents=True, exist_ok=True)


def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = IMAGES_PATH / f"{fig_id}.{fig_extension}"
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)
    
# //////////////////////////////////////
# Import the data
# //////////////////////////////////////


# I have created a csv file from a spectrogram image with pixel RGB values in each cell
# I prefer to use the url method below to import the csv file, but I did not have luck with this...?

#url = "http://github.com/eamcghee/MLCourseProjectsFall23/blob/main/DR01_0101_0407pixel_values_2.csv"
#features = pd.read_csv(url)

# I chose to import using pandas read instead:
features = pd.read_csv('DR01_0101_0407pixel_values_3_.csv', on_bad_lines='skip')

# separate into training, validation, and evaluation sets
# 0.60 training; 0.20 validation, 0.20 evaluation.
# 3456 //1- 2071 training, 2072-2761 (689) validation, 2762-3455 (693) evaluation

# I assigned labels/classes to each of the training set 1-2071 using the rule: if values exceed 20 for all of F1-F6 at a given time interval, return Yes for class
# # classify Yes classes as a swell dispersion event # predict (Yes) as a class / (No) as not a class
    
#X_train and y_train: features and labels used for training
# A csv of training data with reference time and features was saved as DR01_0101_0407_pixel_values_x_train
# A csv of training data with reference time and labels only was saved as DR01_0101_0407_pixel_values_y_train
x_train = pd.read_csv('DR01_0101_0407_pixel_values_x_train_neg1_notime.csv', on_bad_lines='skip')
y_train = pd.read_csv('DR01_0101_0407_pixel_values_y_train_neg1_notime.csv', on_bad_lines='skip')

#x_train = np.empty(shape = (1000,9))
#x_train.fill(1)

#y_train = np.empty(shape = (1000,))
#y_train.fill(1)

#X_val and y_val: features and labels used for validation (hyperparameter tuning)
# A csv of training data with reference time and features was saved as DR01_0101_0407_pixel_values_x_val
# A csv of training data with reference time and labels only was saved as DR01_0101_0407_pixel_values_y_val
x_val = pd.read_csv('DR01_0101_0407_pixel_values_x_val_neg1_notime.csv', on_bad_lines='skip')
y_val = pd.read_csv('DR01_0101_0407_pixel_values_y_val_neg1_notime.csv', on_bad_lines='skip')

#x_val = np.empty(shape = (1000,9))
#x_val.fill(1)

#y_val = np.empty(shape = (1000,))
#y_val.fill(1)

#X_test and y_test: features and labels held back for testing
# A csv of training data with reference time and features was saved as DR01_0101_0407_pixel_values_x_test
# A csv of training data with reference time and labels only was saved as DR01_0101_0407_pixel_values_y_test
x_test = pd.read_csv('DR01_0101_0407_pixel_values_x_test_neg1_notime.csv', on_bad_lines='skip')
y_test = pd.read_csv('DR01_0101_0407_pixel_values_y_test_neg1_notime.csv', on_bad_lines='skip')

#x_test = np.empty(shape = (1000,9))
#x_test.fill(1)

#y_test = np.empty(shape = (1000,))
#y_test.fill(1)

# Lines below are to look at the split data
# print('--- TRAINING ---')
# print(x_train.head(), y_train.head())
# print('--- VALIDATION ---')
# print(x_val.head(), y_val.head())
# print('--- HELD-BACK TESTING --- ')
# print(x_test.head(), y_test.head())

#We now have the following variables:
#x_train and y_train: features and labels used for training
#x_val and y_val: features and labels used for validation (hyperparameter tuning)
#x_test and y_test: features and labels held back for testing
#Now we are ready to set up the random forest model!

#In machine learning, it's common to have data separated into features (X) and the target variable (y).
X = pd.read_csv('DR01_0101_0407pixel_values_3_neg1_notime_X.csv', on_bad_lines='skip')
y = pd.read_csv('DR01_0101_0407pixel_values_3_neg1_notime_y.csv', on_bad_lines='skip')

feature_names = ['F1', 'F2', 'F3', 'F4', 'F5', 'F6'] #'['F1', 'F2', 'F3', 'F4', 'F5', 'F6]
classes = ['1', '-1'] # These are the 1 = Yes (exists); -1 = No (does not exist)


# /////////////////
#2. Set up and run the random forest

### MODIFY HYPERPARAMETERS WITHIN THIS CELL
fd = {
    "tree_number": 15,    # number of trees to "average" together to create a random forest
    "tree_depth": 5,      # maximum depth allowed for each tree
    "node_split": 20,     # minimum number of training samples needed to split a node
    "leaf_samples": 1,    # minimum number of training samples required to make a leaf node
    "criterion": 'gini',  # information gain metric, 'gini' or 'entropy'
    "bootstrap": False,   # whether to perform "bagging=bootstrap aggregating" or not
    "max_samples": None,  # number of samples to grab when training each tree IF bootstrap=True, otherwise None 
    "random_state": 13    # set random state for reproducibility
}

### Default values are retained below for reference
# fd = {
#     "tree_number": 15,    # number of trees to "average" together to create a random forest
#     "tree_depth": 5,      # maximum depth allowed for each tree
#     "node_split": 20,     # minimum number of training samples needed to split a node
#     "leaf_samples": 1,    # minimum number of training samples required to make a leaf node
#     "criterion": 'gini',  # information gain metric, 'gini' or 'entropy'
#     "bootstrap": False,   # whether to perform "bagging=bootstrap aggregating" or not
#     "max_samples": None,  # number of samples to grab when training each tree IF bootstrap=True, otherwise None 
#     "random_state": 13    # set random state for reproducibility
# }

forest = RandomForestClassifier(
                           n_estimators = fd["tree_number"],
                           random_state = fd["random_state"],
                           min_samples_split = fd["node_split"],
                           min_samples_leaf = fd["leaf_samples"],
                           criterion = fd["criterion"],
                           max_depth = fd["tree_depth"],
                           bootstrap = fd["bootstrap"],
                           max_samples = fd["max_samples"])


#Issue with 3.7 vs 3.9

# train the random forest

forest.fit(x_train, y_train.values.ravel()) # Runs the forest classifier
y_pred = forest.predict(x_train)


# check how well we did on the training data using the accuracy score and a confusion matrix.
     
acc = metrics.accuracy_score(y_train, y_pred)
print("training accuracy: ", np.around(acc*100), '%')

# #%%

# sys.exit(0)
# def confusion_matrix(predclasses, targclasses):
#   class_names = np.unique(targclasses)
#   table = []
#   for pred_class in class_names:
#     row = []
#     for true_class in class_names:
#         row.append(100 * np.mean(predclasses[targclasses == true_class] == pred_class))
#     table.append(row)
#   class_titles_t = classes
#   class_titles_p = classes
#   conf_matrix = pd.DataFrame(table, index=class_titles_p, columns=class_titles_t)
#   display(conf_matrix.style.background_gradient(cmap='Greens').format("{:.1f}"))

# confusion_matrix(y_train, y_pred)


#  try the validation data, to make sure our model applies to data it hasn't seen before

y_pred_val = forest.predict(x_val)

acc = metrics.accuracy_score(y_val, y_pred_val)
print("validation accuracy: ", np.around(acc*100), '%')
#confusion_matrix(y_val, y_pred_val)

#Continue tuning the hyperparameters until youre happy with the validation accuracy. 
#Once you are, you can once (and only once!) apply your model on the testing data. 

y_pred_test = forest.predict(x_test)

acc = metrics.accuracy_score(y_test, y_pred_test)
print("validation accuracy: ", np.around(acc*100), '%')
#confusion_matrix(y_test, y_pred_test)

# //////////////////////
#3. Explain and Interpret


local_path = '/Users/em/Documents/'

fig_savename = 'swell_tree'
tree_to_plot = 2 # Enter the value of the tree that you want to see!

tree = forest[tree_to_plot] # Obtain the tree to plot
tree_numstr = str(tree_to_plot) # Adds the tree number to filename

complete_savename = fig_savename + '_' + tree_numstr + '.dot'

Filepath = local_path + complete_savename
print(Filepath)

export_graphviz(tree,
                out_file=Filepath,
                filled=True,
                proportion=False,
                leaves_parallel=False,
                class_names=classes,
                feature_names=feature_names)

#Source.from_file(local_path + complete_savename)

# Read the DOT file
dot_file = "swell_tree_2.dot"

# Create a Graphviz object
dot = graphviz.Source.from_file(dot_file)

# Save the DOT file as a PNG image
output_file = "outputswell_tree_2.png"
dot.render(output_file, format="png")

print(f"DOT file converted to {output_file}")


#%%
# feature importance, which shows how important each feature was to the 
# random forest's predictions.

def calc_importances(rf, feature_list):
    ''' Calculate feature importance '''
    # Get numerical feature importances
    importances = list(rf.feature_importances_)

    # List of tuples with variable and importance
    feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]

    # Sort the feature importances by most important first
    feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)

    # Print out the feature and importances 
    print('')
    [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]
    print('')

    return importances

def plot_feat_importances(importances, feature_list):
    ''' Plot the feature importance calculated by calc_importances ''' 
    plt.figure()
    # Set the style
    plt.style.use('fivethirtyeight')
    # list of x locations for plotting
    x_values = list(range(len(importances)))
    # Make a bar chart
    plt.barh(x_values, importances)
    # Tick labels for x axis
    plt.yticks(x_values, feature_list)
    # Axis labels and title
    plt.xlabel('Importance'); plt.ylabel('Variable'); plt.title('Variable Importances')
    
    
plot_feat_importances(calc_importances(forest, feature_names),  feature_names)

# Permutation importance shows which features cause the largest drop in skill 
# if they are randomly shuffled.

# Single-pass permutation
permute = permutation_importance(
    forest, X, y, n_repeats=20, random_state=fd["random_state"])

# Sort the importances
sorted_idx = permute.importances_mean.argsort()

def plot_perm_importances(permute, sorted_idx, feature_list):
    ''' Plot the permutation importances calculated in previous cell '''
    # Sort the feature list based on 
    new_feature_list = []
    for index in sorted_idx:  
        new_feature_list.append(feature_list[index])

    fig, ax = plt.subplots()
    ax.boxplot(permute.importances[sorted_idx].T,
            vert=False, labels=new_feature_list)
    ax.set_title("Permutation Importances")
    fig.tight_layout()
    
plot_perm_importances(permute, sorted_idx, feature_names)

# 4. Run random forest classifier on the testing data

y_pred_test = forest.predict(X_test)

acc = metrics.accuracy_score(y_test, y_pred_test)
print("held-back testing accuracy: ", np.around(acc*100), '%')
confusion_matrix(y_test, y_pred_test)

# 

#%%
#calculate the Mean Absolute Error (MAE) for training and test sets












