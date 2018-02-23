#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import dt_functions as dt

def main(trainData_path, validationData_path, testData_path, pruningFactor):

     dTree, dTree_pruned, accuracy, accuracy_pruned, nodeInfo = dt.analyze(trainData_path, validationData_path, testData_path, pruningFactor)
     print('\nAccuracy of Decision Tree on Train | Validation | Test Data: ',accuracy)
     print('\nAccuracy on Decision Tree (Post-pruning) on Train | Validation | Test Data: ',accuracy_pruned)
     print('\nTree and other details written to file:  DecisionTree_Results.txt & DecisionTree_Pruned_Results.txt')

     return


trainData_path = sys.argv[1]
validationData_path = sys.argv[2]
testData_path = sys.argv[3]
pruning_factor = float(sys.argv[4]) 
main(trainData_path,validationData_path,validationData_path,pruning_factor)