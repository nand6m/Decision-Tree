#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
import copy


### Check for any redundant data instances 
def checkRedundancy(dataSet):
    labels = list(dataSet)
    temp = dataSet[labels[:-1]]
    duplicate_index = np.where(temp.duplicated(subset=None, keep='first') == True)
    dataSet = dataSet.drop(duplicate_index[0])
    return dataSet


### Read Datasets from CSV Files
def readDatasets(trainData_path, testData_path, validationData_path):
    # Read CSV Files
    d_train = pd.read_csv(trainData_path, index_col=False)
    d_test = pd.read_csv(testData_path, index_col=False)
    d_valid = pd.read_csv(validationData_path, index_col=False)
    
    # Check for Null Entries
    d_train = d_train.dropna(axis=1, how='any')
    d_test = d_test.dropna(axis=1, how='any')
    d_valid = d_valid.dropna(axis=1, how='any')
    
    # Check fro redundancies
    d_train = checkRedundancy(d_train)
    
    return d_train, d_test, d_valid 


### Computes Entropy of a vector (Attribute/Class)
def computeEntropy(dataSet):
    target = dataSet['Class'].values
    p_ones = np.sum(target)/(np.size(target) + 1e-13)
    p_zeros = 1.0 - p_ones
    H = -1.0*(p_ones*np.log2(p_ones + 1e-13) + p_zeros*np.log2(p_zeros + 1e-13))
    
    return H


### Computes Conditional Entropy (Class given an attribute)
def computeCondEntropy(dataSet):
    column_labels = list(dataSet)
    H_A = np.zeros(len(column_labels)-1, dtype = float)
    P = np.zeros((len(column_labels)-1,2), dtype = float)
    for k in range(len(column_labels)-1):
        data_subset = dataSet[[column_labels[k], 'Class']]
        data_subset_class1 = data_subset.loc[data_subset[column_labels[k]] == 1]
        data_subset_class0 = data_subset.loc[data_subset[column_labels[k]] == 0]
        
        H_1 = computeEntropy(data_subset_class1)
        H_0 = computeEntropy(data_subset_class0)
        P[k,0] = data_subset_class0.shape[0]/(data_subset.shape[0] + 1e-13)
        P[k,1] = data_subset_class1.shape[0]/(data_subset.shape[0] + 1e-13)
        
        H_A[k] = P[k,0]*H_0 + P[k,1]*H_1
    H_A = np.asarray(H_A)  
    
    return H_A, P
  


### Computes information-gain for each attribute and finds best feature
def findBestFeature(dataSet, labels):
    
    entropy = computeEntropy(dataSet)
    condEntropy, dataDistribution = computeCondEntropy(dataSet)     
    infoGain = entropy - condEntropy
    try:
        maxInfoGainLoc = np.argmax(infoGain)
    except:
        maxInfoGainLoc = 0
        return
    bestFeature = labels[maxInfoGainLoc]
    
    leafData = dataSet.drop(bestFeature, axis=1)
    leafData0 = leafData.loc[dataSet[bestFeature] == 0]
    leafData1 = leafData.loc[dataSet[bestFeature] == 1]
    leafData = [leafData0, leafData1]
    
    return bestFeature, leafData, len(leafData0), len(leafData1)


### Build decision tree using training data
def getDecisionTree(dataSet):
    
    classList = dataSet['Class'].values
    
    # No Class information - Stop splitting and randomly assign a class
    if len(classList) == 0:
        return np.int64(np.random.randint(low=0, high=2))
    
    # All instances from same class - Stop Splitting and assign the particular class
    if np.mod(np.sum(classList),len(classList)) == 0:
        return classList[0]  
    
    labels = list(dataSet)
    bestFeat, leafData, leftDataSize, rightDataSize = findBestFeature(dataSet, labels)
    
    # Recursively build decision tree
    decisionTree = {bestFeat: {}}
    for k in range(len(leafData)):
        decisionTree[bestFeat][k] = getDecisionTree(leafData[k])
        
    return decisionTree


### Build decision tree using training data
def getNodeInfo(dataSet, nodeInfo = None):
    
    if nodeInfo == None:
        nodeInfo = {}
        
    classList = dataSet['Class'].values
    
    # No Class information - Stop splitting and randomly assign a class
    if len(classList) == 0:
        return np.int64(np.random.randint(low=0, high=2))
    
    # All instances from same class - Stop Splitting and assign the particular class
    if np.mod(np.sum(classList),len(classList)) == 0:
        return classList[0]  
    
    labels = list(dataSet)
    bestFeat, leafData, leftDataSize, rightDataSize = findBestFeature(dataSet, labels)
    
    tempData = pd.concat(leafData)
    if bool(len(tempData)):
        #tempstr = {}
        #tempstr[bestFeat] =  int((np.sum(tempData['Class'].values)/len(tempData['Class'].values))>=0.5)
        #nodeInfo.append(tempstr)
        nodeInfo[bestFeat] =  int((np.sum(tempData['Class'].values)/len(tempData['Class'].values))>=0.5)
        
    # Recursively build decision tree
    for k in range(len(leafData)):
        getNodeInfo(leafData[k], nodeInfo)
        
    return nodeInfo


### Display decision tree in the required format + Count Nodes and leaves
def displayDecisionTree( dTree, file, **kwargs ):
    if bool(len(kwargs)):
        for items in kwargs:
            if items == 'layers':
                layers = kwargs[items]
            elif items == 'node':
                node = kwargs[items]
            else:
                leaf = kwargs[items]
    else:
        layers = 0
        node = [] 
        leaf = []

    if type(dTree) == dict:
        for key in dTree:
            tempTree = dTree[key]
            node.append(key)
            layers = layers + 1
            for k in range(len(tempTree)):
                if type(tempTree[k]) == dict:            
                    file.write((' | '*(layers-1) + key + ' = ' + str(k) + ' : \n'))
                else:
                    file.write((' | '*(layers-1) + key + ' = ' + str(k) +' : ' + str(tempTree[k])) + '\n')
                    leaf.append(tempTree[k])
                node, leaf = displayDecisionTree(tempTree[k], file, layers = layers, node = node, leaf = leaf)
                
    return node, leaf


### Predit a class of a new dataset using decision tree
def predictClass(decisionTree, data):
    
    predictions = np.zeros(len(data),)
    for k in range(len(data)):
        predictions[k] = predictSingleData(decisionTree, data.iloc[k])
    predictions = np.asarray([int(predictions[k]) for k in range(len(data))])
    
    return predictions

  
 ### Predit a class of a new instance using decision tree   
def predictSingleData(decisionTree, data):
    
    if type(decisionTree) == dict:
        for key in decisionTree:
            node = key
            tempTree = decisionTree[key]
            if type(tempTree[data[node]]) == dict:
                prediction = int(predictSingleData(tempTree[data[node]], data))
                
            else:
                prediction = tempTree[data[node]]
    else:
        prediction = decisionTree
        
    return prediction


### Save results to a Text File
def displayMetrics(dataSet, dTree, pruning):
    accuracy = np.zeros((len(dataSet),))
    for k in range(len(dataSet)):
        temp_dataset = dataSet[k]
        prediction = predictClass(dTree, temp_dataset)
        accuracy[k] = 100.0*np.sum(temp_dataset['Class'] == prediction)/len(temp_dataset)
    
    if pruning:
        file= open('./DecisionTree_Pruned_Results.txt','w')
    else:
        file = open('./DecisionTree_Results.txt','w')
    

  
 
    
    if pruning:
        file.write('Decision Tree with Pruning' + '\n')
        file.write('__________________________' + '\n')
    else:
        file.write('Decision Tree without Pruning' + '\n')
        file.write('_____________________________' + '\n')
    
    file.write('\n')
    
    
    
    nodes, leaves = displayDecisionTree(dTree, file)
    
    if pruning:
        file.write('\n\nPost-Pruning Accuracy' + '\n')
        file.write('________________________' + '\n')
    else:
        file.write('\n\nPre-Pruning Accuracy' + '\n')
        file.write('_________________________' + '\n')
    
    file.write('\n')
    file.write('Number of Training Instances:      ' +  str(len(dataSet[0])) + '\n')
    file.write('Number of Training Attributes:     ' + str(len(list(dataSet[0]))) + '\n')
    file.write('Total number of nodes in the tree: ' + str(len(nodes) + len(leaves)) + '\n')
    file.write('Number of leaf nodes in the tree:  ' + str(len(leaves)) + '\n')
    file.write('Accuracy of the model on Training DataSet:   ' + str(accuracy[0]) + '%' + '\n')
    
    
    file.write('\n\n')
    file.write('Number of Validation Instances:      ' +  str(len(dataSet[1])) + '\n')
    file.write('Number of Validation Attributes:     ' + str(len(list(dataSet[1]))) + '\n')
    file.write('Accuracy of the model on Validation DataSet:   ' + str(accuracy[1]) +'%' + '\n')
    
    
    file.write('\n\n')
    file.write('Number of Testing Instances:      ' + str(len(dataSet[2])) + '\n')
    file.write('Number of Testing Attributes:     ' + str(len(list(dataSet[2]))) + '\n')
    file.write('Accuracy of the model on Testing DataSet:   ' + str(accuracy[2]) + '%' + '\n')
    
    
    file.close()
    
    return accuracy



def pruneDTree(dTree, dataset, nodeInfo, pruningFactor):
    pruneTree = copy.deepcopy(dTree)
    if pruningFactor == 0:
        return dTree
    else:
        labels = list(dataset)
        labels = labels[:-1]
        numNodes = int(np.round(pruningFactor*len(nodeInfo)))
        prediction = predictClass(dTree, dataset)
        apriori_accuracy = 100.0*np.sum(dataset['Class'] == prediction)/len(dataset)
        print('Number of nodes being pruned: ', numNodes)
        print('Validation Accuracy: ',apriori_accuracy, '%')
        for k in range(numNodes):
            rand_num = int(np.random.randint(0,len(labels),(1,)))
            print('Pruning Node:  ',labels[rand_num])
            tempTree = findNode(copy.deepcopy(pruneTree), labels[rand_num], nodeInfo)
            prediction = predictClass(tempTree, dataset)
            accuracy = 100.0*np.sum(dataset['Class'] == prediction)/len(dataset)
            if accuracy >= apriori_accuracy:
                print('Accuracy Improvement Noted')
                pruneTree = tempTree
            del labels[rand_num]
    
    return pruneTree



def findNode(pruneTree, node, nodeInfo):
    if type(pruneTree) == dict:
        for key in pruneTree:
            tempTree = pruneTree[key]
            if key == node:
                for k in range(len(tempTree)):
                    tempTree[k] = nodeInfo[node]
                pruneTree[key] = tempTree
            else:
                for k in range(len(tempTree)):
                    findNode(tempTree[k], node, nodeInfo)
    
    return pruneTree

### Main Function to call other functions
def analyze(trainData_path, validationData_path, testData_path, pruningFactor):
    d_train, d_test, d_valid = readDatasets(trainData_path, testData_path, validationData_path)
    decisionTree = getDecisionTree(d_train)
    nodeInfo = getNodeInfo(d_train)
    accuracy = displayMetrics([d_train, d_valid, d_test], decisionTree, False)
    decisionTree_pruned = pruneDTree(decisionTree, d_valid, nodeInfo, pruningFactor)
    if pruningFactor > 0:
        accuracy_pruned = displayMetrics([d_train, d_valid, d_test], decisionTree_pruned, True)
    return decisionTree, decisionTree_pruned, accuracy, accuracy_pruned, nodeInfo









