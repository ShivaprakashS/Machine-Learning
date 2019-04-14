# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 22:53:41 2018

@author: Sathyaraj Natesan
"""
import sys
import pandas as pan
import math as mat
import copy as cop
import random as ran


def measure_entropy(class_labels):
    total_one_count = class_labels.sum().sum()
    total_rows = class_labels.shape[0]
    total_zero_count = total_rows - total_one_count
    if (total_rows == total_one_count) or (total_rows == total_zero_count):
        return 0
    entropy = -(total_one_count/total_rows)*mat.log(total_one_count/total_rows, 2) - (total_zero_count/total_rows)*mat.log(total_zero_count/total_rows,2)
    return entropy

def measure_variance_impurity(class_labels):
    total_one_count = class_labels.sum().sum()
    total_rows = class_labels.shape[0]
    total_zero_count = total_rows - total_one_count
    if (total_rows == total_one_count) or (total_rows == total_zero_count):
        return 0
    variance_impurity = (total_one_count*total_zero_count)/(total_rows*total_rows)
    return variance_impurity

def measure_information_gain(input_attributes):
    total_rows = input_attributes.shape[0]
    total_one_count = input_attributes[input_attributes[input_attributes.columns[0]] == 1].shape[0]
    total_zero_count = input_attributes[input_attributes[input_attributes.columns[0]] == 0].shape[0]
    root_entropy = measure_entropy(input_attributes[['Class']])
    ones_entropy = measure_entropy(input_attributes[input_attributes[input_attributes.columns[0]] == 1][['Class']])
    zeroes_entropy = measure_entropy(input_attributes[input_attributes[input_attributes.columns[0]] == 0][['Class']])
    infoGain = root_entropy - (total_one_count/total_rows)*ones_entropy - (total_zero_count/total_rows)*zeroes_entropy
    return infoGain

def measure_variance_impurity_gain(input_attributes):
    total_rows = input_attributes.shape[0]
    total_one_count = input_attributes[input_attributes[input_attributes.columns[0]] == 1].shape[0]
    total_zero_count = input_attributes[input_attributes[input_attributes.columns[0]] == 0].shape[0]
    
    root_vi = measure_variance_impurity(input_attributes[['Class']])
    ones_vi = measure_variance_impurity(input_attributes[input_attributes[input_attributes.columns[0]] == 1][['Class']])
    zeroes_vi = measure_variance_impurity(input_attributes[input_attributes[input_attributes.columns[0]] == 0][['Class']])
    vi_gain = root_vi - (total_one_count/total_rows)*ones_vi - (total_zero_count/total_rows)*zeroes_vi
    return vi_gain
    

def select_next_attribute(data, heuristicType):
    maxGain = -mat.inf
    for attr in data.columns:
        if attr == 'Class':
            continue
        if(heuristicType == "IG"):
            currentGain = measure_information_gain(data[[attr, 'Class']])
        else:
            currentGain = measure_variance_impurity_gain(data[[attr, 'Class']])
        if maxGain < currentGain:
            maxGain = currentGain
            selected_attribute = attr
    return selected_attribute

class Node:
    def __init__(this):
        this.left = None
        this.right = None
        this.attribute = None
        this.value = None
        this.nodeType = None  
        this.zero_count = None
        this.one_count = None
        this.nodeId = None
        this.label = None

    
    def put_values(this, attribute, nodeType, value = None, one_count = None, zero_count = None):
        this.attribute = attribute
        this.nodeType = nodeType
        this.value = value
        this.zero_count = zero_count
        this.one_count = one_count


class Tree:
    def __init__(this):
        this.root = Node()
        this.root.put_values('', 'R')
        this.nodeCount = 1
        
    def build_decision_tree(this, training_data, node, heuristicType):
        total_one_count = training_data['Class'].sum()
        total_rows = training_data.shape[0]
        total_zero_count = total_rows - total_one_count        
        if (training_data.shape[1] == 1) or (total_rows == total_one_count) or (total_rows == total_zero_count):
            node.nodeType = 'L'
            if (total_one_count <= total_zero_count):
                node.label = 0
            else:
                node.label = 1
            return        
        else:        
            selected_attribute = select_next_attribute(training_data, heuristicType)
            node.left = Node()
            node.right = Node()
            node.left.nodeId = this.nodeCount
            this.nodeCount=this.nodeCount+1
            node.right.nodeId = this.nodeCount
            this.nodeCount = this.nodeCount+1
            
            node.left.put_values(selected_attribute, 'I', 0, training_data[(training_data[selected_attribute]==0) & (training_data['Class']==1) ].shape[0], training_data[(training_data[selected_attribute]==0) & (training_data['Class']==0) ].shape[0])
            node.right.put_values(selected_attribute, 'I', 1, training_data[(training_data[selected_attribute]==1) & (training_data['Class']==1) ].shape[0], training_data[(training_data[selected_attribute]==1) & (training_data['Class']==0) ].shape[0])
            this.build_decision_tree( training_data[training_data[selected_attribute]==0].drop([selected_attribute], axis=1), node.left, heuristicType)
            this.build_decision_tree( training_data[training_data[selected_attribute]==1].drop([selected_attribute], axis=1), node.right, heuristicType)
            
    def print_tree_hierarchy(this, node,level):
        if(node.left is None and node.right is not None):
            for i in range(0,level):    
                print("| ",end="")
            level = level + 1
            print("{} = {} : {}".format(node.attribute, node.value,(node.label if node.label is not None else "")))
            this.print_tree_hierarchy(node.right,level)
        elif(node.right is None and node.left is not None):
            for i in range(0,level):    
                print("| ",end="")
            level = level + 1
            print("{} = {} : {}".format(node.attribute, node.value,(node.label if node.label is not None else "")))
            this.print_tree_hierarchy(node.left,level)
        elif(node.right is None and node.left is None):
            for i in range(0,level):    
                print("| ",end="")
            level = level + 1
            print("{} = {} : {}".format(node.attribute, node.value,(node.label if node.label is not None else "")))
        else:
            for i in range(0,level):    
                print("| ",end="")
            level = level + 1
            print("{} = {} : {}".format(node.attribute, node.value, (node.label if node.label is not None else "")))
            this.print_tree_hierarchy(node.left,level)
            this.print_tree_hierarchy(node.right,level)
    
    def print_tree(this):
        this.print_tree_hierarchy(this.root.left,0)
        this.print_tree_hierarchy(this.root.right,0)
        
    def node_count(this,node):
        if(node.left is not None and node.right is not None):
            return 2 + this.node_count(node.left) + this.node_count(node.right)
        return 0

    def leaf_count(this,node):
        if(node.left is None and node.right is None):
            return 1
        return this.leaf_count(node.left) + this.leaf_count(node.right)
        
    def infer_class_label(this, data, root):
        if root.label is not None:
            return root.label
        elif data[root.left.attribute][data.index.tolist()[0]] == 1:
            return this.infer_class_label(data, root.right)
        else:
            return this.infer_class_label(data, root.left)
    
    def pruneTree(this, lIterator, kIterator, validation_data):
        bestTree = this
        for i in range(1,lIterator):
            copiedTree = cop.deepcopy(this)
            m = ran.randint(1,kIterator)
            for j in range(1,m):
                numOfNonleafNodes = this.node_count(copiedTree.root) - this.leaf_count(copiedTree.root)
                if(numOfNonleafNodes == 0):
                    return bestTree
                randomLocation = ran.randint(1, numOfNonleafNodes)
                radomNode = find_node(copiedTree.root,randomLocation)
                if(radomNode is not None):
                    radomNode.left = None
                    radomNode.right = None
                    radomNode.nodeType = "L"
                    if(radomNode.zero_count >= radomNode.one_count):
                        radomNode.label = 0
                    else:
                        radomNode.label = 1
            if accuracy_estimator(validation_data, copiedTree) > accuracy_estimator(validation_data, bestTree):
                bestTree = copiedTree
        return bestTree

def find_node(node, location):
    returnedNode = None
    if(node.nodeType != "L"):
        if(node.nodeId == location):
            return node
        else:
            returnedNode = find_node(node.left,location)
            if (returnedNode is None):
                returnedNode = find_node(node.right,location)
            return returnedNode
    else:
        return returnedNode

def accuracy_estimator(data, tree):
    correctCount = 0
    for i in data.index:
        val = tree.infer_class_label(data.iloc[i:i+1, :].drop(['Class'], axis=1),tree.root)
        if val == data['Class'][i]:
            correctCount = correctCount + 1
    return correctCount/data.shape[0]*100


def main(argv):
    if (len(argv) == 7):
        l_value = int(sys.argv[1])    
        k_value = int(sys.argv[2])
        train_data_csv = sys.argv[3]
        validation_data_csv = sys.argv[4]
        test_data_csv_ = sys.argv[5]
        is_print = sys.argv[6]
    else:
        sys.exit("Please provide input arguments in this format:\n <L?:Integer> <K?:Integer> <training_set?:file path> <validation_set?:file path> " + 
              "<test_set?:file path> <to_print?:yes/no>")
    
    training_data = pan.read_csv(train_data_csv)
    test_data = pan.read_csv(test_data_csv_)
    validation_data = pan.read_csv(validation_data_csv)
    
    ig_model_tree = Tree()
    ig_model_tree.build_decision_tree(training_data, ig_model_tree.root, "IG")
    ig_model_prunedtree = ig_model_tree.pruneTree(l_value, k_value, validation_data)
    if('yes' == is_print):
        print("\n********** Tree Model - Information Heuristic - Pre Prune ***********\n")
        ig_model_tree.print_tree()
        print("\n********** Tree Model - Information Heuristic - Post Prune ***********\n")
        ig_model_prunedtree.print_tree()
    
    vig_model_tree = Tree()
    vig_model_tree.build_decision_tree(training_data,vig_model_tree.root, "VI")
    vig_model_prunedtree = vig_model_tree.pruneTree(l_value, k_value, validation_data)
    if('yes' == is_print):
        print("\n\n^^^^^^^^ Tree Model - Variance Impurity Heuristic - Pre Prune ^^^^^^^^^\n")
        vig_model_tree.print_tree()
        print("\n^^^^^^^^ Tree Model - Variance Impurity Heuristic - Post Prune ^^^^^^^^^\n")
        vig_model_prunedtree.print_tree()

    print("\n\n^^^^^^^^^ Accuracy Metrics - Pre Pruning^^^^^^^^^^^^\n")
    print("Information Gain Heuristic -> "+str(accuracy_estimator(test_data, ig_model_tree))+"%")
    print("Variance Impurity Heuristic -> "+str(accuracy_estimator(test_data, vig_model_tree))+"%")
    print("\n\n^^^^^^^^^ Accuracy Metrics - Post Pruning^^^^^^^^^^^^\n")
    print("Information Gain Heuristic -> "+str(accuracy_estimator(test_data, ig_model_prunedtree))+"%")
    print("Variance Impurity Heuristic -> "+str(accuracy_estimator(test_data, vig_model_prunedtree))+"%")

if __name__ == "__main__":
    main(sys.argv)
