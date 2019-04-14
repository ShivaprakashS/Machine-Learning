from math import *
from numpy import *
import os
import re
import sys
from collections import Counter



''' inital zero list for the spam and ham vocabulary'''
def initialzerolist():
    length = len(listcounteritems)
    zero=list([0])
    clist = zero * length
    return clist

''' 0 and 1 for the combined spam and ham vocabulary so that the list can be updated with 1 when word occurs. It is a type of binary classification.'''
def traintestwordlist(listcounteritems, combineddata): 
    tlist = list()
    for words in combineddata:
        clist = initialzerolist()
        clist.insert(0,1)
        for word in listcounteritems:
            words1=combineddata[words]
            if word in words1:
                clist[listcounteritems.index(word)] = 1
        tlist.append(clist)
    return tlist

'''Gradient ascent in logistic regression for the number of iterations, lambda value and learning rate thereby updating the weights'''
def gradient_ascent(learningrate, numIteration, lambda1, matrices, values):
    values_transpose=mat(values).transpose()
    y = mat(matrices)
    dim = shape(y)
    dimcolumn = dim[1]
    weight = zeros((dimcolumn,1))
    for i in range(0, numIteration):
        '''Signmoid function: 1+exp(-(y*weight))'''
        error = (values_transpose - (1/(1+exp(-(y*weight)))))
        weight += learningrate*(y.transpose()* error - lambda1*weight)
    return weight

'''calculating the number of correctly classified spam and ham vocabulary after adding weights. Accuracy in percentage'''
def classificationaccuracy(spamtestlength, hamtestlength, weight, testmatrix):
    addweights = mat(testmatrix) * weight
    count = 0
    ''' Accuracy of files in spamtestlength'''
    for row in range(0, spamtestlength):
        if addweights[row]<0:
            count += 1
    ''' Accuracy from spamlength to total length of combined data'''        
    for row in range(spamtestlength+1,spamtestlength+hamtestlength):
        if addweights[row]>0:
            count += 1        
    return count/(spamtestlength+hamtestlength) * 100
'''reading the spam and ham files in train and test folder and deriving each words from the documents and adding to list. With stopwords list, words are filtered from the text files'''    
def read(path,stopwordstxt,stopwordsyn):
    stopwords = list()
    keyvalue = dict()
    words = list()
    readfile = open(stopwordstxt, 'r')
    for line in readfile.readlines():
        stopwords.append(line.strip())
    for file in os.listdir(path):
        readfile = open(path+"/"+file,encoding = "Latin-1")
        listofwords = re.split("\W+", readfile.read().strip())
        if stopwordsyn == 'yes':   
         wordsstop = list()
         for word in listofwords:
             if word not in stopwords:
                 wordsstop.append(word)
         keyvalue[file] = wordsstop
         words.extend(wordsstop)
        else:
         keyvalue[file] = listofwords
         words.extend(listofwords) 
    return words, keyvalue


if __name__ == "__main__":
    spamtrain = sys.argv[1]
    hamtrain = sys.argv[2]
    spamtest = sys.argv[3]
    hamtest = sys.argv[4]
    stopwordstxt = sys.argv[5]
    lambda1 = float(sys.argv[6])
    stopwordsyn = sys.argv[7]
    learningrate = float(sys.argv[8])
    numIteration = int(sys.argv[9])
    spamtrainpath= spamtrain + "/" + "spam"
    hamtrainpath= hamtrain + "/" + "ham"
    spamtestpath= spamtest + "/" + "spam"
    hamtestpath= hamtest + "/" + "ham"
    spamvocabulary, spamdic = read(spamtrainpath,stopwordstxt,stopwordsyn)
    hamvocabulary, hamdic = read(hamtrainpath,stopwordstxt,stopwordsyn)
    testspamvocabulary, testspamdic = read(spamtestpath,stopwordstxt,stopwordsyn)
    testhamvocabulary, testhamdic = read(hamtestpath,stopwordstxt,stopwordsyn)
    dictcounteritems = list(set(spamvocabulary)|set(hamvocabulary))
    combinetrain = spamdic.copy()
    combinetrain.update(hamdic)
    combinetest = testspamdic.copy()
    combinetest.update(testhamdic)
    spamlength = len(spamdic)
    hamlength= len(hamdic)
    spamtestlength = len(testspamdic)
    hamtestlength = len(testhamdic)
    values = list()
    for i in range(0,spamlength):
        values.append(0)
    for j in range(0,hamlength):
        values.append(1) 
    listcounteritems = list(dictcounteritems)    
    matrices = traintestwordlist(listcounteritems, combinetrain)
    weight = gradient_ascent(learningrate, numIteration, lambda1, matrices, values)   
    testmatrix = traintestwordlist(listcounteritems, combinetest)
    accuracy = classificationaccuracy(spamtestlength, hamtestlength, weight, testmatrix)
    print ("Accuracy= "+str(accuracy))
