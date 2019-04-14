import os
import sys
import numpy
from collections import Counter
from math import *
import re

'''calculate priors for spam and ham'''
def calculatepriors(spamlength,hamlength):
    priorsp = spamlength/(spamlength + hamlength)
    priorh = hamlength/(spamlength + hamlength)
    return priorsp, priorh
'''calculate length of spam and ham files'''
def spamvocabularylength(spamvocabulary):
    spamvoccount = len(spamvocabulary)
    return spamvoccount
def hamvocabularylength(hamvocabulary):
    hamvoccount = len(hamvocabulary)
    return hamvoccount
'''calculate conditional probability of spam and ham files and used LAPLACE SMOOTHING'''
def spamconditionalprobability(dictcounteritems, spamdic, spamvocabulary):
    spamconditionalprob = dict() 
    spamvoccount = spamvocabularylength(spamvocabulary)    
    for word in dictcounteritems:
        wordgivenspam = 0
        if word in spamdic:
            wordgivenspam = spamdic[word]
        spamconditionalprob[word] = (wordgivenspam + 1)/(spamvoccount + len(dictcounteritems)) 
    return spamconditionalprob

def hamconditionalprobability(dictcounteritems, hamdic, hamvocabulary):
    hamconditionalprob = dict()
    hamvoccount = hamvocabularylength(hamvocabulary)
    for word in dictcounteritems:
        wordgivenham = 0
        if word in hamdic:
            wordgivenham = hamdic[word]
        hamconditionalprob[word] = (wordgivenham + 1)/(hamvoccount + len(dictcounteritems))   
    return hamconditionalprob
'''calculate the accuracy of correctly classfied files'''
def classificationaccuracy(dictcounteritems, testspamdic, testhamdic, priorsp, priorh, spamconditionalprob, hamconditionalprob):
    count = 0
    tuple1 =(testspamdic, testhamdic)
    length = len(tuple1)
    for index in range(0,length):
        indexes = tuple1[index]
        for file in indexes:
            '''log of priors of spam and ham'''
            logpriorspam, logpriorham = log(priorsp),log(priorh)
            fileindex=indexes[file]
            for word in fileindex:
                if word in dictcounteritems:
                    '''log of conditional probabilites of spam and ham'''
                    logspamcondition = log(spamconditionalprob[word])
                    loghamcondition = log(hamconditionalprob[word])
                    '''encountering each word and updating the logarithmic conditional probability'''
                    logpriorspam += logspamcondition
                    logpriorham += loghamcondition
                    '''Calculate the accuracy of the complete train and test files using the accuracy formula'''
            if (index == 1 and logpriorspam <= logpriorham) | (index == 0 and logpriorspam >= logpriorham):
                count+=1
    total = len(testspamdic) + len(testhamdic)            
    return count/total*100

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
         words.extend(keyvalue[file])
        else:
         keyvalue[file] = listofwords
         words.extend(keyvalue[file]) 
    return words, keyvalue

if __name__ == "__main__":
    spamtrain = sys.argv[1]
    hamtrain = sys.argv[2]
    spamtest = sys.argv[3]
    hamtest = sys.argv[4]
    stopwordstxt=sys.argv[5]
    stopwordsyn = sys.argv[6]
    spamtrainpath= spamtrain + "/" + "spam"
    hamtrainpath= hamtrain + "/" + "ham"
    spamtestpath= spamtest + "/" + "spam"
    hamtestpath= hamtest + "/" + "ham"
    spamvocabulary, spamdic = read(spamtrainpath,stopwordstxt,stopwordsyn)
    hamvocabulary, hamdic = read(hamtrainpath,stopwordstxt,stopwordsyn)
    testspamvocabulary, testspamdic = read(spamtestpath,stopwordstxt,stopwordsyn)
    testhamvocabulary, testhamdic = read(hamtestpath,stopwordstxt,stopwordsyn)
    spamlength = len(spamdic)
    hamlength = len(hamdic)
    spamdic = Counter(spamvocabulary)
    hamdic = Counter(hamvocabulary)
    dictcounteritems = list(set(spamvocabulary)|set(hamvocabulary))
    priorsp, priorh = calculatepriors(spamlength, hamlength)
    spamconditionalprob= spamconditionalprobability(dictcounteritems, spamdic, spamvocabulary)
    hamconditionalprob= hamconditionalprobability(dictcounteritems, hamdic, hamvocabulary)
    accuracy=classificationaccuracy(dictcounteritems, testspamdic, testhamdic, priorsp, priorh, spamconditionalprob, hamconditionalprob)
    print ("Accuracy= "+str(accuracy))
      

