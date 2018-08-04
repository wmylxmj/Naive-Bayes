# -*- coding: utf-8 -*-
"""
Created on Sat Aug  4 15:40:38 2018

@author: wmy
"""

import numpy as np

def LoadDataSet():
    WordDataList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    ClassVector = [0, 1, 0, 1, 0, 1]
    return WordDataList, ClassVector

def CreatVocabularyList(dataset):
    #creat a new set
    VocabularySet = set([])
    for document in dataset:
        #with the set
        VocabularySet = VocabularySet | set(document)
    return list(VocabularySet)

def GetWordAppearVector_WordSetModel(vocabularylist, inputset):
    ReturnVector = [0] * len(vocabularylist)
    for word in inputset:
        if word in vocabularylist:
            ReturnVector[vocabularylist.index(word)] = 1
        else:
            print("the world: %s is not in my vocabulary!" % word)
    return ReturnVector

listOPosts, listClasses = LoadDataSet()
myVecabList = CreatVocabularyList(listOPosts)

print(listOPosts)
print(listClasses)
print(myVecabList)
print(len(myVecabList))
print(GetWordAppearVector_WordSetModel(myVecabList, listOPosts[0]))
print(len(GetWordAppearVector_WordSetModel(myVecabList, listOPosts[0])))
print(GetWordAppearVector_WordSetModel(myVecabList, listOPosts[3]))
print(len(GetWordAppearVector_WordSetModel(myVecabList, listOPosts[3])))

def TrainNaiveBayesTwoClass(trainmatrix, trainclasslable):
    #number of rows
    NumberTrainDocuments = len(trainmatrix)
    #the number of words in the line 
    NumberWords = len(trainmatrix[0])
    #the number of abnormal files = sum(trainclasslable)
    '''P(C1)'''
    ProbAbnormal = sum(trainclasslable)/float(NumberTrainDocuments)
    #a vector
    '''
    Prob0Number = np.zeros(NumberWords)
    Prob1Number = np.zeros(NumberWords)
    '''
    Prob0Number = np.ones(NumberWords)
    Prob1Number = np.ones(NumberWords)
    #a number
    '''
    Prob0Denom = 0.0
    Prob1Denom = 0.0
    '''
    Prob0Denom = 2.0
    Prob1Denom = 2.0
    for i in range(NumberTrainDocuments):
        if trainclasslable[i] == 1:
            #if appeared, then number ++
            Prob1Number += trainmatrix[i]
            #the sum of C1
            Prob1Denom += sum(trainmatrix[i])
        else:
            #if appeared, then number ++
            Prob0Number += trainmatrix[i]
            #the sum of C0
            Prob0Denom += sum(trainmatrix[i])
    '''P(Wi|C1)'''
    #Prob1Vector = Prob1Number / Prob1Denom
    Prob1Vector = np.log(Prob1Number/Prob1Denom)
    '''P(Wi|C0)'''
    #Prob0Vector = Prob0Number / Prob0Denom
    Prob0Vector = np.log(Prob0Number/Prob0Denom)
    return Prob0Vector, Prob1Vector, ProbAbnormal

trainMat = []
for postinDoc in listOPosts:
    trainMat.append(GetWordAppearVector_WordSetModel(myVecabList, postinDoc))
p0V, p1V, pAb = TrainNaiveBayesTwoClass(trainMat, listClasses)

print(pAb)
print(p0V)
print(p1V)

def ClassifyNaiveBayesTwoClass(testarrray, prob0vector, prob1vector, probc1):
    Prob1 = sum(testarrray * prob1vector) + np.log(probc1)
    Prob0 = sum(testarrray * prob0vector) + np.log(1.0 - probc1)
    if Prob1 > Prob0:
        return 1
    else:
        return 0
    
def TestNaiveBayesTwoClass_WordSetModel(testwordlist):
    WordDataList, ClassVector = LoadDataSet()
    VocabularyList = CreatVocabularyList(WordDataList)
    TrainMatrix = []
    for indoc in WordDataList:
        TrainMatrix.append(GetWordAppearVector_WordSetModel(VocabularyList, indoc))
    P0V, P1V, PAb = TrainNaiveBayesTwoClass(np.array(TrainMatrix), np.array(ClassVector))
    TestEntry = testwordlist[:]
    TestFile = np.array(GetWordAppearVector_WordSetModel(VocabularyList, TestEntry))
    ClassifyResult = ClassifyNaiveBayesTwoClass(TestFile, P0V, P1V, PAb)
    return ClassifyResult

print(TestNaiveBayesTwoClass_WordSetModel(['love', 'garbage']))
    
def GetWordAppearVector_BagOfWords(vocabularylist, inputset):
    ReturnVector = [0] * len(vocabularylist)
    for word in inputset:
        if word in vocabularylist:
            ReturnVector[vocabularylist.index(word)] += 1
        else:
            print("the world: %s is not in my vocabulary!" % word)
    return ReturnVector

def TestNaiveBayesTwoClass_BagOfWords(testwordlist):
    WordDataList, ClassVector = LoadDataSet()
    VocabularyList = CreatVocabularyList(WordDataList)
    TrainMatrix = []
    for indoc in WordDataList:
        TrainMatrix.append(GetWordAppearVector_BagOfWords(VocabularyList, indoc))
    P0V, P1V, PAb = TrainNaiveBayesTwoClass(np.array(TrainMatrix), np.array(ClassVector))
    TestEntry = testwordlist[:]
    TestFile = np.array(GetWordAppearVector_BagOfWords(VocabularyList, TestEntry))
    ClassifyResult = ClassifyNaiveBayesTwoClass(TestFile, P0V, P1V, PAb)
    return ClassifyResult

print(TestNaiveBayesTwoClass_BagOfWords(['love', 'garbage']))

