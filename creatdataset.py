# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 15:02:56 2018

@author: wmy
"""

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

def GetWordAppearVector(vocabularylist, inputset):
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
print(GetWordAppearVector(myVecabList, listOPosts[0]))
print(GetWordAppearVector(myVecabList, listOPosts[3]))
