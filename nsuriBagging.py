# loading the data with pandas
import pandas as panda
import hashlib as cryptoService
import re as regex
import nsuriDecisionTree as decitree
import numpy as np
import random
import math
import accuracy as acc
import FileProcessing as filproces

#######################################################################################################################
## checking if im able to access all the functions from the other imported decision tree files
def test():
    treeObjectCreated = decitree.DecisionTreeCreation(my_data, None)
    decitree.DisplayingTheTreeObjectCreated(treeObjectCreated)
    print "The classification of the given sample obserrvation data is in progress"
    x = decitree.ClassifyTheSampleGiven(my_data[1:len(my_data[0])-1], treeObjectCreated)
    print "THe label classified as  : " + str(x)

#######################################################################################################################

## now we have successfully tested the working performance of the algorithm that was written

## crete n number of trees
def CreateAbaggingMethodology(dataPassed, nNumberOfEnsembleTrees,depthOfWeakLearnerTree):
    print " We are creating these many weak classifiers : " + str(nNumberOfEnsembleTrees)
    DictOfModelsTrained = {}
    totalRecords = len(dataPassed)
    #sizeToSample = random.sample(range(1, totalRecords), math.ceil(float(0.8) * (totalRecords)))

    for i in range(0,nNumberOfEnsembleTrees):
        #rows = random.sample(dataPassed.index, sizeToSample)
        #sampledData = dataPassed.ix[rows]
        #my_sampledData = sampledData.values.tolist()
        sampledData  = dataPassed.sample(frac=0.63, replace=True)
        my_sampledData = sampledData.values.tolist()
        print str(len(my_sampledData)) + "Sampled data size"
        DictOfModelsTrained[i] = decitree.DecisionTreeCreation(my_sampledData, depthOfWeakLearnerTree)
    ## All the trees are trained

    return DictOfModelsTrained
#######################################################################################################################
def classifyForEachEnsembler(ensembleOfModelTrained,MarixToStoreClassificaionReuslts,my_Test_data):
    print "Going to check the classification accuracy after ensembling"

    for key in ensembleOfModelTrained.keys():
        modelToClassifyOn = ensembleOfModelTrained[key]
        for i in range(0,len(my_Test_data)):
            observation = my_Test_data[i]
            result = None
            dataToSend = [observation[0:len(observation)-1]]
            #print dataToSend
            #print observation[1:len(observation)-1]
            result = decitree.ClassifyTheSampleGiven(observation[0:len(observation)-1], modelToClassifyOn)
            MarixToStoreClassificaionReuslts[i,key]=result
        print "$$$$$$$$$$$$$$$$$$$$ - classified on Moddel "+ str(key) +"- $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"


    print "Done with classification, now you may calculate the accuracy"

    cols = MarixToStoreClassificaionReuslts.size / len(my_Test_data)
    matrixcolsCount = len(MarixToStoreClassificaionReuslts[0])
    #print cols
    for k in range(0,len(my_Test_data)):
        observation1 = my_Test_data[k]
        realValue = observation1[len(observation1)-1]
        MarixToStoreClassificaionReuslts[k,matrixcolsCount-2] = 5
        MarixToStoreClassificaionReuslts[k, matrixcolsCount - 1] = 5
        resultDict = uniqueCounts(MarixToStoreClassificaionReuslts[k])
        CountOf0 = 0; CountOf1=0
        for keey in resultDict.keys():
            if keey == 0:
                CountOf0 = resultDict[keey]
            elif keey == 1:
                CountOf1 = resultDict[keey]
            else:
                continue
        if CountOf0  >= CountOf1:
            MarixToStoreClassificaionReuslts[k, matrixcolsCount - 1] = 0
        else:
            MarixToStoreClassificaionReuslts[k, matrixcolsCount - 1] = 1
        MarixToStoreClassificaionReuslts[k, matrixcolsCount - 2] = realValue


    #print MarixToStoreClassificaionReuslts
   # print my_Test_data[1]
    #print type(MarixToStoreClassificaionReuslts[k, matrixcolsCount - 2])
    return MarixToStoreClassificaionReuslts

####################################################################################################################
def uniqueCounts(rowRecord):
    frequenciesOfLabelColAsDict4 = {}
    for rowI in range(0,len(rowRecord)):
        if rowRecord[rowI] in frequenciesOfLabelColAsDict4:
            frequenciesOfLabelColAsDict4[rowRecord[rowI]] = frequenciesOfLabelColAsDict4[rowRecord[rowI]]+1
        else:
            frequenciesOfLabelColAsDict4[rowRecord[rowI]] =1
    return frequenciesOfLabelColAsDict4

#######################################################################################################################

def DispalyAllGeneratedtrees(DictOfModelsTrained):
    for model in DictOfModelsTrained.keys():
        print "************************ - " + str(model) + " - ***********************************"
        decitree.DisplayingTheTreeObjectCreated(DictOfModelsTrained[model])

########################################################################################################################


###################################################### Touch only these #################################################################

def BaggingHook(nNumberOfEnsembleTrees,depthToBeTrained,datapath=None):
    #path1 = "C:/Users/Naren Suri/Documents/Python Scripts/DecisionTree/mushrooms/"
    import os
    print os.getcwd()
    print __file__
    cwd = os.getcwd()
    nNumberOfEnsembleTrees = nNumberOfEnsembleTrees
    depthToBeTrained = depthToBeTrained
    path1 = os.getcwd()+"/"
    #path1 = "C:/Users/Naren Suri/Documents/Python Scripts/DecisionTree/mushrooms/"
    try:
        if datapath == None:
            print "if you tried giving some path, which is not recognized or has some problem, so im using the current working directory to run the code"
            path1 = os.getcwd()+"/"
            print path1 + " the path that im considering that the files exist at"
        else:
            print "i got the path you gave, you need not to give the file names. \n file names are iternally taken care \n if you want to give different names please change the code in nsuriBagging.py"
            path1 = datapath
            print datapath
            print path1
    except:
        print "there is some problem with path being given to my code, please look in to the code Adaboosting.py"
        path1 = os.getcwd()+"/"
        pass;  
        
    
    FileName1 = "agaricuslepiotatrain1.csv"
    FileName2 = "agaricuslepiotatest1.csv"
    #########################################################################################################################
    my_data = filproces.FileProcessing(path1,FileName1)
    #print my_data.head()
    my_train_data = my_data # sending as the data frame

    my_Test_data = filproces.FileProcessing(path1,FileName2)
    #print my_Test_data.head()
    my_Test_data_array =  my_Test_data.values.tolist() ## chainging the test data in to arrays, as there is nothing to sample from it

    DictOfModelsTrained =  CreateAbaggingMethodology(my_train_data,nNumberOfEnsembleTrees,depthToBeTrained)
    DispalyAllGeneratedtrees(DictOfModelsTrained)
    # now lets classify based on each tree learned
    # create a data frame that can store the ensembled results
    ensembleValueByEachWeakClassifier = np.zeros((len(my_Test_data_array), nNumberOfEnsembleTrees+2))
    # print type(ensembleValueByEachWeakClassifier)
    resultantMatrix = classifyForEachEnsembler(DictOfModelsTrained,ensembleValueByEachWeakClassifier,my_Test_data_array)
    print "********************* From Bagging ***************************"    
    print "Number of Esemble Trees / Bags larned : " + str(nNumberOfEnsembleTrees)
    print "depth trained in each tree is  : " + str(depthToBeTrained)
    acc.calculateAccuracy(resultantMatrix)

























