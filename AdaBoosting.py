# loading the data with pandas
import pandas as panda
import numpy as np
import random
import math
import accuracy as acc
import FileProcessing as filproces
import adaTree as adtree


def AdaBoostprocess(nNumberOfEnsembleTrees,depthToBeTrained,data,DtAndDtPlus):
    # createa a alpha dictionry
    AlpahEachTreeWeight = {}
    TrainedModelTrees = {}
    # create a matrix of lists to store the results
    ResultsClassification = [[-1 for x in range(3)] for y in range(len(data))]    
    for k in range(0,nNumberOfEnsembleTrees):
        # call the first taining on the data
          TreeTrainedModel = adtree.DecisionTreeCreation(my_train_data,depthToBeTrained,DtAndDtPlus)       
          TrainedModelTrees[k] = TreeTrainedModel
          adtree.DisplayingTheTreeObjectCreated(TreeTrainedModel)
          print "\n ##################################################################### \n"

          # now call the classification on the each observation and store that for the accuracy calculation
          for rowI in range(0,len(data)):
                
            observation = data[rowI]
            ResultsClassification[rowI][0] = observation[len(observation)-2] # real value
                
                
            if observation[len(observation)-2] == 1:## modified -1 or +1 value
                     ResultsClassification[rowI][1] = 1
            else:
                ResultsClassification[rowI][1] = -1
                    
            # now lets alcualte the classification value for each row
            classificiationResult = adtree.ClassifyTheSampleGiven(observation, TreeTrainedModel)
                
            if classificiationResult == 1:## modified -1 or +1 value
                     ResultsClassification[rowI][2] = 1
            else:
                ResultsClassification[rowI][2] = -1
                    
          # so all the training examles are tested now
          # lets calculate the alpha and store it in the dictionary
          # for alpha calculation we need the accuracy on the obained results
          errorE = errorrate(ResultsClassification)
          if errorE ==0:
              errorE = 0.000001
          AlphaForTHisTree = 0.5 * np.log(float(1-float(errorE)) / errorE)
          AlpahEachTreeWeight[k] =  AlphaForTHisTree
          #################################################################
          # we have alpha for the trained tree
          # now lets update the Dt and Dt+1 values which are 0 and 1 indicices
          # first lets update the Dt Vector values
          DtAndDtPlus = UpdateDTvector(ResultsClassification,AlphaForTHisTree,DtAndDtPlus)
          ################################################################
           
    return (AlpahEachTreeWeight,TrainedModelTrees)

###############################################################################

def UpdateDTvector(ResultsClassification,AlphaForTHisTree,DtAndDtPlus):
    # DtAndDtPlus[0] - DtAndDtPlus is Dt+1
    # DtAndDtPlus[1] - DtAndDtPlus is Dt
    DtSum = 0
    for k in range(0, len(ResultsClassification)):
        DtAndDtPlus[k][0] = DtAndDtPlus[k][1] * math.exp(-1 *(AlphaForTHisTree * ResultsClassification[k][1] * ResultsClassification[k][2]))
        DtSum = DtSum + DtAndDtPlus[k][0]
    
    # Now lets normalize and update the all Dt with Dt+1 values
    # it seems i dont need 2col size dt's, but for now lets do it
    
    for d in range(0,len(DtAndDtPlus)):
        DtAndDtPlus[d][0] = float(DtAndDtPlus[d][0])/DtSum 
        DtAndDtPlus[d][1] = float(DtAndDtPlus[d][0])
    # after updating all the Dt Vector return it now
        
    return DtAndDtPlus
        
##############################################################################    
    
    
    

def errorrate(ResultsClassification):
    # calulcate the probablity value of the badly classified
    correctlyClassified =0
    wronglyClassified = 0
    errorRate=0
    
    for i in range(0,len(ResultsClassification)):
        if ResultsClassification[i][1] == ResultsClassification[i][2]:
            # correctly classified
            correctlyClassified = correctlyClassified +1
        else:
            wronglyClassified = wronglyClassified+1
    
    errorRate = float(wronglyClassified)/(correctlyClassified+wronglyClassified)

    return errorRate

#################################################################################

def processColsForAdaDVec(my_data, my_Test_data):
    my_data = my_data
    my_Test_data = my_Test_data

    my_Test_data_array = my_Test_data.values.tolist()  ## chainging the test data in to arrays, as there is nothing to sample from it

    # we have train and test data now.
    # lets try to add the new column called index to the dataFrame Train and send it to the adaTree Training - Which is a bit different from the general decision tree

    my_data['DtIndex'] = range(0, len(my_data))  # sending as the data frame

    ## create the D matrix to store the weights
    cols = 2
    rows = len(my_data)
    defaultValueOfD = float(1) / len(my_data)
    DtAndDtPlus = [[defaultValueOfD for x in range(cols)] for y in range(rows)]
    my_train_data = my_data.values.tolist()
    return (my_train_data, my_Test_data_array, DtAndDtPlus)


###############################################################################
def TestDataAdaBoost(AlpahEachTreeWeight, TrainedModelTrees, data, nNumberOfEnsembleTrees):
    ResultsClassificationTest = [[-1 for x in range(nNumberOfEnsembleTrees + 2)] for y in range(len(data))]
    # real value , predicted value in the ResultsClassificationTest

    # classifying each sample with each tree , and resutls stored
    for rowI in range(0, len(data)):
        observation = data[rowI]
        if observation[len(observation) - 1] == 1:  ## modified -1 or +1 value
            ResultsClassificationTest[rowI][len(ResultsClassificationTest[rowI]) - 2] = 1
        else:
            ResultsClassificationTest[rowI][len(ResultsClassificationTest[rowI]) - 2] = -1
        for CurrentTreeModelForTest in TrainedModelTrees.keys():
            # call the first taining on the data
            TreeTrainedModel = TrainedModelTrees[CurrentTreeModelForTest]
            # now call the classification on the each observation and store that for the accuracy calculation
            # now lets alcualte the classification value for each row
            classificiationResult = adtree.ClassifyTheSampleGiven(observation, TreeTrainedModel)
            if classificiationResult == 1:  ## modified -1 or +1 value
                ResultsClassificationTest[rowI][CurrentTreeModelForTest] = 1
            else:
                ResultsClassificationTest[rowI][CurrentTreeModelForTest] = -1

                ## Finally we deciding the class it belongs to based on the Sigma of tree weight * predicted value
        for row in range(0, len(ResultsClassificationTest)):
            resultvalue = 0
            # predicted valuue to be kept in the last column
            for eachCol in range(0, len(ResultsClassificationTest[row]) - 2):
                resultvalue = resultvalue + AlpahEachTreeWeight[eachCol] * ResultsClassificationTest[row][eachCol]

            ResultsClassificationTest[row][len(ResultsClassificationTest[row]) - 1] = resultvalue
            ## Finished the final prediction and stored in the results
        return ResultsClassificationTest
#################################################################################
def startTheAdabootHook(nNumberOfEnsembleTrees,depthToBeTrained,datapath=None):
    # path1 = "C:/Users/Naren Suri/Documents/Python Scripts/DecisionTree/mushrooms/"
    import os
    print os.getcwd()
    print __file__
    cwd = os.getcwd()

    ###############################################################################
   
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
            print "i got the path you gave, you need not to give the file names. \n file names are iternally taken care \n if you want to give different names please change the code in AdaBoosting and nsuriBagging.py"
            path1 = datapath
            print path1
    except:
        print "there is some problem with path being given to my code, please look in to the code Adaboosting.py"
        path1 = os.getcwd()+"/"
        pass;       
    
    global FileName1
    global FileName2
    global my_data
    global my_Test_data
    global my_train_data
    global my_Test_data_array
    
    # path1 = "C:/Users/Naren Suri/Documents/Python Scripts/DecisionTree/mushrooms/"
    #path1 = os.getcwd() + "/"
    FileName1 = "agaricuslepiotatrain1.csv"
    #FileName1 = "insurance.csv"
    FileName2 = "agaricuslepiotatest1.csv"
    ###############################################################################

    ###############################################################################
    my_data = filproces.FileProcessing(path1, FileName1)
    #print my_data.head()
    my_Test_data = filproces.FileProcessing(path1, FileName2)
    #print my_Test_data.head()
    ##################################################################################


    (my_train_data,my_Test_data_array,DtAndDtPlus) = processColsForAdaDVec(my_data,my_Test_data) #(my_data,my_Test_data)
    (AlpahEachTreeWeight,TrainedModelTrees) = AdaBoostprocess(nNumberOfEnsembleTrees,depthToBeTrained,my_train_data,DtAndDtPlus)
    ResultsClassificationTest = TestDataAdaBoost(AlpahEachTreeWeight,TrainedModelTrees,my_Test_data_array,nNumberOfEnsembleTrees)
    # Now lets calculate the accuracy
    print "********************** FROM ADABoost ********************************"    
    print "Toatla Ensembles Of trees : " + str(nNumberOfEnsembleTrees)
    print "Depth of each tree of Each Ensemble : " + str(depthToBeTrained)
    acc.AdacalculateAccuracy(ResultsClassificationTest)


#adtree.DecisionTreeCreation(my_train_data,depthToBeTrained,DtAndDtPlus)
#adtree.testingFuncs(my_train_data,depthToBeTrained,DtAndDtPlus)