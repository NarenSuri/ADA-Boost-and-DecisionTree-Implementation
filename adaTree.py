# lets create an object that helps to store the nodes of each decision tree
# Each node of the tree should store dataResultsAtThatNode,OnTrueForwardNode,OnFalseForwardNode
# BestEntropyForSplitAtThatNode, OnwhichColumnSplit, ValueAtSplit
global depth
depth=0
CurrentDepth = 0
depthCriteria=None
###############################################################################
class Node:
    def __init__(self, dataResultsAtThatNode=None, OnTrueNextNode=None, OnFalseNextNode=None,
                 BestEntropyForSplitAtThatNode=None, OnwhichColumnSplit= -1, ValueAtSplit=None,depth=None,AnyExtraInfo=None,LeafNodeResultsFrequencyProfile=None,computeClassDominanceStats=None,*args):
        ## i want to store all the below information in the every node of the tree
        self.dataResultsAtThatNode = dataResultsAtThatNode
        self.OnTrueNextNode = OnTrueNextNode
        self.OnFalseNextNode = OnFalseNextNode
        self.BestEntropyForSplitAtThatNode = BestEntropyForSplitAtThatNode
        self.OnwhichColumnSplit = OnwhichColumnSplit
        self.ValueAtSplit = ValueAtSplit
        self.AnyExtraInfo=AnyExtraInfo
        self.depth = depth
        self.LeafNodeResultsFrequencyProfile = LeafNodeResultsFrequencyProfile
        self.computeClassDominanceStats = computeClassDominanceStats
        for count, thing in enumerate(args):
            self.AnyExtraInfo= self.AnyExtraInfo+str(thing)
###############################################################################

def calculateLogOf2(val):
    from math import log
    return log(val)/log(2)

###############################################################################
def CalculateEntropy(dataPassed,Dvector): 
    frequenciesOfLabelColAsDict={}
    totalsum=0
    for eachrow in dataPassed:
        NoOfCols = len(eachrow) ;  Label = eachrow[NoOfCols-2] # skipping the last Dt-index column
        DIndex = eachrow[NoOfCols-1]
        if Label in frequenciesOfLabelColAsDict:
            totalsum = Dvector[DIndex][1]+totalsum
            frequenciesOfLabelColAsDict[Label] = frequenciesOfLabelColAsDict[Label]+Dvector[DIndex][1]
        else:
            frequenciesOfLabelColAsDict[Label]=Dvector[DIndex][1]
            totalsum = Dvector[DIndex][1]+totalsum
    
    from math import log
    ## loop through each key in the dictionary and calculate its entropy value
    ## sigma all the keys of the last column
    entropy =0.0
    for eachkey in frequenciesOfLabelColAsDict.keys():
        entropy = entropy - ((float(frequenciesOfLabelColAsDict[eachkey]) / totalsum ) * calculateLogOf2(float(frequenciesOfLabelColAsDict[eachkey]) / totalsum ))
    return entropy

###############################################################################
def CalculateEntropy0(dataPassed):
    #print "Caluclating the Entropy"
    # first lets count all the records in the dataset passed and also store frequencies of last label column
    frequenciesOfLabelColAsDict={}
    for eachrow in dataPassed:
        NoOfCols = len(eachrow) 
        Label = eachrow[NoOfCols-2] # skipping the last Dt-index column
        if Label in frequenciesOfLabelColAsDict:
            frequenciesOfLabelColAsDict[Label] = frequenciesOfLabelColAsDict[Label]+1
        else:
            frequenciesOfLabelColAsDict[Label]=1
    from math import log
    ## loop through each key in the dictionary and calculate its entropy value
    ## sigma all the keys of the last column
    entropy =0.0
    for eachkey in frequenciesOfLabelColAsDict.keys():
        entropy = entropy - ((float(frequenciesOfLabelColAsDict[eachkey]) / len(dataPassed) ) * calculateLogOf2(float(frequenciesOfLabelColAsDict[eachkey]) / len(dataPassed) ))
    return entropy

###############################################################################

def DivideDataInToTwoSplitsOnColumnAndValue(data,columnToSplitOn,ValueToSplitOn):
    ## Split by the value asked. If the value is less or geater should be placed in the oter bag
    # so we generate two bags, 1. with the value we need 2. the rest
    ResultantSplitData1 =[]
    ResultantSplitData2 = []
    for eachRow in data:
        if eachRow[columnToSplitOn] == ValueToSplitOn:
            ResultantSplitData1.append(eachRow)
        else:
            ResultantSplitData2.append(eachRow)
    return (ResultantSplitData1,ResultantSplitData2)

###############################################################################

def computeClassDominanceStats(dataToProcess4):
    frequenciesOfLabelColAsDict4 = {}
    for eachrow4 in dataToProcess4:
        NoOfCols4 = len(eachrow4)
        Label4 = eachrow4[NoOfCols4 - 1]
        if Label4 in frequenciesOfLabelColAsDict4:
            frequenciesOfLabelColAsDict4[Label4] = frequenciesOfLabelColAsDict4[Label4] + 1
        else:
            frequenciesOfLabelColAsDict4[Label4] = 1
    return frequenciesOfLabelColAsDict4

###############################################################################


def DecisionTreeCreation(dataToProcess,depthCriteria,Dvector):
    # even before proceeding further we should check the stopping condition of the decsision tree progression. If the data gave is null, then tree shoud be stopped from getting in to infinite loop
    global CurrentDepth
    global depth
    if len(dataToProcess) == 0:
        #print "Data is finished and reached the leaf node"
        return Node # return what ever we have to the recursion fucnion which called this iteration.
        #  return the node and trace back to the previous call of the recursion stack, which is basically stp this loop
         #  if the data is finished the leaf node is to be returned which happens here

    else:
        #  lets set all the tracking variables
        TotalColsInDataToProcess = len(dataToProcess[0])  
        # now for each column try to cacluate the entropy to decide which is the best column to choose
        # lets start the calculation of the entropy of the last column first
        contemporaryEntropyValue = CalculateEntropy(dataToProcess,Dvector)  # we have the entropy of the final set
        #  but we need the other things for Information Gain Calculation
        ColsCountList = range(0,TotalColsInDataToProcess-2) # -2 becaue we added the extra indexcolumn to the end of the data frame
        #print ColsCountList
        CurrentBestGain=0.0
        CurrentBestColumn=None
        CurrentBestSplitValue=None
        CurrentBestSplitSets = None

        for colInProcessing in ColsCountList:
            # now after we take each column, we should understand how many unique values are there in that column.
            #since we are doing a binary style of 1 vs all, we should choose one type and rest all to other bag
            # repeat this process for various types of labels in that particular column
            # To get the list of all the unique keys in that particular column lets create a dict
            # first lets count all the records in the dataset passed and also store frequencies of last label column
            frequenciesOfLabelColAsDict1 = {}
            for eachrow1 in dataToProcess:
                #print "The column in processing is :  "+ str(colInProcessing)
                Label1 = eachrow1[colInProcessing]
                if Label1 in frequenciesOfLabelColAsDict1:
                    frequenciesOfLabelColAsDict1[Label1] = frequenciesOfLabelColAsDict1[Label1] + 1
                else:
                    frequenciesOfLabelColAsDict1[Label1] = 1
            # So now we have the dictionary of all the keys or labels in that particular column
            # so now lets calculate the Informatioon gain using entropy of  the each sub type keys
            for keyInColumnUnderProcessing  in frequenciesOfLabelColAsDict1.keys():
                # now lets get the split of the data in to the pieces. Like as discused above, break the data in to the binary style of need value group vs all other values in to other goup
                (ResultData1, ResultData2) = DivideDataInToTwoSplitsOnColumnAndValue(dataToProcess, colInProcessing, keyInColumnUnderProcessing)
                # Now we have the splits of the data that we need. #so lets calculate the Information Gain,
                # First lets get the entropy of the each sets we got now
                
                ########################***********************########################
                # This is the important piece of the AdaBoost.
                # instead of just the weighted avaerage among the spits, we use the Di value
                # So first for each Result splits we have now, lets calculate the Sigma Xi / sigma Xi + Sigma Yi
                # [[],[],[]] List of lists is the shape of the ResultData
                sigmaXi=0;sigmaYi=0                
                for r in range(0,len(ResultData1)):
                    DiIndex = ResultData1[r][len(ResultData1[r])-1]
                    sigmaXi = sigmaXi+Dvector[DiIndex][1]
                    
                for l in range(0,len(ResultData2)):
                    DiIndex = ResultData2[l][len(ResultData2[l])-1]
                    sigmaYi = sigmaYi+Dvector[DiIndex][1]
                
                currentColCombiEntropy = (float(sigmaXi)/ (sigmaXi+sigmaYi) * CalculateEntropy(ResultData1,Dvector))+((float(sigmaYi)/ (sigmaXi+sigmaYi)) * CalculateEntropy(ResultData2,Dvector))
                #print " IG is : " + str(currentColCombiEntropy)                
                #######################**************************#######################
                                
                InfromationGain = contemporaryEntropyValue-currentColCombiEntropy

                if len(ResultData1)>0 and InfromationGain > CurrentBestGain and len(ResultData2)>0:
                    # Update all the values we are storing to these values, so that we can use them in processing the further results
                    CurrentBestGain = InfromationGain
                    CurrentBestColumn = colInProcessing
                    CurrentBestSplitValue = keyInColumnUnderProcessing
                    CurrentBestSplitSets = (ResultData1, ResultData2)
                # updated with all the new values and stored the resultant sets also at the each node level
                # this loop will end after each column is tested with their coresponding splits and the best gain, best col, best value and corresponding result sets are found
                else:
                    # print "Couldnt update the resultant new calcualted values as previous are much better in quality"
                    continue
        # once all this is done we take the best one and we go further by fixing this column and the value, to recursively crete the next tee nodes.
        # Now lets connect to the next tree generation process

        if CurrentBestGain>0 and (depthCriteria>CurrentDepth or depthCriteria==None):
                if CurrentDepth != None:
                    CurrentDepth = CurrentDepth+1
                depth = depth +1
                #print "TrueSideDepth - Increment  :  " + str(CurrentDepth)
                OnTrueRightSideBranch =  DecisionTreeCreation(CurrentBestSplitSets[0],depthCriteria,Dvector)
                CombinedresultSets = CurrentBestSplitSets[0] + CurrentBestSplitSets[1]
                depth = depth + 1
                #print "FalseSideDepth - Decrement and then Increment  :  " + str(CurrentDepth)
                OnFalseLeftSideBranch = DecisionTreeCreation(CurrentBestSplitSets[1],depthCriteria,Dvector)
                CurrentDepth = CurrentDepth - 1
                return Node(dataResultsAtThatNode=CurrentBestSplitSets, OnTrueNextNode=OnTrueRightSideBranch, OnFalseNextNode=OnFalseLeftSideBranch,BestEntropyForSplitAtThatNode=CurrentBestGain, OnwhichColumnSplit= CurrentBestColumn, ValueAtSplit=CurrentBestSplitValue,depth=None,AnyExtraInfo="nsuri",computeClassDominanceStats=computeClassDominanceStats(CombinedresultSets))

        else:
            frequenciesOfLabelColAsDict2 = {}
            for eachrow2 in dataToProcess:
                NoOfCols2 = len(eachrow2)
                Label2 = eachrow2[NoOfCols2 - 2] # becaue of the index colum added
                if Label2 in frequenciesOfLabelColAsDict2:
                    frequenciesOfLabelColAsDict2[Label2] = frequenciesOfLabelColAsDict2[Label2] + 1
                else:
                    frequenciesOfLabelColAsDict2[Label2] = 1
            return Node(dataResultsAtThatNode=CurrentBestSplitSets,LeafNodeResultsFrequencyProfile=frequenciesOfLabelColAsDict2)

###############################################################################

def DisplayingTheTreeObjectCreated(treeObjectCreated,indentation='**'):
   if treeObjectCreated.LeafNodeResultsFrequencyProfile==None:
      ## priting the respecive subtree root - node
      columnToCheckOn =  'OnColumn - ['+ str(treeObjectCreated.OnwhichColumnSplit)+"]"
      ValueToCheckFor = ': Check For - ['+ str(treeObjectCreated.ValueAtSplit)+"]"
      DoesItMeetConditons = ' - Meets condtion ?  '
      print columnToCheckOn + ValueToCheckFor + DoesItMeetConditons
      #### priting the child nodes of the above root node
      print '\n '+indentation+'If True --->',
      DisplayingTheTreeObjectCreated(treeObjectCreated.OnTrueNextNode,indentation+'****')
      #### priting the child nodes of the above root node
      print '\n '+indentation+'If False --->',
      DisplayingTheTreeObjectCreated(treeObjectCreated.OnFalseNextNode,indentation+'****')
   else:
       for key in treeObjectCreated.LeafNodeResultsFrequencyProfile.keys():
           print "Class Type : [" + str(key) + "] is : ["+str(treeObjectCreated.LeafNodeResultsFrequencyProfile[key])+ "] In number!! "

###############################################################################

def ClassifyTheSampleGiven(ObsevationTestSample, TreeToTestOn):

    if TreeToTestOn.LeafNodeResultsFrequencyProfile == None:
        #print "Going Forward in the tree traversal"
        # lets check if the column at which the current node to be checked matches the ccondition we want
        valueAtColumnToTestOn = ObsevationTestSample[TreeToTestOn.OnwhichColumnSplit]
        if(valueAtColumnToTestOn==TreeToTestOn.ValueAtSplit):
            nextToProcessNode = TreeToTestOn.OnTrueNextNode
            return ClassifyTheSampleGiven(ObsevationTestSample, nextToProcessNode)
        else:
            nextToProcessNode = TreeToTestOn.OnFalseNextNode
            return ClassifyTheSampleGiven(ObsevationTestSample, nextToProcessNode)
    else:
        classificationDoneIs=None
        BestMaxVal = 0
        #print  "computeClassDominanceStats" + str(TreeToTestOn.computeClassDominanceStats)
        #print TreeToTestOn.LeafNodeResultsFrequencyProfile
        #print type(TreeToTestOn.LeafNodeResultsFrequencyProfile)
        feqDict = TreeToTestOn.LeafNodeResultsFrequencyProfile
        for key in feqDict.keys():
            x = feqDict[key]
            if x > BestMaxVal:
                BestMaxVal= x
                classificationDoneIs = key
            #print classificationDoneIs

        return classificationDoneIs

###############################################################################
def setGlobalVars():
    global CurrentDepth
    CurrentDepth=0
    global depthCriteria
    depthCriteria=None
setGlobalVars()

################################################################################
my_data = [['a','b','b']]
#if __name__ == '__main__':
Dvector = None

def testingFuncs(my_train_data,depthToBeTrained,DtAndDtPlus):
    print "testing each function pieces with neccessary ada boost changes!!"
    print CalculateEntropy(my_train_data)
    TreeToTestOn =  DecisionTreeCreation(my_train_data,depthToBeTrained,DtAndDtPlus)
    ObsevationTestSample= my_train_data[0]
    print ClassifyTheSampleGiven(ObsevationTestSample, TreeToTestOn)
    







