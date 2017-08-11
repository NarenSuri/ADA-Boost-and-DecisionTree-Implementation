
def calculateAccuracy(resultantMatrix):
    realValueColumn= len(resultantMatrix[0])-2
    PredictValueColumn = len(resultantMatrix[0]) - 1
    TP=0; FP=0;TN=0;FN=0
    for k in range(0, len(resultantMatrix)-1):
       if resultantMatrix[k][realValueColumn] == resultantMatrix[k][PredictValueColumn]:
           TP = TP+1
       elif resultantMatrix[k][realValueColumn] > resultantMatrix[k][PredictValueColumn]:
           FN= FN+1
       elif resultantMatrix[k][realValueColumn] < resultantMatrix[k][PredictValueColumn]:
            FP = FP+1
       else:
           TN = TN+1
    print "Tota tested records are : " + str(len(resultantMatrix))
    print str(TP) + "  True Positive"
    print str(TN) + "  True Neg"
    print str(FP) + "  False Positive"
    print str(FN) + "  Fal  Neg"
    x = float(TP+TN) / (TP+TN+FP+FN)
    print "Accuracy for the Bagging Model is :" + str(float(TP+TN) / (TP+TN+FP+FN) )
    print "So the accuracy is : " + str(x * 100)+ "%"


def AdacalculateAccuracy(resultantMatrix):
    realValueColumn= 0
    PredictValueColumn = 1
    TP=0; FP=0;TN=0;FN=0
    for k in range(0, len(resultantMatrix)):
       if resultantMatrix[k][realValueColumn] == resultantMatrix[k][PredictValueColumn]:
           TP = TP+1
       elif resultantMatrix[k][realValueColumn] > resultantMatrix[k][PredictValueColumn]:
           FN= FN+1
       elif resultantMatrix[k][realValueColumn] < resultantMatrix[k][PredictValueColumn]:
            FP = FP+1
       else:
           TN = TN+1
    print "Tota tested records are : " + str(len(resultantMatrix))
    print str(TP) + "  True Positive"
    print str(TN) + "  True Neg"
    print str(FP) + "  False Positive"
    print str(FN) + "  Fal  Neg"
    x = float(TP+TN) / (TP+TN+FP+FN)
    print "Accuracy for the Bagging Model is :" + str(float(TP+TN) / (TP+TN+FP+FN) )
    print "So the accuracy is : " + str(x * 100)+ "%"