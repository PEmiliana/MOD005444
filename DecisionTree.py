
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas  # import the library which reads the data from our csv file
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier  # import sklearn's decision tree classifier
import numpy
from sklearn import metrics

df = pandas.read_csv("trainingData100k.csv") #read the file with the training data

df['Success'] = df['Success'] #look for the Success row in our data set

features = ['InitialSeparation', 'OvertakingSpeed', 'OncomingSpeed'] # define the headers of our rows

X = df[features] # stores the features rows
y = df['Success'] # stores the Success row

dtree = DecisionTreeClassifier() #DecisionTreeClassifier is a class capable of performing multi-class classification on a dataset. 
#In case that there are multiple classes with the same and highest probability, the classifier will predict the class with the lowest index amongst those classes.

dtree = dtree.fit(X.values, y) #added ".values" to X to fix the warning for "invalid features name"

tree.plot_tree(dtree, feature_names=features) #build the tree



# predictionData =[ #data set to be predicted - this needs to match the size of correctResult to work correct
# [209.2 , 18.4, 18.9],
# [208.9 , 25.2, 26.4],
# [184.1 , 28.6, 24.4],
# [255.8 , 23.1, 20.8],
# [232.2 , 27.4, 31.8],
# [275.8 , 28.0, 27.9],
# [186.4 , 29.7, 24.7],
# [268.6 , 30.7, 23.1],
# [278.7 , 18.3, 19.5],
# [204.7 , 29.0, 19.0],
# ]

# correctResult = [ #the correct result (can or can not overtake) - this needs to match the size of predictionData to work correct
# 1,
# 0,
# 0,
# 1,
# 0,
# 1,
# 0,
# 1,
# 1,
# 1,
# ]

#20 sets of data to be predicted
# predictionData = [
# [223.6, 19.4, 30.3],
# [249.7, 21.1, 21.4],
# [161.6, 27.8, 23.2],
# [264.2, 28.8, 18.2],
# [256.8, 19.9, 27.6],
# [242.2, 31.8, 21.9],
# [171.1, 32.9, 20.1],
# [161.2, 20.6, 19.6],
# [273.8, 21.7, 22.2],
# [265.1, 32.2, 22.2],
# [189.1, 25.0, 27.2],
# [186.4, 20.6, 31.2],
# [268.6, 21.2, 32.3],
# [209.8, 27.2, 19.9],
# [185.9, 24.8, 25.4],
# [232.4, 21.0, 24.3],
# [251.9, 28.0, 20.4],
# [210.1, 20.4, 30.2],
# [245.6, 32.1, 32.6],
# [222.3, 29.7, 18.7],
# ]

#20 sets of data to be predicted
# correctResult = [
# 1,
# 1,
# 0,
# 1,
# 1,
# 0,
# 0,
# 1,
# 1,
# 1,
# 1,
# 0,
# 1,
# 1,
# 0,
# 1,
# 1,
# 1,
# 0,
# 1,
# ]

#50 sets of data to be predicted
predictionData = [
[235.6, 21.8, 31.2],
[215.7, 30.9, 28.8],
[261.3, 21.8, 23.5],
[274.2, 26.1, 26.6],
[270.6, 23.8, 28.0],
[182.2, 29.9, 31.9],
[169.2, 30.0, 20.3],
[260.8, 27.3, 30.0],
[253.6, 29.4, 29.4],
[151.8, 19.2, 27.2],
[235.7, 24.4, 31.8],
[215.5, 29.0, 19.4],
[158.1, 31.9, 28.2],
[252.8, 22.5, 25.3],
[208.2, 31.3, 22.8],
[167.5, 25.4, 21.3],
[256.2, 26.0, 19.3],
[173.3, 24.0, 32.8],
[202.8, 26.3, 27.7],
[235.1, 28.7, 30.9],
[179.8, 30.4, 22.5],
[203.1, 23.2, 29.8],
[211.7, 31.7, 28.0],
[235.3, 18.1, 31.0],
[183.6, 21.5, 23.4],
[265.8, 24.9, 28.9],
[239.7, 20.5, 19.7],
[218.0, 25.9, 21.5],
[201.2, 18.2, 30.6],
[235.1, 18.9, 21.1],
[183.2, 31.6, 23.3],
[193.3, 22.5, 21.5],
[276.9, 20.6, 24.8],
[181.0, 24.9, 31.9],
[177.2, 22.2, 20.4],
[241.9, 18.6, 23.1],
[181.1, 22.0, 19.1],
[152.2, 23.4, 20.7],
[152.3, 25.7, 30.2],
[160.6, 20.8, 27.1],
[253.2, 19.7, 32.3],
[164.1, 29.0, 24.9],
[170.0, 18.6, 19.9],
[199.9, 20.9, 22.6],
[266.2, 25.9, 20.5],
[248.5, 28.1, 24.8],
[229.8, 31.8, 21.8],
[252.6, 18.9, 32.6],
[256.5, 26.8, 24.3],
[150.1, 21.1, 32.4],
]
#50 sets of data to be predicted
correctResult = [ 
1,
0,
1,
1,
1,
0,
0,
1,
1,
1,
1,
1,
0,
1,
0,
0,
1,
0,
0,
0,
0,
1,
0,
1,
1,
1,
1,
1,
1,
1,
0,
1,
1,
0,
0,
1,
1,
0,
0,
0,
1,
0,
1,
1,
1,
1,
1,
1,
1,
0,
 ]

predictedTrue = 0 # variable to store the number of correct predictions which is used to calculate the % of correct guesses at the end
for i in range(len(predictionData)): #loops through the entire data to be predicted
  prediction = bool(dtree.predict([predictionData[i]])) #stores the prediction made by the decision tree in a BOOLEAN format so it can be compared to the results later on
  actualPredictionBool = bool(correctResult[i]) #store the actual correct result (can or can not overtake)
  print("Predicted: " + str(prediction) + " | Actual result: " + str(actualPredictionBool)) # display the correct result against the predicted result
  if  prediction == correctResult[i]: #checks if the prediction is the same as the correct answer from our test dataset
    predictedTrue += 1 #if the prediction was the same as the real answer add one to the predictedTrue variable so we know how much has been predicted right

totalNumberOfData = len(predictionData) #store the length of our list of data
calculatePercentage = (predictedTrue*100)/totalNumberOfData #calculate the % of correct predictions by dividing the predictions * 100 by the total amount of data to be predicted
print("Predicted correct = " + str(calculatePercentage) + "% (" + str(predictedTrue) + " out of " + str(totalNumberOfData) + ")") #print the % of correct predictions (the calculatePercentage variable is converted to string so it can be displayed in the print statement)

actual = numpy.random.binomial(1,.9,size = 1000)
predicted = numpy.random.binomial(1,.9,size = 1000)

confusion_matrix = metrics.confusion_matrix(actual, predicted)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])

cm_display.plot()
plt.show()
#Two  lines to make our compiler able to draw:
plt.savefig(sys.stdout.buffer)
sys.stdout.flush()
