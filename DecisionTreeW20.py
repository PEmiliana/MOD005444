
import sys

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pandas
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

df = pandas.read_csv("trainingData100k.csv")

df['Success'] = df['Success']
features = ['InitialSeparation', 'OvertakingSpeed', 'OncomingSpeed']

X = df[features]
y = df['Success']

dtree = DecisionTreeClassifier()

dtree = dtree.fit(X.values, y)
tree.plot_tree(dtree, feature_names=features)

#Two  lines to make our compiler able to draw:
plt.savefig("plot.png")

sys.stdout.flush()


predictionData = [
 [223.6, 19.4, 30.3],
 [249.7, 21.1, 21.4],
 [161.6, 27.8, 23.2],
 [264.2, 28.8, 18.2],
 [256.8, 19.9, 27.6],
 [242.2, 31.8, 21.9],
 [171.1, 32.9, 20.1],
 [161.2, 20.6, 19.6],
 [273.8, 21.7, 22.2],
 [265.1, 32.2, 22.2],
 [189.1, 25.0, 27.2],
 [186.4, 20.6, 31.2],
 [268.6, 21.2, 32.3],
 [209.8, 27.2, 19.9],
 [185.9, 24.8, 25.4],
 [232.4, 21.0, 24.3],
 [251.9, 28.0, 20.4],
 [210.1, 20.4, 30.2],
 [245.6, 32.1, 32.6],
 [222.3, 29.7, 18.7],
 ]

correctResult = [
 1,
 1,
 0,
 1,
 1,
 0,
 0,
 1,
 1,
 1,
 1,
 0,
 1,
 1,
 0,
 1,
 1,
 1,
 0,
 1,
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


