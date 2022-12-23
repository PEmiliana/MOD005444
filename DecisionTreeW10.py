import sys

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pandas
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

df = pandas.read_csv("trainingData20k.csv")

df['Success'] = df['Success']
features = ['InitialSeparation', 'OvertakingSpeed', 'OncomingSpeed']

X = df[features]
y = df['Success']

dtree = DecisionTreeClassifier()

dtree = dtree.fit(X.values, y)

tree.plot_tree(dtree, feature_names=features)


#Two  lines to make our compiler able to draw:
plt.savefig("plot20kx10.png")

sys.stdout.flush()

predictionData =[
 [209.2 , 18.4, 18.9],
 [208.9 , 25.2, 26.4],
 [184.1 , 28.6, 24.4],
 [255.8 , 23.1, 20.8],
 [232.2 , 27.4, 31.8],
 [275.8 , 28.0, 27.9],
 [186.4 , 29.7, 24.7],
 [268.6 , 30.7, 23.1],
 [278.7 , 18.3, 19.5],
 [204.7 , 29.0, 19.0],
 ]

correctResult = [
 1,
 0,
 0,
 1,
 0,
 1,
 0,
 1,
 1,
 1,
 ]


predictedTrue = 0
for i in range(len(predictionData)):
  prediction = bool(dtree.predict([predictionData[i]]))
  actualPredictionBool = bool(correctResult[i])
  print("Predicted: " + str(prediction) + " | Actual result: " + str(actualPredictionBool)) #
  if  prediction == correctResult[i]:
    predictedTrue += 1

totalNumberOfData = len(predictionData)
calculatePercentage = (predictedTrue*100)/totalNumberOfData
print("Predicted correct = " + str(calculatePercentage) + "% (" + str(predictedTrue) + " out of " + str(totalNumberOfData) + ")") 


