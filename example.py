# Example of using Neural Network to train some MNIST data
# typically takes about 5min to train NN on full dataset

from NeuralNetwork import *
import numpy as np
import time

#-----------------------//----------------------------

#Load in the some training images
fnTrain = 'train.csv'
fnTest = 'test.csv'
# files of form: Col0 = label, Col1-784 are pixels of 28by28 image
dataTrain = np.loadtxt(fnTrain,delimiter=',',skiprows=1) #5000 training examples
dataTest = np.loadtxt(fnTest,delimiter=',',skiprows=1) #1000 test examples

#Training Set
X = dataTrain[:,1:]
y = dataTrain[:,0]
# Change any labels of 0 to another number >0
a = np.where(y==0)
y[a] = 10

# Testing set
XTest = dataTest[:,1:]
yTest = dataTest[:,0]
# Change any labels of 0 to another number >0
a = np.where(yTest==0)
yTest[a] = 10

#-----------------------//----------------------------

#Using NN with 1 hidden layer of 1000 units to classify some MNIST data
#Train Neural Network and see how it performed

#NN structure
s = [np.shape(X)[1],1000,10]

#Randomly initializing thetas
ep_init1 = np.sqrt(6)/np.sqrt(s[0] + s[1])
ep_init2 = np.sqrt(6)/np.sqrt(s[1] + s[2])
Theta1 = randInitialWeights(s[0],s[1],ep_init1)
Theta2 = randInitialWeights(s[1],s[2],ep_init2)
guess = np.append(Theta1.ravel(), Theta2.ravel())

#To record amount of time taken to train NN
getTime = time.time
t = getTime()

# Train neural network (optimizes using conjugate gradient)
lam = 0.1
print 'Optimizing with lambda =' + str(lam) + '...'
NN = NueralNetwork(s,X,y,lam)
opt = NN.optimize(guess)

dt = getTime() - t
dt = dt/60.
print 'Took ' + str('%0.2f' %dt) + ' min to train data'

#-----------------------//----------------------------

#Test samples and print accuracy
nnParams = opt.x
print 'TRAIN:'
predictionsTrain,AccTrain = NN.determineAccuracy(nnParams,X,y)
print 'TEST:'
predictionsTest,AccTest = NN.determineAccuracy(nnParams,XTest,yTest)

