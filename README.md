NeuralNetwork
=============

Neural Network class with backpropagation, arbitrary layers/hidden units, and completely vectorized for efficiency

MISC NOTES: 

The Neural Network class that NeuralNetwork.py creates a Neural Network with the following as inputs:
- LayerLengths (NN structure): LayerLengths is to be given in the format of a list: [s0, s1, ...,sj,...] where sj is the number of units in the j-th layer. So for example creating a NN as NueralNetwork(s,X,y,lam) with s = [784,1000,500,10] would create a NN with 784 units in the first input layer, 1000 units in the first hidden layer, 500 unites in the second hidden layer, and 10 units in the last output layer. 
- X (data): X is the mXn array of input data where the (i,j) element is the j-th feature of the i-th element
- y (labels): y is the mX1 array of input labels
- lam (regularization): lam is the regularization parameter of the cost function. See nnCost for more details in NeuralNetwork.py

---------------------//--------------------------

IMPROVEMENTS:

Please feel free to email me or post here about any suggestions for efficiency gains that would be significant (>=2X) or if you have errors/suggestions/question/comments about the code. I mainly wrote this code because I couldn't find open source and efficient NN that was vectorized and could have arbitrary layers/units, so if you have any further improvements I'd be more than happy to incoorporate them. I also wrote this over the course of a day so forgive any minor bugs or lack of comments. 

Happy hacking,
Alejandro


