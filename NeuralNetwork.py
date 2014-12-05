# NOTES:
#     - below we have code that creates a Neural Network class 
#     - cost function is a sigmoid-log cost function for Neural Networks
#     - utilizing backpropagation
#     - CG optimization is used through optimize unless specified
#     - completely vectorized implementation
#
#     by Alejandro Cortese
#     email ajc383@cornell.edu

import numpy as np
import scipy as sp
import scipy.io
from scipy import optimize
from scipy.optimize import minimize

def sigmoid(z):
    np.seterr(over='ignore')
    return 1./(1 + np.exp(-z))

def sigmoidGrad(z):
    return sigmoid(z)*(1 - sigmoid(z))

#Randomly initializes weights for theta matrices
def randInitialWeights(Lin, Lout, epsilon_init):
    return np.random.random((Lout,Lin+1))*2*epsilon_init - epsilon_init

#Rolls back up thetas
def reshapeThetas(nnParams,LayerLengths):
    s = LayerLengths
    numLayers = len(s)
    totalParamsBefore = 0
    Thetas = []
    for j in range(numLayers - 1):
        numParams = s[j+1]*(s[j]+1)
        Thetas.append(np.reshape(nnParams[totalParamsBefore:totalParamsBefore + numParams],(s[j+1],s[j]+1)))
        totalParamsBefore += numParams
    return Thetas

#Creates a nueral network class of arbitraty layers and units according to LayerLengths
class NueralNetwork:
    def __init__(self,LayerLengths,X,y,lam):
        #[s0, s1, ...,sj,...] where sj is the number of units in the j-th layer
        self.LayerLengths = LayerLengths  
        
        self.X = np.append(np.ones([np.shape(X)[0],1]),X,axis=1) #adds bias components
        self.y = y #labels
        self.lam = lam #regularization lambda
        
        self.numLabels = LayerLengths[-1] #number of Labels
        self.numLayers = len(LayerLengths) #number of layers
        self.m = np.shape(X)[0] #number of samples
        
        
        #Make labels into unit vectors
        Y = np.zeros([LayerLengths[-1],np.shape(y)[0]])
        for i in range(len(y)):
            Y[y[i] - 1,i] = 1
        self.Y = Y
        
    # Compute cost for NN
    def nnCost(self,nnParams):
        
        #Roll back up Thetas
        Thetas = reshapeThetas(nnParams,self.LayerLengths)
        
        #Forward Propagation
        zLayers, aLayers = self.forwardPropagation(self.X,Thetas)
        h = aLayers[-1]

        #Compute cost without regularization
        J = (1./self.m)*(np.einsum('ij,ji',-self.Y,np.log(h)) - np.einsum('ij,ji',1 - self.Y,np.log(1 - h)))

        #Theta1 and Theta2 summations for regularization
        for i in range(len(Thetas)):
            Theta = Thetas[i]
            Theta[0,:] = 0
            TSum = np.sum(Theta**2)
            J += (self.lam/(2.*self.m))*(TSum)

        #return the cost
        return J

    #Compute derivative of NN using back propagation
    def nnCostDer(self,nnParams):
        
        #Roll back up thetas
        Thetas = reshapeThetas(nnParams,self.LayerLengths)
        Thetas_grad = []
        for i in range(len(Thetas)):
            Thetas_grad.append(np.zeros(np.shape(Thetas[i])))

        #Forward propagation
        zLayers,aLayers = self.forwardPropagation(self.X,Thetas)
        h = aLayers[-1]

        #Backpropagation
        currentd = aLayers[-1].T - self.Y
        Deltas = []
        Deltas.append(np.einsum('ji,ik',currentd,aLayers[-2]))
        for j in range(2,len(aLayers)):
            zj = zLayers[-j]
            Theta = Thetas[-j + 1]

            zj = np.hstack((np.ones((np.shape(zj)[0],1)),zj))
            deltaj = Theta.T.dot(currentd) * sigmoidGrad(zj.T)
            currentd = deltaj[1:,:]

            Delta = np.einsum('ji,ik',deltaj,aLayers[-j - 1])
            Delta = Delta[1:,:]
            Deltas.insert(0,Delta)

        #Unregularized
        for i in range(len(Thetas_grad)):
            Thetas_grad[i] = (1./self.m)*Deltas[i]

        #Regularizing but not regularize bias column
        regs = Thetas
        for i in range(len(regs)):
            reg = (self.lam/self.m)*regs[i]
            reg[:,0] = np.zeros([np.shape(Thetas[i])[0]])
            Thetas_grad[i] += reg

        #Unravel the thetas
        toreturn = np.array([])
        for Theta_grad in Thetas_grad:
            toreturn = np.append(toreturn,Theta_grad.ravel())

        return toreturn
    
    def optimize(self,guess,optMethod='CG',maxiterations=50,display=False):
        opt = optimize.minimize(self.nnCost,guess,method=optMethod,jac=self.nnCostDer,options={'maxiter':maxiterations,'disp':display})
        return opt
    
    #Forward propagation to compute layers
    def forwardPropagation(self,Xprop,Thetas):
        currentLayer = Xprop
        zLayers = [np.array([0])]
        aLayers = [Xprop]
        for j in range(self.numLayers - 1):
            zj = currentLayer.dot(Thetas[j].T)
            zLayers.append(zj)

            aj = sigmoid(zj)
            #Add bias component when needed
            if j != self.numLayers - 2:
                aj = np.append(np.ones([np.shape(aj)[0],1]),aj,axis=1)
            aLayers.append(aj)
            
            currentLayer = aj
            
        return zLayers,aLayers
    
    #Predict labels based on forward propagation
    def predict(self,nnParams,Xtest):
        p = np.zeros([np.shape(Xtest)[0]])

        #Roll back up Thetas
        Thetas = reshapeThetas(nnParams,self.LayerLengths)

        #Forward propagation
        zLayers,aLayers = self.forwardPropagation(Xtest,Thetas)
        h = aLayers[-1]
        
        #Find most likely label
        for i in range(np.shape(h)[0]):
            p[i] = h[i,:].argmax() + 1
            
        #Return predictions
        return p
    
    #DetermineAccuracy
    def determineAccuracy(self,nnParams,Xtest,ytest):
        #Add bias components
        Xtest = np.append(np.ones([np.shape(Xtest)[0],1]),Xtest,axis=1)
        
        #Determine predictions and compare to labels
        predictions = self.predict(nnParams,Xtest)
        guesses = predictions - ytest.T
        
        #Determine accuracy
        sucesscount = 0
        for i, guess in enumerate(guesses):
            if guess ==0: sucesscount += 1
        accuracy = sucesscount*100./len(ytest)
        
        #Print results
        print 'Success rate of: ' + str(accuracy) + '%'
        
        #Return predictions and accuracy
        return predictions,accuracy
