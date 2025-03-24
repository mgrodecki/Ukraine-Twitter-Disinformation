## NN - FF and BP
## Gates

#############################################
## FF - BP
## Labels: 0 and 1 where 0 is not at risk for heart disease
## Data is 3D
        
#####################################################     

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


## DATAset
#https://drive.google.com/file/d/1JjkJ4q0MMGJP8jht2ZI9qDEvV5Jjfu0r/view?usp=sharing

filename="StudentSummerProgramData_Numeric_2NumLabeled_3Cols.csv"

## !! Update this to YOUR path
DF = pd.read_csv("C:/Machine_learning/"+str(filename))
#print(DF)

## Set y to the label. Check the shape!
y = np.array(DF.iloc[:,0]).T
y = np.array([y]).T
#print("y is\n", y)

## Normalize the data (not the label!)
## normalized_df=(df-df.min())/(df.max()-df.min())
DF=DF.iloc[:, [1, 2, 3]]
DF=(DF-DF.mean())/DF.std()
#print(DF)
X = np.array(DF)
#print("X is\n", X)

InputColumns = 3
n = len(DF) ## number of rows of entire X

LR = .1
LRB = .1
#................................................
 

class NeuralNetwork(object):
    def __init__(self):
        
        self.InputNumColumns = InputColumns  ## columns
        self.OutputSize = 1 ## Categories
        self.HiddenUnits = 4  ## one layer with h units
        self.n = n  ## number of training examples, n
        
        #print("Initialize NN\n")
        #Random W1
        self.W1 = np.random.randn(self.InputNumColumns, self.HiddenUnits) # c by h  
       
        #print("INIT W1 is\n", self.W1)
        
        ##-----------------------------------------
        ## NOTE ##
        ##
        ## The following are all random. However, you can comment this out
        ## and can set any weights and biases by hand , etc.
        ##
        ##---------------------------------------------
        
        self.W2 = np.random.randn(self.HiddenUnits, self.OutputSize) # h by o 
        #print("W2 is:\n", self.W2)
        
        self.b = np.random.randn(1, self.HiddenUnits)
        #print("The b's are:\n", self.b)
        ## biases for layer 1
        
        self.c = np.random.randn(1, self.OutputSize)
        #print("The c is\n", self.c)
        ## bias for last layer
        
        
    def FeedForward(self, X):
        #print("FeedForward\n\n")
        self.z = (np.dot(X, self.W1)) + self.b 
        #X is n by c   W1  is c by h -->  n by h
        #print("Z1 is:\n", self.z)
        
        self.h = self.Sigmoid(self.z) #activation function    shape: n by h
        #print("H is:\n", self.h)
        
        self.z2 = (np.dot(self.h, self.W2)) + self.c # n by h  @  h by o  -->  n by o  
        #print("Z2 is:\n", self.z2)
        
        ## Using Softmax for the output activation
        output = self.Sigmoid(self.z2)  
        #print("output Y^ is:\n", output)
        return output
        
    def Sigmoid(self, s, deriv=False):
        if (deriv == True):
            return s * (1 - s)
        return 1/(1 + np.exp(-s))
    
    def BackProp(self, X, y, output):
        #print("\n\nBackProp\n")
        self.LR = LR
        self.LRB=LRB  ## LR for biases
        
        # Y^ - Y
        self.output_error = output - y    
        #print("Y^ - Y\n", self.output_error)
        
        ## NOTE TO READER........................
        ## Here - we DO NOT multiply by derivative of Sig for y^ b/c we are using 
        ## cross entropy and softmax for the loss and last activation
        # REMOVED # self.output_delta = self.output_error * self.Sigmoid(output, deriv=True) 
        ## So the above line is commented out...............
        
        self.output_delta = self.output_error 
          
        ##(Y^ - Y)(W2)
        self.D_Error_W2 = self.output_delta.dot(self.W2.T) #  D_Error times W2
        #print("W2 is\n", self.W2)
        #print(" D_Error times W2\n", self.D_Error_W2)
        
        ## (H)(1 - H) (Y^ - Y)(Y^)(1-Y^)(W2)
        ## We still use the Sigmoid on H
        
        self.H_D_Error_W2 = self.D_Error_W2 * self.Sigmoid(self.h, deriv=True) 
        
        ## Note that * will multiply respective values together in each matrix
        #print("Derivative sig H is:\n", self.Sigmoid(self.h, deriv=True))
        #print("self.H_D_Error_W2 is\n", self.H_D_Error_W2)
        
        ################------UPDATE weights and biases ------------------
        #print("Old W1: \n", self.W1)
        #print("Old W2 is:\n", self.W2)
        #print("X transpose is\n", X.T)
        
        ##  XT  (H)(1 - H) (Y^ - Y)(Y^)(1-Y^)(W2)
        self.X_H_D_Error_W2 = X.T.dot(self.H_D_Error_W2) ## this is dW1
        
        ## (H)T (Y^ - Y) - 
        self.h_output_delta = self.h.T.dot(self.output_delta) ## this is for dW2
        
        #print("the gradient :\n", self.X_H_D_Error_W2)
        #print("the gradient average:\n", self.X_H_D_Error_W2/self.n)
        
        #print("Using sum gradient........\n")
        self.W1 = self.W1 - self.LR*(self.X_H_D_Error_W2) # c by h  adjusting first set (input -> hidden) weights
        self.W2 = self.W2 - self.LR*(self.h_output_delta) 
        
        
        #print("The sum of the b update is\n", np.mean(self.H_D_Error_W2, axis=0))
        #print("The b biases before the update are:\n", self.b)
        self.b = self.b  - self.LRB*np.mean(self.H_D_Error_W2, axis=0)
        #print("The H_D_Error_W2 is...\n", self.H_D_Error_W2)
        #print("Updated bs are:\n", self.b)
        
        self.c = self.c - self.LR*np.mean(self.output_delta, axis=0)
        #print("Updated c's are:\n", self.c)
        
        #print("The W1 is: \n", self.W1)
        #print("The W1 gradient is: \n", self.X_H_D_Error_W2)
        #print("The W1 gradient average is: \n", self.X_H_D_Error_W2/self.n)
        #print("The W2 gradient  is: \n", self.h_output_delta)
        #print("The W2 gradient average is: \n", self.h_output_delta/self.n)
        #print("The biases b gradient is:\n",np.mean(self.H_D_Error_W2, axis=0 ))
        #print("The bias c gradient is: \n", np.mean(self.output_delta, axis=0))
        ################################################################
        
    def TrainNetwork(self, X, y):
        output = self.FeedForward(X)
        #print("Output in TNN\n", output)
        self.BackProp(X, y, output)
        return output

#-------------------------------------------------------------------        
MyNN = NeuralNetwork()

TotalLoss=[]
AvgLoss=[]
Epochs=500

for i in range(Epochs): 
    print("RUN: ", i)
    output=MyNN.TrainNetwork(X, y)
   
    #print("The y is ...\n", y)
    #print("The output is: \n", output)
    output=np.where(output > 0.5, 1, 0)
    #print('Prediction y^ is', output)
    ## Using Categorical Cross Entropy...........
    #loss = np.mean(-y * np.log(output))  ## We need y to place the "1" in the right place
    loss=np.sum(np.square(output-y))
    avgLoss=np.mean(np.square(output-y))
    #print("The current loss is\n", loss)
    #print("The current average loss is\n", avgLoss)
    TotalLoss.append(loss)
    AvgLoss.append(avgLoss)


###################-output and vis----------------------    
import matplotlib.pyplot as plt

print("\nPredicted Label  True Label\n")
for i in range(0, len(y)):
    print("\t", output[i], "\t", y[i])
plt.plot(range(1, len(TotalLoss) + 1), TotalLoss)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Neural Network: 1 hidden layer with 4 nodes')
plt.show()

print("\nConfusion Matrix:\n", confusion_matrix(output, y))
