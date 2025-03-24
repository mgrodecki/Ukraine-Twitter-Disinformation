## regression
## patterned after dr. Gates lectures and example code
## ###################################################      

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


filename="StudentSummerProgramData_Numeric_3_cont.csv"

## !! Update this to YOUR path
DF = pd.read_csv("C:/Machine_learning/"+str(filename))
#print(DF)

X = DF.iloc[:,1:3]
X = np.array(X)
#print("X is\n", X)

y = DF.iloc[:,0]
y = np.array([y]).T
#print("y is\n", y)

InputColumns = 2

#plt.scatter(X[:,0], X[:,1])
#plt.show()

W = np.random.random(InputColumns)/200
#print("W is\n", W)
#print(W.shape)
W = np.array([W])
#print(W)
b = 0
LR = 0.000001
epochs = 100

AllErrors = []
AverageErrors = []
n = len(X)
#print(n)

for i in range(epochs):
    print("Epoch: ", i)
    y_hat = X@W.T + b
    #print("y_hat is\n", y_hat)

    Error = y_hat - y
    #print("Error is\n", Error)
    MeanError =  np.mean(Error)
    #print("The current mean error is\n", MeanError)
    AllErrors.append(Error)
    AverageErrors.append(MeanError)

    #derivatives
    dL_dW = (np.mean(X * Error, axis = 0))/10
    dL_dW = np.array([dL_dW])
    dL_db = np.mean(Error)

    W = W - (LR * dL_dW)
    #print(W)
    b = b - (LR * dL_db)
    #print(b)

print("\nMean Error\n")
for i in range(0, len(AverageErrors)):
    print("\t", AverageErrors[i])
plt.plot(range(1, len(AverageErrors) + 1), AverageErrors)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Regression')
plt.show()


