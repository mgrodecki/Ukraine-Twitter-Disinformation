## regression with Sigmoid activation
## patterned after dr. Gates lectures and example code
## ###################################################     

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

filename="StudentSummerProgramData_Numeric_3_cont.csv"

## !! Update this to YOUR path
DF = pd.read_csv("C:/Machine_learning/"+str(filename))
#print(DF)

X = DF.iloc[:,1:3]
#print("X is\n", X)
## Normalize the data (not the label!)
## or min/max
## normalized_X=(X-X.min())/(X.max()-X.min())
X=(X-X.mean())/X.std()
X = np.array(X)
#print("Normalized X is\n", X)


y = DF.iloc[:,0]
y = np.array([y]).T
#print("y is\n", y)
#update labels so that they're 0 (if y<3.5) or 1 (if y >= 3.5)
y = np.where(y >= 3.5, 1, 0)
#print("binary y is\n", y)

InputColumns = 2

#plt.scatter(X[:,0], X[:,1])
#plt.show()

W = np.random.random(InputColumns)-0.5
print("W is\n", W)
#print(W.shape)
W = np.array([W])
#print(W)
b = 0
LR= 0.1
LRB = 0.1
epochs = 500

AllErrors = []
AvgLoss = []
TotalLoss = []
n = len(X)
#print(n)

def Sigmoid(s, deriv=False):
    if (deriv == True):
        return s * (1 - s)
    return 1/(1 + np.exp(-s))

for i in range(epochs):
    print("Epoch: ", i)
    z = X@W.T + b
    y_hat = Sigmoid(z)
    #print("y_hat is\n", y_hat)

    Error = y_hat - y
    #print("Error is\n", Error)

    #derivatives
    #dL_dW = np.mean(X * Error, axis = 0)
    #print("dL_dW is\n", dL_dW)

    dL_dW = np.mean(Error * Sigmoid(y_hat, deriv=True) * X, axis = 0)
    #print("dL_dW is\n", dL_dW)

    dL_dW = np.array([dL_dW])

    #dL_db = np.mean(Error)
    #print("dL_db is\n", dL_db)

    dL_db = np.mean(Error * Sigmoid(y_hat, deriv=True))
    #print("dL_db is\n", dL_db)

    W = W - LR * dL_dW
    #print(W)
    b = b - LRB * dL_db
    #print(b)
    z = X@W.T + b
    output = Sigmoid(z)

    #print("The output is: \n", output)
    output=np.where(output > 0.5, 1, 0)
    #print('Prediction y^ is', output)
    loss=np.sum(np.square(output-y))
    avgLoss=np.mean(np.square(output-y))
    #print("The current loss is\n", loss)
    TotalLoss.append(loss)
    AvgLoss.append(avgLoss)


print("\nPredicted Label  True Label\n")
for i in range(0, len(y)):
    print("\t", output[i], "\t", y[i])
plt.plot(range(1, len(TotalLoss) + 1), TotalLoss)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Regression with Sigmoid activation')
plt.show()

print("\nConfusion Matrix:\n", confusion_matrix(output, y))
