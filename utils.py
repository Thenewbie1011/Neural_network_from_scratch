import numpy as np
#1. Defining the sigmoid function
def sigmoid(X):
    return 1/(1+np.exp(-X))
#2. Derivative of the sigmoid function
def sigmoidDerivative(X):
    sigmoid_res=sigmoid(X)
    return (sigmoid_res*(1-sigmoid_res))
#3. Defining the ReLU function
def relu(X):
    return np.maximum(0,X)
#4. Derivative of the ReLU function
def reluDerivative(X):
    return (X>0).astype(float)
#5. Binary cross entropy formula
def binaryCrossEntropy(y_true,y_pred):
    epsilon=1e-15
    y_pred=np.clip(y_pred,epsilon,1-epsilon)
    #Returns loss for each individual example
    loss=-(y_true*np.log(y_pred)+((1-y_true)*np.log(1-y_pred)))
    #Averages them
    final_loss=np.mean(loss)
    return final_loss
#6. Binary cross entropy derivative
def binaryCrossEntropyDerivative(y_true,y_pred):
    epsilon=1e-15
    y_pred=np.clip(y_pred,epsilon,1-epsilon)
    deriv=(y_pred-y_true)/(y_pred*(1-y_pred))
    return deriv
#7. Accuracy
def accuracy(y_true,y_pred):
    return np.mean(y_true==y_pred)
#8. Leaky relu
def leaky_relu(x):
    return np.where(x>0,x,0.01*x)
def leaky_relu_deriv(x):
    return np.where(x>0,1.0,0.01).astype("float")