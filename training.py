import numpy as np
import matplotlib.pyplot as plt
from neural_network_scratch import NeuralNetworks
import utils as ut
np.random.seed(42)
#Training data
X=np.array([
    [0.90,0.80,0.20],
    [0.85,0.65,0.30],
    [0.75,0.70,0.40],
    [0.60,0.55,0.50],
    [0.30,0.25,0.90],
    [0.20,0.35,0.80],
    [0.10,0.15,0.95],
    [0.40,0.30,0.70],
    [0.95,0.90,0.15],
    [0.80,0.75,0.25],
    [0.35,0.40,0.85],
    [0.15,0.20,0.90],
    [0.70,0.60,0.35],
    [0.25,0.30,0.75],
    [0.55,0.50,0.45],
    [0.45,0.55,0.60]
])
y=np.array([
    [1],
    [1],
    [1],
    [1],
    [0],
    [0],
    [0],
    [0],
    [1],
    [1],
    [0],
    [0],
    [1],
    [0],
    [1],
    [0]
])
X_train=X[:12]
y_train=y[:12]
X_test=X[12:]
y_test=y[12:]
learning_rate=0.1
epochs=5000
model=NeuralNetworks(input_features=3,hidden_layers=[6,4],output_layer=1)
loss_history=[]
accuracy_history=[]
for epoch in range(epochs):
    #Performing forward propagation
    y_pred=model.forward(X_train)
    #Computing the loss at the end of forward propagation
    loss=ut.binaryCrossEntropy(y_train,y_pred)
    loss_history.append(loss)
    #Computing training accuracy
    train_preds=(y_pred>=0.5).astype("int")
    training_accuracy=ut.accuracy(y_train,train_preds)*100
    accuracy_history.append(training_accuracy)
    #Performing backward propagation
    model.backward(y_train)
    #Performing gradient descent
    model.update(learning_rate)
    if(epoch%500==0):
        print(f"Epoch: {epoch}, Loss={loss:.6f}, Accuracy={training_accuracy:.2f}%")
#Plotting the loss curve
plt.figure(figsize=(12,8))
plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training loss")
plt.grid(True)
plt.savefig("images/training_loss.png")
plt.show()
#Plotting the accuracy curve
plt.figure(figsize=(12,8))
plt.plot(accuracy_history)
plt.xlabel("Epoch")
plt.ylabel("Accuracy %")
plt.title("Training Accuracy")
plt.grid(True)
plt.savefig("images/training_accuracy.png")
plt.show()
#Predictions
predictions=model.predict(X_test)
print("Predictions: \n")
print(predictions)
#Prediction accuracy
print("Prediction accuracy%: \n")
pred_acc=ut.accuracy(y_test,predictions)
final_pred_acc=pred_acc*100
print(final_pred_acc)