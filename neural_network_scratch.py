import utils as ut
import numpy as np
class NeuralNetworks:
    '''This works in the following way. The input features are the number of features input into the neural network. 
    Hidden layers is presented as a list [a,b,c,...] where a represents the number of neurons in the first layer, b in the 
    second layer, c in the third layer and so on. The output layer is an integer representing the number of neurons in the output
    layer'''
    def __init__(self,input_features,hidden_layers,output_layer):
        #Step 1 - Storing the architecture
        self.input_features=input_features
        self.hidden_layers=hidden_layers
        self.output_layer=output_layer
        #Step 2 - initializing the weights between each neuron to the layer next to it
        #This line converts everything to a list so if there are 3 input features, 4 neurons in the first layer, 2 in the second, 3 in the third
        #And the output layer has 3, then layers=[3,4,2,3]
        layers=[input_features]+hidden_layers+[output_layer]
        self.weights=[]
        for i in range(len(layers)-1):
            '''This is a technique called as He initialization. First, let us understand what exactly a weight initialization is. We typically need
            the weights between each neuron to the layer next to it to:
            1. Not be the same or all 0
            2. Have some variance
            np.random.randn provides weights which aren't the same and produces a guassian distribution centered around 0, leading to 50%
            of weights being <0 and 50% of weights being >0. 
            Furthermore, a good initialization technique tries to ensure that all the input values to each layer to be centered around 0. Meaning
            that the mean of the values should be 0. 
            While applying ReLU, the negative values are turned to 0. For values being centered around 0, this leads to essentially 
            "half of the values" vanishing, which reduces the variance. This keeps happening during each epoch till the variance becomes extremely
            small, which is NOT what we want. Using this principle, He devised an initialization technique called as He initialization. He realized
            that as ReLU throws away "half of the values", the weights can be decided to produce twice the variance. 
            In the general formula, it is 2/fan_in where fan_in is: the number of neurons in layer i that feed into a single neuron in layer i+1.
            We are also taking the square root here because when a variance is multiplied by a scalar, the variance gets scalaed by the square
            of the multiplier. The term 2/layers[i] is the variance which we'd want, taking the square root of it gives the standard deviation and that
            is what we multiply with random.randn. Due to the law of variance, we get back the variance itself.   '''
            W=np.random.randn(layers[i],layers[i+1])*np.sqrt(2/layers[i])
            self.weights.append(W)
        '''We now create the bias vector. The concept behind this is that each neuron will have a bias and when the bias vector of a layer
        is brought up, it means that the bias vector of a layer contains the biases of the individual neurons in the specified layer'''
        self.biases=[]
        for i in range(1,len(layers)):
            #Common practice to set bias to 0
            b=np.zeros((1,layers[i]))
            self.biases.append(b)
    #Performing forward propagation, here X is the training data
    def forward(self,X):
        self.func_rel=[]
        '''The activation of the input layer would be the input data itself. The activation list stores
        the activation values at each layer. At layer 0, it is the input data itself. Note that the formula is Wixi+bi where 
        xi is the input from the previous layer and wi is the weight matrix for the particular layer in question. '''
        self.activation=[X]
        #We choose to iterate over the length of list containing the weights matrix, as each element in the list is a matrix
        for i in range(len(self.weights)):
            '''This is a point of confusion but worked it out. So the activation of the previous layer is used as input to the current layer.
            For i=0, we just do w1.X+b as X is the input and the activation. We append the result to func_rel and then we do the activation for
            the first layer by calling whichever function is needed. The result of the activation becaomes activation[1] and is used in the 
            next iteration when i=1 and so on'''
            z=np.dot(self.activation[i],self.weights[i])+self.biases[i]
            self.func_rel.append(z)
            '''This decides which activation function to use based on which layer its dealing with. The output layer uses a sigmoid activation function
            as we are dealing with binary classification while the hidden layers use the ReLU activation function'''
            if(i==len(self.weights)-1):
                A=ut.sigmoid(z)
            else:
                '''Why was leaky relu chosen? When using the regular relu, the vanishing gradient problem was encountered. This happened in the
                second hidden layer due to the inputs to the neurons being negative and ReLU outputting 0, which created a problem during gradient
                descent. In order to fix this, leaky relu was used which happens to be a modification of relu. When the input is positive, it returns
                the input itself. If the input is negative, instead of outputting 0, it returns a very small non zero value instead. That value is
                dictated by the parameter alpha (a) which is typically taken as 0.01. Alpha is multiplied with the negative input to get a very small
                non zero value. This allows the vanishing gradient problem to disappear. '''
                A=ut.leaky_relu(z)
            self.activation.append(A)
        return self.activation[-1]
    #Now, we perform backpropagation
    def backward(self,y_true):
        #Storing the gradients - weights and biases
        dW=[]
        db=[]
        #Number of training examples
        examples=y_true.shape[0]
        #Step 1 - differentiate the loss function with respect to the network's output
        dl_dA=ut.binaryCrossEntropyDerivative(y_true,self.activation[-1])
        #Going backwards across layers
        for i in reversed(range(len(self.weights))):
            if(i==len(self.weights)-1):
                #Step 2 - combine the output of dl_dA with the current layer's activation derivative.
                #This essentially measures how much the overall loss change if the raw, pre activation input of a particular neuron is changed a bit
                dl_dZ=dl_dA*ut.sigmoidDerivative(self.func_rel[i])
            else:
                #If not the output layer, we use the ReLU activation function
                dl_dZ=dl_dA*ut.leaky_relu_deriv(self.func_rel[i])
            #Step 3.1 - How much would changing the weight of a neuron affect the overall loss. This is done by
            #Multiplying the value we got in step 2 with the input to the neuron
            dl_dW=np.dot(self.activation[i].T,dl_dZ)/examples
            #Step 3.2 - How much would changing the bias of a neuron affect the overall loss
            '''This step was a bit hard to understand. Each neuron has its own bias, as has been established before. Each example would
            also pass through each neuron, meaning that each neuron contributes a certain amount of bias to it. Consider the formula
            z=wx+b. If b is changed by a certain amount, z is changed by the same amount and we've already established what dl_dZ is. 
            Hence, dl_db is equivalent to dl_dZ. However, the bias of the neuron is applicable to all the training examples. Hence, just 
            investigating one example isn't enough, all must be. Hence, for each column (neuron), we sum up the change produced by 
            dl_dZ. Then we find the average. '''
            dl_db =np.sum(dl_dZ,axis=0,keepdims=True)/examples
            #Storing dl_dW and dl_db
            dW.insert(0,dl_dW)
            db.insert(0,dl_db)
            #Step 4 - Obtaining the gradient for each neuron by multiplying the output weight with the gradient of the output neuron
            #And then sum them up
            if(i>0):
                #This answers the question "if the previous layer's output (after ReLU/sigmoid was applied) changed a little, 
                # how much would the loss change"
                dl_dA=np.dot(dl_dZ,self.weights[i].T)
        self.dW=dW
        self.db=db
    #Now updating the weights and biases
    def update(self,learning_rate):
        for i in range(len(self.weights)):
            self.weights[i]-=learning_rate*self.dW[i]
            self.biases[i]-=learning_rate*self.db[i]
    #Now making predictions
    def predict(self,X,threshold=0.5):
        #Performs forward propagation, used strictly in predictions
        probabilities=self.forward(X)
        #Converts the probabilities into predictions based on the threshold. Default is 0.5
        predictions=(probabilities>=0.5).astype("int")
        return predictions