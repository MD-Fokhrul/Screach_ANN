# Package imports
import numpy as np
import pandas as pd

#from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt




# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


y_train = y_train.reshape((-1,1))
y_test = y_test.reshape((-1,1))



# sigmoid function 
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def mean_squared_error(predictions, labels):
    N = labels.size 
    mse = ((predictions - labels)**2).sum() / (2*N)
    return mse

def accuracy(predictions, labels):
    predicions_correct = predictions.round() == labels
    accuracy = predicions_correct.mean()
    
    return accuracy


# relu function
    '''
def relu(Z):
    R = np.maximum(0, Z)
    return R


def relu_derivative(Z):
    Z[Z >= 0] = 1
    Z[Z < 0]  = 0
    return Z
    '''
# hyperparameter define
m = X_train.shape[0]
lr = 0.1
#epochs = 10000

nn_input = X_train.shape[1]
nn_output = 1
nn_hdim =10

def build_model(nn_hdim, epochs=2000):
    
    # Initialize the parameters to random values. We need to learn these.
    np.random.seed(0)
    w1 = np.random.randn(nn_input, nn_hdim)
    b1 = np.zeros((1, nn_hdim))
    w2 = np.random.randn(nn_hdim, nn_output)
    b2 = np.zeros((1, nn_output))

    # This is what we return at the end
    model = {}
    monitoring = {"mean_squared_error": [], "accuracy": []}
    
    # Gradient descent. For each (batch)...
    for e in range(epochs):

        # forward
        a2 = sigmoid(np.dot(X_train, w1) + b1)        
        a3 = sigmoid(np.dot(a2, w2) + b2)   
        
        
        acc = accuracy(a3, y_train)
        mse = mean_squared_error(a3, y_train)
        monitoring["accuracy"].append(acc)
        monitoring["mean_squared_error"].append(mse)
         
        # backpropagation
        delta_2 = ((a3 - y_train) * a3 * (1 - a3)) / m              
        delta_1 = (np.dot(delta_2, w2.T) * a2 * (1 - a2)) / m
        
        db2 = np.sum(a = delta_2, axis =0, keepdims = True)/ m
        db1 = np.sum(a = delta_1, axis =0, keepdims = True)/ m
        dw2 = np.dot(a2.T, delta_2)
        dw1 = np.dot(X_train.T, delta_1) 
        #np.sum(self.dZ[str(i)], axis = 1, keepdims = True)
        
        
        # weight updates
        w2 -= lr * dw2   
        w1 -= lr * dw1
        
        b2 -= lr * db2   
        b1 -= lr * db1        

      

    
        # Assign new parameters to the model
        model = { 'W1': w1, 'b1': b1, 'W2': w2, 'b2': b2}
        
        if e % 100 == 0:
          print("Loss after iteration %i: %f  and accuracy %f"  %(e, mse, acc)  )
    
    return model, monitoring

model, monitoring = build_model(10)

#Test Data

# just forward
def predict_(model):
    
    a2 = sigmoid(np.dot(X_test, model['W1']) + model['b1'])       
    a3 = sigmoid(np.dot(a2, model['W2']) + model['b2'])
    
    acc = accuracy(a3, y_test)
    print("Accuracy: {}".format(acc))
    return a3

y_pred = predict_(model)
y_pred = (y_pred > 0.5)
acc = accuracy(y_pred, y_test)
print("Accuracy: {}".format(acc))
'''
# for extra visualization


from mlxtend.plotting import plot_confusion_matrix
#from sklearn.metrics import confusion_matrix
import matplotlib
from sklearn.metrics import confusion_matrix, precision_score, recall_score
#cm = confusion_matrix(y_test, y_pred)
precision = precision_score(y_test, y_pred)

recall = recall_score(y_test, y_pred)

font = {
    'family': 'Times New Roman',
    'size': 12,
}
matplotlib.rc('font', **font)
cm = confusion_matrix(y_test, y_pred)
plot_confusion_matrix(conf_mat=cm, figsize=(4, 4), cmap=plt.cm.Greens)
'''

plt.plot(range(len(monitoring['accuracy'])), monitoring['accuracy'])
plt.title("accuracy")
plt.show()


plt.plot(range(len(monitoring['mean_squared_error'])), monitoring['mean_squared_error'])
plt.title("mean_squared_erro")
plt.show()
