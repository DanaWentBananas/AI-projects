import numpy as np
import copy


#find a training set

xtrain,ytrain,xtest,ytest

#About the dataset
mtrain = xtrain.shape[0]
mtest = xtest.shape[0]
numpx = xtrain.shape[1]
print ("Number of training examples: mtrain = " + str(mtrain))
print ("Number of testing examples: mtest = " + str(mtest))
print ("Height/Width of each image: numpx = " + str(numpx))
print ("Each image is of size: (" + str(numpx) + ", " + str(numpx) + ", 3)")
print ("xtrain shape: " + str(xtrain.shape))
print ("ytrain shape: " + str(ytrain.shape))
print ("xtest shape: " + str(xtest.shape))
print ("ytest shape: " + str(ytest.shape))

#Reshape data from (numPx, numPx, 3) to (numPx*numPx∗3, 1)
xtrain = xtrain.reshape(xtrain.shape[0],-1).T
xtest = xtest.reshape(xtest.shape[0],-1).T

#Standarize them
xtrain=xtrain/255
xtest=xtest/255

#Functions

def sigmoid(z):
    s = 1/1+np.exp(-z)
    return s

def initialize(dim):
    w = np.zeros((dim,1))
    b = 0.0
    return w,b

def propagation(w,b,x,y):
    m = x.shape[1]
    
    #activation function
    A = sigmoid(np.dot(w.T,x)+b)
    
    #cost function
    cost = -1/m*(np.dot(y,np.log(A).T)+np.dot((1-y),np.log(1-A).T))
    
    #backward propagation
    dw = 1/m*np.dot(x,(A-y).T)
    db = 1/m*np.sum(A-y)
    
    cost = np.squeeze(np.array(cost))
    
    grads {'dw':dw,
           'db':db}
    
    return grads,cost

#optimization
#learning w and b by minimizing cost function
def optimize(w,b,x,y,iterations=100,learningrate=0.009,printcost=False):
    
    w=copy.deepcopy(w)
    b=copy.deepcopy(b)
    
    costs=[]
    
    w = w-learningrate*dw
    b = b-learningrate*db
    
    if i%100==0:
        costs.append(cost)
        
        if printcost:
            print("cost after iteration %i:%f" %(i,cost))
            
    params = {"w":w,
              "b":b}
    grads = {"dw":dw,
             "db":db}
    
    return params,grads,costs

#prediction

def predict(w,b,x):
    
    m = x.shape[1]
    ypredict = np.zeros((1,m))
    
    A = sigmoid(np.dot(w.T,x)+b)
    
    for i in range(A.shape[1]):
        if A[0,i]>0.5:
            ypredict[0,i]=1
        else:
            ypredict[0,i]=0
    
    return ypredict

#full model function
def model(xtrain,train,xtest,ytest,iterations=2000,learningrate=0.5,printcost=False):
    
    #initialize weight and bias
    w,b = initialize(xtrain.shape[0])
    
    #gradient descent
    params,grads,costs = optimize(w,b,xtrain,ytrain,iterations,learningrate,printcost)
    
    #get weights and bias from params dictionary
    w = params['w']
    b = params['b']
    
    #predict
    ypredicttest = predict(w,b,xtest)
    ypredicttrain = predict(w,b,xtrain)
    
    #accuracies
    trainacc = 100 - np.mean(np.abs(ypredicttrain - ytrain)) * 100
    testacc = 100 - np.mean(np.abs(ypredicttest - ytest)) * 100

    
    if printcost:
        print(f'train accuracy: {trainacc} %')
        print(f'test accuracy: {testacc} %')
        
    d = {'costs':costs,
         'ypredicttest':ypredicttest,
         'ypredicttrain':ypredicttrain,
         'w':w,
         'b':b,
         'learningrate':learningrate,
         'iterations':iterations}
        
    return d

#plotting learning curve

mymodel = model(.......)

costs = np.squeeze(mymodel['costs'])
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =" + str(mymodel["learningrate"]))
plt.show()