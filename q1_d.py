import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

#%%
import numpy as np
#%% import data
data = pd.read_csv("test.csv")
data['x']= data['x']
#%% cost function MSE
def cost_function(X, y, theta):
    m = len(y)
    h = np.dot(X, theta)
    error = h - y
    cost = (1/(2*m)) * np.sum(np.power(error,2))
    return cost
#%% Normal Equation
m= len(data['x'])
x= (np.array(data['x'])).reshape(m,1)
y= (np.array(data['y'])).reshape(m,1)
X = np.insert(x, 0, np.ones([m]), axis=1)

XTX= np.linalg.inv(np.dot(X.T, X))
theta= np.linalg.multi_dot([XTX, X.T, y])
#%% calculate total cost
total_cost= cost_function(X,y,theta)
#%% plot linear regression on data
plt.figure(figsize=(10, 10))
plt.title("regression Normal Equation")
plt.xlabel("x")
plt.ylabel("y")
h = np.dot(X, theta)
plt.plot(x, h, 'r')
plt.plot(x, y, 'o')
plt.show()