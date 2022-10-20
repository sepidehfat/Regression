import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

#%%
import numpy as np
#%% import data
data = pd.read_csv("test.csv")
data['x']= data['x']*5
#%% normalize
normal_data= (data - data.max())/(data.max()-data.min())
#%% cost function MSE
def cost_function(X, y, theta):
    m = len(y)
    h = np.dot(X, theta)
    error = h - y
    cost = (1/(2*m)) * np.sum(np.power(error,2))
    return cost
#%% Gradient descent
import numpy as np
m= len(normal_data['x'])
x_o= (np.array(data['x'])).reshape(m,1)
y_o= (np.array(data['y'])).reshape(m,1)

x= (np.array(normal_data['x'])).reshape(m,1)
y= (np.array(normal_data['y'])).reshape(m,1)

X = np.insert(x, 0, np.ones([m]), axis=1)
X_o= np.insert(x_o, 0, np.ones([m]), axis=1)


theta= np.array([[0],[0]])
prev_cost= 0
tolerance= 100000
costs= []
while(abs(tolerance)>0.00001):
    h = np.dot(X, theta)
    # diff=h-y
    theta = theta - 0.1/m * np.dot(X.T,h-y)
    cost= cost_function(X,y,theta)
    # print("cost:", cost)
    costs.append(cost)
    tolerance= abs(cost-prev_cost)/cost
    # print("tolerance", tolerance)
    prev_cost= cost;
#%% total cost
total_cost= cost_function(X_o,y_o,theta)
#%% plot linear regression on normaized data
plt.figure(figsize=(10, 10))
plt.title("regression on normalized data")
plt.plot(x, h, 'r')
plt.plot(x, y, 'o')
plt.show()
#%% plot linear regression on original data
plt.figure(figsize=(10, 10))
plt.title("regression on original data")
h_o= np.dot(X_o, theta)
plt.plot(x_o, h_o, 'r')
plt.plot(x_o, y_o, 'o')
plt.show()
#%% print cost
plt.title("cost")
plt.plot(list(range(len(costs))),costs)
plt.show()

