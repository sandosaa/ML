import pandas as pd
import numpy as np

data = pd.read_csv("Hours Studied Classes.csv")
x = data['Hours_Studied'].to_numpy()
y = data['Passed'].to_numpy()
y_pred=0
def sigmoid(x):
    return 1/(1+np.exp(-x))

def logistic_regression(x,y,L=1,epochs=1000):
    m,b=0,0
    
    for i in range(epochs):
        y_pred = sigmoid(b + m *x) 

        m_calc=np.dot(x,(y_pred-y))/len(data)
        b_calc=np.sum((y_pred-y))/len(data)

        m -= L * m_calc
        b -= L * b_calc 

    return y_pred,m,b

def cross_entropy(y,y_pred):
    error = -(1/len(data))*sum(y*np.log(y_pred+1e-9)+(1-y)*np.log(1-y_pred+1e-9))
    return error
print(len(data))

y_pred,m,b = logistic_regression(x,y)
print(f'The linear equation: y = {m:.3f}x + {b:.3f}')
print(f'With error = {cross_entropy(y,y_pred):.3f}')
print(y_pred)