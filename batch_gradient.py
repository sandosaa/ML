import pandas as pd

data=pd.read_csv("Housing.csv")
print(data['price'],data['area'])

mean_x = data['area'].mean()
std_x = data['area'].std()
mean_y = data['price'].mean()
std_y = data['price'].std()

data['area'] = (data['area'] - mean_x) / std_x
data['price'] = (data['price'] - mean_y) / std_y

def loss_func(m,b,points):
    total_error=0
    for i in range(len(points)):
        x=points.iloc[i].area
        y=points.iloc[i].price

        total_error+=(y-(m*x+b))**2
    return total_error/(2.0*len(points))

def gradient(m_now,b_now,points,L):
    m_gradient, b_gradient = 0,0
    for i in range(len(points)):
        x=points.iloc[i].area
        y=points.iloc[i].price

        m_gradient+= x*((m_now*x+b_now)-y)/len(points)
        b_gradient+= ((m_now*x+b_now)-y)/len(points)

    m = m_now - (m_gradient) * L
    b = b_now - (b_gradient) * L
    return m,b

m = 0
b = 0
L = 0.001
epochs = 3000
for i in range(epochs):
    m,b = gradient(m,b,data,L)

m_real = (std_y / std_x) * m
b_real = (std_y * b) + mean_y - (m_real * mean_x)
   

print(f"The linear equation is: y = {m_real:.3f}x + {b_real:.3f}")
print(f'The error = {loss_func(m,b,data):.3f}')





