import pandas as pd

data=pd.read_csv("Housing.csv")
print(data)

for col in ['area', 'bedrooms', 'bathrooms', 'stories', 'parking', 'price']:
    data[col] = (data[col] - data[col].mean()) / data[col].std()

def loss_func(m1,m2,m3,m4,m5,b,points):
    total_error=0
    for i in range(len(points)):
        x1=points.iloc[i].area
        x2=points.iloc[i].bedrooms
        x3=points.iloc[i].bathrooms
        x4=points.iloc[i].stories
        x5=points.iloc[i].parking
        y=points.iloc[i].price
        total_error+=(y-(m1*x1+m2*x2+m3*x3+m4*x4+m5*x5+b))**2
    return total_error/(2.0*len(points))

def gradient(m1_now,m2_now,m3_now,m4_now,m5_now,b_now,points,L):
    m1_gradient,m2_gradient,m3_gradient,m4_gradient,m5_gradient, b_gradient = 0,0,0,0,0,0
    for i in range(len(points)):
        x1=points.iloc[i].area
        x2=points.iloc[i].bedrooms
        x3=points.iloc[i].bathrooms
        x4=points.iloc[i].stories
        x5=points.iloc[i].parking

        y=points.iloc[i].price

       
        m1_gradient+= x1*(-2*(y-(m1_now*x1+m2_now*x2+m3_now*x3+m4_now*x4+m5_now*x5+b_now)))
        m2_gradient+= x2*(-2*(y-(m1_now*x1+m2_now*x2+m3_now*x3+m4_now*x4+m5_now*x5+b_now)))
        m3_gradient+= x3*(-2*(y-(m1_now*x1+m2_now*x2+m3_now*x3+m4_now*x4+m5_now*x5+b_now)))
        m4_gradient+= x4*(-2*(y-(m1_now*x1+m2_now*x2+m3_now*x3+m4_now*x4+m5_now*x5+b_now)))
        m5_gradient+= x5*(-2*(y-(m1_now*x1+m2_now*x2+m3_now*x3+m4_now*x4+m5_now*x5+b_now)))

        b_gradient+= (-2*(y-(m1_now*x1+m2_now*x2+m3_now*x3+m4_now*x4+m5_now*x5+b_now)))

    m1 = m1_now - (m1_gradient/ len(points)) * L
    m2 = m2_now - (m2_gradient/ len(points)) * L
    m3 = m3_now - (m3_gradient/ len(points)) * L
    m4 = m4_now - (m4_gradient/ len(points)) * L
    m5 = m5_now - (m5_gradient/ len(points)) * L

    b = b_now - (b_gradient/len(points)) * L
    return m1,m2,m3,m4,m5,b

m1,m2,m3,m4,m5 = 0,0,0,0,0
b = 0
L = 0.0001
epochs = 2000
for i in range(epochs):
    m1,m2,m3,m4,m5,b = gradient(m1,m2,m3,m4,m5,b,data,L)

print(m1,m2,m3,m4,m5,b)
print(loss_func(m1,m2,m3,m4,m5,b,data))

