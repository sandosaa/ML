import pandas as pd

data=pd.read_csv("Housing.csv")
print(data)
data['mainroad']=data['mainroad'].replace({'yes': 1, 'no': 0})
data['guestroom']=data['guestroom'].replace({'yes': 1, 'no': 0})
data['basement']=data['basement'].replace({'yes': 1, 'no': 0})
data['hotwaterheating']=data['hotwaterheating'].replace({'yes': 1, 'no': 0})
data['airconditioning']=data['airconditioning'].replace({'yes': 1, 'no': 0})
data['prefarea'] = data['prefarea'].replace({'yes': 1, 'no': 0})
data['furnishingstatus']=data['furnishingstatus'].replace({'furnished':1,'semi-furnished':.5,'unfurnished':0})

def loss_func(m1,m2,m3,m4,m5,m6,m7,m8,m9,m10,m11,m12,b,points):
    total_error=0
    for i in range(len(points)):
        x1=points.iloc[i].area
        x2=points.iloc[i].bedrooms
        x3=points.iloc[i].bathrooms
        x4=points.iloc[i].stories
        x5=points.iloc[i].mainroad
        x6=points.iloc[i].guestroom
        x7=points.iloc[i].basement
        x8=points.iloc[i].hotwaterheating
        x9=points.iloc[i].airconditioning
        x10=points.iloc[i].parking
        x11=points.iloc[i].prefarea
        x12=points.iloc[i].furnishingstatus
        y=points.iloc[i].price
        total_error+=(y-(m1*x1+m2*x2+m3*x3+m4*x4+m5*x5+m6*x6+m7*x7+m8*x8+m9*x9+m10*x10+m11*x11+m12*x12+b))**2
    return total_error/(2.0)

def gradient(m1_now,m2_now,m3_now,m4_now,m5_now,m6_now,m7_now,m8_now,m9_now,m10_now,m11_now,m12_now,b_now,points,L):
    m1_gradient,m2_gradient,m3_gradient,m4_gradient,m5_gradient,m6_gradient,m7_gradient,m8_gradient,m9_gradient,m10_gradient,m11_gradient,m12_gradient, b_gradient = 0,0,0,0,0,0,0,0,0,0,0,0,0
    for i in range(len(points)):
        x1=points.iloc[i].area
        x2=points.iloc[i].bedrooms
        x3=points.iloc[i].bathrooms
        x4=points.iloc[i].stories
        x5=points.iloc[i].mainroad
        x6=points.iloc[i].guestroom
        x7=points.iloc[i].basement
        x8=points.iloc[i].hotwaterheating
        x9=points.iloc[i].airconditioning
        x10=points.iloc[i].parking
        x11=points.iloc[i].prefarea
        x12=points.iloc[i].furnishingstatus
        y=points.iloc[i].price

        def partitial():
            return -2*(y-(m1_now*x1+m2_now*x2+m3_now*x3+m4_now*x4+m5_now*x5+m6_now*x6+m7_now*x7+m8_now*x8+m9_now*x9+m10_now*x10+m11_now*x11+m12_now*x12+b_now))
        
        m1_gradient+= x1*partitial()
        m2_gradient+= x2*partitial()
        m3_gradient+= x3*partitial()
        m4_gradient+= x4*partitial()
        m5_gradient+= x5*partitial()
        m6_gradient+= x6*partitial()
        m7_gradient+= x7*partitial()
        m8_gradient+= x8*partitial()
        m9_gradient+= x9*partitial()
        m10_gradient+= x10*partitial()
        m11_gradient+= x11*partitial()
        m12_gradient+= x12*partitial()

        b_gradient+= partitial()

    m1 = m1_now - m1_gradient * L
    m2 = m2_now - m2_gradient * L
    m3 = m3_now - m3_gradient * L
    m4 = m4_now - m4_gradient * L
    m5 = m5_now - m5_gradient * L
    m6 = m6_now - m6_gradient * L
    m7 = m7_now - m7_gradient * L
    m8 = m8_now - m8_gradient * L
    m9 = m9_now - m9_gradient * L
    m10 = m10_now - m10_gradient * L
    m11 = m11_now - m11_gradient * L
    m12 = m12_now - m12_gradient * L

    b = b_now - b_gradient * L
    return m1,m2,m3,m4,m5,m6,m7,m8,m9,m10,m11,m12,b

m1,m2,m3,m4,m5,m6,m7,m8,m9,m10,m11,m12 = 0,0,0,0,0,0,0,0,0,0,0,0
b = 0
L = 0.001
epochs = 3000

for i in range(epochs):
    m1,m2,m3,m4,m5,m6,m7,m8,m9,m10,m11,m12,b = gradient(m1,m2,m3,m4,m5,m6,m7,m8,m9,m10,m11,m12,b,data,L)

print(m1,m2,m3,m4,m5,m6,m7,m8,m9,m10,m11,m12,b)
print(loss_func(m1,m2,m3,m4,m5,m6,m7,m8,m9,m10,m11,m12,b,data))

