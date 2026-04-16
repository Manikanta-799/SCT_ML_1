import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = {
    'sqft':[1000,1500,1800,2400,3000],
    'bedrooms':[2,3,3,4,4],
    'bathrooms':[1,2,2,3,3],
    'price':[200000,300000,350000,450000,600000]
}

df = pd.DataFrame(data)

X = df[['sqft','bedrooms','bathrooms']]
y = df['price']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

model = LinearRegression()
model.fit(X_train,y_train)

prediction = model.predict([[2000,3,2]])

print("Predicted price =",prediction)
