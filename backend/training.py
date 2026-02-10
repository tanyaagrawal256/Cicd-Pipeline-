import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression

#Load data
df= pd.read_csv(r"/Users/tanyaagrawal/Downloads/mlops_pipeline/data/data.csv")

x = df[['area', 'bedrooms']]
y =  df['price']

#Train model
model = LinearRegression()
model.fit(x, y)

#Save model
with open(r"/Users/tanyaagrawal/Downloads/mlops_pipeline/backened/model/model.pkl", "wb") as f:
    pickle.dump(model, f)
