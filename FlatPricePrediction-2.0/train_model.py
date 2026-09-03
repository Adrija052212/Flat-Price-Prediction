import pandas as pd
import numpy as np
import math
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

data= pd.read_excel(r"C:\ENGINEERING\Flat_Price_Prediction\Dataset\Flat_Price_Multiple_Linear_Regression_100.xlsx")
x= data[
    [
        "Area_Sqft",
        "Facing",
        "Floor",
        "Car_Parking_Sqft",
        "Bedrooms"
    ]
]
y= data["Price_Lakh"]
x_train, x_test, y_train, y_test= train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)  

encoder = OneHotEncoder(
    drop="first",
    handle_unknown="ignore",
    sparse_output=False
)

x_train_facing = encoder.fit_transform(x_train[["Facing"]])
x_test_facing = encoder.transform(x_test[["Facing"]])
# Remove Facing from the original data
x_train_numeric = x_train.drop("Facing", axis=1).values
x_test_numeric = x_test.drop("Facing", axis=1).values
# Combine encoded Facing with the numerical features
x_train_encoded = np.hstack((x_train_numeric, x_train_facing))
x_test_encoded = np.hstack((x_test_numeric, x_test_facing))

model = LinearRegression()
model.fit(x_train_encoded, y_train)
y_pred = model.predict(x_test_encoded)

mae= mean_absolute_error(y_test, y_pred)
rmse= math.sqrt(mean_squared_error(y_test, y_pred))
r2= r2_score(y_test, y_pred)

print("Model Evaluation")
print("-----------------")
print("Mean Absolute Error: ", round(mae, 2))
print("Root Mean Squared Error: ", round(rmse, 2))
print("R2 Score: ", round(r2, 4))

joblib.dump(model, r"C:\ENGINEERING\Flat_Price_Prediction\model\flat_price_model.pkl")
joblib.dump(encoder, r"C:\ENGINEERING\Flat_Price_Prediction\model\encoder.pkl")
print("\nModel saved Successfully")