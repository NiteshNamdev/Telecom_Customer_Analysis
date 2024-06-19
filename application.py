import pandas as pd
import numpy as np
import streamlit as st
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


st.title("Welcome to the Telecom data Prediction")
st.header("Telecom data Prediction")
df = pd.read_csv("user_scores.csv")


# Features and Target Variable
x = df[['engagement_score', 'experience_score']]
y = df['satisfaction_score']

# Split the data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=45)

# Build a Linear Regression Model
regressor = LinearRegression()
regressor.fit(X_train,y_train)

# Predict the satisfaction score
y_pred = regressor.predict(X_test)

# Evaluate the MOdel
mse = mean_squared_error(y_test,y_pred)
r2 = r2_score(y_test,y_pred)

print(x)
print(y)

print(f"mean Squared Error: {mse}")
print(f"R2 Score: {r2}")

# Save the model
with open('linear_regression_model.pkl', 'wb') as file:
    pickle.dump(regressor, file)
    

# Load the train model
with open('linear_regression_model.pkl', 'rb') as file:
    model = pickle.load(file)
    
def display_model_score():
    score = model.score(X_test,y_test)
    st.write(f"score of model is : {score}")
    
def predict(input_data):
    prediction = model.predict(input_data)
    return prediction

def main():
    st.subheader("score of model at TEST data")
    display_model_score()
    
    st.header("input feature")
    feature1 = st.number_input("engagement_score",value=0.00)
    feature2 = st.number_input("experience_score",value=0.00)
    
    input_data = np.array([[feature1,feature2]])
    
    if st.button("predict"):
        prediction = predict(input_data)
        st.success(f"prediction value is : {prediction[0]}")
        
        
        
        
        
        
if __name__ == "__main__":
    main()      





