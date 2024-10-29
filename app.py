import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
from sklearn.model_selection import GridSearchCV

ca_housing = fetch_california_housing()
X = ca_housing.data
y = ca_housing.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

if not os.path.exists('data'):
    os.makedirs('data')

if not os.path.exists('data/california_housing.csv'):
    df = pd.DataFrame(ca_housing.data, columns=ca_housing.feature_names)
    df['MedHouseVal'] = ca_housing.target
    df.to_csv('data/california_housing.csv', index=False)
    print("Dataset saved to CSV")

if not os.path.exists('models'):
    os.makedirs('models')

def train_and_save_models():
    linear = LinearRegression()
    linear.fit(X_train_scaled, y_train)
    
    param_grid_lasso = {
        'alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
    }
    lasso = GridSearchCV(Lasso(), param_grid=param_grid_lasso, cv=3, n_jobs=-1)
    lasso.fit(X_train_scaled, y_train)
    
    mlp = MLPRegressor(
        hidden_layer_sizes=(50, 100),
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=25,
        random_state=42
    )
    mlp.fit(X_train_scaled, y_train)
    
    base_models = [
        ('linear', linear),
        ('lasso', lasso.best_estimator_),
        ('mlp', mlp)
    ]
    stacking_regressor = StackingRegressor(
        estimators=base_models,
        final_estimator=LinearRegression(),
        n_jobs=-1
    )
    stacking_regressor.fit(X_train_scaled, y_train)
    
    joblib.dump(linear, 'models/linear_model.pkl')
    joblib.dump(lasso, 'models/lasso_model.pkl')
    joblib.dump(mlp, 'models/mlp_model.pkl')
    joblib.dump(stacking_regressor, 'models/stacking_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    
    return linear, lasso, mlp, stacking_regressor

def load_models():
    linear = joblib.load('models/linear_model.pkl')
    lasso = joblib.load('models/lasso_model.pkl')
    mlp = joblib.load('models/mlp_model.pkl')
    stacking_regressor = joblib.load('models/stacking_model.pkl')
    return linear, lasso, mlp, stacking_regressor

if os.path.exists('models/linear_model.pkl'):
    linear, lasso, mlp, stacking_regressor = load_models()
else:
    linear, lasso, mlp, stacking_regressor = train_and_save_models()

def evaluate_model(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mae, mse, r2

def check_fit_condition(train_mse, test_mse):
    if test_mse > train_mse * 1.2:
        return "Overfitting"
    elif test_mse < train_mse * 0.8:
        return "Underfitting"
    else:
        return "Good Fit"

y_pred_linear = linear.predict(X_test_scaled)
y_pred_lasso = lasso.predict(X_test_scaled)
y_pred_mlp = mlp.predict(X_test_scaled)
y_pred_stacking = stacking_regressor.predict(X_test_scaled)

linearMAE, linearMSE, linearR2 = evaluate_model(y_test, y_pred_linear)
lassoMAE, lassoMSE, lassoR2 = evaluate_model(y_test, y_pred_lasso)
mlpMAE, mlpMSE, mlpR2 = evaluate_model(y_test, y_pred_mlp)
stackingMAE, stackingMSE, stackingR2 = evaluate_model(y_test, y_pred_stacking)

train_mse_linear = mean_squared_error(y_train, linear.predict(X_train_scaled))
train_mse_lasso = mean_squared_error(y_train, lasso.predict(X_train_scaled))
train_mse_mlp = mean_squared_error(y_train, mlp.predict(X_train_scaled))
train_mse_stacking = mean_squared_error(y_train, stacking_regressor.predict(X_train_scaled))

fit_condition_linear = check_fit_condition(train_mse_linear, linearMSE)
fit_condition_lasso = check_fit_condition(train_mse_lasso, lassoMSE)
fit_condition_mlp = check_fit_condition(train_mse_mlp, mlpMSE)
fit_condition_stacking = check_fit_condition(train_mse_stacking, stackingMSE)

st.title("Dự đoán giá nhà California Housing")

df = pd.DataFrame(ca_housing.data, columns=ca_housing.feature_names)
feature_ranges = df.describe()

medInc = st.number_input("Median Income", 
                        min_value=float(feature_ranges['MedInc']['min']),
                        max_value=float(feature_ranges['MedInc']['max']),
                        )

houseAge = st.number_input("House Age",
                          min_value=float(feature_ranges['HouseAge']['min']),
                          max_value=float(feature_ranges['HouseAge']['max']),
                          )

aveRooms = st.number_input("Average Rooms",
                          min_value=float(feature_ranges['AveRooms']['min']),
                          max_value=float(feature_ranges['AveRooms']['max']),
                          )

aveBedrms = st.number_input("Average Bedrooms",
                           min_value=float(feature_ranges['AveBedrms']['min']),
                           max_value=float(feature_ranges['AveBedrms']['max']),
                           )

population = st.number_input("Population",
                           min_value=float(feature_ranges['Population']['min']),
                           max_value=float(feature_ranges['Population']['max']),
                           )

aveOccup = st.number_input("Average Occupancy",
                          min_value=float(feature_ranges['AveOccup']['min']),
                          max_value=float(feature_ranges['AveOccup']['max']),
                          )

latitude = st.number_input("Latitude",
                         min_value=float(feature_ranges['Latitude']['min']),
                         max_value=float(feature_ranges['Latitude']['max']),
                         )

longitude = st.number_input("Longitude",
                          min_value=float(feature_ranges['Longitude']['min']),
                          max_value=float(feature_ranges['Longitude']['max']),
                          )

if st.button("Predict"):
    new_data = [[medInc, houseAge, aveRooms, aveBedrms, population, aveOccup, latitude, longitude]]
    
    new_data_scaled = scaler.transform(new_data)
    
    prediction_linear = linear.predict(new_data_scaled)[0]
    prediction_lasso = lasso.predict(new_data_scaled)[0]
    prediction_mlp = mlp.predict(new_data_scaled)[0]
    prediction_stacking = stacking_regressor.predict(new_data_scaled)[0]
    
    st.write("### Prediction Results")
    st.write("Note: Values are in units of $100,000")
    predictions_df = pd.DataFrame({
        'Model': ['Linear Regression', 'Lasso', 'MLP', 'Stacking'],
        'Predicted Price': [prediction_linear, prediction_lasso, prediction_mlp, prediction_stacking]
    })
    st.table(predictions_df)
    
    st.write("### Model Evaluation")
    metrics_df = pd.DataFrame({
        'Model': ['Linear Regression', 'Lasso', 'MLP', 'Stacking'],
        'MAE': [linearMAE, lassoMAE, mlpMAE, stackingMAE],
        'MSE': [linearMSE, lassoMSE, mlpMSE, stackingMSE],
        'R²': [linearR2, lassoR2, mlpR2, stackingR2],
        'Fit Condition': [fit_condition_linear, fit_condition_lasso, fit_condition_mlp, fit_condition_stacking]
    })
    st.table(metrics_df)
    
    st.write("### Error Distribution Plots")
    errors_linear = y_test - y_pred_linear
    errors_lasso = y_test - y_pred_lasso
    errors_mlp = y_test - y_pred_mlp
    errors_stacking = y_test - y_pred_stacking
    
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Error Distributions for Different Models')
    
    axs[0, 0].hist(errors_linear, bins=50, edgecolor='black')
    axs[0, 0].set_title('Linear Regression')
    axs[0, 0].set_xlabel('Error')
    axs[0, 0].set_ylabel('Frequency')
    
    axs[0, 1].hist(errors_lasso, bins=50, edgecolor='black')
    axs[0, 1].set_title('Lasso')
    axs[0, 1].set_xlabel('Error')
    axs[0, 1].set_ylabel('Frequency')
    
    axs[1, 0].hist(errors_mlp, bins=50, edgecolor='black')
    axs[1, 0].set_title('MLP')
    axs[1, 0].set_xlabel('Error')
    axs[1, 0].set_ylabel('Frequency')
    
    axs[1, 1].hist(errors_stacking, bins=50, edgecolor='black')
    axs[1, 1].set_title('Stacking')
    axs[1, 1].set_xlabel('Error')
    axs[1, 1].set_ylabel('Frequency')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    st.write("### Actual vs Predicted Values")
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Actual vs Predicted Values for Different Models')
    
    axs[0, 0].scatter(y_test, y_pred_linear, alpha=0.5)
    axs[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    axs[0, 0].set_title('Linear Regression')
    axs[0, 0].set_xlabel('Actual Values')
    axs[0, 0].set_ylabel('Predicted Values')
    
    axs[0, 1].scatter(y_test, y_pred_lasso, alpha=0.5)
    axs[0, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    axs[0, 1].set_title('Lasso')
    axs[0, 1].set_xlabel('Actual Values')
    axs[0, 1].set_ylabel('Predicted Values')
    
    axs[1, 0].scatter(y_test, y_pred_mlp, alpha=0.5)
    axs[1, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    axs[1, 0].set_title('MLP')
    axs[1, 0].set_xlabel('Actual Values')
    axs[1, 0].set_ylabel('Predicted Values')
    
    axs[1, 1].scatter(y_test, y_pred_stacking, alpha=0.5)
    axs[1, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    axs[1, 1].set_title('Stacking')
    axs[1, 1].set_xlabel('Actual Values')
    axs[1, 1].set_ylabel('Predicted Values')
    
    plt.tight_layout()
    st.pyplot(fig)
