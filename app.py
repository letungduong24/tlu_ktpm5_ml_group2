import streamlit as st
import pandas as pd
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
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

if os.path.exists('models/lasso_model.pkl'):
    lasso = joblib.load('models/lasso_model.pkl')
    linear = joblib.load('models/linear_model.pkl')
    mlp = joblib.load('models/mlp_model.pkl')
    stacking_regressor = joblib.load('models/stacking_model.pkl')
else:
    linear = LinearRegression()
    linear.fit(X_train, y_train)

    param_grid_lasso = {
        'alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
    }

    lasso = GridSearchCV(Lasso(), param_grid=param_grid_lasso, cv=3, n_jobs=-1)
    lasso.fit(X_train, y_train)

    mlp = MLPRegressor(
        hidden_layer_sizes=(50, 100),      
        max_iter=500,                      
        early_stopping=True,                 
        validation_fraction=0.1,             
        n_iter_no_change=25,                 
        random_state=42                     
    )
    mlp.fit(X_train, y_train)

    base_models = [
        ('linear', linear),
        ('lasso', lasso.best_estimator_),
        ('mlp', mlp)
    ]

    stacking_regressor = StackingRegressor(estimators=base_models, final_estimator=LinearRegression(), n_jobs=-1)
    stacking_regressor.fit(X_train, y_train)
    
    if not os.path.exists('models'):
        os.makedirs('models')
    
    joblib.dump(linear, 'models/linear_model.pkl')
    joblib.dump(lasso, 'models/lasso_model.pkl')
    joblib.dump(mlp, 'models/mlp_model.pkl')
    joblib.dump(stacking_regressor, 'models/stacking_model.pkl')

y_pred_lasso = lasso.predict(X_test)
y_pred_linear = linear.predict(X_test)
y_pred_mlp = mlp.predict(X_test)
y_pred_stacking = stacking_regressor.predict(X_test)

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

lassoMAE, lassoMSE, lassoR2 = evaluate_model(y_test, y_pred_lasso)
linearMAE, linearMSE, linearR2 = evaluate_model(y_test, y_pred_linear)
mlpMAE, mlpMSE, mlpR2 = evaluate_model(y_test, y_pred_mlp)
stackingMAE, stackingMSE, stackingR2 = evaluate_model(y_test, y_pred_stacking)

train_mse_lasso = mean_squared_error(y_train, lasso.predict(X_train))
train_mse_linear = mean_squared_error(y_train, linear.predict(X_train))
train_mse_mlp = mean_squared_error(y_train, mlp.predict(X_train))
train_mse_stacking = mean_squared_error(y_train, stacking_regressor.predict(X_train))

fit_condition_lasso = check_fit_condition(train_mse_lasso, lassoMSE)
fit_condition_linear = check_fit_condition(train_mse_linear, linearMSE)
fit_condition_mlp = check_fit_condition(train_mse_mlp, mlpMSE)
fit_condition_stacking = check_fit_condition(train_mse_stacking, stackingMSE)

st.title("Dự đoán giá nhà với nhiều mô hình")

st.write("Nhập dữ liệu để dự đoán:")
medInc = st.number_input("Median Income")
houseAge = st.number_input("House Age")
aveRooms = st.number_input("Average Rooms")
aveBedrms = st.number_input("Average Bedrooms")
population = st.number_input("Population")
aveOccup = st.number_input("Average Occupancy")
latitude = st.number_input("Latitude")
longitude = st.number_input("Longitude")

if st.button("Dự đoán"):
    new_data = [[medInc, houseAge, aveRooms, aveBedrms, population, aveOccup, latitude, longitude]]
    
    prediction_linear = linear.predict(new_data)[0]
    prediction_lasso = lasso.predict(new_data)[0]
    prediction_mlp = mlp.predict(new_data)[0]
    prediction_stacking = stacking_regressor.predict(new_data)[0]

    st.write("### Kết quả dự đoán")
    st.table(pd.DataFrame({
        'Model': ['Linear Regression', 'Lasso', 'MLP', 'Stacking'],
        'Dự đoán': [prediction_linear, prediction_lasso, prediction_mlp, prediction_stacking]
    }))

    st.write("### Đánh giá mô hình")
    st.table(pd.DataFrame({
        'Model': ['Linear Regression', 'Lasso', 'MLP', 'Stacking'],
        'MAE': [linearMAE, lassoMAE, mlpMAE, stackingMAE],
        'MSE': [linearMSE, lassoMSE, mlpMSE, stackingMSE],
        'R²': [linearR2, lassoR2, mlpR2, stackingR2],
        'Fit Condition': [fit_condition_linear, fit_condition_lasso, fit_condition_mlp, fit_condition_stacking]
    }))

    st.write("### Đồ thị sai số")
    errors_lasso = y_test - y_pred_lasso
    errors_linear = y_test - y_pred_linear
    errors_mlp = y_test - y_pred_mlp
    errors_stacking = y_test - y_pred_stacking

    fig, axs = plt.subplots(1, 4, figsize=(24, 8))

    axs[0].hist(errors_lasso, bins=50, edgecolor='k')
    axs[0].set_title('Lasso - Phân phối sai số')
    axs[0].set_xlabel('Error')
    axs[0].set_ylabel('Frequency')

    axs[1].hist(errors_linear, bins=50, edgecolor='k')
    axs[1].set_title('Linear Regression - Phân phối sai số')
    axs[1].set_xlabel('Error')
    axs[1].set_ylabel('Frequency')

    axs[2].hist(errors_mlp, bins=50, edgecolor='k')
    axs[2].set_title('MLP - Phân phối sai số')
    axs[2].set_xlabel('Error')
    axs[2].set_ylabel('Frequency')

    axs[3].hist(errors_stacking, bins=50, edgecolor='k')
    axs[3].set_title('Stacking - Phân phối sai số')
    axs[3].set_xlabel('Error')
    axs[3].set_ylabel('Frequency')

    st.pyplot(fig)

    st.write("### So sánh giá trị thực và dự đoán")
    fig2, ax = plt.subplots()
    ax.scatter(y_test, y_pred_lasso, color='blue', label='Lasso', alpha=0.5)
    ax.scatter(y_test, y_pred_linear, color='green', label='Linear', alpha=0.5)
    ax.scatter(y_test, y_pred_mlp, color='red', label='MLP', alpha=0.5)
    ax.scatter(y_test, y_pred_stacking, color='orange', label='Stacking', alpha=0.5)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
    ax.set_xlabel('Giá trị thực')
    ax.set_ylabel('Giá trị dự đoán')
    ax.set_title('So sánh giá trị thực và dự đoán')
    ax.legend()
    st.pyplot(fig2)
