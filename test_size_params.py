import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Import dữ liệu
ca_housing = fetch_california_housing()
X = ca_housing.data
y = ca_housing.target

# Các giá trị test_size muốn thử
test_sizes = [0.1, 0.2, 0.3, 0.4, 0.5]

# Khởi tạo các biến để lưu kết quả
results = {
    "test_size": [],
    "linear_MSE": [],
    "lasso_MSE": [],
    "mlp_MSE": [],
    "stacking_MSE": []
}

# Vòng lặp qua các giá trị test_size
for test_size in test_sizes:
    # Chia dữ liệu train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1)
    
    # Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Huấn luyện mô hình
    linear = LinearRegression()
    linear.fit(X_train, y_train)
    
    lasso = Lasso(alpha=0.1)
    lasso.fit(X_train, y_train)

    mlp = MLPRegressor(hidden_layer_sizes=(50,), max_iter=500, early_stopping=True, validation_fraction=0.1, n_iter_no_change=10, random_state=1)
    mlp.fit(X_train, y_train)

    base_models = [('linear', linear), ('lasso', lasso), ('mlp', mlp)]
    stacking_regressor = StackingRegressor(estimators=base_models, final_estimator=LinearRegression(), n_jobs=-1)
    stacking_regressor.fit(X_train, y_train)

    # Dự đoán trên tập test
    y_pred_linear = linear.predict(X_test)
    y_pred_lasso = lasso.predict(X_test)
    y_pred_mlp = mlp.predict(X_test)
    y_pred_stacking = stacking_regressor.predict(X_test)

    # Đánh giá các mô hình
    linear_mse = mean_squared_error(y_test, y_pred_linear)
    lasso_mse = mean_squared_error(y_test, y_pred_lasso)
    mlp_mse = mean_squared_error(y_test, y_pred_mlp)
    stacking_mse = mean_squared_error(y_test, y_pred_stacking)

    # Lưu kết quả
    results["test_size"].append(test_size)
    results["linear_MSE"].append(linear_mse)
    results["lasso_MSE"].append(lasso_mse)
    results["mlp_MSE"].append(mlp_mse)
    results["stacking_MSE"].append(stacking_mse)

# Hiển thị kết quả dưới dạng bảng trong Streamlit
st.write("### Kết quả với các test_size khác nhau")
st.table(pd.DataFrame(results))

# Vẽ biểu đồ MSE của các mô hình theo test_size
plt.figure(figsize=(10, 6))
plt.plot(results["test_size"], results["linear_MSE"], marker='o', label='Linear Regression')
plt.plot(results["test_size"], results["lasso_MSE"], marker='o', label='Lasso')
plt.plot(results["test_size"], results["mlp_MSE"], marker='o', label='MLP')
plt.plot(results["test_size"], results["stacking_MSE"], marker='o', label='Stacking')
plt.xlabel('Test Size')
plt.ylabel('MSE')
plt.title('MSE theo các giá trị Test Size')
plt.legend()
st.pyplot(plt)
