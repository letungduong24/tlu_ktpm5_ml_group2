import streamlit as st
import pandas as pd
import joblib
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Import data, khai báo features và target
ca_housing = fetch_california_housing()
X = ca_housing.data
y = ca_housing.target

# Xử lý dữ liệu
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Danh sách các giá trị cho validation_fraction
validation_fractions = [0.1, 0.2, 0.3, 0.4, 0.5]

# Tạo danh sách để lưu kết quả
results = []

# Vòng lặp qua các giá trị validation_fraction
for fraction in validation_fractions:
    mlp = MLPRegressor(
        hidden_layer_sizes=(50,),
        max_iter=500,
        early_stopping=True,
        validation_fraction=fraction,
        n_iter_no_change=10,
        random_state=1
    )
    
    # Huấn luyện mô hình
    mlp.fit(X_train, y_train)
    
    # Dự đoán trên tập test
    y_pred = mlp.predict(X_test)
    
    # Tính toán các chỉ số hiệu suất
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Lưu kết quả
    results.append({
        'validation_fraction': fraction,
        'MSE': mse,
        'MAE': mae,
        'R²': r2
    })

# Chuyển đổi kết quả thành DataFrame và hiển thị
results_df = pd.DataFrame(results)
st.write("Kết quả cho các giá trị validation_fraction:")
st.table(results_df)
