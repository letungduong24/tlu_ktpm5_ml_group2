import streamlit as st
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os

# Import data, khai báo features và target
ca_housing = fetch_california_housing()
X = ca_housing.data
y = ca_housing.target

# Xử lý dữ liệu
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Huấn luyện Linear Regression
linear = LinearRegression()
linear.fit(X_train, y_train)

# Huấn luyện Lasso
lasso = Lasso()
lasso.fit(X_train, y_train)

# Khởi tạo danh sách để lưu kết quả cho MLPRegressor
mlp_results = []

# Thử nhiều giá trị cho n_iter_no_change
n_iter_no_change_values = [5, 10, 15, 20, 25]

for n_iter in n_iter_no_change_values:
    mlp = MLPRegressor(
        hidden_layer_sizes=(50,),
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=n_iter,
        random_state=1
    )
    
    # Huấn luyện mô hình MLP
    mlp.fit(X_train, y_train)
    
    # Dự đoán trên tập test
    y_pred = mlp.predict(X_test)
    
    # Tính toán các chỉ số hiệu suất
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Lưu kết quả vào danh sách
    mlp_results.append({
        'n_iter_no_change': n_iter,
        'MSE': mse,
        'MAE': mae,
        'R²': r2
    })

# Chuyển đổi kết quả thành DataFrame
results_df = pd.DataFrame(mlp_results)

# Hiển thị kết quả
st.title("Kết quả thử nghiệm với MLPRegressor")
st.write("Kết quả cho các giá trị n_iter_no_change:")
st.table(results_df)

# Huấn luyện mô hình Stacking Regressor
base_models = [
    ('linear', linear),
    ('lasso', lasso),
    ('mlp', mlp)
]

stacking_regressor = StackingRegressor(estimators=base_models, final_estimator=LinearRegression(), n_jobs=-1)
stacking_regressor.fit(X_train, y_train)

# Lưu các mô hình đã huấn luyện
if not os.path.exists('models'):
    os.makedirs('models')

joblib.dump(linear, 'models/linear_model.pkl')
joblib.dump(lasso, 'models/lasso_model.pkl')
joblib.dump(mlp, 'models/mlp_model.pkl')
joblib.dump(stacking_regressor, 'models/stacking_model.pkl')
