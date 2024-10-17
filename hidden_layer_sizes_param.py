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

# Huấn luyện mô hình tuyến tính
linear = LinearRegression()
linear.fit(X_train, y_train)

# Huấn luyện mô hình Lasso
param_grid_lasso = {
    'alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
}
lasso = Lasso(alpha=0.01)  # Sử dụng alpha tốt nhất đã xác định trước
lasso.fit(X_train, y_train)

# Danh sách các giá trị hidden_layer_sizes để thử nghiệm
hidden_layer_sizes_list = [(50,), (100,), (50, 50), (100, 50), (50, 100), (100, 100), (100, 150), (150, 200), (250, 300), (350, 400), (350, 350), (400, 400),  (450, 500)]
results = []

# Vòng lặp qua các giá trị hidden_layer_sizes
for hidden_layer_sizes in hidden_layer_sizes_list:
    mlp = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        random_state=1
    )
    mlp.fit(X_train, y_train)

    # Dự đoán trên tập kiểm tra
    y_pred = mlp.predict(X_test)

    # Tính toán các chỉ số hiệu suất
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Lưu kết quả vào danh sách
    results.append({
        'hidden_layer_sizes': hidden_layer_sizes,
        'MSE': mse,
        'MAE': mae,
        'R²': r2
    })

# Chuyển kết quả sang DataFrame
results_df = pd.DataFrame(results)

# Hiển thị kết quả trên Streamlit
st.title("Kết quả thử nghiệm MLPRegressor")
st.dataframe(results_df)

# Huấn luyện mô hình Stacking
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
