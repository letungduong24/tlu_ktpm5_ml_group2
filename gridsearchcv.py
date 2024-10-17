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

# Import data, khai báo features và target
ca_housing = fetch_california_housing()
X = ca_housing.data
y = ca_housing.target

# Xử lý dữ liệu
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


linear = LinearRegression()
linear.fit(X_train, y_train)

# Tạo danh sách các giá trị cv để thử nghiệm
cv_values = [2, 3, 5, 10]
lasso_results = []

    # Huấn luyện và lưu kết quả cho từng giá trị cv
for cv in cv_values:
    param_grid_lasso = {
        'alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
    }
    lasso = GridSearchCV(Lasso(), param_grid=param_grid_lasso, cv=cv, n_jobs=-1)
    lasso.fit(X_train, y_train)

        # Lưu kết quả
    best_score = lasso.best_score_
    best_params = lasso.best_params_
    lasso_results.append((cv, best_params, best_score))

    # Hiển thị kết quả cho các giá trị cv
    st.write("### Kết quả GridSearchCV cho Lasso với các giá trị cv khác nhau")
    lasso_results_df = pd.DataFrame(lasso_results, columns=['CV', 'Best Params', 'Best Score'])
    st.table(lasso_results_df)

    # Huấn luyện mô hình Lasso với tham số tốt nhất
    best_cv, best_lasso_params, _ = max(lasso_results, key=lambda x: x[2])
    lasso = Lasso(**best_lasso_params)
    lasso.fit(X_train, y_train)

    # Huấn luyện MLP
mlp = MLPRegressor(
hidden_layer_sizes=(50,),      # Kích thước lớp ẩn
    max_iter=500,                  # Số vòng lặp tối đa
    early_stopping=True,           # Bật early stopping
    validation_fraction=0.1,       # Tỷ lệ dữ liệu dùng để validation
    n_iter_no_change=10,           # Số vòng lặp không thay đổi hiệu suất trước khi dừng
    random_state=1                 # Hạt giống để tái lập kết quả
)
mlp.fit(X_train, y_train)

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
    
