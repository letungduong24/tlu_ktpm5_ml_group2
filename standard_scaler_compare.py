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
from sklearn.model_selection import GridSearchCV

# Import data, khai báo features và target
ca_housing = fetch_california_housing()
X = ca_housing.data
y = ca_housing.target

# Các giá trị test_size để kiểm tra
test_sizes = [0.1, 0.2, 0.3, 0.4]
results_by_test_size = []

# Kiểm tra với các giá trị test_size khác nhau
for test_size in test_sizes:
    # Xử lý dữ liệu
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1)

    # Không chuẩn hóa
    # Huấn luyện các mô hình không chuẩn hóa
    linear = LinearRegression().fit(X_train, y_train)
    lasso = GridSearchCV(Lasso(), {'alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]}, cv=3).fit(X_train, y_train)
    mlp = MLPRegressor(hidden_layer_sizes=(50,), max_iter=500, early_stopping=True, validation_fraction=0.1, n_iter_no_change=10, random_state=1).fit(X_train, y_train)

    base_models = [
        ('linear', linear),
        ('lasso', lasso),
        ('mlp', mlp)
    ]

    stacking_regressor = StackingRegressor(estimators=base_models, final_estimator=LinearRegression(), n_jobs=-1).fit(X_train, y_train)

    # Predict và đánh giá không chuẩn hóa
    y_pred_lasso = lasso.predict(X_test)
    y_pred_linear = linear.predict(X_test)
    y_pred_mlp = mlp.predict(X_test)
    y_pred_stacking = stacking_regressor.predict(X_test)

    # Đánh giá mô hình không chuẩn hóa
    results_by_test_size.append({
        'test_size': test_size,
        'model': 'Without Scaling',
        'linear_MAE': mean_absolute_error(y_test, y_pred_linear),
        'linear_MSE': mean_squared_error(y_test, y_pred_linear),
        'linear_R2': r2_score(y_test, y_pred_linear),
        'lasso_MAE': mean_absolute_error(y_test, y_pred_lasso),
        'lasso_MSE': mean_squared_error(y_test, y_pred_lasso),
        'lasso_R2': r2_score(y_test, y_pred_lasso),
        'mlp_MAE': mean_absolute_error(y_test, y_pred_mlp),
        'mlp_MSE': mean_squared_error(y_test, y_pred_mlp),
        'mlp_R2': r2_score(y_test, y_pred_mlp),
        'stacking_MAE': mean_absolute_error(y_test, y_pred_stacking),
        'stacking_MSE': mean_squared_error(y_test, y_pred_stacking),
        'stacking_R2': r2_score(y_test, y_pred_stacking),
    })

    # Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Huấn luyện lại các mô hình với dữ liệu đã chuẩn hóa
    linear_scaled = LinearRegression().fit(X_train_scaled, y_train)
    lasso_scaled = GridSearchCV(Lasso(), {'alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]}, cv=3).fit(X_train_scaled, y_train)
    mlp_scaled = MLPRegressor(hidden_layer_sizes=(50,), max_iter=500, early_stopping=True, validation_fraction=0.1, n_iter_no_change=10, random_state=1).fit(X_train_scaled, y_train)

    base_models_scaled = [
        ('linear', linear_scaled),
        ('lasso', lasso_scaled),
        ('mlp', mlp_scaled)
    ]

    stacking_regressor_scaled = StackingRegressor(estimators=base_models_scaled, final_estimator=LinearRegression(), n_jobs=-1).fit(X_train_scaled, y_train)

    # Predict và đánh giá với dữ liệu đã chuẩn hóa
    y_pred_lasso_scaled = lasso_scaled.predict(X_test_scaled)
    y_pred_linear_scaled = linear_scaled.predict(X_test_scaled)
    y_pred_mlp_scaled = mlp_scaled.predict(X_test_scaled)
    y_pred_stacking_scaled = stacking_regressor_scaled.predict(X_test_scaled)

    # Đánh giá mô hình với dữ liệu đã chuẩn hóa
    results_by_test_size.append({
        'test_size': test_size,
        'model': 'With Scaling',
        'linear_MAE': mean_absolute_error(y_test, y_pred_linear_scaled),
        'linear_MSE': mean_squared_error(y_test, y_pred_linear_scaled),
        'linear_R2': r2_score(y_test, y_pred_linear_scaled),
        'lasso_MAE': mean_absolute_error(y_test, y_pred_lasso_scaled),
        'lasso_MSE': mean_squared_error(y_test, y_pred_lasso_scaled),
        'lasso_R2': r2_score(y_test, y_pred_lasso_scaled),
        'mlp_MAE': mean_absolute_error(y_test, y_pred_mlp_scaled),
        'mlp_MSE': mean_squared_error(y_test, y_pred_mlp_scaled),
        'mlp_R2': r2_score(y_test, y_pred_mlp_scaled),
        'stacking_MAE': mean_absolute_error(y_test, y_pred_stacking_scaled),
        'stacking_MSE': mean_squared_error(y_test, y_pred_stacking_scaled),
        'stacking_R2': r2_score(y_test, y_pred_stacking_scaled),
    })

# Chỉ giữ lại kết quả cho test_size = 0.2 để so sánh
results_for_comparison = [result for result in results_by_test_size if result['test_size'] == 0.2]

# Hiển thị kết quả theo test_size
results_df = pd.DataFrame(results_by_test_size)

# Giao diện nhập liệu
st.title("Dự đoán giá nhà với nhiều mô hình")

st.write("Kết quả đánh giá mô hình theo từng test_size:")
st.table(results_df)

# Hiển thị bảng so sánh cho test_size = 0.2
st.write("Kết quả so sánh giữa có và không chuẩn hóa (test_size = 0.2):")
comparison_df = pd.DataFrame(results_for_comparison)
st.table(comparison_df)

# Tiếp tục với phần nhập liệu và dự đoán như trước
st.write("Nhập dữ liệu để dự đoán:")
medInc = st.number_input("Median Income")
houseAge = st.number_input("House Age")
aveRooms = st.number_input("Average Rooms")
aveBedrms = st.number_input("Average Bedrooms")
population = st.number_input("Population")
aveOccup = st.number_input("Average Occupancy")
latitude = st.number_input("Latitude")
longitude = st.number_input("Longitude")

# Khi người dùng bấm nút Dự đoán
if st.button("Dự đoán"):
    new_data = [[medInc, houseAge, aveRooms, aveBedrms, population, aveOccup, latitude, longitude]]
    
    # Dự đoán với các mô hình không chuẩn hóa
    prediction_linear = linear.predict(new_data)[0]
    prediction_lasso = lasso.predict(new_data)[0]
    prediction_mlp = mlp.predict(new_data)[0]
    prediction_stacking = stacking_regressor.predict(new_data)[0]

    # Dự đoán với các mô hình đã chuẩn hóa
    new_data_scaled = scaler.transform(new_data)
    prediction_linear_scaled = linear_scaled.predict(new_data_scaled)[0]
    prediction_lasso_scaled = lasso_scaled.predict(new_data_scaled)[0]
    prediction_mlp_scaled = mlp_scaled.predict(new_data_scaled)[0]
    prediction_stacking_scaled = stacking_regressor_scaled.predict(new_data_scaled)[0]

    # Hiển thị kết quả dự đoán
    st.write("### Kết quả dự đoán (Không chuẩn hóa)")
    st.table(pd.DataFrame({
        'Model': ['Linear Regression', 'Lasso', 'MLP', 'Stacking'],
        'Dự đoán': [prediction_linear, prediction_lasso, prediction_mlp, prediction_stacking]
    }))
    
    st.write("### Kết quả dự đoán (Đã chuẩn hóa)")
    st.table(pd.DataFrame({
        'Model': ['Linear Regression', 'Lasso', 'MLP', 'Stacking'],
        'Dự đoán': [prediction_linear_scaled, prediction_lasso_scaled, prediction_mlp_scaled, prediction_stacking_scaled]
    }))



    # Hiển thị đồ thị sai số
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

    # Đồ thị so sánh giá trị thực và dự đoán
    st.write("### So sánh giá trị thực và dự đoán")

    fig, axs = plt.subplots(1, 4, figsize=(24, 6))

    # Đồ thị Lasso
    axs[0].scatter(y_test, y_pred_lasso, alpha=0.5)
    axs[0].plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
    axs[0].set_title('Lasso - So sánh giá trị')
    axs[0].set_xlabel('Giá trị thực tế')
    axs[0].set_ylabel('Giá trị dự đoán')

    # Đồ thị Linear Regression
    axs[1].scatter(y_test, y_pred_linear, alpha=0.5)
    axs[1].plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
    axs[1].set_title('Linear Regression - So sánh giá trị')
    axs[1].set_xlabel('Giá trị thực tế')
    axs[1].set_ylabel('Giá trị dự đoán')

    # Đồ thị MLP
    axs[2].scatter(y_test, y_pred_mlp, alpha=0.5)
    axs[2].plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
    axs[2].set_title('MLP - So sánh giá trị')
    axs[2].set_xlabel('Giá trị thực tế')
    axs[2].set_ylabel('Giá trị dự đoán')

    # Đồ thị Stacking
    axs[3].scatter(y_test, y_pred_stacking, alpha=0.5)
    axs[3].plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
    axs[3].set_title('Stacking - So sánh giá trị')
    axs[3].set_xlabel('Giá trị thực tế')
    axs[3].set_ylabel('Giá trị dự đoán')

    st.pyplot(fig)
