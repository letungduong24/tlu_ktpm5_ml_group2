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

# Import data, khai báo features và target
ca_housing = fetch_california_housing()
X = ca_housing.data
y = ca_housing.target

# Xử lý dữ liệu
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Load mô hình
lasso = joblib.load('models/lasso_model.pkl')
linear = joblib.load('models/linear_model.pkl')
mlp = joblib.load('models/mlp_model.pkl')
stacking_regressor = joblib.load('models/stacking_model.pkl')

# Predict với tập dữ liệu test
y_pred_lasso = lasso.predict(X_test)
y_pred_linear = linear.predict(X_test)
y_pred_mlp = mlp.predict(X_test)
y_pred_stacking = stacking_regressor.predict(X_test)

# Đánh giá mô hình
def evaluate_model(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mae, mse, r2

# Đánh giá các mô hình
lassoMAE, lassoMSE, lassoR2 = evaluate_model(y_test, y_pred_lasso)
linearMAE, linearMSE, linearR2 = evaluate_model(y_test, y_pred_linear)
mlpMAE, mlpMSE, mlpR2 = evaluate_model(y_test, y_pred_mlp)
stackingMAE, stackingMSE, stackingR2 = evaluate_model(y_test, y_pred_stacking)

# Giao diện nhập liệu
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

# Khi người dùng bấm nút Dự đoán
if st.button("Dự đoán"):
    new_data = [[medInc, houseAge, aveRooms, aveBedrms, population, aveOccup, latitude, longitude]]
    
    # Dự đoán với các mô hình
    prediction_linear = linear.predict(new_data)[0]
    prediction_lasso = lasso.predict(new_data)[0]
    prediction_mlp = mlp.predict(new_data)[0]
    prediction_stacking = stacking_regressor.predict(new_data)[0]

    # Hiển thị kết quả dự đoán
    st.write("### Kết quả dự đoán")
    st.table(pd.DataFrame({
        'Model': ['Linear Regression', 'Lasso', 'MLP', 'Stacking'],
        'Dự đoán': [prediction_linear, prediction_lasso, prediction_mlp, prediction_stacking]
    }))

    # Hiển thị đánh giá mô hình dưới dạng bảng
    st.write("### Đánh giá mô hình")
    st.table(pd.DataFrame({
        'Model': ['Linear Regression', 'Lasso', 'MLP', 'Stacking'],
        'MAE': [linearMAE, lassoMAE, mlpMAE, stackingMAE],
        'MSE': [linearMSE, lassoMSE, mlpMSE, stackingMSE],
        'R²': [linearR2, lassoR2, mlpR2, stackingR2]
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

    plt.tight_layout()
    st.pyplot(fig)
