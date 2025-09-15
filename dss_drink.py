import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
from pulp import *
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# B1: Đọc và tiền xử lý dữ liệu
def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df['Purchase Date'] = pd.to_datetime(df['Purchase Date'])
    df = df.sort_values('Purchase Date')
    df['Month'] = df['Purchase Date'].dt.to_period('M').dt.to_timestamp()
    return df

# B2: Phân tích dữ liệu
def analyze_data(df):
    # Thống kê mô tả
    stats = df.groupby('Product').agg({
        'Units Sold': ['count', 'mean', 'std', 'min', 'max'],
        'Profit': ['mean', 'sum']
    }).round(2)
    
    print("\nThống kê theo sản phẩm:")
    print(stats)
    
    # Vẽ biểu đồ doanh số theo thời gian
    plt.figure(figsize=(12,6))
    for p in df['Product'].unique():
        sub = df[df['Product']==p]
        plt.plot(sub['Month'], sub['Units Sold'], label=p)
    
    plt.legend()
    plt.title("Doanh số theo thời gian")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# B3: Dự báo với Prophet
def forecast_sales(df, product, periods=6):
    sub = df[df['Product']==product][['Month','Units Sold']].rename(
        columns={'Month':'ds','Units Sold':'y'})
    
    model = Prophet(yearly_seasonality=True, 
                   weekly_seasonality=False,
                   daily_seasonality=False)
    model.fit(sub)
    
    future = model.make_future_dataframe(periods=periods, freq='M')
    forecast = model.predict(future)
    
    # Đánh giá độ chính xác
    y_true = sub['y'].values[-periods:]
    y_pred = forecast['yhat'].values[-2*periods:-periods]
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    print(f"\nKết quả dự báo cho {product}:")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print("\nDự báo 6 tháng tới:")
    print(forecast[['ds','yhat', 'yhat_lower', 'yhat_upper']].tail(periods))
    
    # Vẽ biểu đồ dự báo
    model.plot(forecast)
    plt.title(f"Dự báo doanh số {product}")
    plt.show()
    
    return forecast['yhat'].values[-periods:]

# B4: Tối ưu hóa sản xuất
def optimize_production(demands, profits, resources=None):
    # Tạo mô hình
    model = LpProblem("soft-drink-optimization", LpMaximize)
    
    # Biến quyết định
    vars = {p: LpVariable(p, lowBound=0, upBound=demands[p], cat="Integer") 
            for p in demands}
    
    # Hàm mục tiêu
    model += lpSum(profits[p] * vars[p] for p in vars)
    
    # Ràng buộc công suất
    model += lpSum(vars[p] for p in vars) <= 2000
    
    # Ràng buộc nguyên liệu (nếu có)
    if resources:
        for resource, limit in resources.items():
            model += lpSum(resources[resource][p] * vars[p] 
                         for p in vars) <= limit
    
    # Giải bài toán
    model.solve()
    
    print("\n=== Kế hoạch sản xuất tối ưu ===")
    for p in vars:
        print(f"{p}: {int(vars[p].value())} (chai)")
    print(f"Lợi nhuận tối đa = {model.objective.value():,.0f} (VND)")

def main():
    # Đọc dữ liệu
    df = load_and_preprocess_data("soft_drink_sales.csv")
    
    # Phân tích dữ liệu
    analyze_data(df)
    
    # Dự báo cho từng sản phẩm
    products = ['Cola', 'Beer', 'Green Tea'] 
    demands = {}
    for p in products:
        forecast = forecast_sales(df, p)
        demands[p] = int(forecast[-1])
    
    # Tối ưu hóa với các ràng buộc
    profits = {'Cola': 12, 'Beer': 15, 'Green Tea': 10}
    resources = {
        'Water': {'Cola': 0.8, 'Beer': 0.85, 'Green Tea': 0.9},
        'Sugar': {'Cola': 0.15, 'Beer': 0.05, 'Green Tea': 0.1}
    }
    resource_limits = {
        'Water': 1500,  # lít
        'Sugar': 200    # kg
    }
    
    optimize_production(demands, profits, resource_limits)

if __name__ == "__main__":
    main()
