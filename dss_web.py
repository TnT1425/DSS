import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from pulp import *
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

st.set_page_config(page_title="DSS Đồ Uống", page_icon="🥤", layout="wide")
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

st.sidebar.title("Tùy chọn dữ liệu")
uploaded_file = st.sidebar.file_uploader("Chọn file dữ liệu CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df['Purchase Date'] = pd.to_datetime(df['Purchase Date'])
    df = df.sort_values('Purchase Date')
    df['Month'] = df['Purchase Date'].dt.to_period('M').dt.to_timestamp()

    st.title("🥤 DSS dự báo & tối ưu hóa sản phẩm đồ uống")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 📊 Thống kê mô tả")
        stats = df.groupby('Product').agg({
            'Units Sold': ['count', 'mean', 'std', 'min', 'max'],
            'Profit': ['mean', 'sum']
        }).round(2)
        st.dataframe(stats)
    with col2:
        st.markdown("### 📈 Biểu đồ doanh số")
        fig, ax = plt.subplots(figsize=(5,4))
        for p in df['Product'].unique():
            sub = df[df['Product']==p]
            ax.plot(sub['Month'], sub['Units Sold'], label=p)
        ax.legend()
        ax.set_title("Doanh số theo thời gian")
        st.pyplot(fig)

    st.markdown("---")
    st.markdown("## 🔮 Dự báo doanh số")
    products = st.sidebar.multiselect("Chọn sản phẩm để dự báo", df['Product'].unique(), default=['Cola', 'Beer', 'Green Tea'])
    periods = st.sidebar.slider("Số tháng dự báo", 1, 12, 6)
    demands = {}
    for product in products:
        with st.expander(f"Dự báo cho {product}"):
            sub = df[df['Product']==product][['Month','Units Sold']].rename(
                columns={'Month':'ds','Units Sold':'y'})
            model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
            model.fit(sub)
            future = model.make_future_dataframe(periods=periods, freq='M')
            forecast = model.predict(future)
            y_true = sub['y'].values[-periods:]
            y_pred = forecast['yhat'].values[-2*periods:-periods]
            mae = mean_absolute_error(y_true, y_pred) if len(y_true)==len(y_pred) else None
            rmse = np.sqrt(mean_squared_error(y_true, y_pred)) if len(y_true)==len(y_pred) else None
            st.write(f"MAE: {mae:.2f}" if mae else "Không đủ dữ liệu để tính MAE")
            st.write(f"RMSE: {rmse:.2f}" if rmse else "Không đủ dữ liệu để tính RMSE")
            st.dataframe(forecast[['ds','yhat', 'yhat_lower', 'yhat_upper']].tail(periods))
            fig2 = model.plot(forecast)
            st.pyplot(fig2)
            demands[product] = int(forecast['yhat'].values[-1])

    st.markdown("---")
    st.markdown("## ⚙️ Tối ưu hóa sản xuất")
    profits = {'Cola': 12, 'Beer': 15, 'Green Tea': 10}
    resources = {
        'Water': {'Cola': 0.8, 'Beer': 0.85, 'Green Tea': 0.9},
        'Sugar': {'Cola': 0.15, 'Beer': 0.05, 'Green Tea': 0.1}
    }
    resource_limits = {
        'Water': st.sidebar.number_input("Giới hạn nước (lít)", value=1500),
        'Sugar': st.sidebar.number_input("Giới hạn đường (kg)", value=200)
    }

    if st.button("Tính tối ưu"):
        model = LpProblem("soft-drink-optimization", LpMaximize)
        vars = {p: LpVariable(p, lowBound=0, upBound=demands.get(p,0), cat="Integer") for p in products}
        model += lpSum(profits.get(p,0) * vars[p] for p in vars)
        model += lpSum(vars[p] for p in vars) <= 2000
        for resource, limit in resource_limits.items():
            model += lpSum(resources[resource][p] * vars[p] for p in vars) <= limit
        model.solve()
        st.subheader("Kế hoạch sản xuất tối ưu")
        for p in vars:
            st.write(f"{p}: {int(vars[p].value())} (chai)")
        st.write(f"Lợi nhuận tối đa = {model.objective.value():,.0f} (VND)")
else:
    st.info("Vui lòng tải lên file dữ liệu CSV để bắt đầu.")