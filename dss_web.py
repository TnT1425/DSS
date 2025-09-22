import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from pulp import *
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from dss_style import set_custom_style
import plotly.express as px

set_custom_style()

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

    # --- Bộ lọc sản phẩm và thời gian ---
    product_options = df['Product'].unique()
    selected_products = st.sidebar.multiselect("Chọn sản phẩm", product_options, default=list(product_options))
    min_date = df['Purchase Date'].min()
    max_date = df['Purchase Date'].max()
    selected_date = st.sidebar.date_input("Chọn khoảng thời gian", [min_date, max_date])

    # Lọc dữ liệu theo sản phẩm và thời gian
    filtered_df = df[
        (df['Product'].isin(selected_products)) &
        (df['Purchase Date'] >= pd.to_datetime(selected_date[0])) &
        (df['Purchase Date'] <= pd.to_datetime(selected_date[1]))
    ]

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 📊 Thống kê mô tả")
        stats = filtered_df.groupby('Product').agg({
            'Units Sold': ['count', 'mean', 'std', 'min', 'max'],
            'Profit': ['mean', 'sum']
        }).round(2)
        st.dataframe(stats)
    with col2:
        st.markdown("### 📈 Biểu đồ doanh số (Line chart)")
        if not filtered_df.empty:
            # Tổng hợp số lượng bán theo tháng và sản phẩm
            plot_df = (
                filtered_df.groupby(['Month', 'Product'], as_index=False)['Units Sold'].sum()
            )

            fig = px.line(
                plot_df,
                x="Month",
                y="Units Sold",
                color="Product",
                markers=True,
                labels={
                    "Month": "Thời gian (tháng)",
                    "Units Sold": "Số lượng bán",
                    "Product": "Sản phẩm"
                },
                title="Doanh số bán theo thời gian"
            )
            fig.update_layout(
                legend_title_text='Sản phẩm',
                xaxis_title="Thời gian (tháng)",
                yaxis_title="Số lượng bán",
                hovermode="x unified",
                template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Không có dữ liệu để hiển thị biểu đồ.")

    st.markdown("---")
    st.markdown("## 🔮 Dự báo doanh số")
    products = st.sidebar.multiselect(
        "Chọn sản phẩm để dự báo",
        df['Product'].unique(),
        default=['Cola', 'Beer', 'Green Tea']
    )

    # Cho phép chọn số tháng dự báo dài hơn (tối đa 36 tháng)
    periods = st.sidebar.slider("Số tháng dự báo", 1, 36, 12)

    demands = {}
    for product in products:
        with st.expander(f"Dự báo cho {product}"):
            # Tổng hợp số lượng bán theo tháng cho sản phẩm
            sub = (
                df[df['Product'] == product]
                .groupby('Month', as_index=False)['Units Sold'].sum()
                .rename(columns={'Month': 'ds', 'Units Sold': 'y'})
            )
            if len(sub) > 1:
                model = Prophet(
                    yearly_seasonality=True,
                    weekly_seasonality=False,
                    daily_seasonality=False
                )
                model.fit(sub)

                # Tạo future dataframe kéo dài thêm `periods` tháng
                future = model.make_future_dataframe(periods=periods, freq='M')
                forecast = model.predict(future)

                # Đánh giá sai số nếu đủ dữ liệu lịch sử
                if len(sub) > periods:
                    y_true = sub['y'].values[-periods:]
                    y_pred = forecast['yhat'].values[-2*periods:-periods]
                    mae = mean_absolute_error(y_true, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                    st.write(f"MAE: {mae:.2f}")
                    st.write(f"RMSE: {rmse:.2f}")
                else:
                    st.info("Không đủ dữ liệu lịch sử để tính MAE/RMSE")

                # Hiển thị bảng dự báo các tháng cuối
                st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods))

                # Vẽ biểu đồ dự báo
                fig2 = model.plot(forecast)
                st.pyplot(fig2)

                # Lấy dự báo tháng cuối để đưa vào bài toán tối ưu  
                demands[product] = int(forecast['yhat'].values[-1])
            else:
                st.warning("Không đủ dữ liệu để dự báo cho sản phẩm này.")

    st.markdown("---")
    st.markdown("## ⚙️ Tối ưu hóa sản xuất")

    # Chọn tiêu chí tối ưu hóa
    opt_criteria = st.sidebar.selectbox(
        "Chọn tiêu chí tối ưu hóa",
        [
            "Tối đa hóa lợi nhuận",
            "Tối đa hóa tổng số lượng sản xuất",
            "Tối thiểu hóa chi phí nguyên liệu",
            "Cân bằng sản lượng các sản phẩm"
        ]
    )

    profits = {'Cola': 12, 'Beer': 15, 'Green Tea': 10}
    costs = {'Cola': 5, 'Beer': 6, 'Green Tea': 4}  # ví dụ chi phí sản xuất/đơn vị
    resources = {
        'Water': {'Cola': 0.8, 'Beer': 0.85, 'Green Tea': 0.9},
        'Sugar': {'Cola': 0.15, 'Beer': 0.05, 'Green Tea': 0.1}
    }
    resource_limits = {
        'Water': st.sidebar.number_input("Giới hạn nước (lít)", value=1500),
        'Sugar': st.sidebar.number_input("Giới hạn đường (kg)", value=200)
    }
    max_total = st.sidebar.number_input("Công suất tối đa (chai)", value=2000)

    if st.button("Tính tối ưu"):
        model = LpProblem(
            "soft-drink-optimization",
            LpMaximize if opt_criteria != "Tối thiểu hóa chi phí nguyên liệu" else LpMinimize
        )
        vars = {p: LpVariable(p, lowBound=0, upBound=demands.get(p, 0), cat="Integer") for p in products}

        # Hàm mục tiêu theo lựa chọn
        if opt_criteria == "Tối đa hóa lợi nhuận":
            model += lpSum(profits.get(p, 0) * vars[p] for p in vars)
            # Ràng buộc: công suất + nguyên liệu
            model += lpSum(vars[p] for p in vars) <= max_total
            for resource, limit in resource_limits.items():
                model += lpSum(resources[resource][p] * vars[p] for p in vars) <= limit

        elif opt_criteria == "Tối đa hóa tổng số lượng sản xuất":
            model += lpSum(vars[p] for p in vars)
            # Ràng buộc: chỉ công suất, không ràng buộc nguyên liệu
            model += lpSum(vars[p] for p in vars) <= max_total

        elif opt_criteria == "Tối thiểu hóa chi phí nguyên liệu":
            model += lpSum(costs.get(p, 0) * vars[p] for p in vars)
            # Ràng buộc: nguyên liệu + nhu cầu dự báo
            for resource, limit in resource_limits.items():
                model += lpSum(resources[resource][p] * vars[p] for p in vars) <= limit

        elif opt_criteria == "Cân bằng sản lượng các sản phẩm":
            max_var = LpVariable("max_var", lowBound=0)
            for p in vars:
                model += vars[p] <= max_var
            model += max_var
            # Ràng buộc: công suất + nguyên liệu
            model += lpSum(vars[p] for p in vars) <= max_total
            for resource, limit in resource_limits.items():
                model += lpSum(resources[resource][p] * vars[p] for p in vars) <= limit

        model.solve()
        st.subheader("Kế hoạch sản xuất tối ưu")
        for p in vars:
            st.write(f"{p}: {int(vars[p].value())} (chai)")
        if opt_criteria == "Tối đa hóa lợi nhuận":
            st.write(f"Lợi nhuận tối đa = ${model.objective.value():,.0f}")
        elif opt_criteria == "Tối đa hóa tổng số lượng sản xuất":
            st.write(f"Tổng số lượng sản xuất tối đa = {model.objective.value():,.0f} (chai)")
        elif opt_criteria == "Tối thiểu hóa chi phí nguyên liệu":
            st.write(f"Chi phí nguyên liệu tối thiểu = ${model.objective.value():,.0f}")
        elif opt_criteria == "Cân bằng sản lượng các sản phẩm":
            st.write(f"Sản lượng tối đa của 1 sản phẩm = {model.objective.value():,.0f} (chai)")
else:
    st.info("Vui lòng tải lên file dữ liệu CSV để bắt đầu.")
