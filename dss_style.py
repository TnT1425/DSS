import streamlit as st

def set_custom_style():
    st.set_page_config(
        page_title="DSS Đồ Uống",
        page_icon="🥤",
        layout="wide"
    )

    st.markdown("""
        <style>
        /* Ẩn menu và footer mặc định */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}

        /* Toàn bộ background */
        body {
            background: linear-gradient(135deg, #f9f9f9, #e6f7ff);
            font-family: "Segoe UI", Tahoma, sans-serif;
        }

        /* Tiêu đề chính */
        .big-title {
            font-size: 40px;
            color: #ff6600;
            font-weight: 700;
            text-align: center;
            margin: 20px 0 40px 0;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.15);
        }

        /* Box nội dung */
        .custom-box {
            background: #ffffff;
            border-radius: 15px;
            padding: 25px;
            margin-bottom: 25px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            transition: transform 0.2s ease-in-out;
        }
        .custom-box:hover {
            transform: translateY(-3px);
            box-shadow: 0 6px 16px rgba(0,0,0,0.15);
        }

        /* Nút bấm đẹp hơn */
        .stButton>button {
            background: linear-gradient(135deg, #ff9966, #ff5e62);
            color: white;
            border-radius: 8px;
            border: none;
            padding: 0.6em 1.2em;
            font-weight: 600;
            cursor: pointer;
            transition: background 0.3s ease-in-out;
        }
        .stButton>button:hover {
            background: linear-gradient(135deg, #ff5e62, #ff9966);
        }
        </style>
        
        <div class="big-title">🥤 Dự báo xu hướng & tối ưu hóa sản phẩm</div>
    """, unsafe_allow_html=True)
