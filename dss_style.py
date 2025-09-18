import streamlit as st

def set_custom_style():
    st.set_page_config(page_title="DSS Đồ Uống", page_icon="🥤", layout="wide")
    st.markdown("""
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        .big-title {
            font-size:36px;
            color:#ff6600;
            font-weight:bold;
            text-align:center;
            margin-bottom:30px;
        }
        .custom-box {
            background: #f0f2f6;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 2px 2px 8px #aaa;
        }
        </style>
        <div class="big-title">🥤 DSS Đồ Uống - Giao diện đẹp hơn</div>
        <div class="custom-box">
            <b>Chào mừng bạn đến với hệ thống DSS dự báo và tối ưu hóa sản phẩm đồ uống!</b>
        </div>
    """, unsafe_allow_html=True)