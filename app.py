import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

# 設定網頁標題
st.set_page_config(page_title="酒類資料集模型預測", layout="wide")

# 1. 載入資料集
@st.cache_data
def load_data():
    wine = load_wine()
    df = pd.DataFrame(wine.data, columns=wine.feature_names)
    df['target'] = wine.target
    return df, wine

df, wine_data = load_data()

# 2. 側邊欄設計
st.sidebar.title("模型設定")

# 下拉選單選擇模型
model_name = st.sidebar.selectbox(
    "選擇預測模型",
    ("KNN", "羅吉斯迴歸", "Random Forest", "XGBoost")
)

st.sidebar.markdown("---")
st.sidebar.subheader("🍷 酒類資料集資訊")
st.sidebar.info(wine_data.DESCR.split("##")[0]) # 顯示部分描述資訊

# 3. 右側 Main 區
st.title("🍷 酒類資料集 (Wine Dataset) 探索與預測")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 資料集前 5 筆內容")
    st.dataframe(df.head())

with col2:
    st.subheader("📈 特徵統計值資訊")
    st.write(df.describe())

st.markdown("---")

# 4. 預測邏輯
st.subheader(f"🚀 模型預測：{model_name}")

if st.button("按下按鈕進行預測"):
    # 資料分割
    X = df.drop('target', axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 根據選擇初始化模型
    if model_name == "KNN":
        model = KNeighborsClassifier(n_neighbors=5)
    elif model_name == "羅吉斯迴歸":
        model = LogisticRegression(max_iter=10000)
    elif model_name == "Random Forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_name == "XGBoost":
        model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')

    # 訓練模型
    with st.spinner(f"正在訓練 {model_name} 模型..."):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

    # 顯示結果
    st.success(f"### 預測準確度：{acc:.2%}")
    
    st.write("#### 分類報告 (Classification Report):")
    report = classification_report(y_test, y_pred, target_names=wine_data.target_names, output_dict=True)
    st.table(pd.DataFrame(report).transpose())
    
    # 範例單筆預測結果
    st.info("💡 以上結果是基於全體測試集 (20% 資料) 的統計結果。")
