import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# 設定頁面配置
st.set_page_config(page_title="酒類分類機器學習專題", layout="wide")

# 加載資料集
@st.cache_resource
def load_data():
    wine = load_wine()
    df = pd.DataFrame(data=wine.data, columns=wine.feature_names)
    df['target'] = wine.target
    return wine, df

wine_data, df = load_data()

# --- Sidebar 左側選單 ---
st.sidebar.title("🛠️ 模型選擇")
model_name = st.sidebar.selectbox(
    "選擇一個模型",
    ("KNN", "羅吉斯迴歸", "XGBoost", "隨機森林")
)

st.sidebar.markdown("---")
st.sidebar.subheader("🍷 資料集資訊：酒類 (Wine)")
st.sidebar.write(f"- **總樣本數**: {len(df)}")
st.sidebar.write(f"- **特徵數量**: {len(wine_data.feature_names)}")
st.sidebar.write(f"- **類別數量**: {len(wine_data.target_names)} ({', '.join(wine_data.target_names)})")
st.sidebar.info(wine_data.DESCR.split("\n\n")[0])

# --- Main 區 右側顯示 ---
st.title("🧪 酒類分類預測系統")

st.subheader("📋 資料集預覽 (前 5 筆)")
st.dataframe(df.head())

st.subheader("📈 特徵統計值")
st.write(df.describe())

# --- 訓練與預測 ---
st.markdown("---")
st.subheader(f"🚀 模型訓練與預測: {model_name}")

if st.button("開始執行預測"):
    # 準備資料
    X = wine_data.data
    y = wine_data.target
    
    # 拆分訓練與測試集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 根據選擇的模型初始化
    if model_name == "KNN":
        model = KNeighborsClassifier(n_neighbors=5)
    elif model_name == "羅吉斯迴歸":
        model = LogisticRegression(max_iter=5000)
    elif model_name == "XGBoost":
        model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    elif model_name == "隨機森林":
        model = RandomForestClassifier(n_estimators=100)
    
    # 訓練模型
    with st.spinner('訓練中...'):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
    
    # 顯示結果
    col1, col2 = st.columns(2)
    
    with col1:
        st.success(f"**模型準確度 (Accuracy):** {acc:.4f}")
        st.metric(label="準確度", value=f"{acc*100:.2f}%")
        
    with col2:
        st.write("**前 10 筆測試集預測結果 vs 真實目標:**")
        results_df = pd.DataFrame({
            "真實類別": [wine_data.target_names[i] for i in y_test[:10]],
            "預測類別": [wine_data.target_names[i] for i in y_pred[:10]]
        })
        st.table(results_df)

    st.balloons()
else:
    st.write("點擊上方按鈕開始訓練並檢視結果。")
