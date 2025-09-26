import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from pathlib import Path

# --- 1. 기본 페이지 설정 ---
# Matplotlib의 기본 설정을 사용하므로 별도 폰트 코드가 필요 없습니다.
plt.rcParams["axes.unicode_minus"] = False

st.set_page_config(
    page_title="제조장비 이상탐지 대시보드",
    page_icon="🏭",
    layout="wide",
)

# --- 2. 데이터 로딩 ---
@st.cache_data
def load_data():
    """데이터를 로드하고 기본 전처리를 수행하는 함수"""
    script_path = Path(__file__).parent
    data_path = script_path / "data" / "equipment_anomaly_data.csv"
    try:
        df = pd.read_csv(data_path)
        return df
    except FileNotFoundError:
        st.error(f"데이터 파일을 찾을 수 없습니다. 다음 경로를 확인해주세요: {data_path}")
        return None

df = load_data()

# --- 3. 머신러닝 모델 시뮬레이션 및 위험 점수 계산 함수 ---
def predict_fault_probability(equipment, temp, pressure, vib):
    if equipment == 'Turbine':
        if temp > 100 and pressure > 50: return 0.95
        elif temp > 90: return 0.60
        else: return 0.05
    elif equipment == 'Compressor':
        if vib > 3.0: return 0.90
        elif vib > 2.5: return 0.50
        else: return 0.05
    elif equipment == 'Pump':
        if (70 < temp < 90) and (50 < pressure < 70): return 0.85
        elif temp > 90 or pressure > 70: return 0.40
        else: return 0.05
    return 0.05

def calculate_risk_score(row):
    return predict_fault_probability(row['equipment'], row['temperature'], row['pressure'], row['vibration'])

# --- 4. 사이드바 메뉴 구성 ---
with st.sidebar:
    st.title("🏭 이상탐지 대시보드")
    page = st.radio("메뉴 선택", ["🏠 장비별 이상징후 감시", "📊 상세 분석", "🔮 이상징후 시뮬레이터"])
    st.markdown("---")
    st.write("본 대시보드는 제조 장비의 센서 데이터를 분석하여 잠재적 고장을 사전에 감지하는 것을 목표로 합니다.")

# --- 5. 페이지별 콘텐츠 ---
if df is not None:
    if page == "🏠 장비별 이상징후 감시":
        st.header("Action Plan: 장비별 핵심 모니터링 지표")
        st.info("관리할 장비를 선택하면, 해당 장비의 **핵심 관리 지표**와 **위험 구간**, 그리고 **조치 우선순위**를 확인할 수 있습니다.")

        selected_equipment = st.selectbox("관리할 장비를 선택하세요:", df['equipment'].unique())

        col1, col2 = st.columns([1, 1.5])
        with col1:
            st.subheader(f"✅ [{selected_equipment}] 핵심 관리 지표")
            if selected_equipment == 'Turbine':
                st.metric(label="온도 (Temperature)", value="> 100.0 °C", delta="위험", delta_color="inverse")
                st.metric(label="압력 (Pressure)", value="> 50.0", delta="주의", delta_color="off")
                st.metric(label="습도 (Humidity)", value="< 30.0 %", delta="주의", delta_color="off")
                st.info("모든 터빈 장비의 온도가 100도를 넘지 않도록 집중 관리하세요. 고온-고압 또는 고온-건조 상태는 즉시 점검이 필요합니다.")
            elif selected_equipment == 'Compressor':
                st.metric(label="진동 (Vibration)", value="> 3.0", delta="위험", delta_color="inverse")
                st.metric(label="압력 (Pressure)", value="> 60.0", delta="주의", delta_color="off")
                st.metric(label="온도 (Temperature)", value="< 50.0 °C", delta="주의", delta_color="off")
                st.info("모든 압축기 장비의 진동이 3.0을 넘지 않도록 집중 관리하세요. 특히 저온 상태에서의 고진동은 매우 위험한 신호입니다.")
            elif selected_equipment == 'Pump':
                st.metric(label="온도 (Temperature)", value="70-90 °C 구간", delta="위험", delta_color="inverse")
                st.metric(label="압력 (Pressure)", value="50-70 구간", delta="위험", delta_color="inverse")
                st.info("펌프는 특정 작동 구간(온도 70-90, 압력 50-70)을 벗어날 때 위험합니다. 고압-건조 상태도 주의가 필요합니다.")

        with col2:
            st.subheader("📊 위험 구간 시각화 (Hexbin Plot)")
            df_eq = df[df['equipment'] == selected_equipment]
            x_var, y_var = ('temperature', 'pressure') if selected_equipment != 'Compressor' else ('pressure', 'vibration')
            
            fig, ax = plt.subplots(figsize=(8, 6))
            hb = ax.hexbin(x=df_eq[x_var], y=df_eq[y_var], C=df_eq['faulty'], 
                           gridsize=20, cmap='viridis', reduce_C_function=np.mean)
            cb = fig.colorbar(hb, ax=ax)
            cb.set_label('Faulty Ratio')
            ax.set_title(f'Risk Zone for [{selected_equipment}]')
            ax.set_xlabel(x_var.capitalize())
            ax.set_ylabel(y_var.capitalize())
            st.pyplot(fig)

        st.markdown("---")
        st.header("📋 데이터 기반 조치 우선순위")
        st.warning("아래 목록은 현재 데이터 중 **가장 위험 신호에 근접한 장비**들입니다. 이 장비들부터 우선적으로 점검하세요.")
        df['risk_score'] = df.apply(calculate_risk_score, axis=1)
        priority_df = df[df['risk_score'] > 0.5].sort_values(by='risk_score', ascending=False)
        st.dataframe(priority_df[['equipment', 'location', 'risk_score', 'temperature', 'pressure', 'vibration', 'humidity']].head(20))

    elif page == "📊 상세 분석":
        st.header("상세 분석 (Detailed Analysis)")
        st.info("장비 종류와 지역을 필터링하여 데이터를 심층적으로 탐색할 수 있습니다.")

        col1, col2 = st.columns(2)
        with col1:
            selected_equipments = st.multiselect("분석할 장비 종류를 선택하세요:", df['equipment'].unique(), default=df['equipment'].unique())
        with col2:
            selected_locations = st.multiselect("분석할 지역을 선택하세요:", df['location'].unique(), default=df['location'].unique())

        filtered_df = df[(df['equipment'].isin(selected_equipments)) & (df['location'].isin(selected_locations))]
        st.write(f"**선택된 데이터:** {len(filtered_df)}개")

        with st.expander("▼ 전체 데이터 요약"):
            c1, c2 = st.columns(2)
            with c1:
                st.metric("선택 장비 수", len(filtered_df))
                st.metric("고장률", f"{filtered_df['faulty'].mean():.2%}")
            with c2:
                fig, ax = plt.subplots()
                filtered_df['faulty'].value_counts().plot.pie(autopct='%1.1f%%', labels=['Normal', 'Faulty'], ax=ax)
                ax.set_ylabel('')
                ax.set_title('Normal vs. Faulty Distribution')
                st.pyplot(fig)

        with st.expander("▼ 단일 변수 분석"):
            var_to_analyze = st.selectbox("분석할 변수를 선택하세요:", ['temperature', 'pressure', 'vibration', 'humidity', 'equipment', 'location'])
            
            fig, ax = plt.subplots(figsize=(10, 5))
            if var_to_analyze in ['temperature', 'pressure', 'vibration', 'humidity']:
                sns.boxplot(x='faulty', y=var_to_analyze, data=filtered_df, ax=ax)
                ax.set_xticklabels(['Normal', 'Faulty'])
            else:
                fault_rate = filtered_df.groupby(var_to_analyze)['faulty'].mean().sort_values(ascending=False)
                sns.barplot(x=fault_rate.index, y=fault_rate.values, ax=ax)
                ax.set_ylabel('Faulty Rate')
            ax.set_title(f'Analysis of {var_to_analyze.capitalize()}')
            st.pyplot(fig)
        
        with st.expander("▼ 상관관계 분석"):
            st.write("필터링된 데이터의 상관관계 히트맵입니다.")
            corr_matrix = filtered_df[['temperature', 'pressure', 'vibration', 'humidity']].corr()
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
            ax.set_title('Correlation Heatmap')
            st.pyplot(fig)

    elif page == "🔮 이상징후 시뮬레이터":
        st.header("이상징후 시뮬레이터 (What-If Analysis)")
        st.info("가상의 센서 값을 입력하여 해당 조건에서의 **예상 고장 확률**을 예측해볼 수 있습니다.")

        sim_equipment = st.selectbox("시뮬레이션할 장비를 선택하세요:", df['equipment'].unique())
        
        st.write(f"**[{sim_equipment}]**의 센서 값을 조절해보세요:")
        col1, col2, col3 = st.columns(3)
        with col1:
            sim_temp = st.slider("온도 (Temperature)", min_value=df['temperature'].min(), max_value=df['temperature'].max(), value=df['temperature'].mean())
        with col2:
            sim_pressure = st.slider("압력 (Pressure)", min_value=df['pressure'].min(), max_value=df['pressure'].max(), value=df['pressure'].mean())
        with col3:
            sim_vib = st.slider("진동 (Vibration)", min_value=df['vibration'].min(), max_value=df['vibration'].max(), value=df['vibration'].mean())

        probability = predict_fault_probability(sim_equipment, sim_temp, sim_pressure, sim_vib)

        # Plotly는 한글/영문 설정과 무관하게 잘 작동합니다.
        fig = go.Figure(go.Indicator(
            mode = "gauge+number", value = probability * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "예상 고장 확률 (%)"},
            gauge = { 'axis': {'range': [None, 100]}, 'bar': {'color': "darkblue"},
                'steps' : [ {'range': [0, 40], 'color': "lightgreen"}, {'range': [40, 80], 'color': "gold"}, {'range': [80, 100], 'color': "red"}]}))
        
        st.plotly_chart(fig)

        if probability > 0.8:
            st.error(f"**결과:** 현재 조건은 **매우 위험**합니다. 즉시 점검이 필요한 상태일 확률이 높습니다.")
        elif probability > 0.4:
            st.warning(f"**결과:** 현재 조건은 **주의**가 필요합니다. 잠재적 위험 신호가 감지되었습니다.")
        else:
            st.success(f"**결과:** 현재 조건은 **안정적**입니다. 정상 작동 범위 내에 있습니다.")
else:
    st.error("데이터를 불러오는 데 실패했습니다. 파일 경로를 확인하거나 파일을 다시 업로드해주세요.")