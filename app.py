import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from pathlib import Path

# --- 1. 언어 텍스트 딕셔너리 정의 ---
LANGUAGES = {
    "ko": {
        "page_title": "제조장비 이상탐지 대시보드",
        "sidebar_title": "🏭 이상탐지 대시보드",
        "menu_select": "메뉴 선택",
        "menu_main": "🏠 장비별 이상징후 감시",
        "menu_detail": "📊 상세 분석",
        "menu_simulator": "🔮 이상징후 시뮬레이터",
        "sidebar_info": "본 대시보드는 제조 장비의 센서 데이터를 분석하여 잠재적 고장을 사전에 감지하는 것을 목표로 합니다.",
        
        "main_header": "Action Plan: 장비별 핵심 모니터링 지표",
        "main_info": "관리할 장비를 선택하면, 해당 장비의 **핵심 관리 지표**와 **위험 구간**, 그리고 **조치 우선순위**를 확인할 수 있습니다.",
        "main_selectbox_label": "관리할 장비를 선택하세요:",
        "main_kpi_subheader": "✅ [{equipment}] 핵심 관리 지표",
        "main_viz_subheader": "📊 위험 구간 시각화 (Hexbin Plot)",
        "main_priority_header": "📋 데이터 기반 조치 우선순위",
        "main_priority_warning": "아래 목록은 현재 데이터 중 **가장 위험 신호에 근접한 장비**들입니다. 이 장비들부터 우선적으로 점검하세요.",

        "detail_header": "상세 분석 (Detailed Analysis)",
        "detail_info": "장비 종류와 지역을 필터링하여 데이터를 심층적으로 탐색할 수 있습니다.",
        "detail_multiselect_eq": "분석할 장비 종류를 선택하세요:",
        "detail_multiselect_loc": "분석할 지역을 선택하세요:",
        "detail_filtered_data": "**선택된 데이터:** {count}개",
        "detail_expander_summary": "▼ 전체 데이터 요약",
        "detail_metric_count": "선택 장비 수",
        "detail_metric_fault_rate": "고장률",
        "detail_expander_single": "▼ 단일 변수 분석",
        "detail_selectbox_variable": "분석할 변수를 선택하세요:",
        "detail_expander_corr": "▼ 상관관계 분석",
        "detail_corr_info": "필터링된 데이터의 상관관계 히트맵입니다.",
        
        "sim_header": "이상징후 시뮬레이터 (What-If Analysis)",
        "sim_info": "가상의 센서 값을 입력하여 해당 조건에서의 **예상 고장 확률**을 예측해볼 수 있습니다.",
        "sim_selectbox_label": "시뮬레이션할 장비를 선택하세요:",
        "sim_slider_info": "**[{equipment}]**의 센서 값을 조절해보세요:",
        "sim_gauge_title": "예상 고장 확률 (%)",
        "sim_result_critical": "**결과:** 현재 조건은 **매우 위험**합니다. 즉시 점검이 필요한 상태일 확률이 높습니다.",
        "sim_result_warning": "**결과:** 현재 조건은 **주의**가 필요합니다. 잠재적 위험 신호가 감지되었습니다.",
        "sim_result_stable": "**결과:** 현재 조건은 **안정적**입니다. 정상 작동 범위 내에 있습니다.",
    },
    "en": {
        "page_title": "Equipment Anomaly Detection Dashboard",
        "sidebar_title": "🏭 Anomaly Dashboard",
        "menu_select": "Menu Selection",
        "menu_main": "🏠 Monitoring Dashboard",
        "menu_detail": "📊 Detailed Analysis",
        "menu_simulator": "🔮 Anomaly Simulator",
        "sidebar_info": "This dashboard analyzes sensor data to detect potential equipment failures in advance.",

        "main_header": "Action Plan: Key Monitoring Metrics",
        "main_info": "Select an equipment to see its key metrics, risk zones, and a priority list for action.",
        "main_selectbox_label": "Select Equipment:",
        "main_kpi_subheader": "✅ Key Metrics for [{equipment}]",
        "main_viz_subheader": "📊 Risk Zone Visualization (Hexbin Plot)",
        "main_priority_header": "📋 Data-Driven Priority List",
        "main_priority_warning": "The list below shows equipment closest to risk signals. Please inspect these first.",

        "detail_header": "Detailed Analysis",
        "detail_info": "Filter the data by equipment type and location for a deeper dive.",
        "detail_multiselect_eq": "Select Equipment Type(s):",
        "detail_multiselect_loc": "Select Location(s):",
        "detail_filtered_data": "**Filtered Data:** {count} records",
        "detail_expander_summary": "▼ Overall Summary",
        "detail_metric_count": "Selected Equipment Count",
        "detail_metric_fault_rate": "Faulty Rate",
        "detail_expander_single": "▼ Single Variable Analysis",
        "detail_selectbox_variable": "Select a variable to analyze:",
        "detail_expander_corr": "▼ Correlation Analysis",
        "detail_corr_info": "Correlation heatmap of the filtered data.",
        
        "sim_header": "Anomaly Simulator (What-If Analysis)",
        "sim_info": "Predict the probability of failure under hypothetical sensor values.",
        "sim_selectbox_label": "Select equipment for simulation:",
        "sim_slider_info": "Adjust the sensor values for **[{equipment}]**:",
        "sim_gauge_title": "Predicted Failure Probability (%)",
        "sim_result_critical": "**Result:** The current condition is **CRITICAL**. Immediate inspection is highly recommended.",
        "sim_result_warning": "**Result:** The current condition requires **CAUTION**. A potential risk signal is detected.",
        "sim_result_stable": "**Result:** The current condition is **STABLE**. Operating within the normal range.",
    }
}


# --- 기본 페이지 및 폰트 설정 ---
plt.rcParams["axes.unicode_minus"] = False
# 페이지 제목은 기본 언어(한국어)로 설정하고, 나중에 언어 선택에 따라 동적으로 변경
st.set_page_config(page_title="제조장비 이상탐지 대시보드", page_icon="🏭", layout="wide")


# --- 데이터 로딩 ---
@st.cache_data
def load_data():
    script_path = Path(__file__).parent
    data_path = script_path / "data" / "equipment_anomaly_data.csv"
    try:
        df = pd.read_csv(data_path)
        return df
    except FileNotFoundError:
        st.error(f"데이터 파일을 찾을 수 없습니다: {data_path}")
        return None

df = load_data()


# --- 예측 및 리스크 점수 함수 ---
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


# --- 사이드바 메뉴 구성 ---
with st.sidebar:
    # 언어 선택 기능 추가
    if 'lang' not in st.session_state:
        st.session_state.lang = 'ko'

    selected_lang_display = st.radio("Language", ["한국어 (Korean)", "English"], 
                                     index=0 if st.session_state.lang == 'ko' else 1)
    
    st.session_state.lang = 'ko' if selected_lang_display == "한국어 (Korean)" else 'en'
    
    # 선택된 언어에 맞는 텍스트 딕셔너리를 가져옴
    lang = LANGUAGES[st.session_state.lang]

    # 페이지 제목 동적 변경
    st.set_page_config(page_title=lang["page_title"])

    st.title(lang["sidebar_title"])
    
    page_options = [lang["menu_main"], lang["menu_detail"], lang["menu_simulator"]]
    page = st.radio(lang["menu_select"], page_options)
    
    st.markdown("---")
    st.write(lang["sidebar_info"])


# --- 페이지별 콘텐츠 ---
if df is not None:
    if page == lang["menu_main"]:
        st.header(lang["main_header"])
        st.info(lang["main_info"])
        selected_equipment = st.selectbox(lang["main_selectbox_label"], df['equipment'].unique())

        col1, col2 = st.columns([1, 1.5])
        with col1:
            st.subheader(lang["main_kpi_subheader"].format(equipment=selected_equipment))
            if selected_equipment == 'Turbine':
                st.metric(label="Temperature", value="> 100.0 °C", delta="CRITICAL", delta_color="inverse")
                st.metric(label="Pressure", value="> 50.0", delta="Warning", delta_color="off")
                st.metric(label="Humidity", value="< 30.0 %", delta="Warning", delta_color="off")
                st.info("모든 터빈 장비의 온도가 100도를 넘지 않도록 집중 관리하세요. 고온-고압 또는 고온-건조 상태는 즉시 점검이 필요합니다.")
            elif selected_equipment == 'Compressor':
                st.metric(label="Vibration", value="> 3.0", delta="CRITICAL", delta_color="inverse")
                st.metric(label="Pressure", value="> 60.0", delta="Warning", delta_color="off")
                st.metric(label="Temperature", value="< 50.0 °C", delta="Warning", delta_color="off")
                st.info("모든 압축기 장비의 진동이 3.0을 넘지 않도록 집중 관리하세요. 특히 저온 상태에서의 고진동은 매우 위험한 신호입니다.")
            elif selected_equipment == 'Pump':
                st.metric(label="Temperature", value="70-90 °C Range", delta="CRITICAL", delta_color="inverse")
                st.metric(label="Pressure", value="50-70 Range", delta="CRITICAL", delta_color="inverse")
                st.info("펌프는 특정 작동 구간(온도 70-90, 압력 50-70)을 벗어날 때 위험합니다. 고압-건조 상태도 주의가 필요합니다.")

        with col2:
            st.subheader(lang["main_viz_subheader"])
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
        st.header(lang["main_priority_header"])
        st.warning(lang["main_priority_warning"])
        df['risk_score'] = df.apply(calculate_risk_score, axis=1)
        priority_df = df[df['risk_score'] > 0.5].sort_values(by='risk_score', ascending=False)
        st.dataframe(priority_df[['equipment', 'location', 'risk_score', 'temperature', 'pressure', 'vibration', 'humidity']].head(20))

    elif page == lang["menu_detail"]:
        st.header(lang["detail_header"])
        st.info(lang["detail_info"])

        col1, col2 = st.columns(2)
        with col1:
            selected_equipments = st.multiselect(lang["detail_multiselect_eq"], df['equipment'].unique(), default=df['equipment'].unique())
        with col2:
            selected_locations = st.multiselect(lang["detail_multiselect_loc"], df['location'].unique(), default=df['location'].unique())

        filtered_df = df[(df['equipment'].isin(selected_equipments)) & (df['location'].isin(selected_locations))]
        st.write(lang["detail_filtered_data"].format(count=len(filtered_df)))

        with st.expander(lang["detail_expander_summary"]):
            c1, c2 = st.columns(2)
            with c1:
                st.metric(lang["detail_metric_count"], len(filtered_df))
                st.metric(lang["detail_metric_fault_rate"], f"{filtered_df['faulty'].mean():.2%}")
            with c2:
                fig, ax = plt.subplots()
                filtered_df['faulty'].value_counts().plot.pie(autopct='%1.1f%%', labels=['Normal', 'Faulty'], ax=ax)
                ax.set_ylabel('')
                ax.set_title('Normal vs. Faulty Distribution')
                st.pyplot(fig)

        with st.expander(lang["detail_expander_single"]):
            var_to_analyze = st.selectbox(lang["detail_selectbox_variable"], ['temperature', 'pressure', 'vibration', 'humidity', 'equipment', 'location'])
            
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
        
        with st.expander(lang["detail_expander_corr"]):
            st.write(lang["detail_corr_info"])
            corr_matrix = filtered_df[['temperature', 'pressure', 'vibration', 'humidity']].corr()
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
            ax.set_title('Correlation Heatmap')
            st.pyplot(fig)

    elif page == lang["menu_simulator"]:
        st.header(lang["sim_header"])
        st.info(lang["sim_info"])
        sim_equipment = st.selectbox(lang["sim_selectbox_label"], df['equipment'].unique())
        st.write(lang["sim_slider_info"].format(equipment=sim_equipment))

        col1, col2, col3 = st.columns(3)
        with col1:
            sim_temp = st.slider("Temperature", min_value=df['temperature'].min(), max_value=df['temperature'].max(), value=df['temperature'].mean())
        with col2:
            sim_pressure = st.slider("Pressure", min_value=df['pressure'].min(), max_value=df['pressure'].max(), value=df['pressure'].mean())
        with col3:
            sim_vib = st.slider("Vibration", min_value=df['vibration'].min(), max_value=df['vibration'].max(), value=df['vibration'].mean())

        probability = predict_fault_probability(sim_equipment, sim_temp, sim_pressure, sim_vib)

        fig = go.Figure(go.Indicator(
            mode = "gauge+number", value = probability * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': lang["sim_gauge_title"]},
            gauge = { 'axis': {'range': [None, 100]}, 'bar': {'color': "darkblue"},
                'steps' : [ {'range': [0, 40], 'color': "lightgreen"}, {'range': [40, 80], 'color': "gold"}, {'range': [80, 100], 'color': "red"}]}))
        
        st.plotly_chart(fig)

        if probability > 0.8:
            st.error(lang["sim_result_critical"])
        elif probability > 0.4:
            st.warning(lang["sim_result_warning"])
        else:
            st.success(lang["sim_result_stable"])
else:
    st.error("데이터를 불러오는 데 실패했습니다. 파일 경로를 확인하거나 파일을 다시 업로드해주세요.")
