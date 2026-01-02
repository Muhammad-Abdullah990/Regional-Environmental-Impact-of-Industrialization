# # app.py

# import streamlit as st
# import pandas as pd
# import numpy as np
# import pickle
# import plotly.express as px

# # -------------------------------
# # Page Configuration
# # -------------------------------
# st.set_page_config(
#     page_title="Regional Environmental Impact Dashboard",
#     layout="wide"
# )

# st.title("🌍 Regional Environmental Impact of Industrialization")

# # -------------------------------
# # Load Dataset
# # -------------------------------
# df = pd.read_csv("IDS project data - Sheet1.csv")

# # -------------------------------
# # Load Models & Scalers
# # -------------------------------
# model_sindh = pickle.load(open("aqi_model_sindh.pkl", "rb"))
# model_punjab = pickle.load(open("aqi_model_punjab.pkl", "rb"))

# scaler_sindh = pickle.load(open("scaler_sindh.pkl", "rb"))
# scaler_punjab = pickle.load(open("scaler_punjab.pkl", "rb"))

# # -------------------------------
# # Sidebar Controls
# # -------------------------------
# st.sidebar.header("🔍 Dashboard Controls")
# region = st.sidebar.selectbox("Select Region", ["Sindh", "Punjab"])

# # ============================================================
# # 📊 KEY METRICS
# # ============================================================
# st.subheader("📊 Key Environmental Indicators")

# col1, col2, col3 = st.columns(3)

# if region == "Sindh":
#     col1.metric("Average AQI", round(df["AQI (Sindh)"].mean(), 2))
#     col2.metric("Average CO₂ Emissions", round(df["CO2 Emission (Sindh)"].mean(), 2))
#     col3.metric("Average Deforestation Rate", round(df["Deforestation (Sindh)"].mean(), 2))
# else:
#     col1.metric("Average AQI", round(df["AQI (Punjab)"].mean(), 2))
#     col2.metric("Average CO₂ Emissions", round(df["CO2 Emission (Punjab)"].mean(), 2))
#     col3.metric("Average Deforestation Rate", round(df["Deforestation (Punjab)"].mean(), 2))

# # ============================================================
# # 🔥 REGIONAL HEATMAPS (AQI & CO₂)
# # ============================================================
# st.subheader("🔥 Regional Intensity Heatmaps (AQI & CO₂)")

# heatmap_df = pd.DataFrame({
#     "Sindh AQI": df["AQI (Sindh)"],
#     "Punjab AQI": df["AQI (Punjab)"],
#     "Sindh CO₂": df["CO2 Emission (Sindh)"],
#     "Punjab CO₂": df["CO2 Emission (Punjab)"]
# })

# fig_heatmap = px.imshow(
#     heatmap_df.T,
#     color_continuous_scale="RdYlGn_r",
#     title="AQI & CO₂ Intensity Across Regions"
# )

# st.plotly_chart(fig_heatmap, use_container_width=True)

# # ============================================================
# # 📈 YEARLY TRENDS: INDUSTRIALIZATION VS ENVIRONMENT
# # ============================================================
# st.subheader("📈 Industrial Growth vs Environmental Degradation")

# if region == "Sindh":
#     fig_trend = px.line(
#         df,
#         y=[
#             "Industrial Variable (Sindh)",
#             "CO2 Emission (Sindh)",
#             "AQI (Sindh)"
#         ],
#         title="Sindh: Industrial Growth vs Environmental Impact"
#     )
# else:
#     fig_trend = px.line(
#         df,
#         y=[
#             "Industrial Variable (Punjab)",
#             "CO2 Emission (Punjab)",
#             "AQI (Punjab)"
#         ],
#         title="Punjab: Industrial Growth vs Environmental Impact"
#     )

# st.plotly_chart(fig_trend, use_container_width=True)

# # ============================================================
# # 📍 REGION-WISE COMPARISON
# # ============================================================
# st.subheader("📍 Region-wise Comparison (Punjab vs Sindh)")

# comparison_df = pd.DataFrame({
#     "Region": ["Sindh", "Punjab"],
#     "Average AQI": [
#         df["AQI (Sindh)"].mean(),
#         df["AQI (Punjab)"].mean()
#     ],
#     "Average CO₂ Emissions": [
#         df["CO2 Emission (Sindh)"].mean(),
#         df["CO2 Emission (Punjab)"].mean()
#     ],
#     "Average Deforestation": [
#         df["Deforestation (Sindh)"].mean(),
#         df["Deforestation (Punjab)"].mean()
#     ]
# })

# fig_compare = px.bar(
#     comparison_df,
#     x="Region",
#     y=["Average AQI", "Average CO₂ Emissions", "Average Deforestation"],
#     barmode="group",
#     title="Punjab vs Sindh Environmental Comparison"
# )

# st.plotly_chart(fig_compare, use_container_width=True)

# # ============================================================
# # 🔮 AQI PREDICTION MODULE
# # ============================================================
# st.subheader("🔮 Predict AQI (Machine Learning Model)")

# col1, col2, col3 = st.columns(3)

# co2 = col1.number_input("CO₂ Emission", min_value=0.0)
# deforestation = col2.number_input("Deforestation Rate", min_value=0.0)
# industrial = col3.number_input("Industrial Index", min_value=0.0)

# if st.button("Predict AQI"):
#     input_data = np.array([[co2, deforestation, industrial]])

#     if region == "Sindh":
#         input_scaled = scaler_sindh.transform(input_data)
#         prediction = model_sindh.predict(input_scaled)
#     else:
#         input_scaled = scaler_punjab.transform(input_data)
#         prediction = model_punjab.predict(input_scaled)

#     st.success(f"🌫 Predicted AQI for {region}: **{prediction[0]:.2f}**")

# # ============================================================
# # FOOTER
# # ============================================================
# st.markdown("---")
# st.caption(
#     "Dashboard Features: Heatmaps, Trend Analysis, Regional Comparison, "
#     "Predictive Modeling | Domain: Environmental Impact of Industrialization"
# )









import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px

# -------------------------------
# Setup & Loading
# -------------------------------
st.set_page_config(page_title="Industrial Impact Dashboard", layout="wide")
st.title("🌍 Regional Environmental Impact of Industrialization")

@st.cache_data
def load_data():
    df = pd.read_csv("IDS_Project_Simulation.csv")
    df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day']])
    return df

df = load_data()

# Load Models (using the new high-accuracy versions)
m_s = pickle.load(open("aqi_model_sindh_new.pkl", "rb"))
m_p = pickle.load(open("aqi_model_punjab_new.pkl", "rb"))
s_s = pickle.load(open("scaler_sindh_new.pkl", "rb"))
s_p = pickle.load(open("scaler_punjab_new.pkl", "rb"))

# -------------------------------
# Sidebar
# -------------------------------
region = st.sidebar.selectbox("Select Region for Detailed View", ["Punjab", "Sindh"])
year_range = st.sidebar.slider("Select Year Range", 2014, 2024, (2014, 2024))

# Filter data
filtered_df = df[(df['Region'] == region) & (df['Year'].between(year_range[0], year_range[1]))]

# -------------------------------
# Dashboard Layout
# -------------------------------
col1, col2, col3 = st.columns(3)
col1.metric("Avg AQI", f"{filtered_df['AQI'].mean():.1f}")
col2.metric("Avg CO₂", f"{filtered_df['CO2'].mean():.1f} ppm")
col3.metric("Deforestation Rate", f"{filtered_df['Deforestation'].mean():.2f}")

# 🔥 HEATMAP: Regional Intensity
st.subheader("🔥 Regional Intensity Comparison")
# Prepare data for heatmap (pivot for heat visualization)
heatmap_data = df.groupby(['Region', 'Year'])['AQI'].mean().reset_index()
fig_heat = px.density_heatmap(heatmap_data, x="Year", y="Region", z="AQI", 
                             color_continuous_scale="Viridis", title="Yearly AQI Intensity")
st.plotly_chart(fig_heat, use_container_width=True)

# 📈 TREND LINES
st.subheader("📈 Industrial Growth vs Environmental Degradation")
fig_trend = px.line(filtered_df, x='Date', y=['Industrial_Index', 'CO2', 'AQI'],
                    title=f"{region}: Growth vs Pollution Trends")
st.plotly_chart(fig_trend, use_container_width=True)

# 📍 COMPARISON CHART
st.subheader("📍 Region-wise Comparison (Punjab vs Sindh)")
comp_df = df.groupby('Region')[['AQI', 'CO2', 'Deforestation']].mean().reset_index()
fig_comp = px.bar(comp_df, x='Region', y=['AQI', 'CO2'], barmode='group',
                 color_discrete_sequence=['#FF4B4B', '#1C83E1'])
st.plotly_chart(fig_comp, use_container_width=True)

# 🔮 PREDICTION ENGINE
st.subheader("🔮 High-Accuracy AQI Predictor")
p_col1, p_col2, p_col3 = st.columns(3)
p_ind = p_col1.number_input("Industrial Index", value=180.0)
p_co2 = p_col2.number_input("CO2 Level", value=650.0)
p_def = p_col3.number_input("Deforestation Rate", value=2.5)

if st.button("Calculate Predicted AQI"):
    # Features must match training: [Ind, CO2, Def, Year, Month]
    input_data = np.array([[p_ind, p_co2, p_def, 2025, 1]]) # Predicting for next month
    if region == "Sindh":
        scaled = s_s.transform(input_data)
        res = m_s.predict(scaled)
    else:
        scaled = s_p.transform(input_data)
        res = m_p.predict(scaled)
    
    st.success(f"The Predicted AQI for {region} is: **{res[0]:.2f}**")