# test/pages/2_clinic_dashboard.py
# Redesigned as "Clinic Operations & Management Console" for "Sentinel Health Co-Pilot"
# This page simulates a web interface for Clinic Managers or Lead Clinicians
# at a Facility Node (Tier 2). It focuses on operational oversight, service quality,
# resource management, and local health/environmental conditions impacting the clinic.

import streamlit as st
import pandas as pd
import numpy as np # For np.nan
import os
import logging
from datetime import date, timedelta

# --- Sentinel System Imports ---
from config import app_config # Uses new, redesigned app_config

# Core data loading for simulation. In real system, data comes from Facility Node DB.
from utils.core_data_processing import (
    load_health_records,
    load_iot_clinic_environment_data
    # get_clinic_summary, get_clinic_environmental_summary are called by components
)
# Refactored AI engine (apply_ai_models) might be run at Facility Node on incoming data.
from utils.ai_analytics_engine import apply_ai_models

# Refactored Clinic Component Data Preparation Functions
# These functions now return structured data.
from pages.clinic_components_sentinel.environmental_kpi_calculator import calculate_clinic_environmental_kpis
from pages.clinic_components_sentinel.main_kpi_structurer import structure_main_clinic_kpis_data, structure_disease_specific_kpis_data
from pages.clinic_components_sentinel.epi_data_calculator import calculate_clinic_epi_data
from pages.clinic_components_sentinel.environment_detail_preparer import prepare_clinic_environment_details_data
from pages.clinic_components_sentinel.patient_focus_data_preparer import prepare_clinic_patient_focus_data
from pages.clinic_components_sentinel.supply_forecast_generator import prepare_clinic_supply_forecast_data
from pages.clinic_components_sentinel.testing_insights_analyzer import prepare_clinic_testing_insights_data

# Refactored UI helpers for web reports (Tier 2/3 views)
from utils.ui_visualization_helpers import (
    render_web_kpi_card,
    plot_annotated_line_chart_web,
    plot_bar_chart_web,
    plot_donut_chart_web
    # Other plotters might be used by the data display sections below
)

# Page Configuration (Specific to this Clinic Management View)
st.set_page_config(
    page_title=f"Clinic Console - {app_config.APP_NAME}",
    layout="wide",
    initial_sidebar_state="expanded"
)

logger = logging.getLogger(__name__)
# CSS (assume loaded by app_home or via cache_resource if needed here)

# --- Data Loading and Preparation (Simulating Facility Node Data Access) ---
@st.cache_data(ttl=app_config.CACHE_TTL_SECONDS_WEB_REPORTS, show_spinner="Loading clinic operational data for console...")
def get_clinic_console_data(selected_start_dt: date, selected_end_dt: date):
    """
    Simulates fetching and preparing data for the Clinic Management Console.
    In a real system, this would query an aggregated database at the Facility Node.
    """
    # For simulation: Load raw health records and IoT data, then apply AI enrichment (as if done at Facility Node upon data receipt)
    health_df_raw = load_health_records(file_path=app_config.HEALTH_RECORDS_CSV, source_context="ClinicConsoleSim")
    iot_df_raw = load_iot_clinic_environment_data(file_path=app_config.IOT_CLINIC_ENVIRONMENT_CSV, source_context="ClinicConsoleSim")
    
    # Apply AI models (simulating Facility Node processing)
    # This `health_df_enriched` becomes the main source for period-specific analysis.
    health_df_enriched_full = pd.DataFrame()
    if not health_df_raw.empty:
        # Ensure AI models use relevant features. AI engine itself should handle missing columns gracefully.
        health_df_enriched_full = apply_ai_models(health_df_raw, source_context="ClinicConsoleSim/AIEnrich")[0] # [0] gets the health_df part
    else:
        logger.warning("Clinic Console: Raw health records empty, AI enrichment skipped.")

    # --- Filter data for the selected period ---
    # Health Data
    health_df_period_clinic = pd.DataFrame()
    if not health_df_enriched_full.empty and 'encounter_date' in health_df_enriched_full.columns:
        health_df_enriched_full['encounter_date_obj'] = pd.to_datetime(health_df_enriched_full['encounter_date'], errors='coerce').dt.date
        health_df_period_clinic = health_df_enriched_full[
            (health_df_enriched_full['encounter_date_obj'].notna()) &
            (health_df_enriched_full['encounter_date_obj'] >= selected_start_dt) &
            (health_df_enriched_full['encounter_date_obj'] <= selected_end_dt)
        ].copy()
    
    # IoT Data
    iot_df_period_clinic = pd.DataFrame()
    if not iot_df_raw.empty and 'timestamp' in iot_df_raw.columns:
        iot_df_raw['timestamp_date_obj'] = pd.to_datetime(iot_df_raw['timestamp'], errors='coerce').dt.date
        iot_df_period_clinic = iot_df_raw[
            (iot_df_raw['timestamp_date_obj'].notna()) &
            (iot_df_raw['timestamp_date_obj'] >= selected_start_dt) &
            (iot_df_raw['timestamp_date_obj'] <= selected_end_dt)
        ].copy()
    
    # Base `get_clinic_summary` to feed into KPI structuring and other components
    # This would now be a more sophisticated calculation at the Facility Node
    # For this simulation, we'll re-use the core_data_processing one, but ideally, it's more advanced.
    # Assume get_clinic_summary is part of core_data_processing (from your full file list)
    from utils.core_data_processing import get_clinic_summary
    clinic_summary_for_period = get_clinic_summary(health_df_period_clinic) if not health_df_period_clinic.empty else {}

    return health_df_enriched_full, health_df_period_clinic, iot_df_period_clinic, clinic_summary_for_period


# --- Page Title & Introduction for Clinic Manager ---
st.title("🏥 Clinic Operations & Management Console")
st.markdown(f"**Facility Performance, Patient Care Quality, Resource Oversight, and Environmental Safety within {app_config.APP_NAME}**")
st.markdown("---")

# --- Sidebar Filters for Clinic Manager ---
if os.path.exists(app_config.APP_LOGO_SMALL):
    st.sidebar.image(app_config.APP_LOGO_SMALL, width=180)
st.sidebar.header("🗓️ Console Filters")

# Date Range Selection (Crucial for clinic manager's review)
# Default range, e.g., last 7 days or last month, configurable via app_config
default_days_clinic_console = app_config.WEB_DASHBOARD_DEFAULT_DATE_RANGE_DAYS_TREND # e.g., 30 days
min_hist_date_clinic = date.today() - timedelta(days=365*1) # Allow up to 1 year back for trends
max_hist_date_clinic = date.today()

default_end_clinic = max_hist_date_clinic
default_start_clinic = default_end_clinic - timedelta(days=default_days_clinic_console - 1)
if default_start_clinic < min_hist_date_clinic: default_start_clinic = min_hist_date_clinic

selected_start_date_console, selected_end_date_console = st.sidebar.date_input(
    "Select Date Range for Review:",
    value=[default_start_clinic, default_end_clinic],
    min_value=min_hist_date_clinic,
    max_value=max_hist_date_clinic,
    key="clinic_console_date_range_v1"
)
if selected_start_date_console > selected_end_date_console:
    st.sidebar.error("Error: Start date must be before end date.")
    selected_start_date_console = selected_end_date_console # Correct to avoid error

# Load data based on selected date range
full_hist_health_df_clinic, period_health_df_clinic, period_iot_df_clinic, clinic_period_summary_data = get_clinic_console_data(
    selected_start_date_console, selected_end_date_console
)
period_str_clinic_console = f"{selected_start_date_console.strftime('%d %b %Y')} - {selected_end_date_console.strftime('%d %b %Y')}"

# --- Display Main Sections using prepared data ---

# Section 1: Key Performance Indicators
st.header(f"Performance Snapshot: {period_str_clinic_console}")
if clinic_period_summary_data:
    main_kpis_list = structure_main_clinic_kpis_data(clinic_period_summary_data, period_str_clinic_console)
    disease_kpis_list = structure_disease_specific_kpis_data(clinic_period_summary_data, period_str_clinic_console)
    
    if main_kpis_list:
        kpi_cols_main = st.columns(len(main_kpis_list) if len(main_kpis_list) <= 4 else 4) # Max 4 per row
        for i, kpi_data in enumerate(main_kpis_list):
            with kpi_cols_main[i % 4]:
                render_web_kpi_card(
                    title=kpi_data["title"], value=kpi_data["value_str"], icon=kpi_data["icon"],
                    status_level=kpi_data["status_level"], units=kpi_data.get("units",""), help_text=kpi_data.get("help_text","")
                )
    if disease_kpis_list:
        st.markdown("##### Disease-Specific & Supply Indicators")
        kpi_cols_disease = st.columns(len(disease_kpis_list) if len(disease_kpis_list) <= 4 else 4)
        for i, kpi_data in enumerate(disease_kpis_list):
            with kpi_cols_disease[i % 4]:
                render_web_kpi_card(
                    title=kpi_data["title"], value=kpi_data["value_str"], icon=kpi_data["icon"],
                    status_level=kpi_data["status_level"], units=kpi_data.get("units",""), help_text=kpi_data.get("help_text","")
                )
else:
    st.warning("Core clinic summary data for KPIs could not be generated for the selected period.")
st.markdown("---")

# Section 2: Environmental KPIs (using data calculator and web KPI renderer)
env_kpi_data_dict = calculate_clinic_environmental_kpis(period_iot_df_clinic, period_str_clinic_console)
if env_kpi_data_dict and any(pd.notna(v) for k,v in env_kpi_data_dict.items() if "avg_" in k): # Check if any avg value is not NaN
    st.subheader(f"Clinic Environment Snapshot")
    cols_env_kpi_console = st.columns(4) # Display up to 4 key env KPIs
    with cols_env_kpi_console[0]:
        render_web_kpi_card("Avg. CO2", f"{env_kpi_data_dict.get('avg_co2_ppm_overall', 'N/A'):.0f}" if pd.notna(env_kpi_data_dict.get('avg_co2_ppm_overall')) else "N/A", units="ppm", icon="💨", status_level=env_kpi_data_dict.get('co2_status_level',"NEUTRAL"))
    with cols_env_kpi_console[1]:
        render_web_kpi_card("Avg. PM2.5", f"{env_kpi_data_dict.get('avg_pm25_ugm3_overall', 'N/A'):.1f}" if pd.notna(env_kpi_data_dict.get('avg_pm25_ugm3_overall')) else "N/A", units="µg/m³", icon="🌫️", status_level=env_kpi_data_dict.get('pm25_status_level',"NEUTRAL"))
    with cols_env_kpi_console[2]:
        render_web_kpi_card("Avg. Waiting Occupancy", f"{env_kpi_data_dict.get('avg_waiting_room_occupancy_persons', 'N/A'):.1f}" if pd.notna(env_kpi_data_dict.get('avg_waiting_room_occupancy_persons')) else "N/A", units="ppl", icon="👨‍👩‍👧‍👦", status_level=env_kpi_data_dict.get('occupancy_status_level',"NEUTRAL"))
    with cols_env_kpi_console[3]:
        render_web_kpi_card("High Noise Alerts (Rooms)", str(env_kpi_data_dict.get('noise_rooms_at_high_alert_count',0)), icon="🔊", units="rooms", status_level="HIGH_CONCERN" if env_kpi_data_dict.get('noise_rooms_at_high_alert_count',0)>0 else "ACCEPTABLE")
else:
    st.info("No environmental monitoring data available for this period to display snapshot KPIs.")
st.markdown("---")


# --- Tabbed Interface for Deeper Dives ---
st.header("Operational Deep Dive Areas")
tab_titles_clinic_console = [
    "📈 Local Epi Intel",
    "🔬 Testing Insights",
    "💊 Supply Management",
    "🧍 Patient Focus & Review",
    "🌿 Environment Details"
]
tab_epi, tab_testing, tab_supply, tab_patients, tab_env_details = st.tabs(tab_titles_clinic_console)

with tab_epi:
    st.subheader(f"Clinic Epidemiology: {period_str_clinic_console}")
    # Call the data preparation function
    clinic_epi_prepared_data = calculate_clinic_epi_data(period_health_df_clinic, period_str_clinic_console)
    if clinic_epi_prepared_data and not all(v is None for k,v in clinic_epi_prepared_data.items() if k != "reporting_period" and k != "general_notes"):
        # Display Symptom Trends
        symptom_df = clinic_epi_prepared_data.get("symptom_trends_weekly_df")
        if symptom_df is not None and not symptom_df.empty:
            st.plotly_chart(plot_bar_chart_web(
                symptom_df, x_col='week_start_date', y_col='count', color_col='symptom',
                title_bar="Weekly Symptom Frequency (Top 5 Reported)", barmode_web='group', y_axis_is_count=True
            ), use_container_width=True)
        # Display Malaria Positivity (Example)
        malaria_pos_series = clinic_epi_prepared_data.get("malaria_rdt_positivity_weekly_series")
        if malaria_pos_series is not None and not malaria_pos_series.empty:
            st.plotly_chart(plot_annotated_line_chart_web(
                data_series_input=malaria_pos_series, chart_title="Weekly Malaria RDT Positivity Rate",
                y_axis_label="Positivity (%)", target_ref_line=app_config.TARGET_MALARIA_POSITIVITY_RATE
            ), use_container_width=True)
        # Display notes
        if clinic_epi_prepared_data.get("general_notes"):
            for note in clinic_epi_prepared_data["general_notes"]: st.caption(note)
    else:
        st.info("Insufficient data for detailed epidemiological analysis in this period.")

with tab_testing:
    # User might select specific test group from UI; for now, pass default or a fixed example
    testing_insights_data = prepare_clinic_testing_insights_data(
        period_health_df_clinic, clinic_period_summary_data, period_str_clinic_console,
        selected_test_group_display_name_for_detail="All Critical Tests Summary" # Or make this a selectbox
    )
    # Display the structured data (e.g., DataFrames as st.dataframe, metrics, notes)
    # ... This part requires fleshing out how to present each piece from testing_insights_data ...
    # Example for critical tests summary:
    if testing_insights_data.get("critical_tests_summary_df") is not None:
        st.subheader("Critical Tests Performance Summary")
        st.dataframe(testing_insights_data["critical_tests_summary_df"], use_container_width=True)
    # Example for overdue tests:
    if testing_insights_data.get("overdue_pending_tests_df") is not None:
        st.subheader("Overdue Pending Tests")
        st.dataframe(testing_insights_data["overdue_pending_tests_df"].head(10), use_container_width=True)
    if testing_insights_data.get("data_availability_notes"):
        for note in testing_insights_data["data_availability_notes"]: st.caption(f"Note: {note}")


with tab_supply:
    supply_forecast_data_dict = prepare_clinic_supply_forecast_data(
        clinic_historical_health_df=full_hist_health_df_clinic, # Needs full history for rates
        reporting_period_str=period_str_clinic_console,
        use_ai_forecast_model=st.sidebar.checkbox("Use AI Supply Forecast (Simulated)", value=False, key="clinic_supply_ai_toggle_console") # Allow toggle from sidebar
    )
    # Display the structured forecast data
    # ... This part requires fleshing out, e.g., select item, show line chart for its forecast_data_df ...
    # Example for items summary:
    if supply_forecast_data_dict.get("forecast_items_summary_list"):
        st.subheader(f"Supply Outlook ({supply_forecast_data_dict.get('forecast_model_used','N/A')})")
        df_supply_summary = pd.DataFrame(supply_forecast_data_dict["forecast_items_summary_list"])
        st.dataframe(df_supply_summary, use_container_width=True)
    # Placeholder for detailed chart if an item is selected (UI needs item selector)
    # selected_item_forecast = st.selectbox("View Forecast for Item:", options=df_supply_summary['item_name'].tolist())
    # if selected_item_forecast and supply_forecast_data_dict.get("forecast_data_df") is not None: ... plot chart ...
    if supply_forecast_data_dict.get("data_availability_notes"):
        for note in supply_forecast_data_dict["data_availability_notes"]: st.caption(f"Note: {note}")


with tab_patients:
    patient_focus_data_dict = prepare_clinic_patient_focus_data(
        period_health_df_clinic, period_str_clinic_console
    )
    # Display patient load and flagged cases
    # ... This part needs fleshing out for presentation ...
    if patient_focus_data_dict.get("patient_load_by_condition_df") is not None:
        st.subheader("Patient Load by Key Condition")
        # This df is date, condition, unique_patients_count - good for stacked bar chart
        load_df_pf = patient_focus_data_dict["patient_load_by_condition_df"]
        st.plotly_chart(plot_bar_chart_web(
            load_df_pf, x_col='period_start_date', y_col='unique_patients_count', color_col='condition',
            title_bar="Patient Load by Condition", barmode_web='stack', y_axis_is_count=True,
            color_discrete_map_web=app_config.LEGACY_DISEASE_COLORS_WEB # Using legacy for demo
        ), use_container_width=True)

    if patient_focus_data_dict.get("flagged_patients_for_review_df") is not None:
        st.subheader("Flagged Patients for Clinical Review")
        st.dataframe(patient_focus_data_dict["flagged_patients_for_review_df"].head(15), use_container_width=True)
    if patient_focus_data_dict.get("data_availability_notes"):
        for note in patient_focus_data_dict["data_availability_notes"]: st.caption(f"Note: {note}")


with tab_env_details:
    env_details_data_dict = prepare_clinic_environment_details_data(
        period_iot_df_clinic, True, period_str_clinic_console # Assume IoT globally available if period_iot_df_clinic passed
    )
    # Display environmental trends and latest readings
    # ... This part needs fleshing out for presentation ...
    if env_details_data_dict.get("current_environmental_alerts_summary"):
        st.subheader("Current Environmental Alerts (Based on Latest in Period)")
        for alert in env_details_data_dict["current_environmental_alerts_summary"]:
            st.markdown(f"- **{alert['alert_type']}**: {alert['message']} (Status: {alert['level']})")
    
    co2_trend_series_env = env_details_data_dict.get("hourly_avg_co2_trend_series")
    if co2_trend_series_env is not None and not co2_trend_series_env.empty:
        st.plotly_chart(plot_annotated_line_chart_web(
            data_series_input=co2_trend_series_env, chart_title="Hourly Avg. CO2 Levels", y_axis_label="CO2 (ppm)",
            target_ref_line=app_config.ALERT_AMBIENT_CO2_HIGH_PPM, date_display_format="%H:%M (%d-%b)"
        ), use_container_width=True)

    if env_details_data_dict.get("latest_sensor_readings_by_room_df") is not None:
        st.subheader("Latest Sensor Readings by Room")
        st.dataframe(env_details_data_dict["latest_sensor_readings_by_room_df"], use_container_width=True)
    if env_details_data_dict.get("data_availability_notes"):
        for note in env_details_data_dict["data_availability_notes"]: st.caption(f"Note: {note}")

logger.info(f"Clinic Management Console for period {period_str_clinic_console} generated.")
