# test/pages/1_chw_dashboard.py
# Redesigned as a "CHW Supervisor Operations View" for "Sentinel Health Co-Pilot"
# This page simulates what a CHW Supervisor or Hub Coordinator might see on a
# web interface (Tablet/Laptop at Tier 1 Hub or Tier 2 Facility Node).
# It provides an overview of CHW activities, escalated alerts, and team performance.

import streamlit as st
import pandas as pd
import numpy as np # For np.nan if needed
import os
import logging
from datetime import date, timedelta

# --- Sentinel System Imports ---
# Assuming app_config and utils are in PYTHONPATH or project root
from config import app_config # Uses the new, redesigned app_config
from utils.core_data_processing import ( # For loading base data for simulation
    load_health_records
)
# Refactored CHW component data preparation functions
# These now return structured data, not render UI directly.
from pages.chw_components_sentinel.summary_metrics_calculator import calculate_chw_daily_summary_metrics
from pages.chw_components_sentinel.alert_generator import generate_chw_patient_alerts_from_data
from pages.chw_components_sentinel.epi_signal_extractor import extract_chw_local_epi_signals
from pages.chw_components_sentinel.task_processor import generate_chw_prioritized_tasks_summary # Renamed, focus on summary for supervisor
from pages.chw_components_sentinel.activity_trend_calculator import calculate_chw_activity_trends

# Refactored UI helpers for web reports (Tier 2/3 views)
from utils.ui_visualization_helpers import (
    render_web_kpi_card,
    render_web_traffic_light_indicator, # For alert summaries
    plot_annotated_line_chart_web,
    plot_bar_chart_web
    # Other plotters if needed
)

# Page Configuration (Specific to this Supervisor View)
st.set_page_config(
    page_title=f"CHW Supervisor View - {app_config.APP_NAME}",
    layout="wide",
    initial_sidebar_state="expanded"
)

logger = logging.getLogger(__name__)

# --- CSS Loading (for Web Views) ---
# This assumes app_home.py or another entry point has already loaded global CSS.
# If this page can be run standalone or CSS isn't inherited, uncomment:
# @st.cache_resource
# def load_supervisor_css():
#     css_path = app_config.STYLE_CSS_PATH_WEB # Use web-specific styles
#     if os.path.exists(css_path):
#         with open(css_path, encoding="utf-8") as f:
#             st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
# load_supervisor_css()

# --- Data Loading (Simulating Supervisor Access to CHW Data) ---
# For a real system, this data would come from synced PEDs via a Hub/Facility Node.
# Here, we load the main health records and then filter for a specific CHW/Team and period.

@st.cache_data(ttl=app_config.CACHE_TTL_SECONDS_WEB_REPORTS, show_spinner="Loading operational data for supervisor view...")
def get_supervisor_view_data(selected_date: date, chw_id_filter: Optional[str] = None, team_id_filter: Optional[str] = None):
    """
    Simulates fetching and preparing data for the supervisor view.
    In a real system, this would query an aggregated database at a Hub or Facility Node.
    """
    # For simulation, load all health records
    # TODO: In a real Tier 1/2 system, `load_health_records` might point to a local aggregated DB view for CHW activities.
    #       The PED does not load this file directly.
    health_df_all = load_health_records(file_path=app_config.HEALTH_RECORDS_CSV, source_context="SupervisorViewSim")
    if health_df_all.empty:
        logger.error("Supervisor View: Raw health records load failed. Cannot proceed.")
        return pd.DataFrame(), pd.DataFrame(), {} # Empty daily, period, and pre-calc KPIs

    # Filter for the selected date for "daily" views
    # Ensure 'encounter_date' is datetime for filtering
    if 'encounter_date' not in health_df_all.columns or not pd.api.types.is_datetime64_any_dtype(health_df_all['encounter_date']):
         health_df_all['encounter_date'] = pd.to_datetime(health_df_all['encounter_date'], errors='coerce')
    
    daily_df_for_supervisor = health_df_all[health_df_all['encounter_date'].dt.date == selected_date].copy()
    
    # TODO: Implement CHW ID / Team ID filtering if those columns exist in the simulated data
    # For now, supervisor sees all CHW activity for that date/period for simplicity.
    # if chw_id_filter and 'chw_id_column' in daily_df_for_supervisor:
    #     daily_df_for_supervisor = daily_df_for_supervisor[daily_df_for_supervisor['chw_id_column'] == chw_id_filter]
    # Placeholder for `get_chw_summary` data. In a real system, this might be pre-calculated daily.
    # Here, we pass an empty dict and let `calculate_chw_daily_summary_metrics` work from daily_df if needed.
    pre_calculated_kpis_for_day = {} 
    # For trend analysis, supervisor usually looks at a longer period.
    # For simulation, we pass the whole dataset (it will be date-filtered by the trend calc function).
    period_df_for_trends_supervisor = health_df_all.copy()

    return daily_df_for_supervisor, period_df_for_trends_supervisor, pre_calculated_kpis_for_day

# --- Page Title & Introduction for Supervisor ---
st.title("🧑‍🏫 CHW Supervisor Operations View")
st.markdown(f"**Team Performance, Alert Management, and Field Support Overview for {app_config.APP_NAME}**")
st.markdown("---")

# --- Sidebar Filters for Supervisor ---
# Supervisor might want to look at specific dates, CHWs, or zones.
if os.path.exists(app_config.APP_LOGO_SMALL): # Use smaller logo for consistency in sub-pages
    st.sidebar.image(app_config.APP_LOGO_SMALL, width=180)
st.sidebar.header("🗓️ Supervisor Filters")

# Date selection for daily snapshot
# Determine overall min/max dates from a full (simulated) dataset for the date picker
# This would ideally come from the bounds of available synced CHW data.
# For demo, we'll use a fixed range or derive from app_config's typical trend days.
min_hist_date = date.today() - timedelta(days=app_config.WEB_DASHBOARD_DEFAULT_DATE_RANGE_DAYS_TREND * 2)
max_hist_date = date.today()
default_supervisor_view_date = max_hist_date

selected_date_supervisor = st.sidebar.date_input(
    "View Daily Summary For:",
    value=default_supervisor_view_date,
    min_value=min_hist_date,
    max_value=max_hist_date,
    key="supervisor_daily_date_selector_v1"
)

# TODO: Add CHW ID or Team ID selector if data supports it
# selected_chw_supervisor = st.sidebar.selectbox("Filter by CHW ID:", options=["All CHWs"] + available_chw_ids, ...)
# selected_zone_supervisor = st.sidebar.selectbox("Filter by Zone:", options=["All Zones"] + available_zones, ...)
# For now, we use placeholder values.
active_chw_filter_placeholder = None # "CHW001"
active_zone_filter_placeholder = None # "ZoneA"

# Load data based on selected date
daily_chw_data_sup, period_chw_data_sup, pre_calc_kpis_sup = get_supervisor_view_data(
    selected_date_supervisor,
    active_chw_filter_placeholder
)

# --- Section 1: Daily Snapshot Summary Metrics (using refactored calculator) ---
st.header(f"📈 Daily Activity Snapshot: {selected_date_supervisor.strftime('%A, %B %d, %Y')}")
if not daily_chw_data_sup.empty:
    # Pass pre_calc_kpis_sup (can be empty), and daily_chw_data_sup for detailed calcs
    summary_metrics_data = calculate_chw_daily_summary_metrics(
        chw_daily_kpi_input_data=pre_calc_kpis_sup, # This might be enhanced if daily summaries are pre-aggregated
        chw_daily_encounter_df=daily_chw_data_sup,
        for_date=selected_date_supervisor
    )
    
    # Display these metrics using render_web_kpi_card
    cols_kpi_sup_1 = st.columns(4)
    with cols_kpi_sup_1[0]:
        render_web_kpi_card(title="Total Visits", value_str=str(summary_metrics_data.get("visits_count",0)), icon="👥", units="visits",
                           status_level="NEUTRAL" ) # Status can be refined based on targets
    with cols_kpi_sup_1[1]:
        render_web_kpi_card(title="High Priority Follow-ups", value_str=str(summary_metrics_data.get("high_ai_prio_followups_count",0)), icon="🎯", units="tasks",
                           status_level="MODERATE_CONCERN" if summary_metrics_data.get("high_ai_prio_followups_count",0) > 5 else "ACCEPTABLE")
    with cols_kpi_sup_1[2]:
        avg_risk = summary_metrics_data.get("avg_risk_of_visited_patients", np.nan)
        render_web_kpi_card(title="Avg. Risk (Visited Patients)", value_str=f"{avg_risk:.1f}" if pd.notna(avg_risk) else "N/A", icon="📈",
                           status_level="HIGH_RISK" if pd.notna(avg_risk) and avg_risk > app_config.RISK_SCORE_MODERATE_THRESHOLD else "ACCEPTABLE")
    with cols_kpi_sup_1[3]: # Worker self-fatigue - new concept
        fatigue_code = summary_metrics_data.get("worker_self_fatigue_level_code", "NOT_ASSESSED")
        fatigue_status_map = {"HIGH": "HIGH_CONCERN", "MODERATE": "MODERATE_CONCERN", "LOW": "GOOD_PERFORMANCE", "NOT_ASSESSED": "NEUTRAL"}
        render_web_kpi_card(title="CHW Team Fatigue (Avg.)", value_str=fatigue_code.replace("_"," ").title(), icon="😓",
                           status_level=fatigue_status_map.get(fatigue_code, "NEUTRAL"),
                           help_text="Average self-reported or AI-derived fatigue level for CHWs on this day. (Simulated for supervisor view).")
    
    cols_kpi_sup_2 = st.columns(4)
    # Add more relevant supervisor KPIs like "Fever Cases Identified", "Critical SpO2 Cases", "Pending Critical Referrals"
    with cols_kpi_sup_2[0]:
        render_web_kpi_card("Fever Cases Identified", value_str=str(summary_metrics_data.get("fever_cases_identified_count",0)), icon="🔥", units="cases")
    with cols_kpi_sup_2[1]:
        render_web_kpi_card("Critical SpO2 Cases", value_str=str(summary_metrics_data.get("critical_spo2_cases_identified_count",0)), icon="💨", units="cases",
                           status_level="HIGH_CONCERN" if summary_metrics_data.get("critical_spo2_cases_identified_count",0) > 0 else "ACCEPTABLE")
    # Add more cards...

else:
    st.info(f"No CHW activity data found for {selected_date_supervisor.strftime('%B %d, %Y')} "
            f"{('for CHW ' + active_chw_filter_placeholder) if active_chw_filter_placeholder else ''}.")

st.markdown("---")

# --- Section 2: Escalated Alerts & High Priority Task Summary ---
st.header("🚨 Alerts & Priority Task Overview")
# This uses `generate_chw_patient_alerts_from_data` and potentially `generate_chw_prioritized_tasks_summary`
# For supervisors, show a summary or a list of CRITICAL alerts escalated.
alert_data_for_supervisor = generate_chw_patient_alerts_from_data(
    patient_alerts_tasks_df=daily_chw_data_sup, # Daily data acts as source for alerts here
    chw_daily_encounter_df=daily_chw_data_sup,
    for_date=selected_date_supervisor,
    chw_zone_context=active_zone_filter_placeholder or "All Supervised Zones",
    max_alerts_to_return=5 # Supervisor sees top few critical/high priority
)

if alert_data_for_supervisor:
    st.subheader(f"Top Escalated Patient Alerts ({selected_date_supervisor.strftime('%d %b %Y')}):")
    for alert_item in alert_data_for_supervisor:
        if alert_item.get("alert_level") == "CRITICAL": # Supervisor prioritizes CRITICAL
            render_web_traffic_light_indicator(
                message=f"Patient {alert_item['patient_id']}: {alert_item['primary_reason']}",
                status_level="HIGH_RISK", # Mapping "CRITICAL" to "HIGH_RISK" color
                details_text=f"{alert_item['brief_details']} | Context: {alert_item['context_info']} | Prio Score: {alert_item['raw_priority_score']:.0f}"
            )
    # Could add a table of high-priority tasks for the team if `generate_chw_prioritized_tasks_summary` output is suitable
    # Example: tasks_summary_df = generate_chw_prioritized_tasks_summary(...)
    # if tasks_summary_df is not None and not tasks_summary_df.empty:
    #    st.dataframe(tasks_summary_df[['patient_id', 'task_description', 'priority_score', 'due_date', 'status']])

else:
    st.info("No critical patient alerts escalated or high priority tasks needing immediate attention for this day/filter.")

st.markdown("---")

# --- Section 3: Local Epi Signals (Supervisor might monitor specific zones) ---
st.header("🌿 Local Epidemiological Signals")
# For this, we might let supervisor select a zone if not already filtered.
# Default to supervisor's overall area for now.
current_zone_for_epi_view = active_zone_filter_placeholder or "All Supervised Zones"

epi_signals_data = extract_chw_local_epi_signals(
    chw_daily_encounter_df=daily_chw_data_sup,
    pre_calculated_chw_kpis=pre_calc_kpis_sup,
    for_date=selected_date_supervisor,
    chw_zone_context=current_zone_for_epi_view
)

if epi_signals_data:
    col_epi1, col_epi2 = st.columns(2)
    with col_epi1:
        render_web_kpi_card(title="New Symptomatic (Key Cond.)", value_str=str(epi_signals_data.get("new_symptomatic_cases_key_conditions_count",0)), icon="🤒",
                            help_text=f"Keywords: {epi_signals_data.get('symptomatic_keywords_monitored', 'N/A')}")
        render_web_kpi_card(title=f"New Malaria Cases ({current_zone_for_epi_view})", value_str=str(epi_signals_data.get("new_malaria_cases_today_count",0)), icon="🦟")
    with col_epi2:
        render_web_kpi_card(title=f"Pending TB Contact Traces ({current_zone_for_epi_view})", value_str=str(epi_signals_data.get("pending_tb_contact_traces_count",0)), icon="👥")
        # Display symptom cluster alerts if any
    if epi_signals_data.get("reported_symptom_cluster_alerts"):
        st.subheader("Symptom Cluster Alerts:")
        for cluster in epi_signals_data["reported_symptom_cluster_alerts"]:
            st.warning(f"Cluster Detected: {cluster['symptoms']} (Count: {cluster['count']}) in {cluster.get('location_hint','area')}")
else:
    st.info(f"No specific epidemiological signals to report for {current_zone_for_epi_view} on {selected_date_supervisor.strftime('%d %b')}.")

st.markdown("---")

# --- Section 4: CHW Activity Trends (Supervisor View) ---
st.header("📊 CHW Activity Trends (Periodic Overview)")
# Date range for trends (e.g., last 30 days)
trend_end_sup = selected_date_supervisor # End trends at the selected daily view date
trend_start_sup = trend_end_sup - timedelta(days=app_config.WEB_DASHBOARD_DEFAULT_DATE_RANGE_DAYS_TREND - 1)
# Make sure trend_start is not before historical data allows
if trend_start_sup < min_hist_date : trend_start_sup = min_hist_date


st.markdown(f"Displaying trends from **{trend_start_sup.strftime('%d %b %Y')}** to **{trend_end_sup.strftime('%d %b %Y')}**")
# Using `period_chw_data_sup` which is currently all historical data; it gets filtered by `calculate_chw_activity_trends`
activity_trends_data = calculate_chw_activity_trends(
    chw_historical_health_df=period_chw_data_sup,
    trend_start_date=trend_start_sup,
    trend_end_date=trend_end_sup,
    zone_filter=active_zone_filter_placeholder, # Apply zone filter if selected for trends
    time_period_agg='D' # Daily trends for supervisor view
)

cols_trends_sup = st.columns(2)
with cols_trends_sup[0]:
    visits_trend_df = activity_trends_data.get("patient_visits_trend")
    if visits_trend_df is not None and not visits_trend_df.empty:
        st.plotly_chart(plot_annotated_line_chart_web(
            data_series_input=visits_trend_df.squeeze(), # Ensure it's a Series if single column df
            chart_title="Daily Patient Visits", y_axis_label="# Patients", y_axis_is_count=True
        ), use_container_width=True)
    else: st.caption("No patient visit trend data for selected period/filter.")

with cols_trends_sup[1]:
    prio_trend_df = activity_trends_data.get("high_priority_followups_trend")
    if prio_trend_df is not None and not prio_trend_df.empty:
        st.plotly_chart(plot_annotated_line_chart_web(
            data_series_input=prio_trend_df.squeeze(),
            chart_title="Daily High Priority Follow-ups", y_axis_label="# Follow-ups", y_axis_is_count=True
        ), use_container_width=True)
    else: st.caption("No high-priority follow-up trend data for selected period/filter.")

logger.info(f"CHW Supervisor View for date {selected_date_supervisor} (User: Supervisor) generated.")
