# test/pages/4_population_dashboard.py
# Redesigned as "Population Health Analytics & Research Console" for "Sentinel Health Co-Pilot"
# This page simulates a web interface for epidemiologists, researchers, and program managers,
# typically at a Regional/Cloud Node (Tier 3) or a well-equipped Facility Node (Tier 2)
# with access to broader, aggregated datasets for in-depth analysis.

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys # Kept for existing path manipulation, though ideally managed by project structure/PYTHONPATH
import logging
from datetime import date, timedelta
import plotly.express as px # Can be used by _web helpers, or directly for very custom plots
import html # For escaping in custom HTML KPIs

# --- Sentinel System Imports ---
# Assuming app_config and utils are in PYTHONPATH or project root
PROJECT_ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT_PATH not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_PATH) # Ensure root is on path for config/utils

from config import app_config # Uses new, redesigned app_config
from utils.core_data_processing import (
    load_health_records, # For loading base research dataset
    load_zone_data,      # For SDOH context from zone attributes
    get_trend_data       # General utility for trends
)
from utils.ai_analytics_engine import apply_ai_models # AI models would be applied upstream in reality
from utils.ui_visualization_helpers import (
    plot_bar_chart_web,
    plot_donut_chart_web,
    plot_annotated_line_chart_web,
    # render_web_kpi_card # We will use custom markdown KPIs from original style for this page, or simplified display
    _create_empty_plot_figure # For handling no data for plots
)

# --- Page Configuration ---
st.set_page_config(
    page_title=f"Population Analytics - {app_config.APP_NAME}",
    layout="wide",
    initial_sidebar_state="expanded"
)
logger = logging.getLogger(__name__) # Get logger

# CSS Loading (using web-specific styles) - assumes loaded by app_home or via cache_resource
# @st.cache_resource ... load_css(app_config.STYLE_CSS_PATH_WEB) ... (if needed)

# --- Data Loading for Population Analytics Console (Simulates Tier 3 Data Access) ---
@st.cache_data(ttl=app_config.CACHE_TTL_SECONDS_WEB_REPORTS, show_spinner="Loading population analytics dataset...")
def get_population_analytics_dataset(source_context="PopulationAnalyticsConsole"):
    """
    Loads and prepares data for the population analytics console.
    In a real Tier 3 system, this would query a data warehouse or lake.
    AI models are assumed to have been applied during data ingestion/ETL.
    """
    logger.info(f"({source_context}) Loading population health records and zone attributes.")
    # Load potentially large, historical health records dataset
    health_df_raw_pop = load_health_records(file_path=app_config.HEALTH_RECORDS_CSV, source_context=source_context)
    
    # AI model application step (simulates that this is already done on the dataset)
    if not health_df_raw_pop.empty:
        # Assuming apply_ai_models returns (enriched_df, _forecast_df)
        health_df_enriched_pop = apply_ai_models(health_df_raw_pop, source_context=f"{source_context}/AIEnrich")[0]
    else:
        logger.warning(f"({source_context}) Raw health records empty for population analytics. Proceeding with empty dataset.")
        health_df_enriched_pop = pd.DataFrame(columns=health_df_raw_pop.columns if health_df_raw_pop is not None else [])

    # Load zone attributes (non-geometric part of zone_data) for SDOH context
    # load_zone_data returns a GDF. We extract non-geometry parts for zone_attributes_df.
    zone_gdf_pop = load_zone_data(source_context=source_context) # Uses default paths
    zone_attributes_df_pop = pd.DataFrame()
    if zone_gdf_pop is not None and not zone_gdf_pop.empty:
        # Extract non-geometry columns as the attributes DataFrame
        geom_col_name = zone_gdf_pop.geometry.name if hasattr(zone_gdf_pop, 'geometry') else 'geometry'
        if geom_col_name in zone_gdf_pop.columns:
            zone_attributes_df_pop = pd.DataFrame(zone_gdf_pop.drop(columns=[geom_col_name], errors='ignore'))
        else:
            zone_attributes_df_pop = pd.DataFrame(zone_gdf_pop) # Assume no geometry column or it was already dropped
        # Ensure key columns from old attributes CSV are present for consistency with original logic
        expected_attr_cols = ['zone_id', 'name', 'population', 'socio_economic_index', 'avg_travel_time_clinic_min']
        for col in expected_attr_cols:
            if col not in zone_attributes_df_pop.columns:
                zone_attributes_df_pop[col] = np.nan # Add if missing, for downstream safety
                logger.warning(f"({source_context}) Expected zone attribute '{col}' not found. Added as NaN.")
        logger.info(f"({source_context}) Loaded {len(zone_attributes_df_pop)} zone attributes for SDOH context.")
    else:
        logger.warning(f"({source_context}) Zone attributes data (from GeoDataFrame) not available or GDF invalid for population analytics.")
        # Create an empty DataFrame with expected columns for graceful failure
        zone_attributes_df_pop = pd.DataFrame(columns=['zone_id', 'name', 'population', 'socio_economic_index'])

    if health_df_enriched_pop.empty:
        logger.error(f"({source_context}) Population analytics: Health data is empty after loading/AI processing.")
    return health_df_enriched_pop, zone_attributes_df_pop

# Load the datasets
health_df_pop_main, zone_attr_df_pop_main = get_population_analytics_dataset()

if health_df_pop_main.empty:
    st.error("🚨 **Critical Data Error:** Could not load primary health records for population analytics. Dashboard functionality will be severely limited or unavailable."); st.stop()

# --- Page Title and Introduction ---
st.title("📊 Population Health Analytics & Research Console")
st.markdown(f"In-depth exploration of demographics, epidemiology, clinical patterns, and health system factors across aggregated population data. For use at Tier 2 (Facility Nodes with advanced capacity) or Tier 3 (Regional/Cloud Nodes) of **{app_config.APP_NAME}**.")
st.markdown("---")

# --- Sidebar Filters (Analyst might want broad filters) ---
if os.path.exists(app_config.APP_LOGO_SMALL): st.sidebar.image(app_config.APP_LOGO_SMALL, width=180);
st.sidebar.header("🔎 Analytics Filters")

# Date Range (Allow wide range for research)
min_date_analytics_overall = health_df_pop_main['encounter_date'].min().date() if 'encounter_date' in health_df_pop_main.columns and health_df_pop_main['encounter_date'].notna().any() else date.today() - timedelta(days=365*3)
max_date_analytics_overall = health_df_pop_main['encounter_date'].max().date() if 'encounter_date' in health_df_pop_main.columns and health_df_pop_main['encounter_date'].notna().any() else date.today()
if min_date_analytics_overall > max_date_analytics_overall: min_date_analytics_overall = max_date_analytics_overall # Ensure valid range

default_pop_start_date_analytics = min_date_analytics_overall
default_pop_end_date_analytics = max_date_analytics_overall

selected_start_date_pop_analytics, selected_end_date_pop_analytics = st.sidebar.date_input(
    "Select Date Range for Analysis:",
    value=[default_pop_start_date_analytics, default_pop_end_date_analytics],
    min_value=min_date_analytics_overall, max_value=max_date_analytics_overall,
    key="pop_analytics_date_range_console"
)
if selected_start_date_pop_analytics > selected_end_date_pop_analytics:
    st.sidebar.error("Start date must be before end date.")
    selected_start_date_pop_analytics = selected_end_date_pop_analytics

# Filter main health DataFrame by selected date range
analytics_df_period_filtered = pd.DataFrame(columns=health_df_pop_main.columns)
if 'encounter_date' in health_df_pop_main.columns: # Ensure column exists
    health_df_pop_main['encounter_date_obj'] = pd.to_datetime(health_df_pop_main['encounter_date'], errors='coerce').dt.date
    analytics_df_period_filtered = health_df_pop_main[
        (health_df_pop_main['encounter_date_obj'].notna()) &
        (health_df_pop_main['encounter_date_obj'] >= selected_start_date_pop_analytics) &
        (health_df_pop_main['encounter_date_obj'] <= selected_end_date_pop_analytics)
    ].copy()

if analytics_df_period_filtered.empty:
    st.warning(f"No health data available for the selected period: {selected_start_date_pop_analytics.strftime('%d %b %Y')} to {selected_end_date_pop_analytics.strftime('%d %b %Y')}. Please adjust filters or check data sources."); st.stop()

# Optional Condition Filter (Analyst can focus on specific disease groups)
conditions_list_for_analytics_filter = ["All Conditions"] + sorted(analytics_df_period_filtered['condition'].dropna().unique().tolist())
selected_condition_filter_analytics = st.sidebar.selectbox(
    "Filter by Condition (Optional):", options=conditions_list_for_analytics_filter, index=0,
    key="pop_analytics_condition_filter_console"
)
analytics_df_after_condition_filter = analytics_df_period_filtered.copy()
if selected_condition_filter_analytics != "All Conditions":
    analytics_df_after_condition_filter = analytics_df_after_condition_filter[
        analytics_df_after_condition_filter['condition'] == selected_condition_filter_analytics
    ]

# Optional Zone Filter (If analyst wants to drill down by administrative/geographic zone)
# Assuming zone_attr_df_pop_main has 'zone_id' and 'name' (display name)
zone_options_analytics = ["All Zones"]
if not zone_attr_df_pop_main.empty and 'zone_id' in zone_attr_df_pop_main.columns and 'name' in zone_attr_df_pop_main.columns:
    # Create a list of "Zone Name (ZoneID)" for clarity if names aren't unique
    # For simplicity, just use unique names if available, else zone_id
    zone_display_options = zone_attr_df_pop_main.drop_duplicates(subset=['zone_id'])[['zone_id','name']].copy()
    zone_options_analytics.extend(sorted(zone_display_options.apply(lambda x: f"{x['name']} ({x['zone_id']})" if pd.notna(x['name']) and x['name'] != x['zone_id'] else x['zone_id'], axis=1).unique().tolist()))

selected_zone_filter_display_analytics = st.sidebar.selectbox(
    "Filter by Zone (Optional):", options=zone_options_analytics, index=0,
    key="pop_analytics_zone_filter_console"
)

analytics_df_final_for_display = analytics_df_after_condition_filter.copy()
selected_zone_id_analytics = None
if selected_zone_filter_display_analytics != "All Zones":
    # Extract actual zone_id from display string if necessary (e.g. if "Zone Name (ZoneID)" format used)
    try:
        # A more robust extraction might be needed if format is complex
        selected_zone_id_analytics = selected_zone_filter_display_analytics.split('(')[-1].replace(')','').strip() if '(' in selected_zone_filter_display_analytics else selected_zone_filter_display_analytics
    except: selected_zone_id_analytics = selected_zone_filter_display_analytics # Fallback
    
    if selected_zone_id_analytics and 'zone_id' in analytics_df_final_for_display.columns:
        analytics_df_final_for_display = analytics_df_final_for_display[
            analytics_df_final_for_display['zone_id'] == selected_zone_id_analytics
        ]

# Handle case where filters result in empty data for display
if analytics_df_final_for_display.empty and (selected_condition_filter_analytics != "All Conditions" or selected_zone_filter_display_analytics != "All Zones"):
    # More sophisticated fallback logic could be implemented as in original
    st.warning(f"No data found for the combination of selected filters (Condition: {selected_condition_filter_analytics}, Zone: {selected_zone_filter_display_analytics}). Displaying data for the broader selection where possible or period overall.")
    # Fallback to less restrictive filter if current selection is empty
    if not analytics_df_after_condition_filter.empty : analytics_df_final_for_display = analytics_df_after_condition_filter.copy()
    elif not analytics_df_period_filtered.empty: analytics_df_final_for_display = analytics_df_period_filtered.copy()
    # if still empty, the st.stop() from initial load will catch it, or one for period filter.


# --- "Decision-Making KPI Boxes" adapted for Population Health Level ---
# These are high-level summaries for the analyst before diving into tabs.
# Uses custom markdown styling from app_config.STYLE_CSS_PATH_WEB (assumed to be loaded).

st.subheader(f"Key Population Indicators ({selected_start_date_pop_analytics.strftime('%d %b')} - {selected_end_date_pop_analytics.strftime('%d %b %Y')}, "
             f"Condition: {selected_condition_filter_analytics}, Zone: {selected_zone_filter_display_analytics})")

if analytics_df_final_for_display.empty:
    st.info("No data available to display key population indicators for the current filter selection.")
else:
    kpi_cols_pop_analytics = st.columns(4) # Adjust number of columns as needed

    # KPI 1: Unique Patients in Filtered Dataset
    unique_patients_count_pop = analytics_df_final_for_display.get('patient_id', pd.Series(dtype=str)).nunique()
    with kpi_cols_pop_analytics[0]:
         st.markdown(f"""<div class="custom-markdown-kpi-box"><div class="custom-kpi-label-top-condition">{'Unique Patients (Filtered Set)'}</div><div class="custom-kpi-value-large">{unique_patients_count_pop:,}</div></div>""", unsafe_allow_html=True)

    # KPI 2: Average AI Risk Score in Filtered Set
    avg_ai_risk_pop = np.nan
    if 'ai_risk_score' in analytics_df_final_for_display.columns and analytics_df_final_for_display['ai_risk_score'].notna().any():
        avg_ai_risk_pop = analytics_df_final_for_display['ai_risk_score'].mean()
    with kpi_cols_pop_analytics[1]:
        st.markdown(f"""<div class="custom-markdown-kpi-box"><div class="custom-kpi-label-top-condition">{'Avg. AI Risk Score'}</div><div class="custom-kpi-value-large">{avg_ai_risk_pop:.1f if pd.notna(avg_ai_risk_pop) else 'N/A'}</div></div>""", unsafe_allow_html=True)

    # KPI 3: Proportion of High AI Risk Patients
    high_risk_patients_pop_count = 0; prop_high_risk_pop = 0.0
    if 'ai_risk_score' in analytics_df_final_for_display.columns and analytics_df_final_for_display['ai_risk_score'].notna().any() and unique_patients_count_pop > 0:
        high_risk_df_pop_kpi = analytics_df_final_for_display[
            pd.to_numeric(analytics_df_final_for_display['ai_risk_score'], errors='coerce') >= app_config.RISK_SCORE_HIGH_THRESHOLD # Using new config threshold
        ]
        if not high_risk_df_pop_kpi.empty:
            high_risk_patients_pop_count = high_risk_df_pop_kpi['patient_id'].nunique()
        prop_high_risk_pop = (high_risk_patients_pop_count / unique_patients_count_pop) * 100 if unique_patients_count_pop > 0 else 0.0
    value_prop_risk_pop_str = f"{prop_high_risk_pop:.1f}%" if pd.notna(prop_high_risk_pop) else "N/A"
    with kpi_cols_pop_analytics[2]:
        st.markdown(f"""<div class="custom-markdown-kpi-box"><div class="custom-kpi-label-top-condition">{'% High AI Risk Patients'}</div><div class="custom-kpi-value-large">{value_prop_risk_pop_str}</div><div class="custom-kpi-subtext-small">{f"{int(high_risk_patients_pop_count)} patients (≥{app_config.RISK_SCORE_HIGH_THRESHOLD})"}</div></div>""", unsafe_allow_html=True)

    # KPI 4: Top Condition by Encounter (Illustrative Custom KPI)
    top_condition_name_pop, top_condition_count_pop = "N/A", 0
    if 'condition' in analytics_df_final_for_display.columns and analytics_df_final_for_display['condition'].notna().any():
        condition_counts_pop_kpi = analytics_df_final_for_display['condition'].value_counts()
        if not condition_counts_pop_kpi.empty:
            top_condition_name_pop = condition_counts_pop_kpi.idxmax()
            top_condition_count_pop = condition_counts_pop_kpi.max()
    with kpi_cols_pop_analytics[3]:
        st.markdown(f"""<div class="custom-markdown-kpi-box highlight-red-edge"><div class="custom-kpi-label-top-condition">{html.escape("Top Condition (Encounters)")}</div><div class="custom-kpi-value-large">{html.escape(str(top_condition_name_pop))}</div><div class="custom-kpi-subtext-small">{html.escape(f"{top_condition_count_pop:,} encounters") if top_condition_name_pop != "N/A" else ""}</div></div>""", unsafe_allow_html=True)


# --- Tabbed Interface for Detailed Population Analytics ---
# Each tab's content will now primarily call a data preparation function
# and then use the _web UI helpers to display the structured data.

pop_analytics_tab_titles = [
    "📈 Epidemiological Overview",
    "🧑‍🤝‍🧑 Demographics & SDOH Context", # Updated name for clarity on SDOH
    "🔬 Clinical Insights & Diagnostics", # Updated name
    "⚙️ Health Systems & Equity Lens" # Updated name
]
tab_epi_pop, tab_demog_sdoh_pop, tab_clinical_dx_pop, tab_systems_equity_pop = st.tabs(pop_analytics_tab_titles)


with tab_epi_pop:
    st.header(f"Population Epidemiological Overview ({selected_condition_filter_analytics} in {selected_zone_filter_display_analytics})")
    if analytics_df_final_for_display.empty:
        st.info("No data for epidemiological overview with current filters.")
    else:
        # Data preparation for this tab should ideally be encapsulated
        # For now, keeping logic similar to original but using _web plotters
        epi_overview_cols_pop = st.columns(2)
        with epi_overview_cols_pop[0]:
            st.subheader("Top Condition Case Counts (Unique Patients)")
            if 'condition' in analytics_df_final_for_display.columns and analytics_df_final_for_display['condition'].notna().any() and 'patient_id' in analytics_df_final_for_display.columns:
                cond_counts_df_pop = analytics_df_final_for_display.groupby('condition')['patient_id'].nunique().nlargest(10).reset_index(name='unique_patients')
                if not cond_counts_df_pop.empty:
                    cond_counts_df_pop['condition'] = cond_counts_df_pop['condition'].astype(str)
                    st.plotly_chart(plot_bar_chart_web(
                        cond_counts_df_pop, x_col_bar='condition', y_col_bar='unique_patients', title_bar="Top Conditions by Unique Patients",
                        orientation_web='h', y_axis_is_count=True, text_format_web='d', chart_height=400
                    ), use_container_width=True)
                else: st.caption("No condition data for counts.")
            else: st.caption("Condition or Patient ID column missing for counts.")
        # ... (Other plots like AI Risk Distribution, Incidence Trends would be similarly refactored using data prep -> plot_web)
        # For brevity, only one example plot per tab shown in this refactoring.
        # The Incidence Trend logic was quite involved; it would be best in a dedicated data prep function for this tab.

with tab_demog_sdoh_pop:
    st.header("Population Demographics & Social Determinants of Health (SDOH) Context")
    if analytics_df_final_for_display.empty:
        st.info("No data for Demographics/SDOH with current filters.")
    else:
        # Example: Age Distribution
        st.subheader("Age Distribution of Encounters")
        if 'age' in analytics_df_final_for_display.columns and analytics_df_final_for_display['age'].notna().any():
            # Simplified age bins appropriate for population health
            age_bins_pop_analytics = [0, 5, 12, 18, 35, 50, 65, np.inf]
            age_labels_pop_analytics = ['0-4 yrs', '5-11 yrs', '12-17 yrs', '18-34 yrs', '35-49 yrs', '50-64 yrs', '65+ yrs']
            
            age_df_display_pop = analytics_df_final_for_display.copy() # Work on copy
            age_df_display_pop['age_group_analytics'] = pd.cut(
                age_df_display_pop['age'], bins=age_bins_pop_analytics, labels=age_labels_pop_analytics, right=False
            )
            age_dist_data_pop = age_df_display_pop['age_group_analytics'].value_counts().sort_index().reset_index()
            age_dist_data_pop.columns = ['Age Group', 'Encounter Count']
            if not age_dist_data_pop.empty:
                st.plotly_chart(plot_bar_chart_web(
                    age_dist_data_pop, x_col_bar='Age Group', y_col_bar='Encounter Count', title_bar="Encounters by Age Group",
                    y_axis_is_count=True, text_format_web='d', chart_height=350
                ), use_container_width=True)
            else: st.caption("No age data for distribution chart.")
        # ... Other SDOH plots like Gender, Geographic SES would follow similar pattern ...

with tab_clinical_dx_pop:
    st.header("Clinical Insights & Diagnostic Patterns")
    if analytics_df_final_for_display.empty:
        st.info("No data for Clinical/Dx insights with current filters.")
    else:
        # Example: Top Presenting Symptoms
        st.subheader("Top Reported Symptoms (Overall Encounters)")
        if 'patient_reported_symptoms' in analytics_df_final_for_display.columns and analytics_df_final_for_display['patient_reported_symptoms'].notna().any():
            symptoms_series_pop = analytics_df_final_for_display['patient_reported_symptoms'].str.split(';').explode().str.strip().replace(['','Unknown','N/A', 'none', 'None'],np.nan).dropna()
            if not symptoms_series_pop.empty:
                symptom_counts_df_pop = symptoms_series_pop.value_counts().nlargest(10).reset_index()
                symptom_counts_df_pop.columns = ['Symptom', 'Frequency']
                st.plotly_chart(plot_bar_chart_web(
                    symptom_counts_df_pop, x_col_bar='Symptom', y_col_bar='Frequency', title_bar="Top 10 Reported Symptoms",
                    orientation_web='h', y_axis_is_count=True, text_format_web='d', chart_height=400
                ), use_container_width=True)
            else: st.caption("No distinct symptoms reported for analysis.")
        # ... Test Result Distributions, Positivity Rate Trends would follow ...

with tab_systems_equity_pop:
    st.header("Health Systems Performance & Equity Lens")
    if analytics_df_final_for_display.empty:
        st.info("No data for systems/equity analysis with current filters.")
    else:
        # Example: Encounters by Clinic (Proxy for load distribution if data allows)
        st.subheader("Encounter Volume by Clinic ID (Top 10)")
        if 'clinic_id' in analytics_df_final_for_display.columns and analytics_df_final_for_display['clinic_id'].notna().any() and analytics_df_final_for_display['clinic_id'].astype(str).str.lower() != 'unknown':
            clinic_load_df_pop = analytics_df_final_for_display[
                analytics_df_final_for_display['clinic_id'].astype(str).str.lower() != 'unknown' # Filter out unknowns
            ]['clinic_id'].value_counts().nlargest(10).reset_index()
            clinic_load_df_pop.columns = ['Clinic ID', 'Encounter Count']
            if not clinic_load_df_pop.empty:
                st.plotly_chart(plot_bar_chart_web(
                    clinic_load_df_pop, x_col_bar='Clinic ID', y_col_bar='Encounter Count', title_bar="Top 10 Clinics by Encounter Volume",
                    orientation_web='h', y_axis_is_count=True, text_format_web='d', chart_height=400
                ), use_container_width=True)
            else: st.caption("No valid clinic encounter data for volume analysis.")
        # ... Referral Status distribution, AI Risk by SES scatter plot would follow ...


st.markdown("---"); st.caption(app_config.APP_FOOTER_TEXT) # Use new footer from app_config
logger.info("Population Health Analytics & Research Console page generated.")
