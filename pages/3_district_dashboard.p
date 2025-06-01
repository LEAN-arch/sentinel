# test/pages/3_district_dashboard.py
# Redesigned as "District Health Strategic Command Center" for "Sentinel Health Co-Pilot"
# This page simulates a web interface for District Health Officers (DHOs) and their teams,
# typically accessed at a Facility Node (Tier 2) or Regional/Cloud Node (Tier 3).
# It focuses on strategic oversight, population health management, resource allocation,
# and intervention planning across multiple zones.

import streamlit as st
import pandas as pd
import geopandas as gpd # Still needed if GDF is passed around and type-checked
import numpy as np # For np.nan
import os
import logging
from datetime import date, timedelta

# --- Sentinel System Imports ---
from config import app_config # Uses new, redesigned app_config

# Core data loading for simulation. In real system, data comes from Facility/Cloud Node DB.
from utils.core_data_processing import (
    load_health_records,
    load_zone_data, # This now returns the base GDF (geoms + attributes)
    load_iot_clinic_environment_data,
    enrich_zone_geodata_with_health_aggregates, # This is key for creating the district_gdf
    get_district_summary_kpis, # Calculates overall district KPIs from enriched GDF
    hash_geodataframe # For caching GDFs
)
# Refactored AI engine (apply_ai_models) would typically enrich raw health_df before aggregation.
from utils.ai_analytics_engine import apply_ai_models

# Refactored District Component Data Preparation & Rendering Functions
from pages.district_components_sentinel.kpi_structurer_district import structure_district_kpis_data
# map_display_district still renders the map directly, but uses the new plot_layered_choropleth_map_web
from pages.district_components_sentinel.map_display_district_web import render_district_interactive_map_web
from pages.district_components_sentinel.trend_calculator_district import calculate_district_trends_data
from pages.district_components_sentinel.comparison_data_preparer_district import prepare_zonal_comparison_data, get_intervention_criteria_options as get_comparison_criteria_options # reusing criteria getter logic pattern
from pages.district_components_sentinel.intervention_data_preparer_district import identify_priority_zones_for_intervention, get_intervention_criteria_options

# Refactored UI helpers for web reports
from utils.ui_visualization_helpers import (
    render_web_kpi_card,
    plot_annotated_line_chart_web,
    plot_bar_chart_web
    # Potentially others if comparison/intervention tabs need more complex visualizations
)

# Page Configuration
st.set_page_config(
    page_title=f"DHO Command Center - {app_config.APP_NAME}",
    layout="wide",
    initial_sidebar_state="expanded"
)

logger = logging.getLogger(__name__)
# CSS (assume loaded by app_home or via cache_resource if needed)

# --- Data Loading and Preparation (Simulating DHO Access at Facility/Cloud Node) ---
@st.cache_data(
    ttl=app_config.CACHE_TTL_SECONDS_WEB_REPORTS,
    hash_funcs={gpd.GeoDataFrame: hash_geodataframe, pd.DataFrame: pd.util.hash_pandas_object}, # Add pd.DataFrame for completeness
    show_spinner="Loading and enriching district-level operational data..."
)
def get_dho_command_center_data(
    # Parameters for date filtering trends if historical data is also loaded
    # For the enriched GDF itself, it often represents the "latest" snapshot.
    ):
    """
    Simulates fetching, enriching, and preparing all data needed for the DHO Command Center.
    This is a heavy function as it simulates the entire data pipeline feeding the DHO view.
    """
    logger.info("DHO Command Center: Starting data acquisition and enrichment...")
    # 1. Load base data sources
    #    In a real Tier 2/3 node, this might be querying different tables/views from an aggregated DB.
    health_df_raw_full = load_health_records(file_path=app_config.HEALTH_RECORDS_CSV, source_context="DHODataPrep/HealthFull")
    iot_df_full = load_iot_clinic_environment_data(file_path=app_config.IOT_CLINIC_ENVIRONMENT_CSV, source_context="DHODataPrep/IoTFull")
    
    # 2. Apply AI models to raw health data (as if done at ingest or batch processing)
    #    This enriched_health_df forms the basis for health aggregations.
    if not health_df_raw_full.empty:
        # apply_ai_models returns a tuple (enriched_df, supply_forecast_df), we only need enriched_df here
        enriched_health_df_full = apply_ai_models(health_df_raw_full, source_context="DHODataPrep/AIEnrich")[0]
    else:
        enriched_health_df_full = pd.DataFrame()
        logger.warning("DHO Command Center: Raw health data is empty. Proceeding with empty health data.")

    # 3. Load base zone data (geometries + attributes)
    #    This returns a GeoDataFrame.
    base_zone_gdf = load_zone_data(source_context="DHODataPrep/ZoneBase") # Uses default paths from app_config

    if base_zone_gdf is None or base_zone_gdf.empty:
        logger.error("DHO Command Center: Base zone geographic data failed to load. Critical for DHO view.")
        # Return empty structures for downstream robustness
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}, {} # GDF, Health, IoT, Overall KPIs, All Criteria

    # 4. Enrich the zone GDF with health and IoT aggregates
    #    This is a key step to create the `district_gdf_main_enriched`
    district_gdf_enriched = enrich_zone_geodata_with_health_aggregates(
        base_zone_gdf, enriched_health_df_full, iot_df_full, source_context="DHODataPrep/ZoneEnrich"
    )
    if district_gdf_enriched is None or district_gdf_enriched.empty: # Should return a GDF even if enrichment adds nothing
        logger.warning("DHO Command Center: Zone GDF enrichment resulted in empty or None. Using base GDF.")
        district_gdf_enriched = base_zone_gdf # Fallback if enrichment somehow fails critically

    # 5. Calculate overall district KPIs from the enriched GDF
    #    This function resides in core_data_processing.
    overall_district_kpis = get_district_summary_kpis(district_gdf_enriched, source_context="DHODataPrep/DistrictKPIs")

    # 6. Prepare options for criteria-based filtering (used in comparison and intervention tabs)
    #    The actual criteria options (especially for intervention) can be quite specific.
    #    Comparison might use a simpler set of available metrics.
    #    Re-using the get_intervention_criteria_options structure is a good pattern.
    available_filter_criteria = get_intervention_criteria_options(district_gdf_enriched.head() if not district_gdf_enriched.empty else None)


    # enriched_health_df_full and iot_df_full are returned for potential trend calculations
    return district_gdf_enriched, enriched_health_df_full, iot_df_full, overall_district_kpis, available_filter_criteria

# --- Load all necessary data ---
# This single call simulates the data aggregation and preparation at a Tier 2/3 node.
district_gdf, hist_health_df, hist_iot_df, district_kpis_summary, all_criteria_opts_dict = get_dho_command_center_data()


# --- Page Title & Introduction for DHO ---
st.title(f"🌍 {app_config.APP_NAME} - District Health Strategic Command Center")
current_data_timestamp_str = f"Data aggregated up to: {pd.Timestamp('now').strftime('%d %b %Y, %H:%M')}" # Simulated "as of"
st.markdown(f"**Comprehensive Zonal Health Intelligence, Resource Oversight, and Intervention Planning. {current_data_timestamp_str}**")
st.markdown("---")

# --- Sidebar Filters for DHO (Primarily for Trend Analysis Date Range) ---
if os.path.exists(app_config.APP_LOGO_SMALL):
    st.sidebar.image(app_config.APP_LOGO_SMALL, width=180)
st.sidebar.header("🗓️ Analysis Filters")

# Date Range for Trend Analysis Tab
default_days_dho_trends = app_config.WEB_DASHBOARD_DEFAULT_DATE_RANGE_DAYS_TREND * 3 # e.g., 90 days for DHO trends
min_hist_date_dho = date.today() - timedelta(days=365*2) # Up to 2 years back
max_hist_date_dho = date.today()

default_end_dho_trends = max_hist_date_dho
default_start_dho_trends = default_end_dho_trends - timedelta(days=default_days_dho_trends - 1)
if default_start_dho_trends < min_hist_date_dho: default_start_dho_trends = min_hist_date_dho

selected_start_date_dho_trends, selected_end_date_dho_trends = st.sidebar.date_input(
    "Select Date Range for Trend Analysis:",
    value=[default_start_dho_trends, default_end_dho_trends],
    min_value=min_hist_date_dho,
    max_value=max_hist_date_dho,
    key="dho_trends_date_selector_v1"
)
if selected_start_date_dho_trends > selected_end_date_dho_trends:
    st.sidebar.error("Trend start date must be before end date.")
    selected_start_date_dho_trends = selected_end_date_dho_trends # Correct

# --- Main Display Sections ---

# Section 1: District-Wide Key Performance Indicators
st.header("📊 District Performance Overview")
if district_kpis_summary:
    structured_dho_kpis = structure_district_kpis_data(
        district_kpis_summary,
        district_gdf, # Pass GDF for context like total zones
        reporting_period_str=current_data_timestamp_str
    )
    if structured_dho_kpis:
        # Display KPIs in rows of 4 (or adjust dynamically)
        num_kpis = len(structured_dho_kpis)
        kpi_cols_per_row = 4
        for i in range(0, num_kpis, kpi_cols_per_row):
            cols = st.columns(kpi_cols_per_row)
            for j, kpi_data in enumerate(structured_dho_kpis[i : i + kpi_cols_per_row]):
                with cols[j]:
                    render_web_kpi_card(
                        title=kpi_data["title"], value=kpi_data["value_str"], icon=kpi_data["icon"],
                        status_level=kpi_data["status_level"], units=kpi_data.get("units",""), help_text=kpi_data.get("help_text","")
                    )
    else: st.info("District KPIs could not be structured.")
else:
    st.warning("District-wide KPIs are currently unavailable. Check data aggregation processes.")
st.markdown("---")


# --- Tabbed Interface for Detailed Analysis ---
st.header("🔍 Detailed District Analysis Areas")
tab_titles_dho = [
    "🗺️ Interactive District Map",
    "📈 District-Wide Trends",
    "🆚 Zonal Comparative Analysis",
    "🎯 Intervention Planning"
]
tab_map, tab_trends, tab_comparison, tab_interventions = st.tabs(tab_titles_dho)

with tab_map:
    st.subheader("Spatial Distribution of Key Metrics")
    # The map rendering component now handles the selectbox for metric choice
    # Default metric to show could be based on urgency or a DHO preference (e.g., avg_risk_score)
    render_district_interactive_map_web(
        district_gdf_main_enriched=district_gdf, # Pass the enriched GDF
        default_selected_metric_key="avg_risk_score", # Default to avg_risk_score column name
        reporting_period_str=current_data_timestamp_str
    )

with tab_trends:
    st.subheader(f"Health & Environmental Trends ({selected_start_date_dho_trends.strftime('%d %b %Y')} - {selected_end_date_dho_trends.strftime('%d %b %Y')})")
    # Filter historical data for the selected trend period for DHO view
    # This logic could be part of get_dho_command_center_data or done here if hist_health_df is full history.
    # For now, assume hist_health_df and hist_iot_df are full and filter them here.
    
    trend_health_df_dho = pd.DataFrame()
    if not hist_health_df.empty and 'encounter_date' in hist_health_df:
         hist_health_df['encounter_date'] = pd.to_datetime(hist_health_df['encounter_date'], errors='coerce')
         trend_health_df_dho = hist_health_df[
             (hist_health_df['encounter_date'].dt.date >= selected_start_date_dho_trends) &
             (hist_health_df['encounter_date'].dt.date <= selected_end_date_dho_trends)
         ].copy()

    trend_iot_df_dho = pd.DataFrame()
    if hist_iot_df is not None and not hist_iot_df.empty and 'timestamp' in hist_iot_df:
        hist_iot_df['timestamp'] = pd.to_datetime(hist_iot_df['timestamp'], errors='coerce')
        trend_iot_df_dho = hist_iot_df[
            (hist_iot_df['timestamp'].dt.date >= selected_start_date_dho_trends) &
            (hist_iot_df['timestamp'].dt.date <= selected_end_date_dho_trends)
        ].copy()

    district_trend_data_output = calculate_district_trends_data(
        filtered_health_for_trends_dist=trend_health_df_dho,
        filtered_iot_for_trends_dist=trend_iot_df_dho,
        trend_start_date=selected_start_date_dho_trends,
        trend_end_date=selected_end_date_dho_trends,
        reporting_period_str=f"{selected_start_date_dho_trends.strftime('%b %Y')} - {selected_end_date_dho_trends.strftime('%b %Y')}"
    )
    
    # Display the calculated trends
    # Disease Incidence
    if district_trend_data_output.get("disease_incidence_trends"):
        st.markdown("##### Key Disease Incidence Trends (Weekly New Unique Patients)")
        disease_trends = district_trend_data_output["disease_incidence_trends"]
        cols_disease_dho = st.columns(min(len(disease_trends), 2) or 1) # Max 2 charts per row
        idx_disease_dho = 0
        for cond_name, series_data in disease_trends.items():
            if series_data is not None and not series_data.empty:
                with cols_disease_dho[idx_disease_dho % 2]:
                    st.plotly_chart(plot_annotated_line_chart_web(
                        data_series_input=series_data, chart_title=f"New {cond_name} Cases", y_axis_is_count=True
                    ), use_container_width=True)
                idx_disease_dho += 1
    # Other trends (AI Risk, Steps, CO2)
    trend_series_map_dho = {
        "Avg. Patient AI Risk Score": district_trend_data_output.get("avg_patient_ai_risk_trend_series"),
        "Avg. Patient Daily Steps": district_trend_data_output.get("avg_patient_daily_steps_trend_series"),
        "Avg. Clinic CO2 Levels": district_trend_data_output.get("avg_clinic_co2_trend_series")
    }
    cols_other_trends_dho = st.columns(min(sum(1 for s in trend_series_map_dho.values() if s is not None and not s.empty), 2) or 1)
    idx_other_dho = 0
    for title, series in trend_series_map_dho.items():
        if series is not None and not series.empty:
            with cols_other_trends_dho[idx_other_dho % 2]:
                st.plotly_chart(plot_annotated_line_chart_web(series, title, y_axis_is_count="Cases" in title or "Steps" in title), use_container_width=True) # adjust y_is_count
            idx_other_dho +=1
    if district_trend_data_output.get("data_availability_notes"):
        for note in district_trend_data_output["data_availability_notes"]: st.caption(f"Trend Note: {note}")


with tab_comparison:
    st.subheader("Zonal Performance & Profile Comparison")
    comparison_data = prepare_zonal_comparison_data(district_gdf, reporting_period_str=current_data_timestamp_str)
    
    if comparison_data["zonal_comparison_table_df"] is not None:
        st.markdown("###### Zonal Statistics Table (Scrollable)")
        # For Streamlit, styling (heatmap) must be applied before st.dataframe
        # This would ideally be a more sophisticated interactive table in a richer web framework
        # For now, just display the raw DataFrame from prep function. Styling deferred to a potential rich client.
        st.dataframe(comparison_data["zonal_comparison_table_df"], height=400, use_container_width=True)
        
        # Bar chart for comparing a selected metric (UI needs selector for this)
        # For simplicity, let's pick one metric to show by default for the bar chart if config exists
        if comparison_data["comparison_metrics_config"]:
            default_bar_metric_disp_name = "Avg. AI Risk Score (Zone)" # Example default display name
            if default_bar_metric_disp_name not in comparison_data["comparison_metrics_config"]: # Fallback if default not available
                default_bar_metric_disp_name = list(comparison_data["comparison_metrics_config"].keys())[0]

            selected_comp_metric_details = comparison_data["comparison_metrics_config"][default_bar_metric_disp_name]
            metric_col_for_bar = selected_comp_metric_details['col']
            
            df_for_bar = comparison_data["zonal_comparison_table_df"].reset_index() # Use reset index if zone name is index
            # Ensure the x_col (zone identifier) is correct after reset_index
            x_col_for_bar_chart = df_for_bar.columns[0] if df_for_bar.columns[0] != metric_col_for_bar else df_for_bar.columns[1]

            st.markdown(f"###### Visual Comparison: {default_bar_metric_disp_name}")
            st.plotly_chart(plot_bar_chart_web(
                df_for_bar, x_col=x_col_for_bar_chart, y_col=metric_col_for_bar, title_bar=f"{default_bar_metric_disp_name} by Zone",
                sort_values_by_web=metric_col_for_bar, ascending_web=False # Example: sort desc by value
            ), use_container_width=True)
    if comparison_data["data_availability_notes"]:
        for note in comparison_data["data_availability_notes"]: st.caption(f"Comparison Note: {note}")


with tab_interventions:
    st.subheader("Intervention Planning Insights")
    st.markdown("Identify priority zones based on customizable criteria.")
    
    # Get available criteria for multiselect (uses district_gdf to check applicability)
    # `all_criteria_opts_dict` loaded at the top.
    if not all_criteria_opts_dict:
        st.warning("No intervention criteria definitions are available. Check configuration.")
    else:
        selected_intervention_criteria_names = st.multiselect(
            "Select Criteria to Identify Priority Zones (Zones meeting ANY selected criteria will be shown):",
            options=list(all_criteria_opts_dict.keys()),
            default=list(all_criteria_opts_dict.keys())[0:min(2, len(all_criteria_opts_dict))] if all_criteria_opts_dict else None, # Select first two by default if available
            key="dho_intervention_criteria_selector_v1"
        )

        intervention_data = identify_priority_zones_for_intervention(
            district_gdf_main_enriched=district_gdf,
            selected_criteria_display_names=selected_intervention_criteria_names,
            available_criteria_options=all_criteria_opts_dict, # Pass the full dict for lookup
            reporting_period_str=current_data_timestamp_str
        )

        if intervention_data["priority_zones_for_intervention_df"] is not None and \
           not intervention_data["priority_zones_for_intervention_df"].empty:
            st.markdown(f"###### Identified **{len(intervention_data['priority_zones_for_intervention_df'])}** Zone(s) Meeting Criteria:")
            # Display key columns. Actual columns might vary based on selected criteria.
            # The `identify_priority_zones_for_intervention` function now returns a DF with relevant columns.
            st.dataframe(intervention_data["priority_zones_for_intervention_df"], height=400, use_container_width=True)
        elif selected_intervention_criteria_names : # If criteria were selected but no zones met them
             st.success("✅ No zones currently meet the selected high-priority criteria.")
        else: # No criteria selected by user
            st.info("Please select criteria above to identify priority zones.")

        if intervention_data["data_availability_notes"]:
            for note in intervention_data["data_availability_notes"]: st.caption(f"Intervention Note: {note}")

logger.info("DHO Strategic Command Center page generated.")
