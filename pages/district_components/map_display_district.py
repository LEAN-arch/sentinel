# test/pages/district_components/map_display_district.py
# Redesigned for "Sentinel Health Co-Pilot" - LMIC Edge-First System
# This module is responsible for generating the interactive district map visualization
# for display on a DHO's web dashboard (Facility Node - Tier 2, or Cloud - Tier 3).
# It utilizes the refactored plotting utilities.

import streamlit as st # Kept as this component is about rendering a figure in a Streamlit page
import pandas as pd
import numpy as np # Not directly used here, but often imported with pandas
import logging
# Assuming app_config is in the PYTHONPATH or project root.
from config import app_config # Uses new, redesigned app_config
# Using the _web suffixed plotting function from the refactored helpers
from utils.ui_visualization_helpers import plot_layered_choropleth_map_web, _create_empty_plot_figure
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

def render_district_interactive_map(
    district_gdf_main_enriched: pd.DataFrame, # GeoDataFrame (or DataFrame that plot_layered can handle if geometries passed differently)
    default_selected_metric_key: Optional[str] = "population_weighted_avg_ai_risk_score", # Key from map_metric_options below
    reporting_period_str: Optional[str] = "Latest Aggregated Data"
) -> None:
    """
    Renders an interactive choropleth map for district-level health & environment visualization.
    This is intended to be called from a Streamlit page context (e.g., DHO dashboard page).

    Args:
        district_gdf_main_enriched: The enriched GeoDataFrame containing zonal data.
                                    Expected to have geometry and various metric columns.
        default_selected_metric_key: The internal key (column name in GDF) of the metric
                                     to be displayed by default on the map.
        reporting_period_str: String describing the reporting period for context.
    """
    logger.info(f"Rendering district interactive map for period: {reporting_period_str}")

    # This subheader would be set by the calling page.
    # st.subheader(f"🗺️ Interactive Health & Environment Map - {reporting_period_str}")

    if not isinstance(district_gdf_main_enriched, pd.DataFrame) or district_gdf_main_enriched.empty:
        st.warning("District map cannot be displayed: Enriched geographic data is unavailable or empty.")
        # Optionally, display a placeholder image or message.
        st.markdown("*(Map data not loaded. Please check data sources and processing.)*")
        return
    
    # Check for geometry column explicitly for GeoDataFrames
    is_gdf = hasattr(district_gdf_main_enriched, 'geometry')
    active_geom_col = district_gdf_main_enriched.geometry.name if is_gdf and hasattr(district_gdf_main_enriched.geometry, 'name') else 'geometry'

    if is_gdf and (active_geom_col not in district_gdf_main_enriched.columns or
                   district_gdf_main_enriched[active_geom_col].is_empty.all() or
                   not district_gdf_main_enriched[active_geom_col].is_valid.any()):
        st.error("🚨 District map cannot be displayed: Geographic data contains invalid or empty geometries.")
        return


    # Define map metric options. Keys are user-facing display names.
    # 'col' is the actual column name in district_gdf_main_enriched.
    # 'colorscale' and 'format_str' for the plot_layered_choropleth_map_web.
    # These should align with metric names produced by get_district_summary_kpis and enrich_zone_geodata
    map_metric_options = {
        "Avg. AI Risk Score (Zone)": {"col": "avg_risk_score", "colorscale": "OrRd_r", "format_str": "{:.1f}"}, # Note: DHO KPI was population_weighted_avg_ai_risk_score for district level
        "Total Active Key Infections (Zone)": {"col": "total_active_key_infections", "colorscale": "Reds_r", "format_str": "{:.0f}"},
        "Prevalence per 1,000 (Key Inf., Zone)": {"col": "prevalence_per_1000", "colorscale": "YlOrRd_r", "format_str": "{:.1f}"},
        "Facility Coverage Score (Zone)": {"col": "facility_coverage_score", "colorscale": "Greens", "format_str": "{:.1f}%"},
        "Active TB Cases (Zone)": {"col": "active_tb_cases", "colorscale": "Blues_r", "format_str": "{:.0f}"}, # Example, could be any from KEY_CONDITIONS_FOR_ACTION
        "Active Malaria Cases (Zone)": {"col": "active_malaria_cases", "colorscale": "Oranges_r", "format_str": "{:.0f}"},
        "Avg. Patient Steps (Zone)": {"col": "avg_daily_steps_zone", "colorscale": "Cividis_r", "format_str": "{:,.0f}"},
        "Avg. Clinic CO2 (Zone)": {"col": "zone_avg_co2", "colorscale": "Purples_r", "format_str": "{:.0f} ppm"},
        "Zone Population": {"col": "population", "colorscale": "Viridis", "format_str": "{:,.0f}"},
        "Number of Clinics (Zone)": {"col": "num_clinics", "colorscale": "Blues", "format_str":"{:.0f}"},
        "Socio-Economic Index (Zone)": {"col": "socio_economic_index", "colorscale": "Tealgrn_r", "format_str": "{:.2f}"}
    }
    if 'population_density' in district_gdf_main_enriched.columns: # Add if calculated by enrichment
         map_metric_options["Population Density (Pop/SqKm)"] = {"col": "population_density", "colorscale": "Plasma_r", "format_str": "{:,.1f}"}

    # Filter options to only those metrics available and non-null in the GDF
    available_map_metrics_for_selection = {
        display_name: details for display_name, details in map_metric_options.items()
        if details["col"] in district_gdf_main_enriched.columns and district_gdf_main_enriched[details["col"]].notna().any()
    }

    if not available_map_metrics_for_selection:
        st.warning("No metrics with valid data are currently available for map visualization in the provided GDF.")
        return

    # Attempt to set default selection
    # Find display name for default_selected_metric_key
    default_display_name_for_select = default_selected_metric_key # Fallback to key itself if no match
    for dn, det in available_map_metrics_for_selection.items():
        if det["col"] == default_selected_metric_key:
            default_display_name_for_select = dn
            break
    
    # Ensure default is in the list of available options
    if default_display_name_for_select not in available_map_metrics_for_selection:
        if available_map_metrics_for_selection: # If there are any options at all
            default_display_name_for_select = list(available_map_metrics_for_selection.keys())[0]
        else: # Should have been caught above
             st.error("Logic error: No map metrics to select as default."); return

    selected_metric_display_name = st.selectbox(
        "Select Metric to Visualize on Map:",
        options=list(available_map_metrics_for_selection.keys()),
        index=list(available_map_metrics_for_selection.keys()).index(default_display_name_for_select) if default_display_name_for_select in available_map_metrics_for_selection else 0,
        key="dho_map_metric_selector_v3", # Unique key
        help="Choose a metric to visualize its spatial distribution across zones."
    )
    
    selected_metric_config = available_map_metrics_for_selection.get(selected_metric_display_name)
    
    if selected_metric_config:
        map_value_column = selected_metric_config["col"]
        map_color_palette = selected_metric_config["colorscale"]
        # Basic hover columns, can be expanded based on GDF content
        hover_cols_for_map = ['name', map_value_column, 'population', 'num_clinics']
        # Ensure hover_cols actually exist in GDF before passing
        final_hover_cols = [col for col in hover_cols_for_map if col in district_gdf_main_enriched.columns]

        # Prepare facility points GDF if available (example: overlay clinics)
        # This data would need to be loaded and passed to this function.
        # For this example, we assume it's not passed unless explicitly needed.
        # facility_points_data = load_facility_locations_gdf() # Example function call

        map_figure_object = plot_layered_choropleth_map_web(
            gdf_data=district_gdf_main_enriched,
            value_col_name=map_value_column,
            map_title=f"District Map: {selected_metric_display_name}",
            id_col_name='zone_id', # Ensure GDF has 'zone_id'
            color_scale=map_color_palette,
            hover_data_cols=final_hover_cols,
            # facility_points_gdf=facility_points_data, # Pass if available
            # facility_hover_col_name='facility_name', facility_size_col_name='capacity_beds', # Example cols for facilities
            map_height=app_config.WEB_MAP_DEFAULT_HEIGHT, # Using new config
            # mapbox_style can be inherited from theme or app_config.MAPBOX_STYLE_WEB
        )
        st.plotly_chart(map_figure_object, use_container_width=True)
    else:
        st.info("Please select a metric from the dropdown to display the map.")
        # Display an empty placeholder if no metric is somehow selected but options existed
        st.plotly_chart(_create_empty_plot_figure(f"District Map: No Metric Selected", app_config.WEB_MAP_DEFAULT_HEIGHT), use_container_width=True)

# --- Example Usage (If called from a Streamlit page for DHO Dashboard) ---
# This assumes the main DHO dashboard page loads `district_gdf_main_enriched`
# and then calls this rendering function.
#
# Example in `pages/3_district_dashboard.py`:
#
# from pages.district_components import map_display_district # Assuming this file name
# ...
# # After loading district_gdf_main_enriched
# st.subheader("🗺️ Interactive Health & Environment Map of the District") # Title managed by page
# map_display_district.render_district_interactive_map(
#     district_gdf_main_enriched,
#     default_selected_metric_key="avg_risk_score", # Or another relevant default column name
#     reporting_period_str=f"Data as of {pd.Timestamp('today').strftime('%d %b %Y')}" # Example
# )
# st.markdown("---")
