# test/pages/district_components/comparison_tab_district.py
# Redesigned for "Sentinel Health Co-Pilot" - LMIC Edge-First System
# This module prepares data for zonal comparative analysis, to be used in
# DHO web dashboards/reports (Facility Node - Tier 2, or Cloud - Tier 3).
# It focuses on extracting and structuring data for tables and charts.

import pandas as pd
import numpy as np # Not directly used for complex calcs here, but often with pandas
import logging
# Assuming app_config is in the PYTHONPATH or project root.
from config import app_config # Uses new, redesigned app_config
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

def prepare_zonal_comparison_data(
    district_gdf_main_enriched: pd.DataFrame, # GeoDataFrame from core_data_processing
    reporting_period_str: Optional[str] = "Latest Aggregated Data"
) -> Dict[str, Any]:
    """
    Prepares data for zonal comparative analysis tables and charts.

    Args:
        district_gdf_main_enriched: The enriched GeoDataFrame containing aggregated data per zone.
                                    Expected to have 'zone_id', 'name', and various metric columns.
        reporting_period_str: String describing the reporting period/data context.

    Returns:
        Dict[str, Any]: A dictionary containing structured data for comparison.
        Example: {
            "reporting_period": "...",
            "comparison_metrics_config": { // Config for UI to build selectors/format tables
                "Avg. AI Risk Score (Zone)": {"col": "avg_risk_score", "format_str": "{:.1f}" ...} ...
            },
            "zonal_comparison_table_df": pd.DataFrame(...), // Zone as index, metrics as columns
            "data_availability_notes": []
        }
    """
    logger.info(f"Preparing zonal comparison data for period: {reporting_period_str}")

    comparison_output: Dict[str, Any] = {
        "reporting_period": reporting_period_str,
        "comparison_metrics_config": {}, # Will hold {display_name: {col_name, format_str, colorscale_hint}}
        "zonal_comparison_table_df": None,
        "data_availability_notes": []
    }

    if not isinstance(district_gdf_main_enriched, pd.DataFrame) or district_gdf_main_enriched.empty:
        msg = "Zonal comparison requires valid enriched geographic zone data."
        logger.warning(msg)
        comparison_output["data_availability_notes"].append(msg)
        return comparison_output
    
    # Ensure essential 'name' or 'zone_id' for identifying zones
    if 'name' not in district_gdf_main_enriched.columns and 'zone_id' not in district_gdf_main_enriched.columns:
        msg = "Missing 'name' or 'zone_id' column in GDF for zonal identification."
        logger.error(msg)
        comparison_output["data_availability_notes"].append(msg)
        return comparison_output

    # Define metrics suitable for comparison (should align with GDF enrichment outputs)
    # The 'colorscale_hint' is a suggestion for the UI layer if it does heatmapping.
    # 'format_str' helps the UI layer in formatting values.
    comparison_metric_options_config = {
        "Avg. AI Risk Score": {"col": "avg_risk_score", "colorscale_hint": "OrRd_r", "format_str": "{:.1f}"},
        "Total Active Key Infections": {"col": "total_active_key_infections", "colorscale_hint": "Reds_r", "format_str": "{:.0f}"},
        "Prevalence per 1,000 (Key Inf.)": {"col": "prevalence_per_1000", "colorscale_hint": "YlOrRd_r", "format_str": "{:.1f}"},
        "Facility Coverage Score": {"col": "facility_coverage_score", "colorscale_hint": "Greens", "format_str": "{:.1f}%"},
        "Active TB Cases": {"col": "active_tb_cases", "colorscale_hint": "Blues_r", "format_str": "{:.0f}"},
        "Avg. Patient Steps": {"col": "avg_daily_steps_zone", "colorscale_hint": "Cividis_r", "format_str": "{:,.0f}"},
        "Avg. Clinic CO2": {"col": "zone_avg_co2", "colorscale_hint": "Purples_r", "format_str": "{:.0f} ppm"},
        "Population": {"col": "population", "colorscale_hint": "Greys", "format_str": "{:,.0f}"},
        "Number of Clinics": {"col": "num_clinics", "colorscale_hint": "Blues", "format_str":"{:.0f}"},
        "Socio-Economic Index": {"col": "socio_economic_index", "colorscale_hint": "Tealgrn_r", "format_str": "{:.2f}"},
        "Population Density (Pop/SqKm)": {"col": "population_density", "colorscale_hint": "Plasma_r", "format_str": "{:,.1f}"}
    }

    # Filter to available and non-null metrics in the provided GDF
    available_metrics_for_comparison = {}
    for display_name, details in comparison_metric_options_config.items():
        if details["col"] in district_gdf_main_enriched.columns and \
           district_gdf_main_enriched[details["col"]].notna().any():
            available_metrics_for_comparison[display_name] = details
    
    if not available_metrics_for_comparison:
        msg = "No metrics available in the GDF for Zonal Comparison table/chart."
        logger.warning(msg)
        comparison_output["data_availability_notes"].append(msg)
        return comparison_output
    
    comparison_output["comparison_metrics_config"] = available_metrics_for_comparison

    # Prepare the DataFrame for the comparison table
    # Use 'name' for display if available, else 'zone_id'
    zone_identifier_col = 'name' if 'name' in district_gdf_main_enriched.columns else 'zone_id'
    
    cols_for_table = [zone_identifier_col] + [details['col'] for details in available_metrics_for_comparison.values()]
    # Ensure all selected columns actually exist (should be guaranteed by above filter, but good check)
    actual_cols_for_table = [col for col in cols_for_table if col in district_gdf_main_enriched.columns]
    
    zonal_comparison_df = district_gdf_main_enriched[actual_cols_for_table].copy()
    
    # Set the zone identifier as index for a typical comparison table view
    # The UI layer can then decide how to display this index (or reset it for plotting).
    if zone_identifier_col in zonal_comparison_df.columns:
        zonal_comparison_df.set_index(zone_identifier_col, inplace=True, drop=False) # Keep as col too if needed by UI
        # Rename the index to something generic like "Zone" for display, if desired by UI.
        # zonal_comparison_df.index.name = "Zone" 
    
    comparison_output["zonal_comparison_table_df"] = zonal_comparison_df

    logger.info(f"Zonal comparison data prepared with {len(available_metrics_for_comparison)} metrics for {len(zonal_comparison_df)} zones.")
    return comparison_output


# --- Example Usage (for testing or integration into a DHO reporting script) ---
if __name__ == '__main__':
    logger.setLevel(logging.DEBUG)
    logger.info("Running District Comparison Tab component data preparation simulation...")

    # Simulate an enriched GDF (ensure it's a DataFrame for this non-geo test if gpd not installed)
    mock_enriched_zones_data = pd.DataFrame({
        'zone_id': ['ZoneA', 'ZoneB', 'ZoneC', 'ZoneD'],
        'name': ['Northwood District', 'Southville Area', 'Eastgate Community', 'Westend Borough'],
        'population': [12500, 21000, 17500, 8500],
        'avg_risk_score': [65.2, 72.1, 58.9, 75.0],
        'total_active_key_infections': [120, 250, 90, 150],
        'prevalence_per_1000': [9.6, 11.9, 5.1, 17.6],
        'facility_coverage_score': [75.0, 60.5, 82.1, 55.0],
        'active_tb_cases': [10, 22, 5, 15],
        'socio_economic_index': [0.6, 0.4, 0.8, 0.5],
        'num_clinics': [2,1,3,1],
        # 'geometry': [None]*4 # Placeholder if testing without real geometries
        # Missing population_density for this test example
        'avg_daily_steps_zone': [5500, 4800, 6200, 5100],
        'zone_avg_co2': [800, 950, 750, 1100]
    })

    comparison_results = prepare_zonal_comparison_data(
        district_gdf_main_enriched=mock_enriched_zones_data,
        reporting_period_str="Annual District Review 2023"
    )

    print(f"\n--- Prepared Zonal Comparison Data for: {comparison_results['reporting_period']} ---")
    
    print("\n## Comparison Metrics Configuration (for UI Selectors/Formatting):")
    if comparison_results["comparison_metrics_config"]:
        for name, conf in comparison_results["comparison_metrics_config"].items():
            print(f"  Display: '{name}' -> Column: '{conf['col']}', Format: '{conf['format_str']}', Colorscale Hint: '{conf['colorscale_hint']}'")

    print("\n## Zonal Comparison Table DataFrame (Raw Data):")
    if comparison_results["zonal_comparison_table_df"] is not None:
        print(comparison_results["zonal_comparison_table_df"].to_string())
    else:
        print("  (No zonal comparison table data generated)")

    print("\n## Data Availability Notes:")
    if comparison_results["data_availability_notes"]:
        for note in comparison_results["data_availability_notes"]: print(f"  - {note}")
    else:
        print("  (No specific data availability notes)")
