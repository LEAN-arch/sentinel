# test/pages/district_components/intervention_tab_district.py
# Redesigned for "Sentinel Health Co-Pilot" - LMIC Edge-First System
# This module prepares data for identifying priority zones for intervention based on
# selected criteria. Intended for DHO web dashboards/reports (Facility/Cloud Nodes - Tiers 2/3).

import pandas as pd
import numpy as np # For np.nan, pd.Series.any() etc.
import logging
# Assuming app_config is in the PYTHONPATH or project root.
from config import app_config # Uses new, redesigned app_config
from typing import Dict, Any, Optional, List, Callable

logger = logging.getLogger(__name__)

def get_intervention_criteria_options(
    district_gdf_check: Optional[pd.DataFrame] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Defines and returns available intervention criteria options, checking if necessary
    columns exist in a sample of the GDF if provided.

    Args:
        district_gdf_check: Optional DataFrame (ideally a GDF head) to check for column availability.

    Returns:
        Dict[str, Dict[str, Any]]: Dictionary of available criteria.
            Key: User-facing criterion name.
            Value: Dict containing 'lambda_func' (Callable) and 'required_cols' (List[str]).
    """
    criteria_definitions: Dict[str, Dict[str, Any]] = {
        f"High Avg. AI Risk (Zone Score ≥ {app_config.DISTRICT_ZONE_HIGH_RISK_AVG_SCORE})": {
            "lambda_func": lambda df: df.get('avg_risk_score', pd.Series(dtype=float)) >= app_config.DISTRICT_ZONE_HIGH_RISK_AVG_SCORE,
            "required_cols": ['avg_risk_score']
        },
        f"Low Facility Coverage (Zone Score < {app_config.DISTRICT_INTERVENTION_FACILITY_COVERAGE_LOW_PCT}%)": {
            "lambda_func": lambda df: df.get('facility_coverage_score', pd.Series(dtype=float)) < app_config.DISTRICT_INTERVENTION_FACILITY_COVERAGE_LOW_PCT,
            "required_cols": ['facility_coverage_score']
        },
        f"High Key Disease Prevalence (Zone in Top {100 - int(app_config.DISTRICT_INTERVENTION_KEY_DISEASE_PREVALENCE_HIGH_PERCENTILE*100)}% by Prev/1k)": {
            "lambda_func": lambda df: df.get('prevalence_per_1000', pd.Series(dtype=float)) >= df.get('prevalence_per_1000', pd.Series(dtype=float)).quantile(app_config.DISTRICT_INTERVENTION_KEY_DISEASE_PREVALENCE_HIGH_PERCENTILE)
                                    if 'prevalence_per_1000' in df and df['prevalence_per_1000'].notna().any() and len(df['prevalence_per_1000'].dropna()) >= (1/(1-app_config.DISTRICT_INTERVENTION_KEY_DISEASE_PREVALENCE_HIGH_PERCENTILE)) # Ensure enough data for quantile
                                    else pd.Series([False]*len(df), index=df.index),
            "required_cols": ['prevalence_per_1000']
        },
        f"High Absolute TB Burden (Zone Active TB Cases > {app_config.DISTRICT_INTERVENTION_TB_BURDEN_HIGH_ABS})": {
            "lambda_func": lambda df: df.get('active_tb_cases', pd.Series(dtype=float)) > app_config.DISTRICT_INTERVENTION_TB_BURDEN_HIGH_ABS,
            "required_cols": ['active_tb_cases']
        },
        # Example: High Avg. Clinic CO2 in Zone (based on new Sentinel config for ambient alerts)
        f"Poor Avg. Clinic Air Quality (Zone Avg CO2 > {app_config.ALERT_AMBIENT_CO2_HIGH_PPM}ppm)": {
            "lambda_func": lambda df: df.get('zone_avg_co2', pd.Series(dtype=float)) > app_config.ALERT_AMBIENT_CO2_HIGH_PPM,
            "required_cols": ['zone_avg_co2']
        }
        # Add more criteria based on available GDF fields and app_config thresholds
    }

    if district_gdf_check is None or district_gdf_check.empty:
        return criteria_definitions # Return all if no GDF to check against

    available_criteria = {}
    for name, details in criteria_definitions.items():
        if all(col in district_gdf_check.columns for col in details["required_cols"]):
            # Further check: ensure required columns have some non-NA data for meaningful criteria
            if all(district_gdf_check[col].notna().any() for col in details["required_cols"]):
                 available_criteria[name] = details
            else:
                 logger.debug(f"Intervention criterion '{name}' skipped: one of required columns ({details['required_cols']}) has all NaN values.")
        else:
            logger.debug(f"Intervention criterion '{name}' skipped: missing one or more required columns ({details['required_cols']}).")
    return available_criteria


def identify_priority_zones_for_intervention(
    district_gdf_main_enriched: pd.DataFrame, # Using pd.DataFrame for broader compatibility if gpd isn't always present
    selected_criteria_display_names: List[str],
    available_criteria_options: Dict[str, Dict[str, Any]], # Output of get_intervention_criteria_options
    reporting_period_str: Optional[str] = "Latest Aggregated Data"
) -> Dict[str, Any]:
    """
    Identifies priority zones based on selected criteria from the enriched GDF.

    Args:
        district_gdf_main_enriched: The enriched GeoDataFrame (or DataFrame).
        selected_criteria_display_names: List of user-facing names of criteria to apply (from UI selection).
        available_criteria_options: The dictionary of all defined and available criteria.
        reporting_period_str: String describing the reporting period.

    Returns:
        Dict[str, Any]: Dictionary containing:
            "reporting_period": str,
            "applied_criteria": List[str],
            "priority_zones_for_intervention_df": pd.DataFrame of zones meeting criteria,
            "data_availability_notes": List[str]
    """
    logger.info(f"Identifying priority zones for intervention. Criteria: {selected_criteria_display_names}. Period: {reporting_period_str}")

    intervention_output: Dict[str, Any] = {
        "reporting_period": reporting_period_str,
        "applied_criteria": selected_criteria_display_names,
        "priority_zones_for_intervention_df": None,
        "data_availability_notes": []
    }

    if not isinstance(district_gdf_main_enriched, pd.DataFrame) or district_gdf_main_enriched.empty:
        msg = "Intervention planning requires valid enriched geographic zone data."
        logger.warning(msg)
        intervention_output["data_availability_notes"].append(msg)
        return intervention_output
    
    if not selected_criteria_display_names:
        msg = "No intervention criteria selected."
        # Not necessarily an error, just means no filtering.
        intervention_output["data_availability_notes"].append(msg)
        # Return all zones if no criteria selected? Or empty? For now, empty with note.
        intervention_output["priority_zones_for_intervention_df"] = pd.DataFrame(columns=district_gdf_main_enriched.columns)
        return intervention_output

    # Initialize a boolean mask for all zones to False
    combined_filter_mask = pd.Series([False] * len(district_gdf_main_enriched), index=district_gdf_main_enriched.index)
    actually_applied_criteria = []

    for crit_display_name in selected_criteria_display_names:
        criterion_details = available_criteria_options.get(crit_display_name)
        if not criterion_details or 'lambda_func' not in criterion_details:
            logger.warning(f"Criterion '{crit_display_name}' not found or misconfigured in available options. Skipping.")
            intervention_output["data_availability_notes"].append(f"Configuration error for criterion: {crit_display_name}.")
            continue
        
        crit_lambda = criterion_details['lambda_func']
        try:
            current_criterion_mask = crit_lambda(district_gdf_main_enriched)
            if isinstance(current_criterion_mask, pd.Series) and current_criterion_mask.dtype == bool:
                combined_filter_mask = combined_filter_mask | current_criterion_mask.fillna(False) # Apply OR logic
                actually_applied_criteria.append(crit_display_name)
            else:
                logger.error(f"Criterion '{crit_display_name}' did not return a valid boolean Series. Type: {type(current_criterion_mask)}")
                intervention_output["data_availability_notes"].append(f"Execution error for criterion: {crit_display_name}.")
        except Exception as e_apply_crit:
            logger.error(f"Error applying intervention criterion '{crit_display_name}': {e_apply_crit}", exc_info=True)
            intervention_output["data_availability_notes"].append(f"Error applying criterion: {crit_display_name}.")
            
    intervention_output["applied_criteria"] = actually_applied_criteria # Record which ones were actually used

    priority_zones_df = district_gdf_main_enriched[combined_filter_mask].copy()

    if not priority_zones_df.empty:
        # Select key columns relevant for intervention display
        # These should represent why a zone was flagged and its general profile.
        # Prioritize 'name', then key metrics from selection criteria, plus population.
        intervention_cols_to_display = ['name', 'population'] # Start with essentials
        # Add columns related to any applied criteria
        for crit_name in actually_applied_criteria:
            crit_cols = available_criteria_options.get(crit_name, {}).get("required_cols", [])
            intervention_cols_to_display.extend(c for c in crit_cols if c not in intervention_cols_to_display)
        
        # Add a few general informative columns if they exist and aren't already included
        general_info_cols = ['avg_risk_score', 'prevalence_per_1000', 'facility_coverage_score']
        intervention_cols_to_display.extend(c for c in general_info_cols if c in priority_zones_df.columns and c not in intervention_cols_to_display)

        # Ensure all selected columns exist in priority_zones_df
        final_cols_for_output = [col for col in intervention_cols_to_display if col in priority_zones_df.columns]
        if 'zone_id' in priority_zones_df.columns and 'zone_id' not in final_cols_for_output: # Ensure zone_id is there for reference
            final_cols_for_output.insert(0, 'zone_id')


        priority_zones_df_output = priority_zones_df[final_cols_for_output]
        
        # Sort by a primary risk indicator, e.g., avg_risk_score desc, then by prevalence or coverage asc.
        sort_by_intervention = []
        sort_ascending_intervention = []
        if 'avg_risk_score' in final_cols_for_output: sort_by_intervention.append('avg_risk_score'); sort_ascending_intervention.append(False)
        if 'prevalence_per_1000' in final_cols_for_output: sort_by_intervention.append('prevalence_per_1000'); sort_ascending_intervention.append(False)
        if 'facility_coverage_score' in final_cols_for_output: sort_by_intervention.append('facility_coverage_score'); sort_ascending_intervention.append(True)
        
        if sort_by_intervention:
            priority_zones_df_output = priority_zones_df_output.sort_values(by=sort_by_intervention, ascending=sort_ascending_intervention)
        
        intervention_output["priority_zones_for_intervention_df"] = priority_zones_df_output.reset_index(drop=True)
        logger.info(f"Identified {len(priority_zones_df_output)} priority zones for intervention.")
    else:
        intervention_output["data_availability_notes"].append("No zones meet the selected high-priority criteria.")
        intervention_output["priority_zones_for_intervention_df"] = pd.DataFrame(columns=district_gdf_main_enriched.columns)


    return intervention_output


# --- Example Usage (for testing or integration into a DHO reporting script) ---
if __name__ == '__main__':
    logger.setLevel(logging.DEBUG)
    logger.info("Running District Intervention Tab component data preparation simulation...")

    mock_gdf_interv = pd.DataFrame({
        'zone_id': [f'ZN{i}' for i in range(1, 6)],
        'name': ['North', 'South', 'East', 'West', 'Central'],
        'population': [10000, 15000, 8000, 12000, 20000],
        'avg_risk_score': [78, 65, 55, 82, 70], # North, West, Central are >= 70
        'facility_coverage_score': [55, 70, 80, 45, 65], # North, West are < 60
        'prevalence_per_1000': [pd.Series([50,30,20,60,40]).quantile(0.81), 30, 20, 60, 40], # Simulate East being high prev via quantile
        'active_tb_cases': [12, 5, 3, 15, 8], # North, West > 10
        'zone_avg_co2': [1600, 800, 700, 1400, 900] # North, West > 1000 (using ALERT_AMBIENT_CO2_HIGH_PPM as example for poor quality)
        # 'geometry': ... (not strictly needed for this logic test if not testing geo-specific criteria)
    })
    
    reporting_period_interv = "Q4 2023 Intervention Planning"
    
    # First, get available criteria (UI would use this to populate multiselect)
    all_criteria_options = get_intervention_criteria_options(mock_gdf_interv)
    print("\n## Available Intervention Criteria:")
    for name_crit in all_criteria_options.keys(): print(f"  - {name_crit}")

    # Simulate DHO selecting a few criteria
    selected_criteria_names_by_dho = [
        f"High Avg. AI Risk (Zone Score ≥ {app_config.DISTRICT_ZONE_HIGH_RISK_AVG_SCORE})",
        f"Low Facility Coverage (Zone Score < {app_config.DISTRICT_INTERVENTION_FACILITY_COVERAGE_LOW_PCT}%)",
        f"High Absolute TB Burden (Zone Active TB Cases > {app_config.DISTRICT_INTERVENTION_TB_BURDEN_HIGH_ABS})"
    ]
    # Filter selected criteria to only those that are actually available (safety check)
    actual_selected_criteria = [name for name in selected_criteria_names_by_dho if name in all_criteria_options]


    intervention_data_results = identify_priority_zones_for_intervention(
        district_gdf_main_enriched=mock_gdf_interv,
        selected_criteria_display_names=actual_selected_criteria,
        available_criteria_options=all_criteria_options,
        reporting_period_str=reporting_period_interv
    )

    print(f"\n--- Prepared Intervention Data for: {intervention_data_results['reporting_period']} ---")
    print(f"Applied Criteria: {intervention_data_results['applied_criteria']}")
    
    print("\n## Priority Zones for Intervention DataFrame:")
    if intervention_data_results["priority_zones_for_intervention_df"] is not None and \
       not intervention_data_results["priority_zones_for_intervention_df"].empty:
        print(intervention_data_results["priority_zones_for_intervention_df"].to_string())
    else:
        print("  (No priority zones identified based on selected criteria or data insufficient)")

    print("\n## Data Availability Notes:")
    if intervention_data_results["data_availability_notes"]:
        for note in intervention_data_results["data_availability_notes"]: print(f"  - {note}")
    else:
        print("  (No specific data availability notes)")
