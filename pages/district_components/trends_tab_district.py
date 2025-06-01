# test/pages/district_components/trends_tab_district.py
# Redesigned for "Sentinel Health Co-Pilot" - LMIC Edge-First System
# This module calculates district-wide health and environmental trend data
# based on aggregated and filtered time-series information.
# The output is structured data (Pandas Series/DataFrames) intended for:
#   1. Display on a web dashboard/report for District Health Officers (DHOs)
#      at a Facility Node (Tier 2) or Cloud Node (Tier 3).
#   2. Supporting strategic planning, intervention monitoring, and epidemiological surveillance.

import pandas as pd
import numpy as np # Not directly used here now, but common with pandas
import logging
# Assuming app_config is in the PYTHONPATH or project root.
from config import app_config # Uses new, redesigned app_config
from utils.core_data_processing import get_trend_data # Core utility for trend calculation
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

def calculate_district_trends_data(
    filtered_health_for_trends_dist: pd.DataFrame, # Health data for the district and period
    filtered_iot_for_trends_dist: Optional[pd.DataFrame], # IoT data for the district and period
    trend_start_date: Any, # datetime.date object
    trend_end_date: Any, # datetime.date object
    reporting_period_str: str,
    disease_trend_agg_period: str = 'W-Mon', # Weekly for disease incidence
    general_trend_agg_period: str = 'D' # Daily for risk, steps, CO2
) -> Dict[str, Any]:
    """
    Calculates various district-wide health and environmental trends.

    Args:
        filtered_health_for_trends_dist: Filtered health records for the district and period.
        filtered_iot_for_trends_dist: Optional filtered IoT data for the district and period.
        trend_start_date, trend_end_date: Date range for analysis.
        reporting_period_str: String description of the period.
        disease_trend_agg_period: Aggregation period for disease incidence (e.g., 'W-Mon').
        general_trend_agg_period: Aggregation period for other metrics (e.g., 'D').

    Returns:
        Dict[str, Any]: A dictionary where keys are trend metric names and values are
                        Pandas Series/DataFrames or None if data is insufficient.
        Example: {
            "reporting_period": "Oct 2023",
            "disease_incidence_trends": {
                "TB": pd.Series(...), "Malaria": pd.Series(...)
            },
            "avg_patient_ai_risk_trend_series": pd.Series(...),
            "avg_patient_daily_steps_trend_series": pd.Series(...),
            "avg_clinic_co2_trend_series": pd.Series(...),
            "data_availability_notes": []
        }
    """
    logger.info(f"Calculating district-wide trends for period: {reporting_period_str} ({trend_start_date} to {trend_end_date})")

    district_trends_output: Dict[str, Any] = {
        "reporting_period": reporting_period_str,
        "disease_incidence_trends": {}, # Dict: {condition_name: pd.Series}
        "avg_patient_ai_risk_trend_series": None,
        "avg_patient_daily_steps_trend_series": None,
        "avg_clinic_co2_trend_series": None,
        "data_availability_notes": []
    }

    if (filtered_health_for_trends_dist is None or filtered_health_for_trends_dist.empty) and \
       (filtered_iot_for_trends_dist is None or filtered_iot_for_trends_dist.empty):
        msg = "No health or environmental data available for the selected trend period."
        logger.warning(msg)
        district_trends_output["data_availability_notes"].append(msg)
        return district_trends_output

    # --- Disease Incidence Trends (New Cases Identified per specified period) ---
    # Using KEY_CONDITIONS_FOR_ACTION from new config as these are usually high priority for DHOs
    key_conditions_for_district_trends = app_config.KEY_CONDITIONS_FOR_ACTION[:4] # Take top few for example

    if filtered_health_for_trends_dist is not None and not filtered_health_for_trends_dist.empty and \
       all(c in filtered_health_for_trends_dist.columns for c in ['condition', 'patient_id', 'encounter_date']):
        
        # Prepare data for incidence: first occurrence of condition per patient in the *overall dataset*
        # This definition of "new case" depends on the total dataset available, not just the filtered period.
        # If we mean "new cases identified *within this period*", the logic in `get_trend_data` for nunique handles it.
        # For a true "incidence" (first ever recorded), a more complex check against all historical data is needed.
        # For this simulation, we'll assume `get_trend_data` with `agg_func='nunique'` on `patient_id` effectively gives
        # the number of *unique patients presenting with the condition* in each sub-period (day/week).
        
        incidence_trends_data: Dict[str, Optional[pd.Series]] = {}
        for condition_name in key_conditions_for_district_trends:
            # Filter for the specific condition. Allow partial matching for flexibility.
            condition_specific_df = filtered_health_for_trends_dist[
                filtered_health_for_trends_dist['condition'].str.contains(condition_name, case=False, na=False)
            ]
            if not condition_specific_df.empty:
                new_cases_trend_series = get_trend_data(
                    df=condition_specific_df,
                    value_col='patient_id', # Count unique patients for this condition in period
                    date_col='encounter_date',
                    period=disease_trend_agg_period,
                    agg_func='nunique',
                    source_context=f"DistrictTrends/Incidence/{condition_name}"
                )
                if new_cases_trend_series is not None and not new_cases_trend_series.empty:
                    incidence_trends_data[condition_name] = new_cases_trend_series
                else:
                    district_trends_output["data_availability_notes"].append(f"No new {condition_name} case trend data.")
            else:
                district_trends_output["data_availability_notes"].append(f"No data for condition '{condition_name}' in period.")
        district_trends_output["disease_incidence_trends"] = incidence_trends_data
    else:
        district_trends_output["data_availability_notes"].append("Health data missing columns for disease incidence trends (condition, patient_id, encounter_date).")


    # --- Overall AI Risk Score Trend (District-Wide Average) ---
    if filtered_health_for_trends_dist is not None and not filtered_health_for_trends_dist.empty and \
       'ai_risk_score' in filtered_health_for_trends_dist.columns and \
       filtered_health_for_trends_dist['ai_risk_score'].notna().any():
        
        avg_risk_trend_series = get_trend_data(
            df=filtered_health_for_trends_dist,
            value_col='ai_risk_score',
            date_col='encounter_date',
            period=general_trend_agg_period,
            agg_func='mean',
            source_context="DistrictTrends/AvgRisk"
        )
        if avg_risk_trend_series is not None and not avg_risk_trend_series.empty:
            district_trends_output["avg_patient_ai_risk_trend_series"] = avg_risk_trend_series
        else:
            district_trends_output["data_availability_notes"].append("No AI risk score trend data (daily avg).")
    else:
        district_trends_output["data_availability_notes"].append("AI Risk Score data missing or all null for trend analysis.")


    # --- Avg Daily Steps Trend (District-Wide Average from Patient Data - Wellness Proxy) ---
    # This assumes 'avg_daily_steps' is part of the synced health data from CHWs/wearables
    if filtered_health_for_trends_dist is not None and not filtered_health_for_trends_dist.empty and \
       'avg_daily_steps' in filtered_health_for_trends_dist.columns and \
       filtered_health_for_trends_dist['avg_daily_steps'].notna().any():

        avg_steps_trend_series = get_trend_data(
            df=filtered_health_for_trends_dist,
            value_col='avg_daily_steps',
            date_col='encounter_date', # Assuming steps data is tied to encounter date
            period=disease_trend_agg_period, # Often better as weekly average
            agg_func='mean',
            source_context="DistrictTrends/AvgSteps"
        )
        if avg_steps_trend_series is not None and not avg_steps_trend_series.empty:
            district_trends_output["avg_patient_daily_steps_trend_series"] = avg_steps_trend_series
        else:
            district_trends_output["data_availability_notes"].append("No patient steps trend data (weekly avg).")
    else:
        district_trends_output["data_availability_notes"].append("Average daily steps data missing or all null for trends.")
    

    # --- Clinic Environmental Trend (District Average for a key metric, e.g., CO2) ---
    if filtered_iot_for_trends_dist is not None and not filtered_iot_for_trends_dist.empty and \
       'avg_co2_ppm' in filtered_iot_for_trends_dist.columns and \
       'timestamp' in filtered_iot_for_trends_dist.columns and \
       filtered_iot_for_trends_dist['avg_co2_ppm'].notna().any():

        avg_co2_trend_series = get_trend_data(
            df=filtered_iot_for_trends_dist,
            value_col='avg_co2_ppm',
            date_col='timestamp', # IoT data uses 'timestamp'
            period=general_trend_agg_period,
            agg_func='mean',
            source_context="DistrictTrends/AvgCO2"
        )
        if avg_co2_trend_series is not None and not avg_co2_trend_series.empty:
            district_trends_output["avg_clinic_co2_trend_series"] = avg_co2_trend_series
        else:
            district_trends_output["data_availability_notes"].append("No clinic CO2 trend data from IoT (daily avg).")
    else:
        district_trends_output["data_availability_notes"].append("Clinic CO2 data (avg_co2_ppm or timestamp) missing or all null for environmental trends.")
        
    logger.info("District-wide trends data calculation complete.")
    return district_trends_output


# --- Example Usage (for testing or integration into a DHO reporting script) ---
if __name__ == '__main__':
    logger.setLevel(logging.DEBUG)
    logger.info("Running District Trends Tab component data preparation simulation directly...")

    # Sample historical health data
    date_rng_dist_trend = pd.date_range(start='2023-09-01', end='2023-10-31', freq='D')
    num_records_dist_trend = len(date_rng_dist_trend) * 10
    sample_health_district = pd.DataFrame({
        'encounter_date': np.random.choice(date_rng_dist_trend, num_records_dist_trend),
        'patient_id': [f'DP{i%100:03d}' for i in range(num_records_dist_trend)],
        'condition': np.random.choice(app_config.KEY_CONDITIONS_FOR_ACTION + ['Flu', 'Injury'], num_records_dist_trend, 
                                      p=[0.05]*len(app_config.KEY_CONDITIONS_FOR_ACTION) + [0.3, 0.3] ), # Weighted
        'ai_risk_score': np.random.randint(20, 95, num_records_dist_trend),
        'avg_daily_steps': np.random.randint(1000, 12000, num_records_dist_trend),
        'zone_id': np.random.choice(['ZoneX', 'ZoneY', 'ZoneZ'], num_records_dist_trend)
    })
    sample_health_district.sort_values('encounter_date', inplace=True)

    # Sample historical IoT data
    sample_iot_district = pd.DataFrame({
        'timestamp': np.random.choice(date_rng_dist_trend, num_records_dist_trend // 2), # Fewer IoT records
        'avg_co2_ppm': np.random.randint(400, 2000, num_records_dist_trend // 2),
        'zone_id': np.random.choice(['ZoneX', 'ZoneY', 'ZoneZ'], num_records_dist_trend // 2)
    })
    sample_iot_district.sort_values('timestamp', inplace=True)

    start_date_param = pd.Timestamp('2023-10-01').date()
    end_date_param = pd.Timestamp('2023-10-31').date()
    period_str_param = "October 2023 District Trends"

    district_trends_result = calculate_district_trends_data(
        filtered_health_for_trends_dist=sample_health_district,
        filtered_iot_for_trends_dist=sample_iot_district,
        trend_start_date=start_date_param,
        trend_end_date=end_date_param,
        reporting_period_str=period_str_param,
        disease_trend_agg_period='W-Mon',
        general_trend_agg_period='D'
    )

    print(f"\n--- Calculated District Trends Data for: {period_str_param} ---")
    for trend_key, trend_data in district_trends_result.items():
        print(f"\n## {trend_key.replace('_', ' ').title()}:")
        if trend_key == "disease_incidence_trends" and isinstance(trend_data, dict):
            for cond_name, series_data in trend_data.items():
                print(f"  Condition: {cond_name}")
                if series_data is not None and not series_data.empty: print(series_data.to_string())
                else: print("    (No trend data)")
        elif isinstance(trend_data, pd.Series) and not trend_data.empty:
            print(trend_data.to_string())
        elif isinstance(trend_data, list) and trend_key=="data_availability_notes": # notes
            if trend_data:
                for note in trend_data: print(f"  - {note}")
            else: print("  (No data availability notes)")
        elif trend_data is None or (isinstance(trend_data, pd.Series) and trend_data.empty):
            print("  (No trend data or data insufficient)")
        else: # E.g. the reporting_period string
             print(f"  {trend_data}")
