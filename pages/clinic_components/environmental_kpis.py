# test/pages/clinic_components/environmental_kpis.py
# Redesigned for "Sentinel Health Co-Pilot" - LMIC Edge-First System
# This module calculates summary Key Performance Indicators (KPIs) for the
# clinic's environment based on IoT data processed at a Facility Node (Tier 2).
# The output is a structured dictionary of metrics, suitable for:
#   1. Displaying on a simplified web dashboard for clinic managers.
#   2. Feeding into automated alerting systems at the Facility Node for immediate action.
#   3. Generating periodic environmental safety reports.

import pandas as pd
import numpy as np
import logging
# Assuming app_config is in the PYTHONPATH or project root.
from config import app_config # Uses new, redesigned app_config
# core_data_processing.get_clinic_environmental_summary is used
from utils.core_data_processing import get_clinic_environmental_summary
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

def calculate_clinic_environmental_kpis(
    filtered_iot_df_clinic_period: pd.DataFrame, # IoT data for the clinic over a defined period
    # iot_data_was_loaded_initially: bool, # This flag is less relevant here; function assumes valid input or handles empty
    reporting_period_str: str # e.g., "Today", "Last 24 Hours", "2023-10-01 to 2023-10-07"
) -> Dict[str, Any]:
    """
    Calculates and returns key environmental KPIs for the clinic.

    Args:
        filtered_iot_df_clinic_period: DataFrame containing processed IoT sensor readings
                                       for the clinic over the specified reporting period.
                                       Expected to be processed by `get_clinic_environmental_summary`.
        reporting_period_str: A string describing the reporting period for context.

    Returns:
        Dict[str, Any]: A dictionary of calculated environmental KPIs.
        Example: {
            "reporting_period": "Last 24 Hours",
            "avg_co2_ppm_overall": 650.5,
            "co2_status_level": "LOW", // LOW, MODERATE, HIGH based on thresholds
            "co2_rooms_at_very_high_alert_count": 0,
            "avg_pm25_ugm3_overall": 12.3,
            "pm25_status_level": "LOW",
            "pm25_rooms_at_very_high_alert_count": 0,
            "avg_waiting_room_occupancy_persons": 7.2,
            "occupancy_status_level": "MODERATE",
            "occupancy_waiting_room_over_max_flag": False,
            "noise_rooms_at_high_alert_count": 1
        }
    """
    logger.info(f"Calculating clinic environmental KPIs for period: {reporting_period_str}")

    kpi_results = {
        "reporting_period": reporting_period_str,
        "avg_co2_ppm_overall": np.nan, "co2_status_level": "NEUTRAL", "co2_rooms_at_very_high_alert_count": 0,
        "avg_pm25_ugm3_overall": np.nan, "pm25_status_level": "NEUTRAL", "pm25_rooms_at_very_high_alert_count": 0,
        "avg_waiting_room_occupancy_persons": np.nan, "occupancy_status_level": "NEUTRAL", "occupancy_waiting_room_over_max_flag": False,
        "avg_noise_dba_overall": np.nan, "noise_status_level": "NEUTRAL", "noise_rooms_at_high_alert_count": 0
    }

    if filtered_iot_df_clinic_period is None or filtered_iot_df_clinic_period.empty:
        logger.warning("No IoT environmental data provided for KPI calculation.")
        # Return kpi_results with NaNs/defaults indicating no data.
        return kpi_results

    # Utilize the existing get_clinic_environmental_summary from core_data_processing
    # which should be updated to use new app_config thresholds
    clinic_env_summary_data = get_clinic_environmental_summary(
        filtered_iot_df_clinic_period,
        source_context="ClinicKPIs/EnvSummary" # Add context
    )
    
    # --- Populate KPIs from the summary ---

    # CO2 KPIs
    avg_co2 = clinic_env_summary_data.get('avg_co2_overall_ppm', np.nan) # Expecting key from updated get_clinic_environmental_summary
    kpi_results["avg_co2_ppm_overall"] = avg_co2
    kpi_results["co2_rooms_at_very_high_alert_count"] = clinic_env_summary_data.get('rooms_co2_very_high_alert_latest_count', 0) # Using VERY_HIGH threshold
    
    if pd.notna(avg_co2):
        if kpi_results["co2_rooms_at_very_high_alert_count"] > 0 or avg_co2 > app_config.ALERT_AMBIENT_CO2_VERY_HIGH_PPM : # Added avg_co2 check too
            kpi_results["co2_status_level"] = "HIGH_RISK" # Changed from "High"
        elif avg_co2 > app_config.ALERT_AMBIENT_CO2_HIGH_PPM: # Check against general high
            kpi_results["co2_status_level"] = "MODERATE_RISK"
        else: # Assumes CO2_HIGH_PPM is above any "ideal" threshold. For LMICs, "not high" might be "acceptable".
            kpi_results["co2_status_level"] = "ACCEPTABLE"
    else: kpi_results["co2_status_level"] = "NO_DATA"


    # PM2.5 KPIs
    avg_pm25 = clinic_env_summary_data.get('avg_pm25_overall_ugm3', np.nan)
    kpi_results["avg_pm25_ugm3_overall"] = avg_pm25
    kpi_results["pm25_rooms_at_very_high_alert_count"] = clinic_env_summary_data.get('rooms_pm25_very_high_alert_latest_count', 0)
    
    if pd.notna(avg_pm25):
        if kpi_results["pm25_rooms_at_very_high_alert_count"] > 0 or avg_pm25 > app_config.ALERT_AMBIENT_PM25_VERY_HIGH_UGM3:
            kpi_results["pm25_status_level"] = "HIGH_RISK"
        elif avg_pm25 > app_config.ALERT_AMBIENT_PM25_HIGH_UGM3:
            kpi_results["pm25_status_level"] = "MODERATE_RISK"
        else:
            kpi_results["pm25_status_level"] = "ACCEPTABLE"
    else: kpi_results["pm25_status_level"] = "NO_DATA"


    # Waiting Room Occupancy KPIs
    avg_occ = clinic_env_summary_data.get('avg_waiting_room_occupancy_overall_persons', np.nan) # Adjusted key name for clarity
    kpi_results["avg_waiting_room_occupancy_persons"] = avg_occ
    # 'waiting_room_high_occupancy_alert_latest_flag' from summary is boolean: (any room latest reading > TARGET_CLINIC_WAITING_ROOM_OCCUPANCY_MAX)
    kpi_results["occupancy_waiting_room_over_max_flag"] = clinic_env_summary_data.get('waiting_room_high_occupancy_alert_latest_flag', False)
    
    if pd.notna(avg_occ):
        if kpi_results["occupancy_waiting_room_over_max_flag"]:
            kpi_results["occupancy_status_level"] = "HIGH_CONCERN" # Using concern as it's a target breach
        # For LMIC, "moderate" could be > 75% of max target.
        elif avg_occ > (app_config.TARGET_CLINIC_WAITING_ROOM_OCCUPANCY_MAX * 0.75):
            kpi_results["occupancy_status_level"] = "MODERATE_CONCERN"
        else:
            kpi_results["occupancy_status_level"] = "ACCEPTABLE"
    else: kpi_results["occupancy_status_level"] = "NO_DATA"


    # Noise KPIs
    avg_noise = clinic_env_summary_data.get('avg_noise_overall_dba', np.nan) # Key name update for clarity
    kpi_results["avg_noise_dba_overall"] = avg_noise
    kpi_results["noise_rooms_at_high_alert_count"] = clinic_env_summary_data.get('rooms_noise_high_alert_latest_count', 0) # Uses ALERT_AMBIENT_NOISE_HIGH_DBA
    
    if pd.notna(avg_noise):
        if kpi_results["noise_rooms_at_high_alert_count"] > 0 or avg_noise > app_config.ALERT_AMBIENT_NOISE_HIGH_DBA: # Average itself is high
            kpi_results["noise_status_level"] = "HIGH_CONCERN" # High noise can impact communication and patient/staff stress
        # Define moderate if needed, e.g., avg_noise > some "ideal comfort" level lower than ALERT_AMBIENT_NOISE_HIGH_DBA
        # For now, simple high or acceptable.
        else:
            kpi_results["noise_status_level"] = "ACCEPTABLE"
    else: kpi_results["noise_status_level"] = "NO_DATA"

    logger.info(f"Clinic environmental KPIs calculated: {kpi_results}")
    return kpi_results

# --- Example Usage (for testing or integration into a reporting script) ---
if __name__ == '__main__':
    logger.setLevel(logging.DEBUG)
    logger.info("Running Clinic Environmental KPIs component simulation directly...")

    # Simulate iot_df_period that get_clinic_environmental_summary would process
    # For this test, we'll assume get_clinic_environmental_summary is robust and just create a mock output
    # as if it came from that function. In a real test, you'd use sample_iot_df and call it.
    
    # Sample output of get_clinic_environmental_summary
    mock_env_summary_output = {
        'avg_co2_overall_ppm': 750.0, 'rooms_co2_very_high_alert_latest_count': 0,
        'avg_pm25_overall_ugm3': 38.5, 'rooms_pm25_very_high_alert_latest_count': 1,
        'avg_waiting_room_occupancy_overall_persons': 12.5, 'waiting_room_high_occupancy_alert_latest_flag': True,
        'avg_noise_overall_dba': 60.0, 'rooms_noise_high_alert_latest_count': 0
    }
    
    # For the purpose of testing *this* specific module `calculate_clinic_environmental_kpis`,
    # it's more direct to test its logic directly based on the expected input.
    # We create a dummy filtered_iot_df which will then be processed by the actual get_clinic_environmental_summary
    # inside `calculate_clinic_environmental_kpis` (as per its current design).

    sample_iot_data_for_period = pd.DataFrame({
        'timestamp': pd.to_datetime(['2023-10-05T10:00:00', '2023-10-05T11:00:00', '2023-10-05T10:00:00', '2023-10-05T11:00:00']),
        'clinic_id': ['C01', 'C01', 'C01', 'C01'],
        'room_name': ['WaitingArea', 'WaitingArea', 'ConsultRoom1', 'ConsultRoom1'],
        'avg_co2_ppm': [app_config.ALERT_AMBIENT_CO2_HIGH_PPM + 50, # Moderate CO2 in waiting area
                        app_config.ALERT_AMBIENT_CO2_VERY_HIGH_PPM + 100, # Very High in consult (latest)
                        600, 550], # Good in consult room
        'avg_pm25': [app_config.ALERT_AMBIENT_PM25_HIGH_UGM3 - 2, # Acceptable PM2.5
                       15,
                       app_config.ALERT_AMBIENT_PM25_VERY_HIGH_UGM3 + 5, # Very High PM2.5
                       20],
        'waiting_room_occupancy': [app_config.TARGET_CLINIC_WAITING_ROOM_OCCUPANCY_MAX -1,
                                   app_config.TARGET_CLINIC_WAITING_ROOM_OCCUPANCY_MAX + 5, # High occupancy
                                   2, 1], # Occupancy not relevant for consult rooms by this name
        'avg_noise_db': [60, 62, app_config.ALERT_AMBIENT_NOISE_HIGH_DBA + 5, 50] # One room high noise
    })

    # Corrected: Call get_clinic_environmental_summary here for testing setup (to simulate data prep step if not passing the direct summary)
    # But, for simplicity in this standalone unit test of calculate_clinic_environmental_kpis,
    # and because calculate_clinic_environmental_kpis *calls* get_clinic_environmental_summary,
    # we pass sample_iot_data_for_period and let the tested function handle it.
    
    env_kpi_results = calculate_clinic_environmental_kpis(
        filtered_iot_df_clinic_period=sample_iot_data_for_period, # Pass the raw-ish period data
        reporting_period_str="Test Period October 5th"
    )

    print("\n--- Calculated Clinic Environmental KPIs: ---")
    if env_kpi_results:
        for key, value in env_kpi_results.items():
            print(f"  {key.replace('_', ' ').title()}: {value}")
    else:
        print("No environmental KPIs calculated.")
        
    # Test with empty input
    empty_env_kpi_results = calculate_clinic_environmental_kpis(pd.DataFrame(), "Empty Test Period")
    print("\n--- KPIs for Empty Input: ---")
    for key, value in empty_env_kpi_results.items(): print(f"  {key.replace('_', ' ').title()}: {value}")
