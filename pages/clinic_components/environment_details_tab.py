# test/pages/clinic_components/environment_details_tab.py
# Redesigned for "Sentinel Health Co-Pilot" - LMIC Edge-First System
# This module prepares detailed environmental data for trends and latest readings
# from clinic IoT sensors. This data is intended for:
#   1. Display on a simplified web dashboard/report for clinic managers at a Facility Node (Tier 2).
#   2. Investigation into specific environmental concerns highlighted by KPIs.

import pandas as pd
import numpy as np
import logging
# Assuming app_config is in the PYTHONPATH or project root.
from config import app_config # Uses new, redesigned app_config
from utils.core_data_processing import get_trend_data, get_clinic_environmental_summary # Re-using for consistency
from typing import Dict, Any, Optional, List, Tuple

logger = logging.getLogger(__name__)

def prepare_clinic_environment_details_data(
    filtered_iot_df_clinic_period: pd.DataFrame, # IoT data for the clinic over a defined period
    iot_data_was_available_globally: bool, # Flag indicating if IoT data source exists at all
    reporting_period_str: str
) -> Dict[str, Any]:
    """
    Prepares structured data for detailed environmental trends and latest room readings.

    Args:
        filtered_iot_df_clinic_period: DataFrame containing processed IoT sensor readings
                                       for the clinic over the specified reporting period.
        iot_data_was_available_globally: True if the IoT data source generally exists for the system.
        reporting_period_str: A string describing the reporting period for context.

    Returns:
        Dict[str, Any]: A dictionary containing structured environmental detail data.
        Example: {
            "reporting_period": "Last 24 Hours",
            "current_environmental_alerts_summary": [
                {"alert_type": "CO2", "message": "1 room(s) with CO2 > 2500ppm.", "level": "HIGH"}
            ],
            "hourly_avg_co2_trend_series": pd.Series(...), // Index=datetime, Value=avg_co2
            "hourly_avg_waiting_room_occupancy_trend_series": pd.Series(...),
            "latest_sensor_readings_by_room_df": pd.DataFrame(...), // Cols: room_name, timestamp, co2, pm25, etc.
            "data_availability_notes": ["No CO2 trend data to plot."] // If any issues
        }
    """
    logger.info(f"Preparing clinic environment details data for period: {reporting_period_str}")

    env_details_output: Dict[str, Any] = {
        "reporting_period": reporting_period_str,
        "current_environmental_alerts_summary": [],
        "hourly_avg_co2_trend_series": None,
        "hourly_avg_waiting_room_occupancy_trend_series": None,
        "latest_sensor_readings_by_room_df": None,
        "data_availability_notes": []
    }

    if filtered_iot_df_clinic_period is None or filtered_iot_df_clinic_period.empty:
        if iot_data_was_available_globally:
            env_details_output["data_availability_notes"].append(
                f"No clinic environmental IoT data found for the period: {reporting_period_str} to display details and trends."
            )
        else:
            # If IoT data wasn't even available globally, a more general message is typically handled by the caller.
            env_details_output["data_availability_notes"].append(
                "IoT data source appears unavailable for environmental monitoring details."
            )
        return env_details_output

    df_iot_period = filtered_iot_df_clinic_period.copy()
    if 'timestamp' not in df_iot_period.columns or not pd.api.types.is_datetime64_any_dtype(df_iot_period['timestamp']):
        env_details_output["data_availability_notes"].append("Timestamp data missing or invalid in IoT records.")
        return env_details_output # Cannot proceed without valid timestamps

    # 1. Current Environmental Alerts Summary (from latest readings in period)
    #    Leverage get_clinic_environmental_summary for consistency
    env_summary_for_alerts = get_clinic_environmental_summary(
        df_iot_period, source_context="EnvDetailsTab/Alerts"
    )
    
    if env_summary_for_alerts.get('rooms_co2_very_high_alert_latest_count', 0) > 0: # Using VERY_HIGH from new config
        env_details_output["current_environmental_alerts_summary"].append({
            "alert_type": "CO2",
            "message": f"{env_summary_for_alerts['rooms_co2_very_high_alert_latest_count']} room(s) with CO2 > {app_config.ALERT_AMBIENT_CO2_VERY_HIGH_PPM}ppm.",
            "level": "HIGH_RISK"
        })
    if env_summary_for_alerts.get('rooms_pm25_very_high_alert_latest_count', 0) > 0:
        env_details_output["current_environmental_alerts_summary"].append({
            "alert_type": "PM2.5",
            "message": f"{env_summary_for_alerts['rooms_pm25_very_high_alert_latest_count']} room(s) with PM2.5 > {app_config.ALERT_AMBIENT_PM25_VERY_HIGH_UGM3}µg/m³.",
            "level": "HIGH_RISK"
        })
    if env_summary_for_alerts.get('rooms_noise_high_alert_latest_count', 0) > 0:
        env_details_output["current_environmental_alerts_summary"].append({
            "alert_type": "Noise",
            "message": f"{env_summary_for_alerts['rooms_noise_high_alert_latest_count']} room(s) with Noise > {app_config.ALERT_AMBIENT_NOISE_HIGH_DBA}dB.",
            "level": "HIGH_CONCERN"
        })
    if env_summary_for_alerts.get('waiting_room_high_occupancy_alert_latest_flag', False):
        env_details_output["current_environmental_alerts_summary"].append({
            "alert_type": "Occupancy",
            "message": f"High Waiting Room Occupancy: > {app_config.TARGET_CLINIC_WAITING_ROOM_OCCUPANCY_MAX} persons (latest reading in period).",
            "level": "HIGH_CONCERN"
        })
    if not env_details_output["current_environmental_alerts_summary"]:
        env_details_output["current_environmental_alerts_summary"].append({
            "alert_type": "General", "message": "No critical environmental alerts based on latest readings in period.", "level": "ACCEPTABLE"
        })


    # 2. Hourly Trends for Key Environmental Metrics
    #    Ensure timestamp column is valid datetime before passing to get_trend_data
    df_iot_period['timestamp'] = pd.to_datetime(df_iot_period['timestamp'], errors='coerce')
    df_iot_period_cleaned_ts = df_iot_period.dropna(subset=['timestamp'])

    if not df_iot_period_cleaned_ts.empty:
        # CO2 Trend
        if 'avg_co2_ppm' in df_iot_period_cleaned_ts.columns:
            co2_trend = get_trend_data(df_iot_period_cleaned_ts, 'avg_co2_ppm', date_col='timestamp', period='H', agg_func='mean', source_context="EnvDetails/CO2Trend")
            if co2_trend is not None and not co2_trend.empty:
                env_details_output["hourly_avg_co2_trend_series"] = co2_trend
            else: env_details_output["data_availability_notes"].append("No CO2 trend data to plot for the selected period.")
        else: env_details_output["data_availability_notes"].append("CO2 data ('avg_co2_ppm') missing for trend.")

        # Waiting Room Occupancy Trend
        if 'waiting_room_occupancy' in df_iot_period_cleaned_ts.columns:
            occ_trend = get_trend_data(df_iot_period_cleaned_ts, 'waiting_room_occupancy', date_col='timestamp', period='H', agg_func='mean', source_context="EnvDetails/OccupancyTrend")
            if occ_trend is not None and not occ_trend.empty:
                env_details_output["hourly_avg_waiting_room_occupancy_trend_series"] = occ_trend
            else: env_details_output["data_availability_notes"].append("No waiting room occupancy trend data for the selected period.")
        else: env_details_output["data_availability_notes"].append("Occupancy data ('waiting_room_occupancy') column missing for trend.")
    else:
        env_details_output["data_availability_notes"].append("No valid timestamp data available in IoT records for trend calculation.")

    # 3. Latest Sensor Readings by Room (End of Selected Period)
    cols_for_latest_readings_table = [
        'clinic_id', 'room_name', 'timestamp', 'avg_co2_ppm', 'avg_pm25',
        'avg_temp_celsius', 'avg_humidity_rh', 'avg_noise_db',
        'waiting_room_occupancy', 'patient_throughput_per_hour', 'sanitizer_dispenses_per_hour'
    ]
    available_cols_for_latest = [col for col in cols_for_latest_readings_table if col in df_iot_period.columns]

    if all(c in available_cols_for_latest for c in ['timestamp', 'clinic_id', 'room_name']):
        # Ensure timestamp is datetime for sorting before drop_duplicates
        df_iot_period['timestamp'] = pd.to_datetime(df_iot_period['timestamp'], errors='coerce')
        latest_readings_df = df_iot_period.dropna(subset=['timestamp']) \
                                         .sort_values('timestamp') \
                                         .drop_duplicates(subset=['clinic_id', 'room_name'], keep='last')
        
        if not latest_readings_df.empty:
            env_details_output["latest_sensor_readings_by_room_df"] = latest_readings_df[available_cols_for_latest].reset_index(drop=True)
        else:
            env_details_output["data_availability_notes"].append("No distinct room sensor readings available for the latest point in this period after filtering.")
    else:
        missing_cols_msg = "Essential columns (timestamp, clinic_id, room_name) missing from IoT data, cannot display detailed latest room readings."
        env_details_output["data_availability_notes"].append(missing_cols_msg)
        logger.warning(missing_cols_msg)
        
    logger.info("Clinic environment details data preparation complete.")
    return env_details_output


# --- Example Usage (for testing or integration into a reporting script) ---
if __name__ == '__main__':
    logger.setLevel(logging.DEBUG)
    logger.info("Running Clinic Environment Details Tab component data preparation simulation...")

    # Create sample IoT data for a period
    sample_iot_period_data = pd.DataFrame({
        'timestamp': pd.to_datetime(['2023-10-05T10:00:00Z', '2023-10-05T11:00:00Z', '2023-10-05T10:30:00Z', '2023-10-05T11:30:00Z',
                                     '2023-10-05T09:00:00Z' # Earlier record for another room
                                     ]),
        'clinic_id': ['C01', 'C01', 'C01', 'C01', 'C01'],
        'room_name': ['WaitingArea', 'WaitingArea', 'ConsultRoom1', 'ConsultRoom1', 'Lab'],
        'avg_co2_ppm': [1600, app_config.ALERT_AMBIENT_CO2_VERY_HIGH_PPM + 50, 700, 650, 800],
        'avg_pm25': [20, 25, app_config.ALERT_AMBIENT_PM25_VERY_HIGH_UGM3 + 10, 15, 10],
        'avg_temp_celsius': [24, 24.5, 23, 23.5, 22],
        'avg_humidity_rh': [55, 56, 50, 52, 48],
        'avg_noise_db': [60, 65, 50, app_config.ALERT_AMBIENT_NOISE_HIGH_DBA + 2, 45],
        'waiting_room_occupancy': [10, app_config.TARGET_CLINIC_WAITING_ROOM_OCCUPANCY_MAX + 3, np.nan, np.nan, np.nan], # Occupancy mainly for waiting area
        'patient_throughput_per_hour': [5,6,10,12, np.nan],
        'sanitizer_dispenses_per_hour': [2,3,1,1,np.nan]
    })
    
    reporting_str = "Daily Report - 2023-10-05"
    env_details = prepare_clinic_environment_details_data(
        filtered_iot_df_clinic_period=sample_iot_period_data,
        iot_data_was_available_globally=True,
        reporting_period_str=reporting_str
    )

    print(f"\n--- Prepared Clinic Environment Details for: {reporting_str} ---")
    for key, value in env_details.items():
        print(f"\n## {key.replace('_', ' ').title()}:")
        if isinstance(value, pd.DataFrame):
            print(value.to_string())
        elif isinstance(value, pd.Series):
            print(value.to_string())
        elif isinstance(value, list) and key=="current_environmental_alerts_summary":
             for alert_item in value: print(f"  - Type: {alert_item['alert_type']}, Message: {alert_item['message']}, Level: {alert_item['level']}")
        elif isinstance(value, list) and key=="data_availability_notes":
            if value:
                for note in value: print(f"  - {note}")
            else: print("  (No data availability notes)")
        else:
            print(f"  {value}")

    # Test with empty input
    empty_details = prepare_clinic_environment_details_data(pd.DataFrame(), True, "Empty Period Test")
    print(f"\n--- Details for Empty Input: ---")
    for key, val in empty_details.items(): print(f"  {key}: {val}")
