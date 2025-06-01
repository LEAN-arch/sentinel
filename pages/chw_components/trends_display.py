# test/pages/chw_components/trends_display.py
# Redesigned for "Sentinel Health Co-Pilot" - LMIC Edge-First System
# This module focuses on calculating CHW activity trend data over a period.
# This logic is primarily for:
#   1. Use at a Supervisor Hub (Tier 1) or Facility Node (Tier 2) to generate
#      reports or simple web dashboards showing CHW team/individual performance trends.
#   2. Simulation and testing of trend calculation.
# Detailed trend charts are generally not displayed on a CHW's Personal Edge Device (PED).

import pandas as pd
import numpy as np
import logging
# Assuming app_config is in the PYTHONPATH or project root.
from config import app_config # Uses new, redesigned app_config
# The core_data_processing.get_trend_data is assumed to be available and robust.
from utils.core_data_processing import get_trend_data
from typing import Dict, Any, Optional, pd # pd for DataFrame type hint from pandas

logger = logging.getLogger(__name__)

def calculate_chw_activity_trends(
    chw_historical_health_df: pd.DataFrame, # Historical data for a CHW or CHW team
    trend_start_date: Any, # datetime.date object
    trend_end_date: Any, # datetime.date object
    zone_filter: Optional[str] = None, # e.g., "ZoneA" or None for all zones in the df
    time_period_agg: str = 'D' # 'D' for daily, 'W' for weekly
) -> Dict[str, Optional[pd.DataFrame]]:
    """
    Calculates CHW activity trends (e.g., visits, high-priority tasks) over a specified period.

    Args:
        chw_historical_health_df: DataFrame containing historical health records relevant to the CHW/team.
                                  Expected columns: 'encounter_date', 'patient_id', 
                                                  'ai_followup_priority_score', 'zone_id' (optional).
        trend_start_date: The start date for the trend analysis.
        trend_end_date: The end date for the trend analysis.
        zone_filter: Optional. If provided, filters data for a specific zone.
        time_period_agg: Aggregation period ('D' for daily, 'W-Mon' for weekly starting Monday, etc.).

    Returns:
        Dict[str, Optional[pd.DataFrame]]: A dictionary where keys are trend metric names
        (e.g., "daily_patient_visits_trend", "daily_high_priority_followups_trend")
        and values are Pandas DataFrames/Series containing the trend data (index=date, value=metric).
        Returns None for a metric if data is insufficient.
    """
    logger.info(f"Calculating CHW activity trends from {trend_start_date} to {trend_end_date}, zone: {zone_filter or 'All'}, period: {time_period_agg}")

    trends_output: Dict[str, Optional[pd.DataFrame]] = {
        "patient_visits_trend": None,
        "high_priority_followups_trend": None
        # Other trends can be added here, e.g., "avg_risk_score_trend_visited_patients"
    }

    if chw_historical_health_df is None or chw_historical_health_df.empty:
        logger.warning("No historical health data provided for CHW trend calculation.")
        return trends_output
    if trend_start_date > trend_end_date:
        logger.error("Trend period error: Start date is after end date.")
        return trends_output
        
    df_trends_base = chw_historical_health_df.copy()

    # Ensure 'encounter_date' is datetime type and filter by date range
    if 'encounter_date' not in df_trends_base.columns:
        logger.error("Missing 'encounter_date' column for trend analysis.")
        return trends_output
    
    df_trends_base['encounter_date'] = pd.to_datetime(df_trends_base['encounter_date'], errors='coerce')
    df_trends_base.dropna(subset=['encounter_date'], inplace=True)
    if df_trends_base.empty:
        logger.info("No valid encounter dates after cleaning for trend analysis.")
        return trends_output

    # Convert date objects to datetime64[ns] for comparison if necessary, or use .dt.date
    # Assuming trend_start_date and trend_end_date are datetime.date objects
    mask_date_range = (df_trends_base['encounter_date'].dt.normalize() >= pd.to_datetime(trend_start_date)) & \
                      (df_trends_base['encounter_date'].dt.normalize() <= pd.to_datetime(trend_end_date))
    df_period_filtered = df_trends_base[mask_date_range]

    # Apply zone filter if specified
    if zone_filter and 'zone_id' in df_period_filtered.columns:
        df_period_filtered = df_period_filtered[df_period_filtered['zone_id'] == zone_filter]

    if df_period_filtered.empty:
        logger.info(f"No CHW data found for the specified period/zone for trend analysis.")
        return trends_output

    # 1. Trend for Daily/Weekly Patient Visits (Unique Patients)
    if 'patient_id' in df_period_filtered.columns:
        # Using core_data_processing.get_trend_data utility
        visits_trend_series = get_trend_data(
            df=df_period_filtered,
            value_col='patient_id',
            date_col='encounter_date',
            period=time_period_agg,
            agg_func='nunique', # Count unique patients
            source_context="CHWTrends/Visits"
        )
        if visits_trend_series is not None and not visits_trend_series.empty:
            trends_output["patient_visits_trend"] = visits_trend_series.rename("unique_patients_visited").to_frame()


    # 2. Trend for High Priority Follow-ups (Unique Patients with high AI Follow-up Score)
    if 'ai_followup_priority_score' in df_period_filtered.columns and \
       'patient_id' in df_period_filtered.columns and \
       df_period_filtered['ai_followup_priority_score'].notna().any():
        
        # Filter for high priority encounters
        # Using FATIGUE_INDEX_HIGH_THRESHOLD as a proxy for high priority score as per earlier component logic
        high_prio_encounters_df = df_period_filtered[
            df_period_filtered['ai_followup_priority_score'] >= app_config.FATIGUE_INDEX_HIGH_THRESHOLD
        ]
        
        if not high_prio_encounters_df.empty:
            high_prio_trend_series = get_trend_data(
                df=high_prio_encounters_df,
                value_col='patient_id', # Count unique patients needing high prio follow-up
                date_col='encounter_date',
                period=time_period_agg,
                agg_func='nunique',
                source_context="CHWTrends/HighPrio"
            )
            if high_prio_trend_series is not None and not high_prio_trend_series.empty:
                trends_output["high_priority_followups_trend"] = high_prio_trend_series.rename("high_priority_followups_count").to_frame()
        else:
            logger.info("No high priority follow-up encounters found in the period for trend calculation.")
    else:
        logger.info("Data for AI Follow-up Priority Score trend not available or insufficient.")

    # Log completion and what trends were successfully calculated
    calculated_trends = [key for key, value in trends_output.items() if value is not None]
    if calculated_trends:
        logger.info(f"Successfully calculated CHW trends for: {', '.join(calculated_trends)}")
    else:
        logger.info("No CHW trends could be calculated with the provided data/filters.")
        
    return trends_output


# --- Example Usage (for testing or integration into a supervisor reporting script) ---
if __name__ == '__main__':
    logger.setLevel(logging.DEBUG) # Set to DEBUG to see more logs from get_trend_data
    logger.info("Running CHW Trends Display component simulation directly...")

    # Create sample historical data spanning a few weeks
    date_rng = pd.date_range(start='2023-09-15', end='2023-10-15', freq='D')
    num_records = len(date_rng) * 3 # Approx 3 records per day
    
    sample_historical_data = pd.DataFrame({
        'encounter_date': np.random.choice(date_rng, num_records),
        'patient_id': [f'P{i%20:03d}' for i in range(num_records)], # 20 unique patients repeating
        'zone_id': np.random.choice(['ZoneA', 'ZoneB', 'ZoneC'], num_records),
        'ai_followup_priority_score': np.random.randint(30, 100, num_records),
        'condition': np.random.choice(['TB', 'Malaria', 'Wellness', 'Pneumonia'], num_records)
    })
    sample_historical_data.sort_values('encounter_date', inplace=True)

    start_dt = pd.Timestamp('2023-10-01').date()
    end_dt = pd.Timestamp('2023-10-14').date()

    # Calculate Daily Trends for ZoneA
    daily_trends_zone_a = calculate_chw_activity_trends(
        chw_historical_health_df=sample_historical_data,
        trend_start_date=start_dt,
        trend_end_date=end_dt,
        zone_filter="ZoneA",
        time_period_agg='D'
    )
    print(f"\n--- Calculated Daily CHW Activity Trends for ZoneA (Period: {start_dt} to {end_dt}): ---")
    for trend_name, trend_df in daily_trends_zone_a.items():
        if trend_df is not None:
            print(f"\nTrend: {trend_name.replace('_', ' ').title()}")
            print(trend_df.head())
        else:
            print(f"\nTrend: {trend_name.replace('_', ' ').title()} - No data")

    # Calculate Weekly Trends for All Zones
    weekly_trends_all_zones = calculate_chw_activity_trends(
        chw_historical_health_df=sample_historical_data,
        trend_start_date=pd.Timestamp('2023-09-18').date(), # Different range for weekly
        trend_end_date=pd.Timestamp('2023-10-15').date(),
        zone_filter=None, # All zones
        time_period_agg='W-Mon' # Weekly, starting Monday
    )
    print(f"\n--- Calculated Weekly CHW Activity Trends (All Zones): ---")
    for trend_name, trend_df in weekly_trends_all_zones.items():
        if trend_df is not None:
            print(f"\nTrend: {trend_name.replace('_', ' ').title()}")
            print(trend_df)
        else:
            print(f"\nTrend: {trend_name.replace('_', ' ').title()} - No data")
