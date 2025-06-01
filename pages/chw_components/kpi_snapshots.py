# test/pages/chw_components/kpi_snapshots.py
# Redesigned for "Sentinel Health Co-Pilot" - LMIC Edge-First System
# This module calculates key summary metrics for a CHW's daily activity.
# This logic is primarily for:
#   1. Simulating metrics that a PED might track internally.
#   2. Generating a structured summary that could be synced to a Supervisor Hub (Tier 1)
#      or Facility Node (Tier 2) for reporting and performance monitoring.
# Direct UI rendering of "KPI cards" on a PED is replaced by native UI elements
# showing glanceable summaries or triggering contextual alerts.

import pandas as pd
import numpy as np
import logging
# Assuming app_config is in the PYTHONPATH or project root.
from config import app_config # Uses new, redesigned app_config
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

def calculate_chw_daily_summary_metrics(
    chw_daily_kpi_input_data: Dict[str, Any], # Pre-calculated aggregates (e.g., from core_data_processing.get_chw_summary)
    chw_daily_encounter_df: Optional[pd.DataFrame], # Raw daily encounter DataFrame for more detailed calculations if needed
    for_date: Any # datetime.date, for context
) -> Dict[str, Any]:
    """
    Calculates and returns a dictionary of key CHW daily summary metrics.

    Args:
        chw_daily_kpi_input_data (Dict[str, Any]): A dictionary possibly containing
            pre-aggregated values (like those from `get_chw_summary` in core_data_processing).
            Keys expected: 'visits_today', 'avg_patient_risk_visited_today',
                           'patients_fever_visited_today', 'patients_low_spo2_visited_today',
                           'avg_patient_steps_visited_today', 'patients_fall_detected_today'.
        chw_daily_encounter_df (Optional[pd.DataFrame]): The raw DataFrame of CHW encounters for the day.
            Used for metrics not easily pre-aggregated, like 'high_ai_prio_followups_count'.
            Expected columns: 'patient_id', 'ai_followup_priority_score', 'ai_risk_score',
                              'min_spo2_pct', 'vital_signs_temperature_celsius' (or 'max_skin_temp_celsius'),
                              'avg_daily_steps', 'fall_detected_today'.
        for_date (Any): The date these metrics apply to.

    Returns:
        Dict[str, Any]: A dictionary of calculated metrics, structured for reporting or sync.
            Example: {
                "date": "2023-10-01",
                "visits_count": 5,
                "high_ai_prio_followups_count": 2,
                "avg_risk_of_visited_patients": 65.2,
                "fever_cases_identified_count": 1,
                "low_spo2_cases_identified_count": 0,
                "avg_steps_of_visited_patients": 4500.0,
                "fall_events_among_visited_count": 0,
                "worker_self_fatigue_level": "MODERATE" // Example placeholder for worker's own status
            }
    """
    logger.info(f"Calculating CHW daily summary metrics for date: {for_date}")

    metrics_summary = {
        "date": str(for_date),
        "visits_count": 0,
        "high_ai_prio_followups_count": 0,
        "avg_risk_of_visited_patients": np.nan,
        "fever_cases_identified_count": 0, # Patients met by CHW with fever
        "critical_spo2_cases_identified_count": 0, # Patients met with SpO2 < CRITICAL
        "avg_steps_of_visited_patients": np.nan,
        "fall_events_among_visited_count": 0,
        "worker_self_fatigue_level_code": "NOT_ASSESSED" # Placeholder, would come from worker's PED
    }

    # --- Populate from chw_daily_kpi_input_data (pre-aggregates) ---
    if chw_daily_kpi_input_data:
        metrics_summary["visits_count"] = chw_daily_kpi_input_data.get('visits_today', 0)
        metrics_summary["avg_risk_of_visited_patients"] = chw_daily_kpi_input_data.get('avg_patient_risk_visited_today', np.nan)
        # These are counts of PATIENTS, not just encounters with these flags.
        metrics_summary["fever_cases_identified_count"] = chw_daily_kpi_input_data.get('patients_fever_visited_today', 0)
        # Using the "critical" SpO2 threshold for this summary metric for actionability
        # Note: Original kpi_snapshots used 'patients_low_spo2_visited_today' based on SPO2_LOW_THRESHOLD_PCT.
        # Shifting to ALERT_SPO2_CRITICAL_LOW_PCT for a more "snapshot critical" metric.
        # If low but not critical needed, it can be added as another metric.
        metrics_summary["critical_spo2_cases_identified_count"] = chw_daily_kpi_input_data.get('patients_critical_spo2_visited_today', 0) # Assuming this key would be calculated by an updated get_chw_summary
        metrics_summary["avg_steps_of_visited_patients"] = chw_daily_kpi_input_data.get('avg_patient_steps_visited_today', np.nan)
        metrics_summary["fall_events_among_visited_count"] = chw_daily_kpi_input_data.get('patients_fall_detected_today', 0)

    # --- Calculate or refine metrics using chw_daily_encounter_df if available ---
    if chw_daily_encounter_df is not None and not chw_daily_encounter_df.empty:
        df_enc = chw_daily_encounter_df.copy()

        # Unique visits today (can override if pre-agg was just encounter count)
        if 'patient_id' in df_enc:
            metrics_summary["visits_count"] = df_enc['patient_id'].nunique()

        # High AI Priority Follow-ups Count
        if 'ai_followup_priority_score' in df_enc and df_enc['ai_followup_priority_score'].notna().any():
            # Using the high fatigue threshold as an example from new config (FATIGUE_INDEX_HIGH_THRESHOLD often signals general high concern)
            # Or, a dedicated high priority score threshold if app_config defines one e.g. app_config.AI_FOLLOWUP_HIGH_PRIO_THRESHOLD
            metrics_summary["high_ai_prio_followups_count"] = df_enc[
                df_enc['ai_followup_priority_score'] >= app_config.FATIGUE_INDEX_HIGH_THRESHOLD # Using this as a proxy
            ]['patient_id'].nunique()

        # Average risk of visited patients (recalculate if full df is richer)
        if 'ai_risk_score' in df_enc and df_enc['ai_risk_score'].notna().any():
             unique_patient_risk_scores = df_enc.drop_duplicates(subset=['patient_id'])['ai_risk_score']
             if unique_patient_risk_scores.notna().any():
                 metrics_summary["avg_risk_of_visited_patients"] = unique_patient_risk_scores.mean()


        # Fever cases among visited patients (recalculate based on preferred temp col & CRITICAL threshold for impact)
        temp_col_calc = next((tc for tc in ['vital_signs_temperature_celsius', 'max_skin_temp_celsius'] if tc in df_enc and df_enc[tc].notna().any()), None)
        if temp_col_calc:
            metrics_summary["fever_cases_identified_count"] = df_enc[
                df_enc[temp_col_calc] >= app_config.ALERT_BODY_TEMP_FEVER_C # Using general fever for "cases identified"
            ]['patient_id'].nunique()

        # Critically Low SpO2 cases among visited patients
        if 'min_spo2_pct' in df_enc and df_enc['min_spo2_pct'].notna().any():
            metrics_summary["critical_spo2_cases_identified_count"] = df_enc[
                df_enc['min_spo2_pct'] < app_config.ALERT_SPO2_CRITICAL_LOW_PCT
            ]['patient_id'].nunique()
        
        # Avg steps for unique patients visited today
        if 'avg_daily_steps' in df_enc and df_enc['avg_daily_steps'].notna().any():
            unique_patient_steps = df_enc.drop_duplicates(subset=['patient_id'])['avg_daily_steps']
            if unique_patient_steps.notna().any():
                metrics_summary["avg_steps_of_visited_patients"] = unique_patient_steps.mean()

        # Fall events for unique patients visited today
        if 'fall_detected_today' in df_enc and df_enc['fall_detected_today'].notna().any():
             metrics_summary["fall_events_among_visited_count"] = df_enc[
                df_enc['fall_detected_today'] > 0
            ]['patient_id'].nunique() # Counts patients who had at least one fall

    # Round averages for cleaner reporting
    if pd.notna(metrics_summary["avg_risk_of_visited_patients"]):
        metrics_summary["avg_risk_of_visited_patients"] = round(metrics_summary["avg_risk_of_visited_patients"], 1)
    if pd.notna(metrics_summary["avg_steps_of_visited_patients"]):
        metrics_summary["avg_steps_of_visited_patients"] = round(metrics_summary["avg_steps_of_visited_patients"])

    # Placeholder for worker's own fatigue level (this would be derived on worker's PED)
    # Example logic if worker's own 'fatigue_index_today' was part of chw_daily_kpi_input_data
    worker_fatigue_score = chw_daily_kpi_input_data.get('worker_self_fatigue_index_today', np.nan)
    if pd.notna(worker_fatigue_score):
        if worker_fatigue_score >= app_config.FATIGUE_INDEX_HIGH_THRESHOLD:
            metrics_summary["worker_self_fatigue_level_code"] = "HIGH"
        elif worker_fatigue_score >= app_config.FATIGUE_INDEX_MODERATE_THRESHOLD:
            metrics_summary["worker_self_fatigue_level_code"] = "MODERATE"
        else:
            metrics_summary["worker_self_fatigue_level_code"] = "LOW"
    
    logger.info(f"CHW daily summary metrics calculated: {metrics_summary}")
    return metrics_summary


# --- Example Usage (for testing or integration into a reporting script) ---
if __name__ == '__main__':
    logger.setLevel(logging.DEBUG)
    logger.info("Running CHW KPI Snapshots component simulation directly...")

    # Simulate input from core_data_processing.get_chw_summary
    sample_kpi_input = {
        'visits_today': 8,
        'avg_patient_risk_visited_today': 55.3,
        'patients_fever_visited_today': 2,
        'patients_critical_spo2_visited_today': 1, # Assume this would be based on CRITICAL_LOW threshold
        'avg_patient_steps_visited_today': 5230.0,
        'patients_fall_detected_today': 0,
        'worker_self_fatigue_index_today': 65 # Example: moderate fatigue
    }

    # Simulate a more detailed daily encounter DataFrame (optional input)
    sample_encounter_df = pd.DataFrame({
        'patient_id': ['P001', 'P002', 'P001', 'P003', 'P004', 'P005', 'P006', 'P007', 'P008'],
        'ai_followup_priority_score': [85, 50, 85, 92, 40, 60, 70, 30, 75], # Two > FATIGUE_INDEX_HIGH_THRESHOLD (80) -> P001, P003
        'ai_risk_score': [70, 60, 70, 85, 30, 50, 65, 20, 60],
        'min_spo2_pct': [96, 88, 96, 90, 99, 97, 95, 98, 93], # P002 < CRITICAL_LOW (90)
        'vital_signs_temperature_celsius': [37.0, 39.6, 37.0, 37.2, 36.5, 38.1, 38.8, 36.0, 37.1], # P002, P006, P007 have fever
        'avg_daily_steps': [4000, 6000, 4000, 5500, 7000, 3000, 8000, 9000, 4500],
        'fall_detected_today': [0, 0, 0, 0, 0, 1, 0, 0, 0], # P005 had a fall
    })

    calculated_metrics = calculate_chw_daily_summary_metrics(
        chw_daily_kpi_input_data=sample_kpi_input,
        chw_daily_encounter_df=sample_encounter_df,
        for_date=pd.Timestamp('2023-10-02').date()
    )

    print("\n--- Calculated CHW Daily Summary Metrics: ---")
    for key, value in calculated_metrics.items():
        print(f"  {key.replace('_', ' ').title()}: {value}")

    # Test with only pre-aggregated data
    calculated_metrics_pre_agg_only = calculate_chw_daily_summary_metrics(
         chw_daily_kpi_input_data=sample_kpi_input,
         chw_daily_encounter_df=None, # Simulate no raw encounter DF
         for_date=pd.Timestamp('2023-10-03').date()
    )
    print("\n--- Calculated CHW Metrics (Pre-Aggregated Only): ---")
    for key, value in calculated_metrics_pre_agg_only.items():
        print(f"  {key.replace('_', ' ').title()}: {value}")
