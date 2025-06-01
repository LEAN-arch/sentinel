# test/pages/chw_components/alerts_display.py
# Redesigned for "Sentinel Health Co-Pilot" - LMIC Edge-First System
# This module processes daily CHW data to generate structured alert information.
# This logic would be implemented natively on a PED for real-time alerts.
# In this Python context, it's used for simulation or generating data for supervisor reports (Tier 1/2).

import pandas as pd
import numpy as np
import logging
# Assuming app_config is in the PYTHONPATH or project root.
from config import app_config # Uses new, redesigned app_config
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

def generate_chw_patient_alerts_from_data(
    patient_alerts_tasks_df: pd.DataFrame, # This is the primary source, already containing pre-calculated AI scores & some alert flags
    chw_daily_encounter_df: pd.DataFrame,  # Contextual daily data for this CHW
    for_date: Any, # Typically datetime.date, for context in alerts
    chw_zone_context: str, # e.g., "Zone A" or "All Assigned Zones"
    max_alerts_to_return: int = 10 # Limit for reporting
) -> List[Dict[str, Any]]:
    """
    Processes CHW daily data to generate a list of structured patient alerts.
    This list can be used for reports or to simulate alerts.
    
    Args:
        patient_alerts_tasks_df: DataFrame containing patient data potentially pre-flagged
                                 with AI scores or basic alert triggers. Expected columns:
                                 patient_id, ai_risk_score, ai_followup_priority_score,
                                 min_spo2_pct, vital_signs_temperature_celsius (or max_skin_temp_celsius),
                                 condition, fall_detected_today, alert_reason (optional existing).
        chw_daily_encounter_df: Broader daily encounters for this CHW for context (e.g., if alerts_tasks_df is a subset).
        for_date: The date for which these alerts are relevant.
        chw_zone_context: The zone(s) this CHW is covering.
        max_alerts_to_return: Max number of top alerts to return.

    Returns:
        A list of dictionaries, where each dictionary represents an actionable alert.
        Example alert dict:
        {
            "patient_id": "P001",
            "alert_level": "CRITICAL" | "WARNING" | "INFO",
            "primary_reason": "Critical Low SpO2",
            "brief_details": "SpO2: 88%",
            "suggested_action_code": "ACT_CHECK_VITALS_URGENT", // Maps to pictogram/JIT guidance on PED
            "raw_priority_score": 95.5, // Numerical score for sorting
            "context_info": "Condition: Pneumonia | Zone: Zone A"
        }
    """
    logger.info(f"Generating CHW patient alerts for date: {for_date}, zone: {chw_zone_context}")
    
    if patient_alerts_tasks_df is None or patient_alerts_tasks_df.empty:
        logger.info("No patient data provided for alert generation.")
        return []

    processed_alerts: List[Dict[str, Any]] = []
    
    # Use a working copy for modifications
    df_for_alerting = patient_alerts_tasks_df.copy()

    # Determine the temperature column to use for consistency
    temp_col_to_use_alerts = None
    # Check in the broader daily encounter context first, then fallback to alert-specific df
    context_df_for_temp_check = chw_daily_encounter_df if chw_daily_encounter_df is not None and not chw_daily_encounter_df.empty else df_for_alerting
    
    if 'vital_signs_temperature_celsius' in context_df_for_temp_check.columns and context_df_for_temp_check['vital_signs_temperature_celsius'].notna().any():
        temp_col_to_use_alerts = 'vital_signs_temperature_celsius'
    elif 'max_skin_temp_celsius' in context_df_for_temp_check.columns and context_df_for_temp_check['max_skin_temp_celsius'].notna().any():
        temp_col_to_use_alerts = 'max_skin_temp_celsius'

    # --- Alerting Rules (LMIC context: prioritize life-threatening, easily identifiable issues) ---
    # This simulates the logic that would be on the PED.
    # Each rule should assign an `alert_level`, `primary_reason`, `brief_details`, `suggested_action_code`, and `raw_priority_score`.

    for _, record in df_for_alerting.iterrows():
        current_patient_alerts: List[Dict[str, Any]] = []
        patient_id_str = str(record.get('patient_id', 'UnknownPatient'))
        condition_str = str(record.get('condition', 'N/A'))
        zone_str = str(record.get('zone_id', chw_zone_context)) # Use specific zone if available
        context_str = f"Cond: {condition_str} | Zone: {zone_str}"

        # Rule 1: Critical Low SpO2
        spo2_val = record.get('min_spo2_pct')
        if pd.notna(spo2_val) and spo2_val < app_config.ALERT_SPO2_CRITICAL_LOW_PCT:
            current_patient_alerts.append({
                "alert_level": "CRITICAL", "primary_reason": "Critical Low SpO2",
                "brief_details": f"SpO2: {spo2_val:.0f}%",
                "suggested_action_code": "ACTION_SPO2_MANAGE_URGENT", "raw_priority_score": 98,
                "patient_id": patient_id_str, "context_info": context_str
            })
        elif pd.notna(spo2_val) and spo2_val < app_config.ALERT_SPO2_WARNING_LOW_PCT:
             current_patient_alerts.append({
                "alert_level": "WARNING", "primary_reason": "Low SpO2",
                "brief_details": f"SpO2: {spo2_val:.0f}%",
                "suggested_action_code": "ACTION_SPO2_RECHECK_MONITOR", "raw_priority_score": 75,
                "patient_id": patient_id_str, "context_info": context_str
            })

        # Rule 2: High Fever
        if temp_col_to_use_alerts and pd.notna(record.get(temp_col_to_use_alerts)):
            temp_val_alert = record.get(temp_col_to_use_alerts)
            if temp_val_alert >= app_config.ALERT_BODY_TEMP_HIGH_FEVER_C:
                current_patient_alerts.append({
                    "alert_level": "CRITICAL", "primary_reason": "High Fever",
                    "brief_details": f"Temp: {temp_val_alert:.1f}°C",
                    "suggested_action_code": "ACTION_FEVER_MANAGE_URGENT", "raw_priority_score": 95,
                    "patient_id": patient_id_str, "context_info": context_str
                })
            elif temp_val_alert >= app_config.ALERT_BODY_TEMP_FEVER_C:
                current_patient_alerts.append({
                    "alert_level": "WARNING", "primary_reason": "Fever Present",
                    "brief_details": f"Temp: {temp_val_alert:.1f}°C",
                    "suggested_action_code": "ACTION_FEVER_MONITOR_SUPPORT", "raw_priority_score": 70,
                    "patient_id": patient_id_str, "context_info": context_str
                })

        # Rule 3: Fall Detected (if available in data)
        if pd.notna(record.get('fall_detected_today')) and record['fall_detected_today'] > 0:
            current_patient_alerts.append({
                "alert_level": "CRITICAL", "primary_reason": "Fall Detected",
                "brief_details": f"Falls: {int(record['fall_detected_today'])} today",
                "suggested_action_code": "ACTION_FALL_ASSESS_URGENT", "raw_priority_score": 92,
                "patient_id": patient_id_str, "context_info": context_str
            })

        # Rule 4: High AI Follow-up Priority Score
        ai_prio_score = record.get('ai_followup_priority_score')
        if pd.notna(ai_prio_score) and ai_prio_score >= app_config.FATIGUE_INDEX_HIGH_THRESHOLD: # Using FATIGUE_INDEX as example proxy
             current_patient_alerts.append({
                "alert_level": "WARNING", # AI score is advisory, vitals are more direct critical
                "primary_reason": "High AI Follow-up Priority",
                "brief_details": f"AI Prio Score: {ai_prio_score:.0f}",
                "suggested_action_code": "ACTION_REVIEW_CASE_PRIORITY", "raw_priority_score": ai_prio_score,
                "patient_id": patient_id_str, "context_info": context_str
            })
        
        # Rule 5: High AI Risk Score (if not covered by AI Prio)
        ai_risk_s = record.get('ai_risk_score')
        if pd.notna(ai_risk_s) and ai_risk_s >= app_config.RISK_SCORE_HIGH_THRESHOLD and \
           not (pd.notna(ai_prio_score) and ai_prio_score >= app_config.FATIGUE_INDEX_HIGH_THRESHOLD):
            current_patient_alerts.append({
                "alert_level": "INFO", # Usually leads to closer monitoring rather than immediate crisis action if vitals okay
                "primary_reason": "High AI Risk Score",
                "brief_details": f"AI Risk Score: {ai_risk_s:.0f}",
                "suggested_action_code": "ACTION_MONITOR_CLOSELY_RISK", "raw_priority_score": ai_risk_s,
                "patient_id": patient_id_str, "context_info": context_str
            })
            
        # Add other LMIC-relevant rules, e.g.,
        # - Symptoms indicative of dehydration in high heat index + vulnerable patient (child/elderly)
        # - Danger signs for specific key conditions (e.g., TB patient coughing blood)
        # - Severe malnutrition signs

        # Consolidate alerts for this patient: For now, take the one with the highest raw_priority_score.
        # A real PED might show multiple indicators.
        if current_patient_alerts:
            top_alert_for_patient = max(current_patient_alerts, key=lambda x: x['raw_priority_score'])
            processed_alerts.append(top_alert_for_patient)

    # De-duplicate alerts (e.g., if multiple records for same patient show same top alert criteria)
    # For simplicity in this simulation, assume one alert per patient_id from the top scored ones if duplicates exist
    if processed_alerts:
        final_alerts_df = pd.DataFrame(processed_alerts)
        final_alerts_df.sort_values(by="raw_priority_score", ascending=False, inplace=True)
        # Ensure unique patient alert, taking the highest priority one if multiple for same patient
        final_alerts_df.drop_duplicates(subset=["patient_id"], keep="first", inplace=True)
        return final_alerts_df.head(max_alerts_to_return).to_dict(orient='records')
        
    return []


# --- Example Usage (for testing or if this module is run as part of a report script) ---
if __name__ == '__main__':
    # This section would simulate creating sample data and calling the function
    # to demonstrate its output.
    logger.setLevel(logging.DEBUG)
    logger.info("Running CHW Alerts component simulation directly...")

    sample_patients_data = pd.DataFrame({
        'patient_id': ['P001', 'P002', 'P003', 'P004'],
        'encounter_date': pd.to_datetime(['2023-10-01']*4),
        'condition': ['Pneumonia', 'Malaria', 'Healthy', 'TB'],
        'zone_id': ['ZoneA', 'ZoneA', 'ZoneB', 'ZoneA'],
        'age': [67, 5, 30, 45], 'gender': ['Male', 'Female', 'Male', 'Female'],
        'ai_risk_score': [88, 70, 30, 78],
        'ai_followup_priority_score': [90, 65, 20, 82],
        'min_spo2_pct': [88, 92, 99, 94],
        'vital_signs_temperature_celsius': [39.8, 38.5, 37.0, 37.8],
        'max_skin_temp_celsius': [39.6, 38.2, 36.8, 37.5],
        'fall_detected_today': [0, 1, 0, 0],
        'referral_status': ['N/A','Pending','N/A','Pending Urgent'],
    })
    sample_daily_encounters = sample_patients_data # For this test, they are the same

    alerts_output = generate_chw_patient_alerts_from_data(
        patient_alerts_tasks_df=sample_patients_data,
        chw_daily_encounter_df=sample_daily_encounters,
        for_date=pd.Timestamp('2023-10-01').date(),
        chw_zone_context="ZoneA Focus"
    )

    if alerts_output:
        print("\n--- Generated CHW Patient Alerts: ---")
        for alert_item in alerts_output:
            print(f"  Patient: {alert_item['patient_id']}, Level: {alert_item['alert_level']}, Reason: {alert_item['primary_reason']}")
            print(f"    Details: {alert_item['brief_details']}, Action Code: {alert_item['suggested_action_code']}")
            print(f"    Priority: {alert_item['raw_priority_score']:.1f}, Context: {alert_item['context_info']}\n")
    else:
        print("No alerts generated from sample data.")
