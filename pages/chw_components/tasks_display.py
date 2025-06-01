# test/pages/chw_components/tasks_display.py
# Redesigned for "Sentinel Health Co-Pilot" - LMIC Edge-First System
# This module processes CHW data (especially alerts and AI scores) to generate
# a prioritized list of tasks for the CHW.
# This logic primarily simulates what a PED would do to manage and display tasks.
# The output is a structured list of task objects, suitable for:
#   1. Informing the native task list UI on a PED.
#   2. Generating task summary reports for supervisors (Tier 1 Hub / Tier 2 Facility).
#   3. Simulation and testing of task generation and prioritization.

import pandas as pd
import numpy as np
import logging
# Assuming app_config is in the PYTHONPATH or project root.
from config import app_config # Uses new, redesigned app_config
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

def generate_chw_prioritized_tasks(
    patient_alerts_tasks_df: pd.DataFrame, # Source data with AI scores, alert reasons etc.
    chw_daily_encounter_df: Optional[pd.DataFrame], # Broader daily data for context
    for_date: Any, # datetime.date, for context
    chw_zone_context: str, # e.g., "Zone A"
    max_tasks_to_return: int = 20 # Limit for reporting purposes
) -> List[Dict[str, Any]]:
    """
    Generates a prioritized list of CHW tasks from input data.

    Args:
        patient_alerts_tasks_df: DataFrame containing patient data, AI scores,
                                 and potential alert reasons. Expected columns:
                                 patient_id, encounter_date, zone_id, condition, age,
                                 ai_risk_score, ai_followup_priority_score,
                                 alert_reason (from alert generation), referral_status,
                                 min_spo2_pct, fall_detected_today, vital_signs_temperature_celsius (or max_skin_temp_celsius).
        chw_daily_encounter_df: Optional DataFrame of all CHW encounters for the day for richer context.
        for_date: The date for which these tasks are relevant.
        chw_zone_context: The zone context for these tasks.
        max_tasks_to_return: Maximum number of tasks to return.

    Returns:
        List[Dict[str, Any]]: A list of task dictionaries, sorted by priority.
        Example task dict:
        {
            "task_id": "TASK_P001_20231001_VITALCHECK", // Unique ID
            "patient_id": "P001",
            "task_type_code": "TASK_VISIT_FOLLOWUP", // Maps to pictogram
            "task_description": "Follow-up: Critical Low SpO2 Alert",
            "priority_score": 95.0,
            "due_date": "2023-10-01", // Or relative "Today", "Tomorrow"
            "status": "PENDING", // PENDING, IN_PROGRESS, COMPLETED, ESCALATED
            "key_patient_context": "Cond: Pneumonia, SpO2: 88%, Age: 67",
            "suggested_next_action_code": "ACTION_CHECK_VITALS_URGENT" // From alert
        }
    """
    logger.info(f"Generating CHW prioritized tasks for date: {for_date}, zone: {chw_zone_context}")

    if patient_alerts_tasks_df is None or patient_alerts_tasks_df.empty:
        logger.info("No input data provided for task generation.")
        return []

    tasks_list: List[Dict[str, Any]] = []
    df_tasks_input = patient_alerts_tasks_df.copy()

    # Use preferred temperature column name (from alerts_display logic, adapted)
    temp_col_for_tasks_context = None
    context_df_for_temp_tasks = chw_daily_encounter_df if chw_daily_encounter_df is not None and not chw_daily_encounter_df.empty else df_tasks_input
    if 'vital_signs_temperature_celsius' in context_df_for_temp_tasks.columns and context_df_for_temp_tasks['vital_signs_temperature_celsius'].notna().any():
        temp_col_for_tasks_context = 'vital_signs_temperature_celsius'
    elif 'max_skin_temp_celsius' in context_df_for_temp_tasks.columns and context_df_for_temp_tasks['max_skin_temp_celsius'].notna().any():
        temp_col_for_tasks_context = 'max_skin_temp_celsius'

    # Sort by AI Follow-up Priority score primarily, then AI Risk score as secondary
    sort_by_cols_tasks = []
    if 'ai_followup_priority_score' in df_tasks_input.columns and df_tasks_input['ai_followup_priority_score'].notna().any():
        sort_by_cols_tasks.append('ai_followup_priority_score')
    if 'ai_risk_score' in df_tasks_input.columns and df_tasks_input['ai_risk_score'].notna().any():
         # only add if not already primary, or if primary doesn't exist
        if not sort_by_cols_tasks or 'ai_followup_priority_score' not in sort_by_cols_tasks :
            sort_by_cols_tasks.append('ai_risk_score')
        elif 'ai_followup_priority_score' in sort_by_cols_tasks and sort_by_cols_tasks[0] != 'ai_risk_score':
             sort_by_cols_tasks.append('ai_risk_score') # Add as secondary

    if sort_by_cols_tasks:
        df_tasks_sorted = df_tasks_input.sort_values(by=sort_by_cols_tasks, ascending=[False]*len(sort_by_cols_tasks))
    else:
        df_tasks_sorted = df_tasks_input # Process as is if no sortable priority/risk score
        logger.warning("No AI priority/risk scores found for task sorting.")
    
    for index, row in df_tasks_sorted.iterrows():
        patient_id = str(row.get('patient_id', 'UnknownPatient'))
        encounter_dt_str = str(pd.to_datetime(row.get('encounter_date', for_date)).date()) # Ensure a date string
        
        # Define task based on alert reason or high priority scores
        # This logic needs to be more sophisticated based on specific alert types
        task_description = "Review Patient Case" # Default
        task_type_code = "TASK_VISIT_ASSESS"   # Default assessment task
        
        alert_reason_val = str(row.get('alert_reason', '')).lower() # From previous alert generation step
        priority_score = float(row.get('ai_followup_priority_score', row.get('ai_risk_score', 50.0))) # Prioritize follow-up score

        if "critical low spo2" in alert_reason_val or (pd.notna(row.get('min_spo2_pct')) and row.get('min_spo2_pct') < app_config.ALERT_SPO2_CRITICAL_LOW_PCT):
            task_description = "Urgent: Assess Critical Low SpO2"
            task_type_code = "TASK_VISIT_VITALS_URGENT"
            priority_score = max(priority_score, 95) # Boost score
        elif "high fever" in alert_reason_val or (temp_col_for_tasks_context and pd.notna(row.get(temp_col_for_tasks_context)) and row.get(temp_col_for_tasks_context) >= app_config.ALERT_BODY_TEMP_HIGH_FEVER_C):
            task_description = "Urgent: Assess High Fever"
            task_type_code = "TASK_VISIT_VITALS_URGENT"
            priority_score = max(priority_score, 92)
        elif "fall detected" in alert_reason_val or (pd.notna(row.get('fall_detected_today')) and row['fall_detected_today'] > 0):
            task_description = "Assess Recent Fall"
            task_type_code = "TASK_VISIT_FALL_ASSESS"
            priority_score = max(priority_score, 90)
        elif "high ai follow-up priority" in alert_reason_val: # Specific AI Followup
            task_description = "Follow-up: High AI Priority Patient"
            task_type_code = "TASK_VISIT_FOLLOWUP_AI"
            # priority_score already set from ai_followup_priority_score
        elif "high ai risk score" in alert_reason_val or priority_score > app_config.RISK_SCORE_MODERATE_THRESHOLD: # General high risk
            task_description = "Follow-up: High AI Risk Patient"
            task_type_code = "TASK_VISIT_FOLLOWUP_RISK"
            priority_score = max(priority_score, row.get('ai_risk_score',50)) # Ensure risk score contributes
        elif "pending critical referral" in alert_reason_val:
             task_description = f"Follow-up: Pending Critical Referral for {row.get('condition', 'N/A')}"
             task_type_code = "TASK_VISIT_REFERRAL_CHECK"
             priority_score = max(priority_score, 85)
        elif "poor medication adherence" in alert_reason_val:
            task_description = "Counseling: Poor Medication Adherence"
            task_type_code = "TASK_VISIT_ADHERENCE"
            priority_score = max(priority_score, 70)
        
        # Add more task definitions for:
        # - Routine wellness checks (lower priority)
        # - Scheduled medication deliveries
        # - Health education sessions (can be group tasks if PED supports this concept)
        # - Specific disease management follow-ups (e.g., TB DOTS observation) -> These may come from pre-defined schedule not alerts.

        key_patient_context_parts = []
        if pd.notna(row.get('condition')) and str(row.get('condition')).lower() != 'unknown': key_patient_context_parts.append(f"Cond: {row.get('condition')}")
        if pd.notna(row.get('age')): key_patient_context_parts.append(f"Age: {row.get('age'):.0f}")
        if pd.notna(row.get('min_spo2_pct')): key_patient_context_parts.append(f"SpO2: {row.get('min_spo2_pct'):.0f}%")
        if temp_col_for_tasks_context and pd.notna(row.get(temp_col_for_tasks_context)): key_patient_context_parts.append(f"Temp: {row.get(temp_col_for_tasks_context):.1f}°C")

        task_id_str = f"TASK_{patient_id}_{encounter_dt_str}_{task_type_code.split('_')[-1]}"
        
        task_obj = {
            "task_id": task_id_str,
            "patient_id": patient_id,
            "zone_id": str(row.get('zone_id', chw_zone_context)),
            "task_type_code": task_type_code,
            "task_description": task_description,
            "priority_score": round(priority_score, 1),
            "due_date": encounter_dt_str, # For alerts based on today's encounters, due is today
            "status": "PENDING",
            "key_patient_context": " | ".join(key_patient_context_parts) if key_patient_context_parts else "N/A",
            "alert_reason_source": str(row.get('alert_reason', 'Routine/Preventive')) # What triggered this task generation
        }
        tasks_list.append(task_obj)

    # Sort final list by priority score and return top N
    if tasks_list:
        # De-duplicate tasks for the same patient based on task_description (or more robustly, by core alert type)
        # For simplicity, allow multiple task types for same patient if their reasons differ significantly leading to different descriptions.
        # A more complex deduplication would be needed for a real system.
        # Here, simply sort by priority_score for now.
        sorted_tasks = sorted(tasks_list, key=lambda x: x['priority_score'], reverse=True)
        # Further de-duplication might be needed if a single underlying event triggers multiple "reasons"
        # This could involve checking if patient_id + a "root_cause_code" is unique.
        # For this simulation, let's take unique tasks by 'patient_id' and 'task_description' pair (highest priority one)
        
        unique_task_tracker = {} # {(patient_id, task_description): highest_priority_score_task_obj}
        final_unique_tasks = []
        for task in sorted_tasks:
            key = (task['patient_id'], task['task_description'])
            if key not in unique_task_tracker:
                 unique_task_tracker[key] = task
                 final_unique_tasks.append(task)
            # else: task already captured or lower priority one for same (patient, desc) pair

        return final_unique_tasks[:max_tasks_to_return]

    return []


# --- Example Usage (for testing or integration into a reporting script) ---
if __name__ == '__main__':
    logger.setLevel(logging.DEBUG)
    logger.info("Running CHW Tasks Display component simulation directly...")

    # Simulate patient_alerts_tasks_df (this would typically be output from alert generation)
    sample_alerts_tasks_input = pd.DataFrame({
        'patient_id': ['P001', 'P002', 'P003', 'P004', 'P005', 'P001'],
        'encounter_date': pd.to_datetime(['2023-10-05']*6),
        'zone_id': ['ZoneA', 'ZoneA', 'ZoneB', 'ZoneA', 'ZoneC', 'ZoneA'],
        'condition': ['Pneumonia', 'Malaria', 'Wellness', 'TB Exposure', 'Dehydration Suspected', 'Pneumonia'],
        'age': [68, 6, 35, 50, 2, 68],
        'ai_risk_score': [92, 75, 30, 80, 88, 92],
        'ai_followup_priority_score': [95, 80, 25, 88, 90, 95], # Higher score for P001, P003, P004, P005
        'alert_reason': [ # Simulate output from alert generation step
            "Critical Low SpO2; High AI Follow-up Priority", "High AI Follow-up Priority; Fever Present", 
            "Routine Check", "High AI Follow-up Priority; TB Contact", 
            "High AI Follow-up Priority; Dehydration Signs", "Critical Low SpO2"
        ],
        'min_spo2_pct': [87, 93, 98, 96, 91, 87],
        'vital_signs_temperature_celsius': [37.5, 38.9, 37.1, 37.0, 39.0, 37.5],
        'fall_detected_today': [0,0,0,0,1,0] # P005 fall
    })
    current_date = pd.Timestamp('2023-10-05').date()

    prioritized_tasks = generate_chw_prioritized_tasks(
        patient_alerts_tasks_df=sample_alerts_tasks_input,
        chw_daily_encounter_df=sample_alerts_tasks_input, # Using same for this test
        for_date=current_date,
        chw_zone_context="Field Operations Area 1",
        max_tasks_to_return=5
    )

    print(f"\n--- Generated CHW Prioritized Tasks (Top {len(prioritized_tasks)}): ---")
    if prioritized_tasks:
        for i, task_item in enumerate(prioritized_tasks):
            print(f"  Task {i+1}: ID: {task_item['task_id']}")
            print(f"    Patient: {task_item['patient_id']} (Zone: {task_item['zone_id']})")
            print(f"    Type: {task_item['task_type_code']}")
            print(f"    Description: {task_item['task_description']}")
            print(f"    Priority Score: {task_item['priority_score']}")
            print(f"    Due: {task_item['due_date']}, Status: {task_item['status']}")
            print(f"    Context: {task_item['key_patient_context']}")
            print(f"    Source Alert: {task_item['alert_reason_source']}\n")
    else:
        print("No tasks generated from sample data.")
