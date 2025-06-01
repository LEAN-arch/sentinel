# test/pages/chw_components/epi_watch.py
# Redesigned for "Sentinel Health Co-Pilot" - LMIC Edge-First System
# This module focuses on processing CHW daily encounter data to extract
# potential epidemiological signals and key task-related counts.
# This logic is primarily for:
#   1. Simulation of local epi-signal detection that might occur (in simpler form) on a PED or Group Hub.
#   2. Generating structured data for Supervisor Hub (Tier 1) or Facility Node (Tier 2)
#      reports and dashboards to monitor local health trends and CHW activities.

import pandas as pd
import numpy as np
import logging
# Assuming app_config is in the PYTHONPATH or project root.
from config import app_config # Uses new, redesigned app_config
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

def extract_chw_local_epi_signals(
    chw_daily_encounter_df: pd.DataFrame, # CHW's encounters for the day
    pre_calculated_chw_kpis: Optional[Dict[str, Any]] = None, # Optional dict with values like 'tb_contacts_to_trace_today'
    for_date: Any, # datetime.date, for context
    chw_zone_context: str # e.g., "Zone A" or "All Assigned Zones"
) -> Dict[str, Any]:
    """
    Extracts epidemiological signals and task-related counts from CHW's daily data.

    Args:
        chw_daily_encounter_df: DataFrame of the CHW's encounters for the specified day.
                               Expected cols: 'patient_id', 'condition', 'patient_reported_symptoms',
                                              'ai_risk_score', 'age', 'referral_reason', 'referral_status'.
        pre_calculated_chw_kpis: Optional pre-calculated CHW summary metrics.
        for_date: The date for which these signals are relevant.
        chw_zone_context: The zone(s) context for these signals.

    Returns:
        Dict[str, Any]: A dictionary containing key epi signals and counts.
        Example: {
            "date": "2023-10-01",
            "zone_context": "Zone A",
            "new_symptomatic_cases_key_conditions_count": 2, // Unique patients
            "symptomatic_keywords_monitored": "fever|cough|chills...",
            "new_malaria_cases_today_count": 1, // Example specific disease
            "pending_tb_contact_traces_count": 3,
            "high_risk_patients_today_demographics": { // Simplified demographics
                "count": 5,
                "age_groups": {"0-4": 1, "18-44": 2, "65+": 2}, // Count per group
                "gender_distribution": {"Male": 3, "Female": 2} // If gender data available
            },
            "reported_symptom_cluster_alerts": [ // If clusters detected
                {"symptoms": "fever;diarrhea", "count": 3, "location_hint": "Village X (if available)"}
            ]
        }
    """
    logger.info(f"Extracting CHW local epi signals for date: {for_date}, zone: {chw_zone_context}")

    epi_signals = {
        "date": str(for_date),
        "zone_context": chw_zone_context,
        "new_symptomatic_cases_key_conditions_count": 0,
        "symptomatic_keywords_monitored": "", # Will be populated from app_config
        "new_malaria_cases_today_count": 0, # Example, can be made generic for KEY_CONDITIONS_FOR_ACTION
        "pending_tb_contact_traces_count": 0,
        "high_risk_patients_today_demographics": {
            "count": 0, "age_groups": {}, "gender_distribution": {}
        },
        "reported_symptom_cluster_alerts": [] # List to hold potential symptom cluster info
    }

    if chw_daily_encounter_df is None or chw_daily_encounter_df.empty:
        logger.warning("No daily encounter data provided for epi signal extraction.")
        # Populate pending TB contacts if available from pre-calculated KPIs
        if pre_calculated_chw_kpis:
             epi_signals["pending_tb_contact_traces_count"] = pre_calculated_chw_kpis.get('tb_contacts_to_trace_today', 0)
        return epi_signals

    df_enc = chw_daily_encounter_df.copy()

    # 1. New Symptomatic Cases for Key Conditions
    # Using focused KEY_CONDITIONS_FOR_ACTION from new config
    symptomatic_key_conditions = set(app_config.KEY_CONDITIONS_FOR_ACTION) & \
                                 {"TB", "Pneumonia", "Malaria", "Dengue", "Diarrheal Diseases (Severe)"} # Focus on acutely symptomatic ones
    
    # Simple keywords for broad symptom categories (fever, respiratory, GI)
    # On PED, this might be tapping icons for observed symptoms.
    symptom_categories_keywords = {
        "fever_general": "fever|chills|hot",
        "respiratory": "cough|breathless|sore throat|runny nose",
        "gastrointestinal": "diarrhea|vomit|nausea|stomach ache"
    }
    # Combine all keywords for general symptomatic check for this KPI
    all_symptom_keywords_str = "|".join(symptom_categories_keywords.values())
    epi_signals["symptomatic_keywords_monitored"] = all_symptom_keywords_str.replace("|", ", ")

    if 'patient_reported_symptoms' in df_enc.columns and 'condition' in df_enc.columns and 'patient_id' in df_enc.columns:
        reported_symptoms_series = df_enc['patient_reported_symptoms'].astype(str).fillna('')
        symptomatic_df = df_enc[
            df_enc['condition'].apply(lambda x: any(key_cond.lower() in str(x).lower() for key_cond in symptomatic_key_conditions)) &
            (reported_symptoms_series.str.contains(all_symptom_keywords_str, case=False, na=False))
        ]
        epi_signals["new_symptomatic_cases_key_conditions_count"] = symptomatic_df['patient_id'].nunique()

    # 2. Specific Disease Counts (e.g., New Malaria Cases Today)
    # This can be generalized if needed, or specific for high-priority local diseases
    if 'condition' in df_enc.columns and 'patient_id' in df_enc.columns:
        # Example for Malaria
        epi_signals["new_malaria_cases_today_count"] = df_enc[
            df_enc['condition'].str.contains("Malaria", case=False, na=False)
        ]['patient_id'].nunique()
        # Add other key diseases as needed from app_config.KEY_CONDITIONS_FOR_ACTION
        # e.g., new_tb_cases_identified_today_count = df_enc[df_enc['condition'].str.contains("TB", case=False, na=False)]['patient_id'].nunique()

    # 3. Pending TB Contact Traces (Primary source is often `pre_calculated_chw_kpis` from more complex logic)
    # This is a task count more than an epi signal, but often included in CHW daily overview.
    # It could also be derived if CHW flags new TB case and initiates referral for contact tracing.
    if pre_calculated_chw_kpis and 'tb_contacts_to_trace_today' in pre_calculated_chw_kpis:
        epi_signals["pending_tb_contact_traces_count"] = pre_calculated_chw_kpis.get('tb_contacts_to_trace_today', 0)
    elif all(c in df_enc.columns for c in ['condition', 'referral_reason', 'referral_status']):
        # Fallback basic derivation if not pre-calculated (simplified)
        epi_signals["pending_tb_contact_traces_count"] = df_enc[
            df_enc['condition'].str.contains("TB", case=False, na=False) &
            df_enc['referral_reason'].str.contains("Contact Trac", case=False, na=False) & # Partial match
            (df_enc['referral_status'].str.lower() == 'pending')
        ]['patient_id'].nunique()


    # 4. Demographics of High AI Risk Patients Today (simplified for reporting)
    if 'ai_risk_score' in df_enc.columns and 'patient_id' in df_enc.columns:
        high_risk_today_df = df_enc[df_enc['ai_risk_score'] >= app_config.RISK_SCORE_HIGH_THRESHOLD].drop_duplicates(subset=['patient_id'])
        
        if not high_risk_today_df.empty:
            epi_signals["high_risk_patients_today_demographics"]["count"] = len(high_risk_today_df)
            
            if 'age' in high_risk_today_df.columns and high_risk_today_df['age'].notna().any():
                # Simplified age bins for LMIC CHW reporting context
                age_bins_epi = [0, 5, 18, 50, np.inf] # Under 5, Child/Adolescent, Adult, Elderly
                age_labels_epi = ['0-4 yrs', '5-17 yrs', '18-49 yrs', '50+ yrs']
                # Use .loc to avoid SettingWithCopyWarning
                high_risk_today_df_copy = high_risk_today_df.copy()
                high_risk_today_df_copy.loc[:, 'age_group_epi'] = pd.cut(
                    high_risk_today_df_copy['age'], bins=age_bins_epi, labels=age_labels_epi, right=False
                )
                age_group_counts = high_risk_today_df_copy['age_group_epi'].value_counts().to_dict()
                epi_signals["high_risk_patients_today_demographics"]["age_groups"] = age_group_counts
            
            if 'gender' in high_risk_today_df.columns and high_risk_today_df['gender'].notna().any():
                # Simple gender counts, assuming 'Male', 'Female', 'Unknown'
                gender_counts = high_risk_today_df['gender'].value_counts().to_dict()
                epi_signals["high_risk_patients_today_demographics"]["gender_distribution"] = gender_counts

    # 5. Symptom Cluster Detection (Very basic simulation for this context)
    # A real system would use more advanced spatiotemporal clustering.
    # Here, just count co-occurrence of common symptom patterns for reporting.
    if 'patient_reported_symptoms' in df_enc.columns:
        # Example: Look for clusters of "Fever & Diarrhea" or "Fever & Cough"
        symptoms_series = df_enc['patient_reported_symptoms'].astype(str).str.lower()
        fever_diarrhea_count = symptoms_series[
            symptoms_series.str.contains('fever', na=False) & symptoms_series.str.contains('diarrhea', na=False)
        ].count()
        fever_cough_count = symptoms_series[
            symptoms_series.str.contains('fever', na=False) & symptoms_series.str.contains('cough', na=False)
        ].count()

        # Threshold for alerting on a cluster (highly dependent on CHW's daily patient load)
        cluster_threshold = 3 # Example: if 3 or more patients present with this combo
        
        if fever_diarrhea_count >= cluster_threshold:
            epi_signals["reported_symptom_cluster_alerts"].append({
                "symptoms": "Fever & Diarrhea", "count": int(fever_diarrhea_count),
                "location_hint": chw_zone_context # Location would be more granular on PED
            })
        if fever_cough_count >= cluster_threshold:
            epi_signals["reported_symptom_cluster_alerts"].append({
                "symptoms": "Fever & Cough", "count": int(fever_cough_count),
                "location_hint": chw_zone_context
            })

    logger.info(f"CHW local epi signals extracted: {epi_signals}")
    return epi_signals


# --- Example Usage (for testing or integration into a reporting script) ---
if __name__ == '__main__':
    logger.setLevel(logging.DEBUG)
    logger.info("Running CHW Epi Watch component simulation directly...")

    sample_epi_encounters = pd.DataFrame({
        'patient_id': ['P001', 'P002', 'P003', 'P004', 'P005', 'P006', 'P007'],
        'condition': ['Malaria', 'TB', 'Pneumonia', 'Diarrheal Diseases (Severe)', 'Wellness Visit', 'Malaria', 'Malaria'],
        'patient_reported_symptoms': [
            'fever;chills', 'cough;fever', 'cough;breathless', 'fever;diarrhea;vomit', 
            'none', 'fever', 'fever;headache'
        ],
        'ai_risk_score': [75, 85, 90, 80, 20, 65, 78],
        'age': [5, 45, 68, 2, 30, 22, 3],
        'gender': ['Male', 'Female', 'Male', 'Male', 'Female', 'Female', 'Male'],
        'referral_reason': ['N/A', 'Contact Tracing', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A'],
        'referral_status': ['N/A', 'Pending', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A'],
        'zone_id': ['ZoneA', 'ZoneA', 'ZoneB', 'ZoneA', 'ZoneB', 'ZoneA', 'ZoneA']
    })
    sample_date = pd.Timestamp('2023-10-04').date()

    # Simulate pre-calculated KPI for TB contacts
    pre_calc_kpis = {'tb_contacts_to_trace_today': 1} # From sample data P002

    epi_data_output = extract_chw_local_epi_signals(
        chw_daily_encounter_df=sample_epi_encounters,
        pre_calculated_chw_kpis=pre_calc_kpis,
        for_date=sample_date,
        chw_zone_context="ZoneA Daily"
    )

    print("\n--- Extracted CHW Epi Signals: ---")
    for key, value in epi_data_output.items():
        if isinstance(value, dict):
            print(f"  {key.replace('_', ' ').title()}:")
            for sub_key, sub_value in value.items():
                print(f"    {sub_key.replace('_', ' ').title()}: {sub_value}")
        elif isinstance(value, list) and value:
            print(f"  {key.replace('_', ' ').title()}:")
            for item_in_list in value:
                print(f"    - {item_in_list}")
        else:
            print(f"  {key.replace('_', ' ').title()}: {value}")
