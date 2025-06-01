# test/pages/clinic_components/patient_focus_tab.py
# Redesigned for "Sentinel Health Co-Pilot" - LMIC Edge-First System
# This module prepares data related to patient load and flagged high-risk/complex cases
# at the clinic level. This data is intended for:
#   1. Display on a simplified web dashboard/report for clinic managers/clinical leads
#      at a Facility Node (Tier 2) for operational oversight and case review.
#   2. Informing resource allocation and highlighting patients needing urgent clinical attention.

import pandas as pd
import numpy as np
import logging
# Assuming app_config is in the PYTHONPATH or project root.
from config import app_config # Uses new, redesigned app_config
# get_patient_alerts_for_clinic is now a more robust function in core_data_processing
from utils.core_data_processing import get_patient_alerts_for_clinic
from typing import Dict, Any, Optional, List, Tuple

logger = logging.getLogger(__name__)

def prepare_clinic_patient_focus_data(
    filtered_health_df_clinic_period: pd.DataFrame,
    reporting_period_str: str,
    time_period_agg_load: str = 'D' # 'D' for daily, 'W-Mon' for weekly patient load
) -> Dict[str, Any]:
    """
    Prepares data for patient load analysis and flagged patient case review.

    Args:
        filtered_health_df_clinic_period: DataFrame of health records for the clinic and period.
        reporting_period_str: String describing the reporting period.
        time_period_agg_load: Aggregation period for patient load ('D', 'W-Mon').

    Returns:
        Dict[str, Any]: A dictionary containing structured data for patient focus.
        Example: {
            "reporting_period": "Last 7 Days",
            "patient_load_by_condition_df": pd.DataFrame(...), // Cols: date, condition, unique_patients
            "flagged_patients_for_review_df": pd.DataFrame(...), // Output of get_patient_alerts_for_clinic
            "data_availability_notes": []
        }
    """
    logger.info(f"Preparing clinic patient focus data for period: {reporting_period_str}")

    patient_focus_output: Dict[str, Any] = {
        "reporting_period": reporting_period_str,
        "patient_load_by_condition_df": None,
        "flagged_patients_for_review_df": None,
        "data_availability_notes": []
    }

    if filtered_health_df_clinic_period is None or filtered_health_df_clinic_period.empty:
        logger.warning("No health data provided for patient focus data preparation.")
        patient_focus_output["data_availability_notes"].append("No health data available for the selected period.")
        return patient_focus_output

    df_clinic_pf = filtered_health_df_clinic_period.copy()
    # Ensure essential columns for encounter date and patient/condition IDs exist
    if 'encounter_date' not in df_clinic_pf.columns or 'patient_id' not in df_clinic_pf.columns or 'condition' not in df_clinic_pf.columns:
        logger.error("Essential columns (encounter_date, patient_id, condition) missing for patient focus analysis.")
        patient_focus_output["data_availability_notes"].append("Required data columns missing for analysis.")
        return patient_focus_output
        
    df_clinic_pf['encounter_date'] = pd.to_datetime(df_clinic_pf['encounter_date'], errors='coerce')
    df_clinic_pf.dropna(subset=['encounter_date', 'patient_id', 'condition'], inplace=True)


    # 1. Patient Load by Key Condition
    #    Uses KEY_CONDITIONS_FOR_ACTION from new config, more focused list.
    key_conditions_for_load = app_config.KEY_CONDITIONS_FOR_ACTION
    
    # Filter for key conditions and valid patient IDs
    load_analysis_df = df_clinic_pf[
        df_clinic_pf['condition'].str.contains('|'.join(key_conditions_for_load), case=False, na=False) &
        (df_clinic_pf['patient_id'].astype(str).str.lower().isin(['unknown', 'n/a', ''])) == False # Exclude common invalid patient_ids
    ].copy()

    if not load_analysis_df.empty:
        # Group by specified time period and condition, counting unique patients
        # Example: daily unique patient encounters for each key condition
        # For time_period_agg 'D' (Daily):
        #   index = encounter_date (day)
        #   columns = condition_1, condition_2, ...
        #   values = count of unique_patients
        # A simpler output is often: date, condition, unique_patients (long format)
        
        patient_load_summary_df = load_analysis_df.groupby(
            [pd.Grouper(key='encounter_date', freq=time_period_agg_load), 'condition']
        )['patient_id'].nunique().reset_index()
        
        patient_load_summary_df.rename(
            columns={'encounter_date': 'period_start_date', 'patient_id': 'unique_patients_count'},
            inplace=True
        )
        patient_focus_output["patient_load_by_condition_df"] = patient_load_summary_df
        if patient_load_summary_df.empty:
            patient_focus_output["data_availability_notes"].append("No patient load data for key conditions in period after grouping.")
    else:
        patient_focus_output["data_availability_notes"].append("No patients with encounters for key conditions in selected period for load analysis.")


    # 2. Flagged Patient Cases for Clinical Review
    #    This utilizes the more robust get_patient_alerts_for_clinic from core_data_processing,
    #    which is already designed to return a DataFrame.
    flagged_patients_df = get_patient_alerts_for_clinic(
        health_df_period=df_clinic_pf, # Pass the already period-filtered df
        risk_threshold_moderate=app_config.RISK_SCORE_MODERATE_THRESHOLD, # From new config
        source_context="ClinicPatientFocus/FlaggedCases"
    ) # This function now returns a df with alert_reason, priority_score etc.

    if flagged_patients_df is not None and not flagged_patients_df.empty:
        patient_focus_output["flagged_patients_for_review_df"] = flagged_patients_df
        logger.info(f"Generated {len(flagged_patients_df)} flagged patient cases for review.")
    else:
        patient_focus_output["data_availability_notes"].append("No specific patient cases flagged for clinical review in the period based on current criteria.")
        # Ensure an empty DataFrame of expected structure if no alerts
        # This might depend on exact columns returned by get_patient_alerts_for_clinic
        # For now, setting to None is handled, but for strict schema, an empty df with cols is better.
        # Example if specific cols known: patient_focus_output["flagged_patients_for_review_df"] = pd.DataFrame(columns=['patient_id', 'Alert Reason', ...])


    logger.info("Clinic patient focus data preparation complete.")
    return patient_focus_output


# --- Example Usage (for testing or integration into a reporting script) ---
if __name__ == '__main__':
    logger.setLevel(logging.DEBUG)
    logger.info("Running Clinic Patient Focus Tab component data preparation simulation...")

    # Create sample clinic health data for a period
    date_rng_pf = pd.to_datetime(pd.date_range(start='2023-10-01', end='2023-10-07', freq='D'))
    num_records_pf = len(date_rng_pf) * 8 
    
    sample_clinic_data_pf = pd.DataFrame({
        'encounter_date': np.random.choice(date_rng_pf, num_records_pf),
        'patient_id': [f'CP{i%15:03d}' for i in range(num_records_pf)], # 15 unique patients
        'condition': np.random.choice(app_config.KEY_CONDITIONS_FOR_ACTION + ['Wellness Visit', 'Minor Injury'], num_records_pf, p=[0.15]*len(app_config.KEY_CONDITIONS_FOR_ACTION) + [0.3, 0.1]), # Weighted
        'zone_id': np.random.choice(['ClinicMain', 'OutreachSite1'], num_records_pf),
        'age': np.random.randint(1, 90, num_records_pf),
        'ai_risk_score': np.random.randint(20, 100, num_records_pf),
        'ai_followup_priority_score': np.random.randint(10, 100, num_records_pf),
        'min_spo2_pct': np.random.randint(85, 100, num_records_pf),
        'vital_signs_temperature_celsius': np.random.normal(37.5, 1.0, num_records_pf).round(1),
        'referral_status': np.random.choice(['Pending', 'Completed', 'N/A'], num_records_pf)
        # Add other columns expected by get_patient_alerts_for_clinic if necessary for richer test
    })
    sample_clinic_data_pf.sort_values('encounter_date', inplace=True)
    current_reporting_period = "Week 40, 2023 (Oct 1-7)"

    patient_focus_results = prepare_clinic_patient_focus_data(
        filtered_health_df_clinic_period=sample_clinic_data_pf,
        reporting_period_str=current_reporting_period,
        time_period_agg_load='D' # Daily patient load
    )

    print(f"\n--- Prepared Clinic Patient Focus Data for: {current_reporting_period} ---")
    
    print("\n## Patient Load by Condition (Daily):")
    if patient_focus_results["patient_load_by_condition_df"] is not None and \
       not patient_focus_results["patient_load_by_condition_df"].empty:
        print(patient_focus_results["patient_load_by_condition_df"].to_string(index=False))
    else:
        print("  (No patient load data calculated)")

    print("\n## Flagged Patients for Clinical Review:")
    if patient_focus_results["flagged_patients_for_review_df"] is not None and \
       not patient_focus_results["flagged_patients_for_review_df"].empty:
        # Print selected columns for brevity
        cols_to_show_flagged = ['patient_id', 'encounter_date', 'condition', 'AI Risk Score', 'Alert Reason', 'Priority Score']
        # Ensure AI Risk Score matches exact column name from get_patient_alerts_for_clinic output (it may be 'ai_risk_score')
        actual_cols_to_show_flagged = [c for c in cols_to_show_flagged if c in patient_focus_results["flagged_patients_for_review_df"].columns]
        if not actual_cols_to_show_flagged: # if renamed cols from get_patient_alerts... changed
             actual_cols_to_show_flagged = [c for c in ['patient_id', 'Alert Reason', 'Priority Score'] if c in patient_focus_results["flagged_patients_for_review_df"].columns]

        print(patient_focus_results["flagged_patients_for_review_df"][actual_cols_to_show_flagged].head(10).to_string(index=False))
    else:
        print("  (No patients flagged for review)")

    print("\n## Data Availability Notes:")
    if patient_focus_results["data_availability_notes"]:
        for note in patient_focus_results["data_availability_notes"]: print(f"  - {note}")
    else:
        print("  (No specific data availability notes)")
