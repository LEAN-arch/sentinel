# test/pages/clinic_components/epi_module.py
# Redesigned for "Sentinel Health Co-Pilot" - LMIC Edge-First System
# This module calculates clinic-level epidemiological data, including
# symptom trends, test positivity trends, demographic breakdowns, and referral patterns.
# The output is structured data (DataFrames/dictionaries) primarily for:
#   1. Display on a simplified web dashboard/report for clinic managers at a Facility Node (Tier 2).
#   2. Informing local outbreak detection and response strategies.

import pandas as pd
import numpy as np
import logging
# Assuming app_config is in the PYTHONPATH or project root.
from config import app_config # Uses new, redesigned app_config
from utils.core_data_processing import get_trend_data # For trend calculations
from typing import Dict, Any, Optional, List, Tuple

logger = logging.getLogger(__name__)

def calculate_clinic_epi_data(
    filtered_health_df_clinic_period: pd.DataFrame,
    reporting_period_str: str,
    selected_condition_for_demographics: Optional[str] = "All Conditions" # Can be parameterized by UI
) -> Dict[str, Any]:
    """
    Calculates various epidemiological data sets for the clinic over a period.

    Args:
        filtered_health_df_clinic_period: DataFrame of health records for the clinic and period.
        reporting_period_str: String describing the reporting period.
        selected_condition_for_demographics: Condition selected for detailed demographic breakdown.

    Returns:
        Dict[str, Any]: A dictionary containing structured epidemiological data.
        Keys like: "symptom_trends_weekly_df", "malaria_rdt_positivity_weekly_series",
                   "demographics_for_selected_condition_data", "referral_funnel_summary_data".
    """
    logger.info(f"Calculating clinic epi data for period: {reporting_period_str}, demo condition: {selected_condition_for_demographics}")

    epi_data_output: Dict[str, Any] = {
        "reporting_period": reporting_period_str,
        "symptom_trends_weekly_df": None, # DataFrame: week_start_date, symptom, count
        "malaria_rdt_positivity_weekly_series": None, # Series: index=week_start_date, value=positivity_rate
        "demographics_for_selected_condition_data": None, # Dict: {age_df: df, gender_df: df}
        "referral_funnel_summary_data": None, # DataFrame: Stage, Count
        "general_notes": [] # To capture any issues or lack of data
    }

    if filtered_health_df_clinic_period is None or filtered_health_df_clinic_period.empty:
        logger.warning("No health data provided for clinic epi data calculation.")
        epi_data_output["general_notes"].append("No health data available for the period.")
        return epi_data_output

    df_clinic = filtered_health_df_clinic_period.copy()
    if 'encounter_date' not in df_clinic.columns:
        logger.error("Encounter_date column missing, essential for epi calculations.")
        epi_data_output["general_notes"].append("Encounter_date column missing.")
        return epi_data_output
    df_clinic['encounter_date'] = pd.to_datetime(df_clinic['encounter_date'], errors='coerce')
    df_clinic.dropna(subset=['encounter_date'], inplace=True)


    # 1. Symptom Trends (Weekly Top 5 Symptoms)
    #    On PED, CHW enters symptoms (icons/voice). Aggregated here.
    if 'patient_reported_symptoms' in df_clinic.columns and df_clinic['patient_reported_symptoms'].notna().any():
        symptoms_df = df_clinic[['encounter_date', 'patient_reported_symptoms']].copy()
        symptoms_df.dropna(subset=['patient_reported_symptoms'], inplace=True)
        
        # Exclude common non-informative entries
        symptoms_df = symptoms_df[~symptoms_df['patient_reported_symptoms'].str.lower().isin(["unknown", "n/a", "none", ""])]

        if not symptoms_df.empty:
            symptoms_exploded = symptoms_df.assign(symptom=symptoms_df['patient_reported_symptoms'].str.split(';')) \
                                           .explode('symptom')
            symptoms_exploded['symptom'] = symptoms_exploded['symptom'].str.strip().str.title() # Consistent casing
            symptoms_exploded.dropna(subset=['symptom'], inplace=True)
            symptoms_exploded = symptoms_exploded[symptoms_exploded['symptom'] != '']

            if not symptoms_exploded.empty:
                top_5_symptoms_overall = symptoms_exploded['symptom'].value_counts().nlargest(5).index.tolist()
                symptoms_to_trend = symptoms_exploded[symptoms_exploded['symptom'].isin(top_5_symptoms_overall)]

                if not symptoms_to_trend.empty:
                    # Group by week and symptom to get counts
                    symptom_trends_df = symptoms_to_trend.groupby(
                        [pd.Grouper(key='encounter_date', freq='W-Mon'), 'symptom']
                    ).size().reset_index(name='count')
                    symptom_trends_df.rename(columns={'encounter_date': 'week_start_date'}, inplace=True)
                    epi_data_output["symptom_trends_weekly_df"] = symptom_trends_df
                else: epi_data_output["general_notes"].append("Not enough distinct symptom data for top 5 weekly trends.")
            else: epi_data_output["general_notes"].append("No parsable symptom data after cleaning.")
        else: epi_data_output["general_notes"].append("No actionable patient reported symptoms data.")
    else: epi_data_output["general_notes"].append("Patient_reported_symptoms column missing or empty.")


    # 2. Test Positivity Rate Trends (Example: Malaria RDT)
    #    Using a key test from app_config
    malaria_rdt_key = "RDT-Malaria" # Original key from app_config.KEY_TEST_TYPES_FOR_ANALYSIS
    malaria_rdt_display_name = app_config.KEY_TEST_TYPES_FOR_ANALYSIS.get(malaria_rdt_key, {}).get("display_name", malaria_rdt_key)
    
    if 'test_type' in df_clinic.columns and 'test_result' in df_clinic.columns:
        malaria_tests_df = df_clinic[
            (df_clinic['test_type'] == malaria_rdt_key) & # Match on the original key
            (~df_clinic.get('test_result', pd.Series(dtype=str)).isin(['Pending', 'Rejected Sample', 'Unknown', 'Indeterminate', 'N/A', '']))
        ].copy()

        if not malaria_tests_df.empty:
            malaria_tests_df['is_positive'] = (malaria_tests_df['test_result'] == 'Positive')
            
            weekly_malaria_pos_rate_series = get_trend_data(
                df=malaria_tests_df,
                value_col='is_positive', date_col='encounter_date',
                period='W-Mon', agg_func='mean', # Mean of boolean (0/1) gives proportion
                source_context="ClinicEpi/MalariaPos"
            )
            if weekly_malaria_pos_rate_series is not None and not weekly_malaria_pos_rate_series.empty:
                epi_data_output["malaria_rdt_positivity_weekly_series"] = weekly_malaria_pos_rate_series * 100 # Convert to percentage
            else: epi_data_output["general_notes"].append(f"No aggregated weekly positivity data for {malaria_rdt_display_name}.")
        else: epi_data_output["general_notes"].append(f"No conclusive test data for {malaria_rdt_display_name} in period.")
    else: epi_data_output["general_notes"].append(f"Test_type or test_result columns missing for {malaria_rdt_display_name} trends.")


    # 3. Demographic Breakdown for Selected Condition
    #    CHW would collect minimal demographics. This part aggregates them.
    if 'condition' in df_clinic.columns and 'patient_id' in df_clinic.columns:
        demographics_data = {"age_distribution_df": None, "gender_distribution_df": None, "selected_condition_for_demog": selected_condition_for_demographics}
        
        condition_filtered_df_for_demog = df_clinic.copy()
        if selected_condition_for_demographics != "All Conditions":
            condition_filtered_df_for_demog = df_clinic[
                df_clinic['condition'].str.contains(selected_condition_for_demographics, case=False, na=False) # Allow partial match for condition
            ]

        if not condition_filtered_df_for_demog.empty:
            # Get unique patients for the condition in the period (first occurrence for "new cases" context if desired)
            # For general demographic breakdown, unique patients is sufficient.
            unique_patients_for_demog_df = condition_filtered_df_for_demog.drop_duplicates(subset=['patient_id'])

            if not unique_patients_for_demog_df.empty:
                # Age breakdown
                if 'age' in unique_patients_for_demog_df.columns and unique_patients_for_demog_df['age'].notna().any():
                    age_bins_clinic_epi = [0, 5, 18, 35, 50, 65, np.inf]
                    age_labels_clinic_epi = ['0-4', '5-17', '18-34', '35-49', '50-64', '65+']
                    # Use .loc on a copy to avoid SettingWithCopyWarning
                    temp_age_df = unique_patients_for_demog_df.copy()
                    temp_age_df.loc[:, 'age_group_display'] = pd.cut(temp_age_df['age'], bins=age_bins_clinic_epi, labels=age_labels_clinic_epi, right=False)
                    age_dist_df = temp_age_df['age_group_display'].value_counts().sort_index().reset_index()
                    age_dist_df.columns = ['Age Group', 'Patient Count']
                    demographics_data["age_distribution_df"] = age_dist_df
                else: epi_data_output["general_notes"].append("Age data unavailable for demographic breakdown of selected condition.")

                # Gender breakdown
                if 'gender' in unique_patients_for_demog_df.columns and unique_patients_for_demog_df['gender'].notna().any():
                    # Ensure 'gender' column is cleaned (e.g. "Unknown" for blanks/NaNs)
                    temp_gender_df = unique_patients_for_demog_df.copy()
                    temp_gender_df['gender'] = temp_gender_df['gender'].fillna('Unknown').astype(str).str.strip().replace(['', 'nan', 'None'], 'Unknown')
                    gender_dist_df = temp_gender_df[temp_gender_df['gender'] != 'Unknown']['gender'].value_counts().reset_index() # Exclude "Unknown" if desired
                    gender_dist_df.columns = ['Gender', 'Patient Count']
                    demographics_data["gender_distribution_df"] = gender_dist_df
                else: epi_data_output["general_notes"].append("Gender data unavailable for demographic breakdown of selected condition.")
                epi_data_output["demographics_for_selected_condition_data"] = demographics_data
            else: epi_data_output["general_notes"].append(f"No unique patients found for condition '{selected_condition_for_demographics}' for demographic breakdown.")
        else: epi_data_output["general_notes"].append(f"No patients found for condition '{selected_condition_for_demographics}'.")
    else: epi_data_output["general_notes"].append("Condition or patient_id column missing for demographic breakdown.")


    # 4. Referral Funnel Analysis (Simplified for reporting)
    if 'referral_status' in df_clinic.columns and 'encounter_id' in df_clinic.columns:
        # Focus on actionable, conclusive referral statuses. CHW PED might log basic referral made.
        referral_funnel_df = df_clinic[
            df_clinic.get('referral_status', pd.Series(dtype=str)).str.lower().isin(['pending', 'completed', 'initiated', 'service provided', 'attended', 'missed appointment', 'declined', 'unknown'])
        ].copy()

        if not referral_funnel_df.empty:
            total_referrals_initiated_funnel = referral_funnel_df['encounter_id'].nunique() # Assuming each encounter can have one referral process
            
            completed_outcome_list = ['completed', 'service provided', 'attended consult', 'attended followup', 'attended'] # from old config
            if 'referral_outcome' in referral_funnel_df.columns:
                referrals_completed_outcome_funnel = referral_funnel_df[
                    referral_funnel_df['referral_outcome'].str.lower().isin(completed_outcome_list)
                ]['encounter_id'].nunique()
            else: referrals_completed_outcome_funnel = 0

            referrals_still_pending_funnel = referral_funnel_df[
                referral_funnel_df['referral_status'].str.lower() == 'pending'
            ]['encounter_id'].nunique()

            funnel_summary_list = [
                {'Stage': 'Referrals Initiated (Period)', 'Count': total_referrals_initiated_funnel},
                {'Stage': 'Referrals Concluded (Outcome Recorded as Positive)', 'Count': referrals_completed_outcome_funnel},
                {'Stage': 'Referrals Still Pending (Status "Pending")', 'Count': referrals_still_pending_funnel},
            ]
            epi_data_output["referral_funnel_summary_data"] = pd.DataFrame(funnel_summary_list)
        else: epi_data_output["general_notes"].append("No actionable referral records found for funnel analysis.")
    else: epi_data_output["general_notes"].append("Referral status or encounter_id data missing for funnel analysis.")

    logger.info("Clinic epi data calculation complete.")
    return epi_data_output


# --- Example Usage (for testing or integration into a reporting script) ---
if __name__ == '__main__':
    logger.setLevel(logging.DEBUG)
    logger.info("Running Clinic Epi Module data calculation simulation directly...")

    # Create sample clinic data for a period
    date_rng_epi = pd.date_range(start='2023-10-01', end='2023-10-31', freq='D')
    num_records_epi = len(date_rng_epi) * 5 # Approx 5 records per day
    
    sample_clinic_health_data = pd.DataFrame({
        'encounter_date': np.random.choice(date_rng_epi, num_records_epi),
        'patient_id': [f'CP{i%30:03d}' for i in range(num_records_epi)],
        'condition': np.random.choice(['Malaria', 'TB', 'Pneumonia', 'Wellness Visit', 'STI-Syphilis', 'Diabetes'], num_records_epi),
        'patient_reported_symptoms': np.random.choice(
            ['fever;cough', 'fever;chills', 'headache', 'none', 'diarrhea;vomit', 'cough;breathless', 'fever;diarrhea'], 
            num_records_epi
        ),
        'test_type': np.random.choice(['RDT-Malaria', 'Sputum-AFB', 'HIV-Rapid', 'Glucose Test', 'N/A'], num_records_epi),
        'test_result': np.random.choice(['Positive', 'Negative', 'Pending', 'Unknown'], num_records_epi),
        'age': np.random.randint(1, 85, num_records_epi),
        'gender': np.random.choice(['Male', 'Female', 'Unknown'], num_records_epi),
        'referral_status': np.random.choice(['Pending', 'Completed', 'Initiated', 'N/A'], num_records_epi),
        'referral_outcome': np.random.choice(['Attended', 'Missed Appointment', 'Service Provided', 'Pending'], num_records_epi),
        'encounter_id': [f'CE{i:04d}' for i in range(num_records_epi)]
    })
    sample_clinic_health_data.sort_values('encounter_date', inplace=True)
    current_reporting_period_str = "October 2023 Test Data"

    clinic_epi_results = calculate_clinic_epi_data(
        filtered_health_df_clinic_period=sample_clinic_health_data,
        reporting_period_str=current_reporting_period_str,
        selected_condition_for_demographics="Malaria"
    )

    print(f"\n--- Calculated Clinic Epi Data for: {current_reporting_period_str} ---")
    for section_name, section_data in clinic_epi_results.items():
        print(f"\n## {section_name.replace('_', ' ').title()}:")
        if isinstance(section_data, pd.DataFrame):
            print(section_data.to_string(index=False))
        elif isinstance(section_data, pd.Series):
            print(section_data.to_string())
        elif isinstance(section_data, dict): # For demographics
            for sub_key, sub_data in section_data.items():
                print(f"  {sub_key.replace('_', ' ').title()}:")
                if isinstance(sub_data, pd.DataFrame): print(sub_data.to_string(index=False))
                else: print(f"    {sub_data}")
        elif isinstance(section_data, list) and section_name=="general_notes": # for notes
            if section_data:
                for note in section_data: print(f"- {note}")
            else: print("  (No notes or issues)")
        else:
            print(f"  {section_data}")
