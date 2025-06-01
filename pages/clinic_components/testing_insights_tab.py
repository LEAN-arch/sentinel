# test/pages/clinic_components/testing_insights_tab.py
# Redesigned for "Sentinel Health Co-Pilot" - LMIC Edge-First System
# This module prepares detailed data for laboratory testing performance and trends at a clinic.
# This data is intended for:
#   1. Display on a simplified web dashboard/report for clinic/lab managers at a Facility Node (Tier 2).
#   2. Identifying bottlenecks, quality issues (sample rejection), and overdue tests for action.

import pandas as pd
import numpy as np
import logging
# Assuming app_config is in the PYTHONPATH or project root.
from config import app_config # Uses new, redesigned app_config
from utils.core_data_processing import get_trend_data # For TAT/Volume trends
from typing import Dict, Any, Optional, List, Tuple

logger = logging.getLogger(__name__)

def prepare_clinic_testing_insights_data(
    filtered_health_df_clinic_period: pd.DataFrame,
    clinic_service_kpis_summary: Dict[str, Any], # Output from core_data_processing.get_clinic_summary
    reporting_period_str: str,
    # Parameterizing the focus allows for more flexible report generation by the caller
    selected_test_group_display_name_for_detail: Optional[str] = "All Critical Tests Summary" 
) -> Dict[str, Any]:
    """
    Prepares structured data for detailed testing insights, including performance metrics,
    trends, overdue tests, and rejection analysis.

    Args:
        filtered_health_df_clinic_period: DataFrame of health records for the clinic and period.
        clinic_service_kpis_summary: Summary dict from get_clinic_summary, must contain 'test_summary_details'.
        reporting_period_str: String describing the reporting period.
        selected_test_group_display_name_for_detail: The specific test group (display name)
            or "All Critical Tests Summary" to focus on for detailed metrics and trends.

    Returns:
        Dict[str, Any]: A dictionary containing structured testing insights data.
        Example: {
            "reporting_period": "...",
            "focus_area": "All Critical Tests Summary" | "Malaria RDT",
            "critical_tests_summary_df": pd.DataFrame (if focus is "All Critical..."),
            "selected_test_group_metrics": {"Positivity (%)": ..., "Avg TAT (Days)": ...}, (if focus is specific group)
            "selected_test_group_tat_trend_series": pd.Series,
            "selected_test_group_volume_trend_df": pd.DataFrame, // Could have 'Conclusive' and 'Pending' columns
            "overdue_pending_tests_df": pd.DataFrame,
            "sample_rejection_reasons_df": pd.DataFrame, // Rejection Reason, Count
            "data_availability_notes": []
        }
    """
    logger.info(f"Preparing clinic testing insights. Focus: {selected_test_group_display_name_for_detail}, Period: {reporting_period_str}")

    testing_insights_output: Dict[str, Any] = {
        "reporting_period": reporting_period_str,
        "focus_area": selected_test_group_display_name_for_detail,
        "critical_tests_summary_df": None,
        "selected_test_group_metrics": None, # Dict of KPI values
        "selected_test_group_tat_trend_series": None,
        "selected_test_group_volume_trend_df": None,
        "overdue_pending_tests_df": None,
        "sample_rejection_reasons_df": None, # For donut chart data
        "top_rejected_samples_list_df": None, # For table view
        "data_availability_notes": []
    }

    if filtered_health_df_clinic_period is None or filtered_health_df_clinic_period.empty:
        testing_insights_output["data_availability_notes"].append("No health data for the period.")
        return testing_insights_output
    if not clinic_service_kpis_summary or "test_summary_details" not in clinic_service_kpis_summary:
        testing_insights_output["data_availability_notes"].append("Clinic service KPIs or test_summary_details missing.")
        return testing_insights_output

    df_clinic_ti = filtered_health_df_clinic_period.copy()
    detailed_test_stats_from_summary = clinic_service_kpis_summary.get("test_summary_details", {})

    if not detailed_test_stats_from_summary:
        testing_insights_output["data_availability_notes"].append("No detailed test summary statistics available from KPI summary.")
        # Proceed with overdue and rejection analysis which use raw df_clinic_ti
    
    # --- A. Process Focus Area: "All Critical Tests Summary" or Specific Test Group ---
    if selected_test_group_display_name_for_detail == "All Critical Tests Summary":
        crit_summary_list = []
        for group_disp_name, stats in detailed_test_stats_from_summary.items():
            # Find original key to check "critical" flag in app_config
            original_group_key = next((
                key for key, cfg_props in app_config.KEY_TEST_TYPES_FOR_ANALYSIS.items()
                if cfg_props.get("display_name") == group_disp_name
            ), None)
            
            if original_group_key and app_config.KEY_TEST_TYPES_FOR_ANALYSIS.get(original_group_key, {}).get("critical"):
                crit_summary_list.append({
                    "Test Group": group_disp_name,
                    "Positivity (%)": stats.get("positive_rate_perc", 0.0), # Adjusted key
                    "Avg. TAT (Days)": stats.get("avg_tat_days", np.nan),
                    "% Met TAT Target": stats.get("perc_met_tat_target", 0.0),
                    "Pending (Patients)": stats.get("pending_count_patients", 0), # Adjusted key
                    "Rejected (Patients)": stats.get("rejected_count_patients", 0), # Adjusted key
                    "Total Conclusive": stats.get("total_conclusive_tests", 0)
                })
        if crit_summary_list:
            testing_insights_output["critical_tests_summary_df"] = pd.DataFrame(crit_summary_list)
        else:
            testing_insights_output["data_availability_notes"].append("No data for critical tests or none configured.")

    elif selected_test_group_display_name_for_detail in detailed_test_stats_from_summary:
        stats_selected_group = detailed_test_stats_from_summary[selected_test_group_display_name_for_detail]
        testing_insights_output["selected_test_group_metrics"] = {
            "Positivity (%)": stats_selected_group.get("positive_rate_perc", 0.0),
            "Avg. TAT (Days)": stats_selected_group.get("avg_tat_days", np.nan),
            "% Met TAT Target": stats_selected_group.get("perc_met_tat_target", 0.0),
            "Pending (Patients)": stats_selected_group.get("pending_count_patients", 0),
            "Rejected (Patients)": stats_selected_group.get("rejected_count_patients", 0)
        }

        # Find original key and config for the selected display name to get test types for filtering raw data
        original_key_for_selected_group = next((
            key for key, cfg_props in app_config.KEY_TEST_TYPES_FOR_ANALYSIS.items()
            if cfg_props.get("display_name") == selected_test_group_display_name_for_detail
        ), None)

        if original_key_for_selected_group and original_key_for_selected_group in app_config.KEY_TEST_TYPES_FOR_ANALYSIS:
            config_selected_test = app_config.KEY_TEST_TYPES_FOR_ANALYSIS[original_key_for_selected_group]
            # Use original keys (from test_type column in df_clinic_ti) for filtering raw data for trends
            actual_test_keys_for_filtering = config_selected_test.get("types_in_group", [original_key_for_selected_group])
            if isinstance(actual_test_keys_for_filtering, str): actual_test_keys_for_filtering = [actual_test_keys_for_filtering]

            # TAT Trend
            if 'test_turnaround_days' in df_clinic_ti.columns and 'encounter_date' in df_clinic_ti.columns:
                df_tat_trend_src = df_clinic_ti[
                    (df_clinic_ti['test_type'].isin(actual_test_keys_for_filtering)) &
                    (df_clinic_ti['test_turnaround_days'].notna()) &
                    (~df_clinic_ti.get('test_result', pd.Series(dtype=str)).isin(['Pending', 'Unknown', 'Rejected Sample', 'Indeterminate', 'Unknown']))
                ].copy()
                if not df_tat_trend_src.empty:
                    tat_trend_series = get_trend_data(df_tat_trend_src, 'test_turnaround_days', date_col='encounter_date', period='D', agg_func='mean', source_context="TestingInsights/TATTrend")
                    if tat_trend_series is not None and not tat_trend_series.empty:
                        testing_insights_output["selected_test_group_tat_trend_series"] = tat_trend_series
                    else: testing_insights_output["data_availability_notes"].append(f"No aggregated TAT trend data for {selected_test_group_display_name_for_detail}.")
                else: testing_insights_output["data_availability_notes"].append(f"No conclusive tests with TAT data for {selected_test_group_display_name_for_detail}.")
            
            # Volume Trend (Conclusive vs. Pending)
            if 'patient_id' in df_clinic_ti.columns and 'encounter_date' in df_clinic_ti.columns: # patient_id for count/nunique
                df_volume_src = df_clinic_ti[df_clinic_ti['test_type'].isin(actual_test_keys_for_filtering)].copy()
                if not df_volume_src.empty:
                    vol_conclusive_series = get_trend_data(df_volume_src[~df_volume_src.get('test_result', pd.Series(dtype=str)).isin(['Pending', 'Unknown', 'Rejected Sample', 'Indeterminate', 'Unknown'])], 
                                                          'patient_id', date_col='encounter_date', period='D', agg_func='count', source_context="TestingInsights/VolConclusive").rename("Conclusive Tests") # using count here assuming each row is a test
                    vol_pending_series = get_trend_data(df_volume_src[df_volume_src.get('test_result', pd.Series(dtype=str)) == 'Pending'], 
                                                        'patient_id', date_col='encounter_date', period='D', agg_func='count', source_context="TestingInsights/VolPending").rename("Pending Tests")
                    
                    if (vol_conclusive_series is not None and not vol_conclusive_series.empty) or \
                       (vol_pending_series is not None and not vol_pending_series.empty):
                        # Combine into a single DataFrame for easier plotting by a UI component
                        volume_trend_combined_df = pd.concat([vol_conclusive_series, vol_pending_series], axis=1).fillna(0).reset_index()
                        testing_insights_output["selected_test_group_volume_trend_df"] = volume_trend_combined_df
                    else: testing_insights_output["data_availability_notes"].append(f"No test volume data for {selected_test_group_display_name_for_detail}.")
                else: testing_insights_output["data_availability_notes"].append(f"No tests found matching '{selected_test_group_display_name_for_detail}' for volume trend.")
        else:
            testing_insights_output["data_availability_notes"].append(f"Configuration for '{selected_test_group_display_name_for_detail}' not found; cannot generate trends.")
    else:
        testing_insights_output["data_availability_notes"].append(f"No activity data found for test group: '{selected_test_group_display_name_for_detail}'.")


    # --- B. Overdue Pending Tests (All Test Types) ---
    # This logic should use date columns that reflect when a sample was actionable by the lab.
    # Default to 'sample_collection_date', then 'sample_registered_lab_date', then 'encounter_date'.
    date_col_for_pending_duration = 'encounter_date' # Fallback
    if 'sample_collection_date' in df_clinic_ti.columns and df_clinic_ti['sample_collection_date'].notna().any():
        date_col_for_pending_duration = 'sample_collection_date'
    elif 'sample_registered_lab_date' in df_clinic_ti.columns and df_clinic_ti['sample_registered_lab_date'].notna().any():
        date_col_for_pending_duration = 'sample_registered_lab_date'
    
    overdue_src_df = df_clinic_ti[
        (df_clinic_ti.get('test_result', pd.Series(dtype=str)) == 'Pending') &
        (df_clinic_ti[date_col_for_pending_duration].notna())
    ].copy()

    if not overdue_src_df.empty:
        overdue_src_df[date_col_for_pending_duration] = pd.to_datetime(overdue_src_df[date_col_for_pending_duration], errors='coerce')
        overdue_src_df.dropna(subset=[date_col_for_pending_duration], inplace=True) # Critical

        if not overdue_src_df.empty:
            overdue_src_df['days_pending'] = (pd.Timestamp('today').normalize() - overdue_src_df[date_col_for_pending_duration]).dt.days
            
            def get_overdue_threshold(test_type_key_from_df: str) -> int:
                test_conf = app_config.KEY_TEST_TYPES_FOR_ANALYSIS.get(test_type_key_from_df)
                # Add a small buffer (e.g., 1-2 days) to target TAT for "overdue" flag
                buffer = 1 # Days
                if test_conf and 'target_tat_days' in test_conf:
                    return int(test_conf['target_tat_days'] + buffer)
                return int(app_config.OVERDUE_PENDING_TEST_DAYS_GENERAL_FALLBACK + buffer)
            
            overdue_src_df['overdue_if_days_pending_greater_than'] = overdue_src_df['test_type'].apply(get_overdue_threshold)
            
            final_overdue_df = overdue_src_df[
                overdue_src_df['days_pending'] > overdue_src_df['overdue_if_days_pending_greater_than']
            ]
            if not final_overdue_df.empty:
                cols_to_show_overdue = ['patient_id', 'test_type', date_col_for_pending_duration, 'days_pending', 'overdue_if_days_pending_greater_than']
                testing_insights_output["overdue_pending_tests_df"] = final_overdue_df[cols_to_show_overdue].sort_values('days_pending', ascending=False)
            else: testing_insights_output["data_availability_notes"].append("No tests currently pending longer than their target TAT + buffer.")
        else: testing_insights_output["data_availability_notes"].append("No valid pending tests after date cleaning for overdue evaluation.")
    else: testing_insights_output["data_availability_notes"].append("No pending tests found in the period for overdue evaluation.")


    # --- C. Sample Rejection Analysis ---
    if 'sample_status' in df_clinic_ti.columns and 'rejection_reason' in df_clinic_ti.columns:
        rejected_samples_data_df = df_clinic_ti[df_clinic_ti.get('sample_status',pd.Series(dtype=str)) == 'Rejected'].copy()
        if not rejected_samples_data_df.empty:
            # Ensure rejection_reason is clean
            rejected_samples_data_df['rejection_reason'] = rejected_samples_data_df['rejection_reason'].fillna('Unknown').astype(str).str.strip().replace(['','nan','None'], 'Unknown Reason')
            
            rejection_reason_counts = rejected_samples_data_df['rejection_reason'].value_counts().reset_index()
            rejection_reason_counts.columns = ['Rejection Reason', 'Count']
            testing_insights_output["sample_rejection_reasons_df"] = rejection_reason_counts

            cols_for_rejected_list = ['patient_id', 'test_type', 'encounter_date', 'rejection_reason']
             # Add sample_collection_date if more relevant
            if 'sample_collection_date' in rejected_samples_data_df.columns : cols_for_rejected_list.insert(2,'sample_collection_date')
            
            testing_insights_output["top_rejected_samples_list_df"] = rejected_samples_data_df[
                [col for col in cols_for_rejected_list if col in rejected_samples_data_df.columns] # Ensure only existing columns are selected
            ].head(10) # Top 10 examples for a table display
        else:
            testing_insights_output["data_availability_notes"].append("No rejected samples recorded in this period.")
    else:
        testing_insights_output["data_availability_notes"].append("Sample status or rejection reason data missing for analysis.")

    logger.info("Clinic testing insights data preparation complete.")
    return testing_insights_output

# --- Example Usage (for testing or integration into a reporting script) ---
if __name__ == '__main__':
    logger.setLevel(logging.DEBUG)
    logger.info("Running Clinic Testing Insights Tab component data preparation simulation...")

    # Simulate clinic_service_kpis_summary (especially test_summary_details)
    mock_kpi_summary = {
        'test_summary_details': {
            app_config.KEY_TEST_TYPES_FOR_ANALYSIS["RDT-Malaria"]["display_name"]: { # Malaria RDT
                "positive_rate_perc": 15.5, "avg_tat_days": 0.2, "perc_met_tat_target": 98.0,
                "pending_count_patients": 3, "rejected_count_patients": 1, "total_conclusive_tests": 200
            },
            app_config.KEY_TEST_TYPES_FOR_ANALYSIS["Sputum-GeneXpert"]["display_name"]: { # TB GeneXpert
                "positive_rate_perc": 8.2, "avg_tat_days": 1.5, "perc_met_tat_target": 85.0,
                "pending_count_patients": 5, "rejected_count_patients": 2, "total_conclusive_tests": 50
            }
        }
        # Other summary KPIs can be added if needed for test context
    }

    # Simulate filtered_health_df_clinic_period
    num_test_records = 100
    test_dates = pd.to_datetime(pd.date_range(start='2023-10-01', periods=10, freq='D').to_list() * (num_test_records//10))

    sample_clinic_tests_df = pd.DataFrame({
        'encounter_date': test_dates,
        'sample_collection_date': test_dates - pd.Timedelta(hours=1),
        'patient_id': [f'TP{i%20:03d}' for i in range(num_test_records)],
        'test_type': np.random.choice(list(app_config.KEY_TEST_TYPES_FOR_ANALYSIS.keys()), num_test_records),
        'test_result': np.random.choice(['Positive', 'Negative', 'Pending', 'Rejected Sample'], num_test_records, p=[0.15,0.65,0.15,0.05]),
        'test_turnaround_days': np.random.uniform(0.1, 5.0, num_test_records).round(1),
        'sample_status': ['Accepted'] * (num_test_records - 5) + ['Rejected'] * 5, # some rejections
        'rejection_reason': ['Hemolyzed', 'Insufficient Volume', 'Improper Labeling', 'Clotted', 'Delay in Transit'] + ['Unknown Reason'] * (num_test_records - 10) + ['Hemolyzed'] * 5,
    })
    # Make TAT NaN for pending/rejected
    sample_clinic_tests_df.loc[sample_clinic_tests_df['test_result'].isin(['Pending','Rejected Sample']), 'test_turnaround_days'] = np.nan
    # Assign correct sample_status for rejections
    sample_clinic_tests_df.loc[sample_clinic_tests_df['test_result'] == 'Rejected Sample', 'sample_status'] = 'Rejected'


    reporting_period = "October Week 1, 2023"
    
    # Test for "All Critical Tests Summary"
    insights_critical = prepare_clinic_testing_insights_data(
        filtered_health_df_clinic_period=sample_clinic_tests_df,
        clinic_service_kpis_summary=mock_kpi_summary,
        reporting_period_str=reporting_period,
        selected_test_group_display_name_for_detail="All Critical Tests Summary"
    )
    print(f"\n--- Prepared Clinic Testing Insights (Focus: All Critical Tests) for: {reporting_period} ---")
    if insights_critical["critical_tests_summary_df"] is not None:
        print("\nCritical Tests Summary:")
        print(insights_critical["critical_tests_summary_df"].to_string())
    
    # Test for a specific test group, e.g., Malaria RDT
    malaria_rdt_disp_name = app_config.KEY_TEST_TYPES_FOR_ANALYSIS["RDT-Malaria"]["display_name"]
    insights_malaria = prepare_clinic_testing_insights_data(
        filtered_health_df_clinic_period=sample_clinic_tests_df,
        clinic_service_kpis_summary=mock_kpi_summary,
        reporting_period_str=reporting_period,
        selected_test_group_display_name_for_detail=malaria_rdt_disp_name
    )
    print(f"\n--- Prepared Clinic Testing Insights (Focus: {malaria_rdt_disp_name}) for: {reporting_period} ---")
    if insights_malaria["selected_test_group_metrics"] is not None:
        print("\nSelected Test Group Metrics:")
        for m_key, m_val in insights_malaria["selected_test_group_metrics"].items(): print(f"  {m_key}: {m_val}")
    if insights_malaria["selected_test_group_tat_trend_series"] is not None:
        print("\nTAT Trend (Series head):")
        print(insights_malaria["selected_test_group_tat_trend_series"].head())
    if insights_malaria["selected_test_group_volume_trend_df"] is not None:
        print("\nVolume Trend (DataFrame head):")
        print(insights_malaria["selected_test_group_volume_trend_df"].head())

    # Print common sections for the last run (insights_malaria)
    print("\nOverdue Pending Tests:")
    if insights_malaria["overdue_pending_tests_df"] is not None and not insights_malaria["overdue_pending_tests_df"].empty:
        print(insights_malaria["overdue_pending_tests_df"].head().to_string())
    else: print("  (No overdue pending tests found or data insufficient)")

    print("\nSample Rejection Reasons:")
    if insights_malaria["sample_rejection_reasons_df"] is not None and not insights_malaria["sample_rejection_reasons_df"].empty:
        print(insights_malaria["sample_rejection_reasons_df"].to_string())
    else: print("  (No rejection reason data or no rejections)")
    
    print("\nTop Rejected Samples List:")
    if insights_malaria["top_rejected_samples_list_df"] is not None and not insights_malaria["top_rejected_samples_list_df"].empty:
        print(insights_malaria["top_rejected_samples_list_df"].to_string())
    else: print("  (No rejected samples list)")


    print("\nData Availability Notes (from last run):")
    if insights_malaria["data_availability_notes"]:
        for note in insights_malaria["data_availability_notes"]: print(f"  - {note}")
    else: print("  (No specific data availability notes)")
