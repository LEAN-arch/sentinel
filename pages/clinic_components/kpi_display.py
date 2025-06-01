# test/pages/clinic_components/kpi_display.py
# Redesigned for "Sentinel Health Co-Pilot" - LMIC Edge-First System
# This module calculates and structures key clinic performance and disease-specific
# KPIs based on summarized clinic service data.
# The output is primarily for:
#   1. Display on a simplified web dashboard/report for clinic managers at a Facility Node (Tier 2).
#   2. Informing higher-level summaries for DHOs.

import pandas as pd
import numpy as np
import logging
# Assuming app_config is in the PYTHONPATH or project root.
from config import app_config # Uses new, redesigned app_config
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

def structure_main_clinic_kpis_data(
    clinic_service_kpis_summary: Dict[str, Any], # Output from core_data_processing.get_clinic_summary
    reporting_period_str: str
) -> List[Dict[str, Any]]:
    """
    Structures main clinic performance KPIs from a summary dictionary into a list
    of dictionaries, each representing a KPI ready for display or reporting.

    Args:
        clinic_service_kpis_summary: A dictionary containing pre-calculated clinic service KPIs.
            Expected keys (from get_clinic_summary): 'overall_avg_test_turnaround_conclusive_days',
            'perc_critical_tests_tat_met', 'total_pending_critical_tests_patients',
            'sample_rejection_rate_perc'.
        reporting_period_str: String describing the reporting period for context.

    Returns:
        List[Dict[str, Any]]: A list of KPI dictionaries. Example:
        {
            "title": "Overall Avg. TAT", "value_str": "2.1d", "icon": "⏱️",
            "status_level": "ACCEPTABLE", "units": "days",
            "help_text": "Average TAT. Target: <=2 days.", "metric_code": "AVG_TAT_OVERALL"
        }
    """
    logger.info(f"Structuring main clinic KPIs for period: {reporting_period_str}")
    main_kpis_structured: List[Dict[str, Any]] = []

    if not clinic_service_kpis_summary:
        logger.warning("No clinic service KPI summary data provided to structure_main_clinic_kpis_data.")
        return main_kpis_structured

    # 1. Overall Average Test Turnaround Time (TAT)
    overall_tat_val = clinic_service_kpis_summary.get('overall_avg_test_turnaround_conclusive_days', np.nan)
    tat_status = "NO_DATA"
    if pd.notna(overall_tat_val):
        # Using general TARGET_TEST_TURNAROUND_DAYS from new app_config if not more specific target exists
        general_target_tat = app_config.TARGET_TEST_TURNAROUND_DAYS # Default from old config, should be updated in new app_config if this specific KPI needed
        # Let's assume a more generic target for TAT if not critically specified
        # For LMIC, higher TAT might be "acceptable" if critical tests are faster.
        # Example: TAT > 5 days = HIGH_CONCERN, > 3 = MODERATE_CONCERN
        if overall_tat_val > (general_target_tat + 2): tat_status = "HIGH_CONCERN"
        elif overall_tat_val > general_target_tat: tat_status = "MODERATE_CONCERN"
        else: tat_status = "ACCEPTABLE"
    main_kpis_structured.append({
        "metric_code": "AVG_TAT_OVERALL", "title": "Overall Avg. TAT (Conclusive)",
        "value_str": f"{overall_tat_val:.1f}" if pd.notna(overall_tat_val) else "N/A", "units": "days",
        "icon": "⏱️", "status_level": tat_status,
        "help_text": f"Avg. Turnaround Time for all conclusive tests. Target reference: approx {general_target_tat} days."
    })

    # 2. Percentage of Critical Tests Meeting TAT
    perc_met_tat_val = clinic_service_kpis_summary.get('perc_critical_tests_tat_met', 0.0)
    crit_tat_status = "NO_DATA"
    target_overall_crit_tat_pct = app_config.TARGET_OVERALL_TESTS_MEETING_TAT_PCT_FACILITY # From new config
    if pd.notna(perc_met_tat_val):
        if perc_met_tat_val >= target_overall_crit_tat_pct: crit_tat_status = "GOOD_PERFORMANCE" # Explicitly good
        elif perc_met_tat_val >= target_overall_crit_tat_pct * 0.8: crit_tat_status = "MODERATE_CONCERN" # Needs improvement
        else: crit_tat_status = "HIGH_CONCERN" # Significant underperformance
    main_kpis_structured.append({
        "metric_code": "PERC_CRITICAL_TAT_MET", "title": "% Critical Tests TAT Met",
        "value_str": f"{perc_met_tat_val:.1f}" if pd.notna(perc_met_tat_val) else "N/A", "units": "%",
        "icon": "🎯", "status_level": crit_tat_status,
        "help_text": f"Percentage of critical diagnostic tests meeting their defined TAT targets. Target: ≥{target_overall_crit_tat_pct}%."
    })

    # 3. Total Pending Critical Tests (Unique Patients)
    pending_crit_val = clinic_service_kpis_summary.get('total_pending_critical_tests_patients', 0)
    pending_status = "ACCEPTABLE" # Default if 0
    # Define thresholds based on context (e.g., clinic capacity)
    # Example: For a small clinic, >5 might be moderate, >10 high.
    if pd.notna(pending_crit_val):
        if pending_crit_val > 10: pending_status = "HIGH_CONCERN"
        elif pending_crit_val > 3: pending_status = "MODERATE_CONCERN"
    main_kpis_structured.append({
        "metric_code": "PENDING_CRITICAL_TESTS_PATIENTS", "title": "Pending Critical Tests (Patients)",
        "value_str": str(int(pending_crit_val)) if pd.notna(pending_crit_val) else "N/A", "units": "patients",
        "icon": "⏳", "status_level": pending_status,
        "help_text": "Number of unique patients with critical tests results still pending. Aim for minimal backlog."
    })

    # 4. Sample Rejection Rate
    rejection_rate_val = clinic_service_kpis_summary.get('sample_rejection_rate_perc', 0.0)
    rejection_status = "NO_DATA"
    target_rejection_rate_pct = app_config.TARGET_SAMPLE_REJECTION_RATE_PCT_FACILITY # From new config
    if pd.notna(rejection_rate_val):
        if rejection_rate_val > target_rejection_rate_pct * 1.5 : rejection_status = "HIGH_CONCERN" # Significantly above target
        elif rejection_rate_val > target_rejection_rate_pct: rejection_status = "MODERATE_CONCERN"
        else: rejection_status = "GOOD_PERFORMANCE"
    main_kpis_structured.append({
        "metric_code": "SAMPLE_REJECTION_RATE", "title": "Sample Rejection Rate",
        "value_str": f"{rejection_rate_val:.1f}" if pd.notna(rejection_rate_val) else "N/A", "units":"%",
        "icon": "🚫", "status_level": rejection_status,
        "help_text": f"Overall rate of samples rejected by the lab. Target: <{target_rejection_rate_pct}%."
    })
    
    logger.info(f"Structured main clinic KPIs: {main_kpis_structured}")
    return main_kpis_structured


def structure_disease_specific_kpis_data(
    clinic_service_kpis_summary: Dict[str, Any], # Output from core_data_processing.get_clinic_summary
    reporting_period_str: str
) -> List[Dict[str, Any]]:
    """
    Structures disease-specific KPIs (like test positivity) and key drug stockouts
    into a list of dictionaries for reporting.

    Args:
        clinic_service_kpis_summary: A dictionary from get_clinic_summary.
            Must contain 'test_summary_details' and 'key_drug_stockouts_count'.
        reporting_period_str: String describing the reporting period.

    Returns:
        List[Dict[str, Any]]: List of structured KPI dictionaries.
    """
    logger.info(f"Structuring disease-specific and supply KPIs for period: {reporting_period_str}")
    disease_kpis_structured: List[Dict[str, Any]] = []

    if not clinic_service_kpis_summary:
        logger.warning("No clinic service KPI summary data for disease-specific KPIs.")
        return disease_kpis_structured

    test_details_summary = clinic_service_kpis_summary.get("test_summary_details", {})
    if not test_details_summary:
        logger.warning("No 'test_summary_details' found in KPI summary for disease-specific KPIs.")
    
    # Key tests to highlight for positivity rates (using their display names as in test_summary_details)
    # These map to new app_config settings. Icons are LMIC-relevant.
    tests_to_highlight_map = {
        app_config.KEY_TEST_TYPES_FOR_ANALYSIS.get("Sputum-GeneXpert",{}).get("display_name", "TB GeneXpert"): {"icon": "🫁", "target_positive_rate_max_percent": 15.0, "metric_code": "POS_TB_GENEXPERT"}, # Example target
        app_config.KEY_TEST_TYPES_FOR_ANALYSIS.get("RDT-Malaria",{}).get("display_name", "Malaria RDT"): {"icon": "🦟", "target_positive_rate_max_percent": app_config.TARGET_MALARIA_POSITIVITY_RATE, "metric_code": "POS_MALARIA_RDT"}, # From config
        app_config.KEY_TEST_TYPES_FOR_ANALYSIS.get("HIV-Rapid",{}).get("display_name", "HIV Rapid Test"): {"icon": "🩸", "target_positive_rate_max_percent": 5.0, "metric_code": "POS_HIV_RAPID"} # Example national target might be lower
    }

    for test_display_name, props in tests_to_highlight_map.items():
        test_stats = test_details_summary.get(test_display_name, {})
        pos_rate = test_stats.get("positive_rate_perc", np.nan) # Assuming this key from get_clinic_summary
        
        pos_status = "NO_DATA"
        if pd.notna(pos_rate):
            target_max_pos = props.get("target_positive_rate_max_percent", 10.0) # Default target if not in map
            if pos_rate > target_max_pos * 1.25: pos_status = "HIGH_CONCERN" # Significantly above target
            elif pos_rate > target_max_pos : pos_status = "MODERATE_CONCERN"
            else: pos_status = "ACCEPTABLE" # Within or below typical/target range

        disease_kpis_structured.append({
            "metric_code": props["metric_code"], "title": f"{test_display_name} Positivity",
            "value_str": f"{pos_rate:.1f}" if pd.notna(pos_rate) else "N/A", "units":"%",
            "icon": props["icon"], "status_level": pos_status,
            "help_text": f"Positivity rate for {test_display_name}. Target: generally <{props.get('target_positive_rate_max_percent', 10)}% (varies by context)."
        })

    # Key Drug Stockouts
    drug_stockouts_count = clinic_service_kpis_summary.get('key_drug_stockouts_count', 0)
    stockout_status = "GOOD_PERFORMANCE" # No stockouts
    if pd.notna(drug_stockouts_count) and drug_stockouts_count > 0:
        stockout_status = "HIGH_CONCERN" if drug_stockouts_count > 2 else "MODERATE_CONCERN" # Example: >2 key drugs = high concern
    
    disease_kpis_structured.append({
        "metric_code": "KEY_DRUG_STOCKOUTS", "title": "Key Drug Stockouts",
        "value_str": str(int(drug_stockouts_count)) if pd.notna(drug_stockouts_count) else "N/A", "units":"items",
        "icon": "💊", "status_level": stockout_status,
        "help_text": f"Number of key drugs with < {app_config.CRITICAL_SUPPLY_DAYS_REMAINING} days supply. Aim for 0."
    })

    logger.info(f"Structured disease-specific KPIs: {disease_kpis_structured}")
    return disease_kpis_structured


# --- Example Usage (for testing or integration into a reporting script) ---
if __name__ == '__main__':
    logger.setLevel(logging.DEBUG)
    logger.info("Running Clinic KPI Display component logic simulation directly...")

    # Simulate the output of core_data_processing.get_clinic_summary
    mock_clinic_summary = {
        'overall_avg_test_turnaround_conclusive_days': 2.8,
        'perc_critical_tests_tat_met': 75.5,
        'total_pending_critical_tests_patients': 7,
        'sample_rejection_rate_perc': 6.1,
        'key_drug_stockouts_count': 1,
        'test_summary_details': {
            app_config.KEY_TEST_TYPES_FOR_ANALYSIS.get("Sputum-GeneXpert",{}).get("display_name", "TB GeneXpert"): {
                "positive_rate_perc": 12.5, "avg_tat_days": 1.2, "perc_met_tat_target": 95.0,
                "pending_count_patients": 2, "rejected_count_patients": 1, "total_conclusive_tests": 150
            },
            app_config.KEY_TEST_TYPES_FOR_ANALYSIS.get("RDT-Malaria",{}).get("display_name", "Malaria RDT"): {
                "positive_rate_perc": app_config.TARGET_MALARIA_POSITIVITY_RATE + 2.0, # Slightly above target
                 "avg_tat_days": 0.3, "perc_met_tat_target": 99.0,
                "pending_count_patients": 5, "rejected_count_patients": 3, "total_conclusive_tests": 500
            },
            app_config.KEY_TEST_TYPES_FOR_ANALYSIS.get("HIV-Rapid",{}).get("display_name", "HIV Rapid Test"): {
                "positive_rate_perc": 1.8, "avg_tat_days": 0.1, "perc_met_tat_target": 100.0,
                "pending_count_patients": 1, "rejected_count_patients": 0, "total_conclusive_tests": 300
            },
             # Add another non-highlighted test for completeness of test_summary_details
            "Glucose Test": {
                "positive_rate_perc": 22.0, "avg_tat_days": 0.1, "perc_met_tat_target": 100.0,
                "pending_count_patients": 0, "rejected_count_patients": 0, "total_conclusive_tests": 250
            }
        }
    }
    reporting_period = "Last 7 Days (Oct 1-7, 2023)"

    main_kpis = structure_main_clinic_kpis_data(mock_clinic_summary, reporting_period)
    print("\n--- Structured Main Clinic KPIs: ---")
    for kpi in main_kpis:
        print(kpi)

    disease_kpis = structure_disease_specific_kpis_data(mock_clinic_summary, reporting_period)
    print("\n--- Structured Disease-Specific & Supply KPIs: ---")
    for kpi in disease_kpis:
        print(kpi)

    # Test with empty summary
    empty_main_kpis = structure_main_clinic_kpis_data({}, "Empty Period")
    print(f"\n--- Main KPIs from Empty Summary: {empty_main_kpis} ---")
    empty_disease_kpis = structure_disease_specific_kpis_data({}, "Empty Period")
    print(f"--- Disease KPIs from Empty Summary: {empty_disease_kpis} ---")
