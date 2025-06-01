# test/pages/district_components/kpi_display_district.py
# Redesigned for "Sentinel Health Co-Pilot" - LMIC Edge-First System
# This module structures district-wide Key Performance Indicators (KPIs)
# based on aggregated zonal data. The output is intended for:
#   1. Display on a web dashboard/report for District Health Officers (DHOs)
#      at a Facility Node (Tier 2) or Cloud Node (Tier 3).
#   2. Providing a high-level strategic overview of district health status.

import pandas as pd # For pd.notna
import numpy as np
import logging
# Assuming app_config is in the PYTHONPATH or project root.
from config import app_config # Uses new, redesigned app_config
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

def structure_district_kpis_data(
    district_overall_kpis_summary: Dict[str, Any], # Output from core_data_processing.get_district_summary_kpis
    district_enriched_gdf: Optional[pd.DataFrame] = None, # Optional, for context like total zones
    reporting_period_str: Optional[str] = "Latest Aggregated Data"
) -> List[Dict[str, Any]]:
    """
    Structures district-wide KPIs from a summary dictionary into a list of
    dictionaries, each representing a KPI ready for reporting.

    Args:
        district_overall_kpis_summary: Dict containing pre-calculated district KPIs.
            Expected keys like: 'population_weighted_avg_ai_risk_score', 'district_avg_facility_coverage_score',
                               'zones_meeting_high_risk_criteria_count', 'district_overall_key_disease_prevalence_per_1000',
                               'district_total_active_tb_cases', 'district_total_active_malaria_cases',
                               'district_population_weighted_avg_steps', 'district_avg_clinic_co2_ppm'.
                               (Names based on output of updated get_district_summary_kpis)
        district_enriched_gdf: Optional GeoDataFrame of enriched zones, used here to get total zone count.
        reporting_period_str: String describing the reporting period/data context.

    Returns:
        List[Dict[str, Any]]: A list of KPI dictionaries.
        Example:
        {
            "metric_code": "DISTRICT_AVG_POP_RISK", "title": "Avg. Population AI Risk",
            "value_str": "65.2", "units": "score", "icon": "🎯",
            "status_level": "MODERATE_RISK",
            "help_text": "Population-weighted average AI risk score across all zones."
        }
    """
    logger.info(f"Structuring district-wide KPIs for period: {reporting_period_str}")
    district_kpis_structured: List[Dict[str, Any]] = []

    if not district_overall_kpis_summary:
        logger.warning("No district overall KPI summary data provided.")
        return district_kpis_structured

    # Helper to determine total number of zones for percentage calculations
    total_zones_in_district = 0
    if district_enriched_gdf is not None and not district_enriched_gdf.empty and 'zone_id' in district_enriched_gdf.columns:
        total_zones_in_district = district_enriched_gdf['zone_id'].nunique()
    elif district_enriched_gdf is not None and not district_enriched_gdf.empty: # Fallback if no zone_id but GDF exists
        total_zones_in_district = len(district_enriched_gdf)


    # KPI 1: Average Population AI Risk
    avg_pop_risk = district_overall_kpis_summary.get('population_weighted_avg_ai_risk_score', np.nan)
    pop_risk_status = "NO_DATA"
    if pd.notna(avg_pop_risk):
        if avg_pop_risk >= app_config.RISK_SCORE_HIGH_THRESHOLD: pop_risk_status = "HIGH_RISK"
        elif avg_pop_risk >= app_config.RISK_SCORE_MODERATE_THRESHOLD: pop_risk_status = "MODERATE_RISK"
        else: pop_risk_status = "LOW_RISK"
    district_kpis_structured.append({
        "metric_code": "DISTRICT_AVG_POP_RISK", "title": "Avg. Population AI Risk",
        "value_str": f"{avg_pop_risk:.1f}" if pd.notna(avg_pop_risk) else "N/A", "units": "score", "icon": "🎯",
        "status_level": pop_risk_status,
        "help_text": "Population-weighted average AI risk score across all zones."
    })

    # KPI 2: District Facility Coverage Score
    facility_coverage = district_overall_kpis_summary.get('district_avg_facility_coverage_score', np.nan)
    facility_cov_status = "NO_DATA"
    if pd.notna(facility_coverage):
        # LMIC target setting: e.g. >70% = GOOD, 50-70% = MODERATE, <50% = POOR
        if facility_coverage >= 70: facility_cov_status = "GOOD_PERFORMANCE"
        elif facility_coverage >= app_config.DISTRICT_INTERVENTION_FACILITY_COVERAGE_LOW_PCT: facility_cov_status = "MODERATE_CONCERN"
        else: facility_cov_status = "HIGH_CONCERN"
    district_kpis_structured.append({
        "metric_code": "DISTRICT_FACILITY_COVERAGE", "title": "Facility Coverage Score (Avg.)",
        "value_str": f"{facility_coverage:.1f}" if pd.notna(facility_coverage) else "N/A", "units": "%", "icon": "🏥",
        "status_level": facility_cov_status,
        "help_text": "Population-weighted average score reflecting access/capacity of health facilities."
    })

    # KPI 3: High AI Risk Zones
    high_risk_zones_count = district_overall_kpis_summary.get('zones_meeting_high_risk_criteria_count', 0)
    perc_high_risk_zones_str = "N/A"
    high_risk_zones_status = "ACCEPTABLE"
    if total_zones_in_district > 0 and pd.notna(high_risk_zones_count):
        perc_val = (high_risk_zones_count / total_zones_in_district) * 100
        perc_high_risk_zones_str = f"{perc_val:.0f}%"
        if perc_val > 30: high_risk_zones_status = "HIGH_CONCERN" # e.g. >30% of zones are high risk
        elif perc_val > 10: high_risk_zones_status = "MODERATE_CONCERN"
    elif pd.notna(high_risk_zones_count) and high_risk_zones_count > 0: # Have count but no total zones
         perc_high_risk_zones_str = "(% N/A)"
         if high_risk_zones_count > 2: high_risk_zones_status = "HIGH_CONCERN" # More than 2 high risk zones absolute
         elif high_risk_zones_count > 0: high_risk_zones_status = "MODERATE_CONCERN"

    district_kpis_structured.append({
        "metric_code": "HIGH_RISK_ZONES_COUNT", "title": "High AI Risk Zones",
        "value_str": f"{int(high_risk_zones_count) if pd.notna(high_risk_zones_count) else '0'} ({perc_high_risk_zones_str})", "units": "zones", "icon": "⚠️",
        "status_level": high_risk_zones_status,
        "help_text": f"Number (& %) of zones with average AI risk score >= {app_config.DISTRICT_ZONE_HIGH_RISK_AVG_SCORE}."
    })

    # KPI 4: Overall Key Disease Prevalence
    district_prevalence = district_overall_kpis_summary.get('district_overall_key_disease_prevalence_per_1000', np.nan)
    prevalence_status = "NO_DATA"
    # Example thresholds for prevalence in LMIC
    if pd.notna(district_prevalence):
        if district_prevalence > 75: prevalence_status = "HIGH_CONCERN" # e.g., 7.5% of pop with key disease
        elif district_prevalence > 30: prevalence_status = "MODERATE_CONCERN"
        else: prevalence_status = "ACCEPTABLE"
    district_kpis_structured.append({
        "metric_code": "DISTRICT_KEY_DISEASE_PREVALENCE", "title": "Key Disease Prevalence",
        "value_str": f"{district_prevalence:.1f}" if pd.notna(district_prevalence) else "N/A", "units": "/1k pop", "icon": "📈",
        "status_level": prevalence_status,
        "help_text": f"Combined prevalence of key infectious diseases (from '{', '.join(app_config.KEY_CONDITIONS_FOR_ACTION[:3])}...') per 1,000 population."
    })

    # KPIs for Key Disease Burdens
    tb_total = district_overall_kpis_summary.get('district_total_active_tb_cases', 0)
    tb_status = "ACCEPTABLE"
    if pd.notna(tb_total): # Threshold may depend on district size or expected burden
      if total_zones_in_district > 0 and tb_total > (total_zones_in_district * app_config.DISTRICT_INTERVENTION_TB_BURDEN_HIGH_ABS): tb_status = "HIGH_CONCERN"
      elif tb_total > app_config.DISTRICT_INTERVENTION_TB_BURDEN_HIGH_ABS * 0.5: tb_status = "MODERATE_CONCERN" # If more than half of one zone's high threshold
    district_kpis_structured.append({
        "metric_code": "DISTRICT_TOTAL_TB_CASES", "title": "Total Active TB Cases",
        "value_str": str(int(tb_total)) if pd.notna(tb_total) else "N/A", "units": "cases", "icon": "🫁",
        "status_level": tb_status, "help_text": "Total active TB cases reported across the district."
    })
    
    # Similar for Malaria, etc. - can loop through app_config.KEY_CONDITIONS_FOR_ACTION if desired
    malaria_total = district_overall_kpis_summary.get('district_total_active_malaria_cases', 0) # Assuming key in summary matches
    malaria_status = "ACCEPTABLE"
    if pd.notna(malaria_total): # Similar logic to TB for status
        if malaria_total > 50: malaria_status = "HIGH_CONCERN" # Example absolute number
        elif malaria_total > 20: malaria_status = "MODERATE_CONCERN"
    district_kpis_structured.append({
        "metric_code": "DISTRICT_TOTAL_MALARIA_CASES", "title": "Total Active Malaria Cases",
        "value_str": str(int(malaria_total)) if pd.notna(malaria_total) else "N/A", "units": "cases", "icon": "🦟",
        "status_level": malaria_status, "help_text": "Total active Malaria cases reported across the district."
    })

    # KPI: Average Patient Steps (District Pop-Weighted - Worker/Community Wellness Proxy)
    avg_steps_dist = district_overall_kpis_summary.get('district_population_weighted_avg_steps', np.nan)
    steps_status = "NO_DATA"
    if pd.notna(avg_steps_dist):
        if avg_steps_dist >= app_config.TARGET_DAILY_STEPS * 0.9: steps_status = "GOOD_PERFORMANCE"
        elif avg_steps_dist >= app_config.TARGET_DAILY_STEPS * 0.6: steps_status = "MODERATE_CONCERN"
        else: steps_status = "HIGH_CONCERN" # Low activity level
    district_kpis_structured.append({
        "metric_code": "DISTRICT_AVG_PATIENT_STEPS", "title": "Avg. Patient Steps (Pop. Weighted)",
        "value_str": f"{avg_steps_dist:,.0f}" if pd.notna(avg_steps_dist) else "N/A", "units": "steps/day", "icon": "👣",
        "status_level": steps_status,
        "help_text": f"Population-weighted average daily steps from patient data. Target: >{app_config.TARGET_DAILY_STEPS*0.9:,.0f} steps."
    })

    # KPI: Average Clinic CO2 (District Average of Zonal Averages)
    avg_co2_district = district_overall_kpis_summary.get('district_avg_clinic_co2_ppm', np.nan)
    co2_status_dist = "NO_DATA"
    if pd.notna(avg_co2_district):
        if avg_co2_district > app_config.ALERT_AMBIENT_CO2_VERY_HIGH_PPM: co2_status_dist = "HIGH_RISK"
        elif avg_co2_district > app_config.ALERT_AMBIENT_CO2_HIGH_PPM: co2_status_dist = "MODERATE_RISK"
        else: co2_status_dist = "ACCEPTABLE"
    district_kpis_structured.append({
        "metric_code": "DISTRICT_AVG_CLINIC_CO2", "title": "Avg. Clinic CO2 (District)",
        "value_str": f"{avg_co2_district:.0f}" if pd.notna(avg_co2_district) else "N/A", "units": "ppm", "icon": "💨",
        "status_level": co2_status_dist,
        "help_text": f"District average of zonal mean CO2 levels in clinics. Aim for <{app_config.ALERT_AMBIENT_CO2_HIGH_PPM}ppm."
    })

    logger.info(f"Structured district KPIs: Count = {len(district_kpis_structured)}")
    return district_kpis_structured


# --- Example Usage (for testing or integration into a DHO reporting script) ---
if __name__ == '__main__':
    logger.setLevel(logging.DEBUG)
    logger.info("Running District KPI Display component logic simulation directly...")

    # Simulate the output of core_data_processing.get_district_summary_kpis
    mock_district_summary = {
        'population_weighted_avg_ai_risk_score': 68.5,
        'district_avg_facility_coverage_score': 65.2,
        'zones_meeting_high_risk_criteria_count': 3,
        'district_overall_key_disease_prevalence_per_1000': 45.1,
        'district_total_active_tb_cases': 25,
        'district_total_active_malaria_cases': 60,
        'district_population_weighted_avg_steps': 5800.0,
        'district_avg_clinic_co2_ppm': 1250.0
    }
    # Simulate a GDF for total_zones context
    mock_gdf = pd.DataFrame({'zone_id': ['Z1', 'Z2', 'Z3', 'Z4', 'Z5', 'Z6', 'Z7', 'Z8', 'Z9', 'Z10']})

    reporting_period = "Q3 2023 District Summary"

    structured_kpis = structure_district_kpis_data(
        district_overall_kpis_summary=mock_district_summary,
        district_enriched_gdf=mock_gdf, # Pass the mock GDF
        reporting_period_str=reporting_period
    )

    print(f"\n--- Structured District KPIs for: {reporting_period} ---")
    if structured_kpis:
        for kpi_item in structured_kpis:
            print(
                f"  Title: {kpi_item['title']} | Value: {kpi_item['value_str']} {kpi_item.get('units','')} | "
                f"Status: {kpi_item['status_level']} | Code: {kpi_item['metric_code']}"
            )
    else:
        print("  No district KPIs structured.")
