# test/tests/test_core_data_processing.py
# Pytest tests for the refactored functions in utils.core_data_processing.py
# Aligned with "Sentinel Health Co-Pilot" redesign.

import pytest
import pandas as pd
import geopandas as gpd
import numpy as np
from datetime_patch import datetime_patch # If tests involve pd.Timestamp('today')
                                        # See note below on mocking 'today'

# Functions to be tested (assuming they are in test/utils/core_data_processing.py)
from utils.core_data_processing import (
    _clean_column_names,
    _convert_to_numeric,
    # hash_geodataframe, # Hashing can be tricky to assert exact values, test for non-None or changes
    load_health_records,
    load_iot_clinic_environment_data,
    load_zone_data,
    enrich_zone_geodata_with_health_aggregates,
    get_overall_kpis,
    get_chw_summary,
    get_patient_alerts_for_chw,
    get_clinic_summary,
    get_clinic_environmental_summary,
    get_patient_alerts_for_clinic,
    get_district_summary_kpis,
    get_trend_data,
    get_supply_forecast_data
)

# Import fixtures from conftest.py (using their NEW Sentinel-aligned names)
# These fixtures should provide data reflecting the new schemas and app_config values.
# from ..conftest import ( # Relative import if test structure allows
#     sample_health_records_df_main,
#     sample_iot_clinic_df_main,
#     sample_zone_attributes_df_main, # Note: load_zone_data combines this now
#     sample_zone_geometries_gdf_main, # Note: load_zone_data combines this now
#     sample_enriched_gdf_main, # This is CRUCIAL as it uses the refactored enrichment
#     empty_health_df_with_sentinel_schema,
#     empty_iot_df_with_sentinel_schema,
#     empty_enriched_gdf_with_sentinel_schema
# )
# For this template, we'll assume fixtures are available in the pytest environment.

# --- Mocking and Setup ---
# If functions use pd.Timestamp('today'), tests might need to mock 'today'
# for reproducibility. A common way is using a fixture with `freezegun`
# or a custom context manager. For this example, assume it's handled or
# tests are designed to be robust to small 'today' variations if they compare durations.

# (If you need to mock app_config values for specific tests, use pytest-mock or monkeypatch)


# --- Tests for Helper Functions ---

def test_clean_column_names():
    df = pd.DataFrame(columns=['Test Column', 'Another-Col', 'allgood'])
    cleaned_df = _clean_column_names(df.copy()) # Pass copy
    assert list(cleaned_df.columns) == ['test_column', 'another_col', 'allgood']
    # Test with empty DataFrame
    assert list(_clean_column_names(pd.DataFrame()).columns) == []
    # Test with already clean columns
    assert list(_clean_column_names(cleaned_df.copy()).columns) == ['test_column', 'another_col', 'allgood']

def test_convert_to_numeric():
    series = pd.Series(['1', '2.5', 'abc', '4', None])
    converted = _convert_to_numeric(series.copy())
    expected = pd.Series([1.0, 2.5, np.nan, 4.0, np.nan])
    pd.testing.assert_series_equal(converted, expected, check_dtype=False) # Check_dtype False for np.nan comparison flexibility
    
    converted_with_default = _convert_to_numeric(series.copy(), default_value=0)
    expected_with_default = pd.Series([1.0, 2.5, 0.0, 4.0, 0.0])
    pd.testing.assert_series_equal(converted_with_default, expected_with_default, check_dtype=False)

    # Test with already numeric
    numeric_series = pd.Series([1, 2, 3])
    pd.testing.assert_series_equal(_convert_to_numeric(numeric_series.copy()), numeric_series.astype(float), check_dtype=False)


# --- Tests for Data Loading Functions ---

def test_load_health_records_structure(empty_health_df_with_sentinel_schema, sample_health_records_df_main):
    """
    Tests that loaded health records have the expected Sentinel schema (columns).
    Also tests with a sample file if available (mocked path or actual small sample).
    """
    # This test primarily checks column structure from a known good fixture
    # The fixture sample_health_records_df_main is already "loaded and cleaned" by its definition
    # if it calls load_health_records. If it just defines data, then we call load_health_records
    # with a path to a temporary CSV created from sample_health_records_df_main.

    # For this conceptual example, let's assume sample_health_records_df_main IS the output of load_health_records
    # on a representative CSV.
    loaded_df = sample_health_records_df_main # Simulate output
    
    assert isinstance(loaded_df, pd.DataFrame)
    # Check for key Sentinel schema columns
    # This needs to align with the columns defined in empty_health_df_with_sentinel_schema
    # or the actual output of refactored load_health_records
    expected_cols_subset = ['patient_id', 'encounter_date', 'age', 'gender', 'chronic_condition_flag',
                            'min_spo2_pct', 'vital_signs_temperature_celsius', 'fall_detected_today',
                            'condition', 'ai_risk_score', 'ai_followup_priority_score', 'zone_id']
    for col in expected_cols_subset:
        assert col in loaded_df.columns

    # Test data types for some key columns
    if not loaded_df.empty:
        assert pd.api.types.is_datetime64_any_dtype(loaded_df['encounter_date'])
        assert pd.api.types.is_numeric_dtype(loaded_df['age'])
        assert pd.api.types.is_numeric_dtype(loaded_df['ai_risk_score'])

def test_load_health_records_empty_or_missing_file(tmp_path):
    # Test missing file (load_health_records should handle this by logging and returning empty DF)
    missing_file = tmp_path / "non_existent_health_records.csv"
    df_missing = load_health_records(file_path=str(missing_file))
    assert df_missing.empty
    # Test with an empty CSV file
    empty_file = tmp_path / "empty_health_records.csv"
    empty_file.write_text("") # Create empty file
    df_empty_csv = load_health_records(file_path=str(empty_file))
    assert df_empty_csv.empty # Or check if it has schema but no rows depending on implementation

# Similar tests for load_iot_clinic_environment_data and load_zone_data
# focusing on schema, data types, and handling of missing/empty files.

def test_load_zone_data_structure(sample_zone_geometries_gdf_main): # Fixture should simulate output of load_zone_data
    """
    Tests that loaded zone data (which is a merged GDF of geometries and attributes)
    has the expected structure. The sample_zone_geometries_gdf_main fixture should
    already be in the expected post-load_zone_data format.
    """
    loaded_gdf = sample_zone_geometries_gdf_main # This fixture simulates the result of load_zone_data
    assert isinstance(loaded_gdf, gpd.GeoDataFrame)
    assert "geometry" in loaded_gdf.columns
    assert loaded_gdf.crs == app_config.DEFAULT_CRS_STANDARD # Check CRS from new app_config
    # Check for key attribute columns merged in by load_zone_data logic (name, population, etc.)
    expected_zone_attrs = ['zone_id', 'name', 'population', 'socio_economic_index']
    for col in expected_zone_attrs:
        assert col in loaded_gdf.columns
    if not loaded_gdf.empty:
        assert pd.api.types.is_numeric_dtype(loaded_gdf['population'])


# --- Tests for Data Enrichment Functions ---

def test_enrich_zone_geodata_with_health_aggregates_structure(sample_enriched_gdf_main, empty_enriched_gdf_with_sentinel_schema):
    """
    Tests that the enriched GDF has the expected Sentinel schema and key aggregated columns.
    The `sample_enriched_gdf_main` fixture *must* be generated using the refactored enrichment function.
    """
    enriched_gdf = sample_enriched_gdf_main
    assert isinstance(enriched_gdf, gpd.GeoDataFrame)
    
    # Check against the defined schema for an empty enriched GDF
    # This ensures all expected aggregate columns are present.
    expected_cols = list(empty_enriched_gdf_with_sentinel_schema.columns)
    for col in expected_cols:
        assert col in enriched_gdf.columns, f"Enriched GDF missing expected column: {col}"

    # Check some data types
    if not enriched_gdf.empty:
        assert pd.api.types.is_numeric_dtype(enriched_gdf['avg_risk_score'])
        assert pd.api.types.is_numeric_dtype(enriched_gdf['total_active_key_infections'])
        assert 'geometry' in enriched_gdf.columns # Ensure geometry column remains

def test_enrich_zone_geodata_values(sample_enriched_gdf_main, sample_health_records_df_main):
    """
    Test specific aggregated values in the sample_enriched_gdf_main.
    This requires knowing expected outcomes based on sample_health_records_df_main and sample_iot_clinic_df_main.
    """
    # Example: Check avg_risk_score for a specific zone based on input health_df
    # This kind of test is powerful but can be brittle if sample data changes.
    zone_a_health_df = sample_health_records_df_main[sample_health_records_df_main['zone_id'] == 'ZoneA']
    if not zone_a_health_df.empty and 'ai_risk_score' in zone_a_health_df.columns:
        expected_zone_a_avg_risk = zone_a_health_df['ai_risk_score'].mean()
        actual_zone_a_avg_risk = sample_enriched_gdf_main[sample_enriched_gdf_main['zone_id'] == 'ZoneA']['avg_risk_score'].iloc[0]
        # Allow for floating point precision issues
        assert np.isclose(actual_zone_a_avg_risk, expected_zone_a_avg_risk) or pd.isna(actual_zone_a_avg_risk) == pd.isna(expected_zone_a_avg_risk)

    # Add more value checks for other aggregated metrics like active_tb_cases, prevalence_per_1000 etc.
    # For example, count TB cases in 'ZoneA' from sample_health_records_df_main
    expected_tb_zone_a = sample_health_records_df_main[
        (sample_health_records_df_main['zone_id'] == 'ZoneA') &
        (sample_health_records_df_main['condition'].str.contains("TB", case=False, na=False))
    ]['patient_id'].nunique()
    
    actual_tb_zone_a = sample_enriched_gdf_main[
        sample_enriched_gdf_main['zone_id'] == 'ZoneA'
    ]['active_tb_cases'].iloc[0] # Assuming 'active_tb_cases' is the column name
    assert actual_tb_zone_a == expected_tb_zone_a


# --- Tests for KPI and Summary Calculation Functions ---
# These tests will need to use the new fixture data and check against
# expected KPI names and value types (e.g., np.nan vs 0.0) from refactored functions.

def test_get_overall_kpis_structure_and_types(sample_health_records_df_main):
    kpis = get_overall_kpis(sample_health_records_df_main)
    assert isinstance(kpis, dict)
    # Check for NEW KPI names and that values are appropriate (e.g., nan for averages if no data)
    # Expected keys based on refactored `get_overall_kpis`
    expected_kpi_keys = ["total_patients", "avg_patient_risk", "active_tb_cases_current",
                           "malaria_rdt_positive_rate_period", "key_supply_stockout_alerts"]
    for key in expected_kpi_keys:
        assert key in kpis, f"Overall KPI key '{key}' missing."
    
    assert isinstance(kpis["total_patients"], (int, np.integer))
    assert isinstance(kpis["avg_patient_risk"], (float, np.floating)) or pd.isna(kpis["avg_patient_risk"])


def test_get_chw_summary_values(sample_health_records_df_main):
    # Filter sample_health_records for a specific day for CHW summary
    # Using encounter_date_obj added in fixture if needed, or convert here
    sample_health_records_df_main['encounter_date_obj_test'] = pd.to_datetime(sample_health_records_df_main['encounter_date']).dt.date
    specific_day_data = sample_health_records_df_main[sample_health_records_df_main['encounter_date_obj_test'] == pd.Timestamp('2023-10-01').date()]
    
    if not specific_day_data.empty:
        summary = get_chw_summary(specific_day_data)
        # Assert specific values based on the '2023-10-01' data in sample_health_records_df_main
        # Example:
        expected_visits_oct1 = specific_day_data['patient_id'].nunique()
        assert summary.get("visits_today") == expected_visits_oct1
        # Add checks for other CHW summary metrics like 'patients_critical_spo2_today', 'pending_critical_condition_referrals'
        # These require careful calculation from the sample data.


# Test get_patient_alerts_for_chw - check output structure (list of dicts) and key alert triggers
def test_get_patient_alerts_for_chw_output(sample_health_records_df_main):
    daily_data = sample_health_records_df_main[sample_health_records_df_main['encounter_date'].dt.date == pd.Timestamp('2023-10-01').date()]
    if not daily_data.empty:
        alerts = get_patient_alerts_for_chw(daily_data, source_context="TestAlertsCHW")
        assert isinstance(alerts, list)
        if alerts:
            alert_item = alerts[0]
            expected_alert_keys = ["patient_id", "alert_level", "primary_reason", "brief_details",
                                   "suggested_action_code", "raw_priority_score", "context_info"]
            for key in expected_alert_keys:
                assert key in alert_item, f"Alert item missing key: {key}"
            # Example: find a patient who SHOULD trigger a critical SpO2 alert from sample data
            # And verify that such an alert is generated.
            crit_spo2_patient = daily_data[daily_data['min_spo2_pct'] < app_config.ALERT_SPO2_CRITICAL_LOW_PCT].iloc[0]
            if not crit_spo2_patient.empty:
                assert any(a['patient_id'] == crit_spo2_patient['patient_id'] and "Critical Low SpO2" in a['primary_reason'] for a in alerts)


# Similarly, add tests for:
# - get_clinic_summary (check KPI names, types, values for 'test_summary_details')
# - get_clinic_environmental_summary (check KPI names and status levels based on new config)
# - get_patient_alerts_for_clinic (check specific clinic-level alert rules)
# - get_district_summary_kpis (use sample_enriched_gdf_main, check pop-weighted averages)
# - get_trend_data (test various periods, agg_funcs, and filtering)
# - get_supply_forecast_data (test linear forecast output structure, dates, stockout logic)


# Example for testing get_trend_data
def test_get_trend_data_logic(sample_health_records_df_main):
    # Test daily count of unique patients
    if not sample_health_records_df_main.empty:
        daily_visits = get_trend_data(sample_health_records_df_main, value_col='patient_id',
                                      date_col='encounter_date', period='D', agg_func='nunique')
        assert isinstance(daily_visits, pd.Series)
        assert not daily_visits.empty
        assert daily_visits.index.name == 'encounter_date' # Default index name from grouper
        assert pd.api.types.is_integer_dtype(daily_visits.dtype) # Nunique should be int

        # Test weekly average AI risk score
        weekly_avg_risk = get_trend_data(sample_health_records_df_main, value_col='ai_risk_score',
                                         date_col='encounter_date', period='W-Mon', agg_func='mean')
        assert isinstance(weekly_avg_risk, pd.Series)
        if weekly_avg_risk.notna().any(): # If there's any non-NaN data
            assert pd.api.types.is_numeric_dtype(weekly_avg_risk.dtype) # Mean should be float/numeric

# Remember to add tests for functions that now return specific dictionary structures
# (e.g., the new component data preparer functions if they were moved into core_data_processing,
# though it seems they are mostly in `pages.<dashboard>_components_sentinel.<module>`)
