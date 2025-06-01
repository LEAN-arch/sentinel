# test/tests/conftest.py
# Conceptually Refactored for "Sentinel Health Co-Pilot"
# NOTE: This is a guide. Actual implementation requires alignment with
#       finalized app_config.py and data processing function outputs.

import pytest
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon
import numpy as np

# CRITICAL: Ensure app_config is accessible and reflects the NEW system design
# from config import app_config
# For this conceptual pass, we'll reference app_config as if imported.
# Actual test run would need proper import setup.

# Simulate app_config for conceptual demonstration (replace with actual import)
class AppConfigMock:
    # Vital Sign Alerts
    ALERT_SPO2_CRITICAL_LOW_PCT = 90
    ALERT_SPO2_WARNING_LOW_PCT = 94
    ALERT_BODY_TEMP_FEVER_C = 38.0
    ALERT_BODY_TEMP_HIGH_FEVER_C = 39.5
    # Ambient Alerts
    ALERT_AMBIENT_CO2_HIGH_PPM = 1500
    # Risk & Intervention
    RISK_SCORE_HIGH_THRESHOLD = 75
    DISTRICT_ZONE_HIGH_RISK_AVG_SCORE = 70
    DISTRICT_INTERVENTION_FACILITY_COVERAGE_LOW_PCT = 60
    DISTRICT_INTERVENTION_TB_BURDEN_HIGH_ABS = 10
    # KEY_CONDITIONS_FOR_ACTION
    KEY_CONDITIONS_FOR_ACTION = ['TB', 'Malaria', 'Severe Dehydration', 'Pneumonia']
    # KEY_TEST_TYPES_FOR_ANALYSIS & CRITICAL_TESTS_LIST
    KEY_TEST_TYPES_FOR_ANALYSIS = {"RDT-Malaria": {"critical": True, "target_tat_days": 0.5, "display_name": "Malaria RDT"},
                                 "Sputum-GeneXpert": {"critical": True, "target_tat_days": 1, "display_name": "TB GeneXpert"}}
    CRITICAL_TESTS_LIST = ["RDT-Malaria", "Sputum-GeneXpert"]
    # Default CRS
    DEFAULT_CRS_STANDARD = "EPSG:4326"
app_config = AppConfigMock() # Using the mock for this conceptual code

# --- Fixture for Sample Health Records ---
@pytest.fixture(scope="session")
def sample_health_records_df_main():
    """
    Updated DataFrame of health records for Sentinel Health Co-Pilot.
    Includes lean data inputs and values to test new thresholds.
    """
    data = {
        # IDs & Dates (encounter_date is critical)
        'encounter_id': [f'SENC{i:03d}' for i in range(1, 21)],
        'patient_id': [f'SPID{i%10:03d}' for i in range(1, 21)], # 10 unique patients
        'encounter_date': pd.to_datetime(
            ['2023-10-01T08:00', '2023-10-01T10:00', '2023-10-01T14:00', '2023-10-02T09:00', '2023-10-02T11:00',
             '2023-10-02T15:00', '2023-10-03T08:30', '2023-10-03T12:00', '2023-10-04T10:00', '2023-10-04T13:00',
             '2023-10-05T09:30', '2023-10-05T11:30', '2023-10-06T10:00', '2023-10-06T14:30', '2023-10-07T08:00',
             '2023-10-07T12:30', '2023-10-08T09:00', '2023-10-08T11:00', '2023-10-09T10:30', '2023-10-09T13:30']
        ),
        # Demographics (lean)
        'age': [35, 68, 2, 45, 28, 72, 5, 52, 60, 1, 22, 58, 15, 40, 75, 3, 30, 65, 8, 50],
        'gender': ['Female', 'Male', 'Male', 'Female', 'Male', 'Female', 'Female', 'Male', 'Female', 'Male'] * 2, # Sex at birth
        'pregnancy_status': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], # 0=No, 1=Yes, (or specific codes)
        'chronic_condition_flag': [1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1], # 1=Yes
        # Job/Context (Example for CHW records, less for general patients unless relevant)
        # 'exposure_type_code': ['clinic', 'field_household', ...]
        # 'ppe_compliant_flag': [1, 0, 1, ...]
        'ambient_heat_index_c': [30, 33, 28, 35, 31, 38, 29, 32, 36, 30, 29, 33, 31, 37, 34, 28, 30, 39, 32, 35], # Ambient context
        # Sensor Streams (mix of good, warning, critical values based on app_config)
        'hrv_rmssd_ms': [25, 18, 55, 40, 48, 15, 60, 35, 22, 65, 50, 19, 58, 30, 12, 70, 45, 10, 62, 38],
        'min_spo2_pct': [96, 89, 98, 93, 97, 88, 99, 94, 91, 97, 95, 89, 96, 92, 87, 98, 95, 88, 97, 93],
        'vital_signs_temperature_celsius': [37.0, 39.6, 37.5, 38.1, 36.8, 39.8, 38.5, 37.2, 38.8, 36.5, 37.1, 39.0, 36.9, 38.3, 40.0, 37.7, 37.3, 39.9, 38.2, 37.0],
        'movement_activity_level': [2, 1, 3, 2, 2, 1, 3, 2, 1, 3, 2, 1, 3, 2, 1, 3, 2, 1, 3, 2], # e.g., 1=low, 2=med, 3=high
        'fall_detected_today': [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
        # Behavioral / Psychometric (simple versions)
        'signs_of_fatigue_observed_flag': [0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1],
        'rapid_psychometric_distress_score': [1, 8, 2, 5, 3, 9, 1, 6, 7, 2, 3, 8, 2, 5, 9, 1, 4, 8, 2, 6], # 0-10 scale
        # Condition & Test (Simplified for this example)
        'condition': ['Wellness Visit', 'Pneumonia', 'Malaria', 'TB', 'Wellness Visit',
                      'Severe Dehydration', 'Malaria', 'TB', 'Pneumonia', 'Malaria',
                      'TB', 'Severe Dehydration', 'Wellness Visit', 'Pneumonia', 'TB',
                      'Malaria', 'Wellness Visit', 'Severe Dehydration', 'Malaria', 'Pneumonia'],
        'test_type': ['None', 'Chest X-Ray', 'RDT-Malaria', 'Sputum-GeneXpert', 'None',
                      'Electrolytes', 'RDT-Malaria', 'Sputum-GeneXpert', 'Chest X-Ray', 'RDT-Malaria',
                      'Sputum-GeneXpert', 'Electrolytes', 'None', 'Chest X-Ray', 'Sputum-GeneXpert',
                      'RDT-Malaria', 'None', 'Electrolytes', 'RDT-Malaria', 'Chest X-Ray'],
        'test_result': ['N/A', 'Positive', 'Positive', 'Positive', 'N/A',
                        'Abnormal', 'Negative', 'Positive', 'Indeterminate', 'Positive',
                        'Negative', 'Abnormal', 'N/A', 'Positive', 'Positive',
                        'Negative', 'N/A', 'Abnormal', 'Pending', 'Rejected Sample'],
        'test_turnaround_days': [np.nan, 2, 0, 1, np.nan, 0.5, 0, 1, np.nan, 0, 1, 0.5, np.nan, 2, 1, 0, np.nan, 0.5, np.nan, np.nan],
        # AI Scores (will be re-calculated by apply_ai_models in other tests, but good to have source values)
        'ai_risk_score_initial': [20, 85, 70, 90, 25, 95, 65, 88, 78, 60, 82, 93, 15, 80, 92, 68, 22, 94, 72, 70],
        'ai_followup_priority_score_initial': [10, 90, 75, 92, 20, 98, 70, 90, 80, 65, 85, 96, 10, 82, 95, 72, 18, 97, 78, 75],
        # Zone for aggregation
        'zone_id': ['ZoneA', 'ZoneB', 'ZoneA', 'ZoneC', 'ZoneB', 'ZoneA', 'ZoneC', 'ZoneB', 'ZoneA', 'ZoneC'] * 2,
        # Supply chain related fields from original schema (might be less detailed on PED level)
        'item': ['ORS', 'Amoxicillin', 'ACT', 'TB-Regimen', 'Multivitamin'] * 4,
        'item_stock_agg_zone': [10, 20, 5, 2, 50, 8, 15, 3, 1, 40, 9, 18, 4, 0, 45, 7, 12, 2, 1, 30], # Low stock examples
        'consumption_rate_per_day': [1, 2, 0.5, 0.1, 1, 1.2, 2.2, 0.6, 0.2, 1.1, 0.9, 1.8, 0.4, 0.1, 1.2, 0.8, 1.5, 0.3, 0.1, 0.8],
        # Referral info
        'referral_status': ['N/A', 'Completed', 'Pending', 'Initiated', 'N/A', 'Pending', 'Completed', 'Initiated', 'Pending', 'Completed'] * 2
    }
    df = pd.DataFrame(data)
    # Simulate the cleaning that `load_health_records` would do
    # For a real conftest, you might actually call the clean_column_names, _convert_to_numeric helpers
    # on this raw data, or ensure data types are already mostly correct.
    # Ensure essential numeric columns are indeed numeric
    numeric_cols_check = ['age', 'hrv_rmssd_ms', 'min_spo2_pct', 'vital_signs_temperature_celsius', 'ambient_heat_index_c',
                          'movement_activity_level', 'fall_detected_today', 'rapid_psychometric_distress_score',
                          'test_turnaround_days', 'item_stock_agg_zone', 'consumption_rate_per_day']
    for col in numeric_cols_check:
        if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Ensure specific flags are int like
    flag_cols = ['pregnancy_status', 'chronic_condition_flag', 'signs_of_fatigue_observed_flag']
    for col in flag_cols:
        if col in df.columns: df[col] = df[col].astype(int)

    # Add 'encounter_date_obj' as often used for filtering.
    df['encounter_date_obj'] = df['encounter_date'].dt.date
    return df

@pytest.fixture(scope="session")
def sample_iot_clinic_df_main():
    # Update with values hitting new app_config ambient thresholds
    data = {
        'timestamp': pd.to_datetime(['2023-10-01T09:00:00Z', '2023-10-01T10:00:00Z', '2023-10-01T11:00:00Z',
                                     '2023-10-02T09:30:00Z', '2023-10-02T10:30:00Z']),
        'clinic_id': ['C01', 'C01', 'C02', 'C01', 'C02'],
        'room_name': ['Waiting Room A', 'Consultation 1', 'TB Clinic Waiting', 'Waiting Room A', 'Lab'],
        'avg_co2_ppm': [800, 1600, app_config.ALERT_AMBIENT_CO2_VERY_HIGH_PPM + 100, 700, 900],
        'avg_pm25': [10, app_config.ALERT_AMBIENT_PM25_HIGH_UGM3 + 5, 12, 8, app_config.ALERT_AMBIENT_PM25_VERY_HIGH_UGM3+2],
        'avg_noise_dba': [55, 60, app_config.ALERT_AMBIENT_NOISE_HIGH_DBA + 5, 50, 65], # avg_noise_db renamed
        'waiting_room_occupancy': [5, np.nan, 15, 8, np.nan], # 15 > TARGET_CLINIC_WAITING_ROOM_OCCUPANCY_MAX (10)
        'zone_id': ['ZoneA', 'ZoneA', 'ZoneB', 'ZoneA', 'ZoneB']
    }
    df = pd.DataFrame(data)
    df = df.rename(columns={'avg_noise_dba': 'avg_noise_db'}) # Align with potential cleaning in load_iot
    return df

@pytest.fixture(scope="session")
def sample_zone_attributes_df_main():
    # Updated 'name' column
    data = {'zone_id': ['ZoneA', 'ZoneB', 'ZoneC'],
            'name': ['North Zone Alpha', 'South Zone Beta', 'East Zone Gamma'], # Changed from zone_display_name
            'population': [12000, 20000, 9000],
            'socio_economic_index': [0.55, 0.35, 0.75], # Values for testing SES correlations
            'num_clinics': [2, 1, 1],
            'avg_travel_time_clinic_min': [15, 35, 20]}
    return pd.DataFrame(data)

@pytest.fixture(scope="session")
def sample_zone_geometries_gdf_main(sample_zone_attributes_df_main): # Depends on attributes for merge testing
    # Geometries are simple and unchanged in definition but test merge with new attributes
    features = [{"zone_id": "ZoneA", "geometry": Polygon([[0,0],[0,10],[10,10],[10,0],[0,0]])},
                {"zone_id": "ZoneB", "geometry": Polygon([[10,0],[10,10],[20,10],[20,0],[10,0]])},
                {"zone_id": "ZoneC", "geometry": Polygon([[0,-10],[0,0],[10,0],[10,-10],[0,-10]])}]
    df_geom = pd.DataFrame(features)
    gdf = gpd.GeoDataFrame(df_geom, geometry='geometry', crs=app_config.DEFAULT_CRS_STANDARD)
    # Simulate part of load_zone_data: merge attributes to ensure 'name' and other attrs are present
    # This makes this fixture more representative of what `load_zone_data` might produce as a base.
    gdf_merged = gdf.merge(sample_zone_attributes_df_main, on="zone_id", how="left")
    # If 'name_x' or 'name_y' exist due to merge, resolve to 'name'
    if 'name_y' in gdf_merged.columns and 'name_x' in gdf_merged.columns : gdf_merged['name'] = gdf_merged['name_y'].fillna(gdf_merged['name_x'])
    elif 'name_y' in gdf_merged.columns : gdf_merged.rename(columns={'name_y':'name'}, inplace=True)
    elif 'name_x' in gdf_merged.columns : gdf_merged.rename(columns={'name_x':'name'}, inplace=True)

    # Ensure geometry column is named 'geometry' after merge
    if gdf_merged.geometry.name != 'geometry':
        gdf_merged = gdf_merged.rename_geometry('geometry')

    return gdf_merged


@pytest.fixture(scope="session")
def sample_enriched_gdf_main(sample_zone_geometries_gdf_main, sample_health_records_df_main, sample_iot_clinic_df_main):
    """
    Generates a sample enriched GDF by calling the REFFACTORED
    `enrich_zone_geodata_with_health_aggregates` function.
    This ensures the test fixture matches the actual pipeline output.
    """
    from utils.core_data_processing import enrich_zone_geodata_with_health_aggregates # ACTUAL refactored function
    
    # sample_zone_geometries_gdf_main already includes merged attributes and 'name'
    # No need to call load_zone_data here, as the fixture itself prepares a suitable base_gdf.
    base_gdf_for_enrich = sample_zone_geometries_gdf_main.copy()

    # Ensure all columns expected by enrich_zone_geodata (from attributes) are present on base_gdf_for_enrich
    # 'population', 'num_clinics', 'socio_economic_index' are usually critical.
    # sample_zone_geometries_gdf_main should ideally already contain these from its own merge logic.
    for col_check in ['population', 'num_clinics', 'socio_economic_index']:
        if col_check not in base_gdf_for_enrich.columns:
            # This indicates an issue with sample_zone_geometries_gdf_main not fully simulating load_zone_data
            # For robustness, add with default, but the fixture should be self-contained.
            base_gdf_for_enrich[col_check] = 0 if col_check != 'socio_economic_index' else 0.5


    enriched_gdf = enrich_zone_geodata_with_health_aggregates(
        zone_gdf=base_gdf_for_enrich, # This GDF should already have 'name' and other attributes from load_zone_data logic
        health_df=sample_health_records_df_main,
        iot_df=sample_iot_clinic_df_main,
        source_context="ConftestEnrichment" # Add context for logging clarity
    )
    return enriched_gdf

# --- Plotting data fixtures - likely unchanged if plot function signatures are stable ---
@pytest.fixture(scope="session")
def sample_series_data():
    # ... (original content should be fine) ...
    dates = pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05', '2023-01-06', '2023-01-07'])
    values = [10, 12, 9, 15, 13, 18, 16]; return pd.Series(values, index=dates, name="Daily Count")

@pytest.fixture(scope="session")
def sample_bar_df(): # For plot_bar_chart_web
    return pd.DataFrame({'category': ['A', 'B', 'A', 'C', 'B', 'C', 'A', 'B'],
                         'value': [20, 15, 25, 30, 18, 22, 28, 12],
                         'group': ['G1', 'G1', 'G2', 'G1', 'G2', 'G2', 'G1', 'G1']})

# ... other simple plotting fixtures (donut, heatmap) are likely okay ...

# --- Empty Schema Fixtures - CRITICAL TO UPDATE ---
@pytest.fixture
def empty_health_df_with_sentinel_schema(): # Renamed for clarity
    # This MUST match the columns and dtypes (as best as possible)
    # from the refactored `load_health_records` function.
    # Reflecting a selection of lean data model + key processed fields:
    cols = [
        'encounter_id', 'patient_id', 'encounter_date', 'encounter_date_obj',
        'age', 'gender', 'pregnancy_status', 'chronic_condition_flag', # Lean Demographics
        # Job Context examples (might not be in every health_record CSV)
        # 'worker_id', 'exposure_type_code', 'ppe_compliant_flag',
        'ambient_heat_index_c', # Environmental context linked to encounter
        # Sensor Streams
        'hrv_rmssd_ms', 'min_spo2_pct', 'vital_signs_temperature_celsius', 'max_skin_temp_celsius', # Added both temp options
        'movement_activity_level', 'fall_detected_today',
        # Behavioral / Psychometric
        'signs_of_fatigue_observed_flag', 'rapid_psychometric_distress_score',
        # Condition, Test, Treatment - simplified for edge if possible
        'condition', 'test_type', 'test_result', 'test_turnaround_days',
        'item', 'item_stock_agg_zone', 'consumption_rate_per_day', # For supply elements
        'referral_status', # Key outcome/action
        # Processed AI scores
        'ai_risk_score', 'ai_followup_priority_score',
        'zone_id', 'clinic_id',
        # Other potentially useful original fields for higher-tier analytics
        'sample_collection_date', 'sample_registered_lab_date', 'notes', 'rejection_reason', 'sample_status'
    ]
    # Dtypes are hard to enforce perfectly in empty DF, but good for structure:
    # Example: dtypes = {'age': float, 'min_spo2_pct': float, 'encounter_date': 'datetime64[ns]'}
    return pd.DataFrame(columns=cols)


@pytest.fixture
def empty_iot_df_with_sentinel_schema(): # Renamed
    # Must match columns from refactored `load_iot_clinic_environment_data`
    cols = [
        'timestamp', 'clinic_id', 'room_name', 'zone_id',
        'avg_co2_ppm', 'max_co2_ppm', # Retain max if used by specific alerts
        'avg_pm25', 'voc_index', # voc_index might be lower priority for LMIC lean data
        'avg_temp_celsius', 'avg_humidity_rh', 'avg_noise_db',
        'waiting_room_occupancy', 'patient_throughput_per_hour', # Operational
        'sanitizer_dispenses_per_hour' # Hygiene proxy
    ]
    return pd.DataFrame(columns=cols)

@pytest.fixture
def empty_zone_attributes_df_with_sentinel_schema(): # Renamed
    # Matches expected structure after `load_zone_data` prepares attributes
    cols = ['zone_id', 'name', 'population', 'socio_economic_index',
            'num_clinics', 'avg_travel_time_clinic_min']
    return pd.DataFrame(columns=cols)

@pytest.fixture
def empty_enriched_gdf_with_sentinel_schema(): # Renamed
    # CRITICAL: This must match the columns output by the refactored
    # `enrich_zone_geodata_with_health_aggregates`.
    # The list of columns here must be derived from that function's logic.
    # Placeholder using many from previous `empty_gdf_with_schema` - NEEDS VERIFICATION
    # based on actual columns created in refactored `enrich_zone_geodata_with_health_aggregates`
    expected_cols = [
        'zone_id', 'name', 'geometry', # Base GDF columns
        'population', 'socio_economic_index', 'num_clinics', 'avg_travel_time_clinic_min', # Base attributes
        # Aggregated health metrics (examples, verify against enrichment function)
        'total_population_health_data', 'avg_risk_score', 'total_patient_encounters',
        'active_tb_cases', 'active_malaria_cases', # Examples for KEY_CONDITIONS_FOR_ACTION
        # ... other active_{condition}_cases ...
        'hiv_positive_cases', # If still used by name or example of another
        'pneumonia_cases',
        'total_active_key_infections', 'prevalence_per_1000',
        'facility_coverage_score',
        # Aggregated wearable/IoT context per zone
        'avg_daily_steps_zone', 'zone_avg_co2', # IoT metric for clinic zones
        # Population density example
        'population_density' # if enrichment calculates it
    ]
    # Ensure 'geometry' is explicitly set as the geometry column for an empty GDF
    return gpd.GeoDataFrame(columns=expected_cols, geometry='geometry', crs=app_config.DEFAULT_CRS_STANDARD)
