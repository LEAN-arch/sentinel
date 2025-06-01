# test/utils/core_data_processing.py
# Redesigned for "Sentinel Health Co-Pilot" - LMIC Edge-First System
# This module now focuses on robust data loading, cleaning, and aggregation utilities
# primarily intended for:
#   1. Facility Node (Tier 2) and Cloud (Tier 3) backend processing.
#   2. Initial data provisioning and system setup.
#   3. Simulation and testing environments.
# Direct use on Personal Edge Devices (PEDs) is minimal for loading functions;
# however, cleaning and some aggregation *logic* might be adapted for on-device implementations.

import streamlit as st # Kept for @st.cache_data, assuming these utils might be called by higher-tier Streamlit apps.
                       # UI-specific calls like st.error within functions will be minimized or made conditional.
import pandas as pd
import geopandas as gpd
import numpy as np
import os
import logging
# Assuming app_config is in thePYTHONPATH or project root.
# If not, sys.path manipulation might be needed when run outside Streamlit context.
from config import app_config # Now using the redesigned app_config
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

# --- I. Core Helper Functions (Largely Unchanged, Essential for Robustness) ---
def _clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Standardizes column names: lower case, replaces spaces/hyphens with underscores."""
    if not isinstance(df, pd.DataFrame):
        logger.error(f"_clean_column_names expects a pandas DataFrame, got {type(df)}.")
        # Attempt to return original input or a default empty DataFrame if conversion fails badly
        return df if df is not None else pd.DataFrame()
    df.columns = df.columns.str.lower().str.replace(' ', '_').str.replace('-', '_')
    return df

def _convert_to_numeric(series: pd.Series, default_value: Any = np.nan) -> pd.Series:
    """Safely converts a pandas Series to numeric, coercing errors to default_value."""
    if not isinstance(series, pd.Series):
        logger.debug(f"_convert_to_numeric given non-Series type: {type(series)}. Attempting conversion to Series.")
        try:
            series = pd.Series(series, dtype=float if default_value is np.nan else type(default_value))
        except Exception as e_series:
            logger.error(f"Could not convert input of type {type(series)} to Series in _convert_to_numeric: {e_series}")
            length = len(series) if hasattr(series, '__len__') else 1
            dtype_val = type(default_value) if default_value is not np.nan else float
            return pd.Series([default_value] * length, dtype=dtype_val) # Return a default series

    return pd.to_numeric(series, errors='coerce').fillna(default_value)

def hash_geodataframe(gdf: gpd.GeoDataFrame) -> Optional[str]:
    """
    Custom hash function for GeoDataFrames for Streamlit caching.
    More robust handling of empty or invalid geometries and diverse data types.
    """
    if gdf is None or not isinstance(gdf, gpd.GeoDataFrame):
        return None
    try:
        geom_col_name = gdf.geometry.name if hasattr(gdf, 'geometry') and hasattr(gdf.geometry, 'name') else 'geometry'
        non_geom_cols = []
        geom_hash_val = 0

        if geom_col_name in gdf.columns and hasattr(gdf[geom_col_name], 'is_empty') and not gdf[geom_col_name].is_empty.all():
            non_geom_cols = gdf.drop(columns=[geom_col_name], errors='ignore').columns.tolist()
            # Ensure geometries are valid before attempting WKT conversion
            valid_geoms = gdf[geom_col_name][gdf[geom_col_name].is_valid & ~gdf[geom_col_name].is_empty]
            if not valid_geoms.empty:
                geom_hash_val = pd.util.hash_array(valid_geoms.to_wkt().values).sum()
            else: # if all geoms are invalid/empty after filtering
                 geom_hash_val = pd.util.hash_array(gdf[geom_col_name].astype(str).values).sum() # Fallback hash based on string representation
        else:
            non_geom_cols = gdf.columns.tolist() # No valid geometry column found

        # Hash non-geometric part
        if not non_geom_cols: # If only geometry column or empty df
            df_content_hash = 0
        else:
            df_to_hash = gdf[non_geom_cols].copy()
            # Convert datetime/timedelta to int representation for stable hashing
            for col in df_to_hash.select_dtypes(include=['datetime64', 'datetime64[ns]', f"datetime64[ns, {app_config.DEFAULT_CRS}]"]).columns: # Added TZ-aware example
                df_to_hash[col] = pd.to_datetime(df_to_hash[col], errors='coerce').astype('int64') // 10**9 # Nanoseconds to seconds
            for col in df_to_hash.select_dtypes(include=['timedelta64', 'timedelta64[ns]']).columns:
                df_to_hash[col] = df_to_hash[col].astype('int64')
            # Attempt to hash, fall back for unhashable types by converting to string
            try:
                df_content_hash = pd.util.hash_pandas_object(df_to_hash, index=True).sum()
            except TypeError:
                logger.warning("Unhashable type found in GDF, converting to string for hashing.")
                df_content_hash = pd.util.hash_pandas_object(df_to_hash.astype(str), index=True).sum()

        return f"{df_content_hash}-{geom_hash_val}"
    except Exception as e:
        logger.error(f"Robust Hashing GeoDataFrame failed: {e}", exc_info=True)
        # Fallback hashing, not ideal but better than crashing cache.
        return str(gdf.head().to_string()) + str(gdf.shape)


def _robust_merge_agg(
    left_df: pd.DataFrame, right_df: pd.DataFrame, target_col_name: str,
    on_col: str = 'zone_id', default_fill_value: Any = 0.0
) -> pd.DataFrame:
    """
    Robustly merges an aggregated right_df into left_df, handling potential missing
    columns, preserving left_df index, and ensuring target column exists.
    """
    if not isinstance(left_df, pd.DataFrame):
        logger.error(f"Left df in _robust_merge_agg is not a DataFrame: {type(left_df)}")
        return left_df
    if target_col_name not in left_df.columns:
        left_df[target_col_name] = default_fill_value
    else: # Ensure existing target column also fills NaNs with default
        left_df[target_col_name] = left_df[target_col_name].fillna(default_fill_value)

    if not isinstance(right_df, pd.DataFrame) or right_df.empty or on_col not in right_df.columns:
        logger.debug(f"Right_df for {target_col_name} is empty or missing '{on_col}'. Skipping merge.")
        return left_df

    value_col_candidates = [col for col in right_df.columns if col != on_col]
    if not value_col_candidates:
        logger.debug(f"No value column found in right_df for {target_col_name}. Skipping merge.")
        return left_df
    value_col_in_right = value_col_candidates[0] # Take the first non-key column

    # Ensure 'on_col' types are compatible for merging
    try:
        left_df[on_col] = left_df[on_col].astype(str)
        right_df_for_merge = right_df[[on_col, value_col_in_right]].copy()
        right_df_for_merge[on_col] = right_df_for_merge[on_col].astype(str)
    except Exception as e:
        logger.error(f"Type conversion error for '{on_col}' in _robust_merge_agg: {e}")
        return left_df

    temp_agg_col = f"__temp_agg_{target_col_name}_{np.random.randint(0, 1e6)}__"
    right_df_for_merge.rename(columns={value_col_in_right: temp_agg_col}, inplace=True)

    original_index = left_df.index
    original_index_name = left_df.index.name

    # Perform merge. If left_df has a non-default index, reset it for merge and restore later.
    if not isinstance(left_df.index, pd.RangeIndex) or original_index_name is not None:
        left_df_for_merge = left_df.reset_index()
    else:
        left_df_for_merge = left_df

    merged_df = left_df_for_merge.merge(right_df_for_merge, on=on_col, how='left')

    if temp_agg_col in merged_df.columns:
        # Update target_col_name: use merged value if exists, otherwise keep original or default.
        merged_df[target_col_name] = merged_df[temp_agg_col].combine_first(merged_df.get(target_col_name, pd.Series(dtype=type(default_fill_value))))
        merged_df.drop(columns=[temp_agg_col], inplace=True, errors='ignore')

    merged_df[target_col_name].fillna(default_fill_value, inplace=True)

    # Restore original index if it was reset
    if not isinstance(left_df.index, pd.RangeIndex) or original_index_name is not None:
        index_col_to_set_back = original_index_name if original_index_name else 'index' # 'index' is default if unnamed
        if index_col_to_set_back in merged_df.columns:
            merged_df.set_index(index_col_to_set_back, inplace=True)
            if original_index_name : # Only rename if it was originally named
                 merged_df.index.name = original_index_name
        else: # Index was dropped, cannot restore perfectly, log and use default
             logger.warning(f"Original index '{index_col_to_set_back}' lost during merge for {target_col_name}. Resetting to RangeIndex.")
    return merged_df


# --- II. Data Loading and Basic Cleaning Functions ---
# These functions are now primarily for data ingestion at Tier 2 (Facility Node) or Tier 3 (Cloud),
# or for initial data provisioning.
# @st.cache_data decorators are kept, as these might be called by Streamlit apps in Tier 2/3.

@st.cache_data(ttl=app_config.CACHE_TTL_SECONDS_WEB_REPORTS, show_spinner="Loading health records data...")
def load_health_records(file_path: Optional[str] = None, source_context: str = "FacilityNode") -> pd.DataFrame:
    """
    Loads and cleans health records.
    Role: Data ingestion utility for Facility Node, Cloud, or Setup.
    """
    actual_file_path = file_path or app_config.HEALTH_RECORDS_CSV # Fallback to default config path
    logger.info(f"({source_context}) Attempting to load health records from: {actual_file_path}")
    if not os.path.exists(actual_file_path):
        logger.error(f"({source_context}) Health records file not found: {actual_file_path}")
        # Avoid direct st.error for backend use; callee should handle UI error if needed.
        return pd.DataFrame()
    try:
        df = pd.read_csv(actual_file_path, low_memory=False)
        df = _clean_column_names(df)
        logger.info(f"({source_context}) Successfully loaded {len(df)} raw records from {actual_file_path}.")

        # Date conversions (using .get for robustness against missing columns)
        date_cols = ['encounter_date', 'sample_collection_date', 'sample_registered_lab_date', 'referral_date', 'referral_outcome_date']
        for col in date_cols:
            if col in df.columns: df[col] = pd.to_datetime(df.get(col), errors='coerce')
            else: df[col] = pd.NaT # Ensure column exists as datetime if expected

        # Numeric conversions (referencing new, more specific thresholds from app_config for any default values)
        numeric_cols_map = { # Map column to a reasonable default if needed, np.nan is typical
            'test_turnaround_days': np.nan, 'quantity_dispensed': 0, 'item_stock_agg_zone': 0,
            'consumption_rate_per_day': 0.0, 'ai_risk_score': np.nan, 'ai_followup_priority_score': np.nan,
            'vital_signs_bp_systolic': np.nan, 'vital_signs_bp_diastolic': np.nan,
            'vital_signs_temperature_celsius': np.nan, 'min_spo2_pct': np.nan, 'max_skin_temp_celsius': np.nan,
            'avg_spo2': np.nan, 'avg_daily_steps': 0, 'resting_heart_rate': np.nan, 'avg_hrv': np.nan,
            'avg_sleep_duration_hrs': np.nan, 'sleep_score_pct': np.nan, 'stress_level_score': np.nan,
            'fall_detected_today': 0, 'age': np.nan, 'chw_visit': 0, 'tb_contact_traced': 0,
            'patient_latitude': np.nan, 'patient_longitude': np.nan,
            'hiv_viral_load_copies_ml': np.nan # This one specifically handles '<50>' socoerce is fine
        }
        for col, default_val in numeric_cols_map.items():
            if col in df.columns: df[col] = _convert_to_numeric(df.get(col), default_val)
            else: df[col] = default_val # Ensure column exists

        # String-like columns (ensure consistent "Unknown" for missing or varied NA representations)
        string_like_cols = [
            'encounter_id', 'patient_id', 'encounter_type', 'condition', 'diagnosis_code_icd10',
            'test_type', 'test_result', 'item', 'zone_id', 'clinic_id', 'physician_id',
            'notes', 'patient_reported_symptoms', 'gender', 'screening_hpv_status',
            'key_chronic_conditions_summary', 'medication_adherence_self_report',
            'referral_status', 'referral_reason', 'referred_to_facility_id',
            'referral_outcome', 'sample_status', 'rejection_reason'
        ]
        common_na_values = ['', 'nan', 'None', 'N/A', '#N/A', 'np.nan', 'NaT'] # More comprehensive list
        for col in string_like_cols:
            if col in df.columns:
                df[col] = df[col].fillna("Unknown").astype(str).str.strip()
                df[col] = df[col].replace(common_na_values, "Unknown", regex=False)
            else: df[col] = "Unknown" # Ensure column exists

        # Ensure absolutely critical columns for downstream logic are present, even if filled with NA/Unknown
        critical_data_cols = {'patient_id': "Unknown", 'encounter_date': pd.NaT, 'condition': "Unknown", 'test_type': "Unknown"}
        for col, default_val in critical_data_cols.items():
            if col not in df.columns or df[col].isnull().all():
                logger.warning(f"({source_context}) Critical column '{col}' missing or all null in health records. Filling with default.")
                df[col] = default_val

        logger.info(f"({source_context}) Health records loaded and basic cleaning complete for {len(df)} records.")
        return df
    except Exception as e:
        logger.error(f"({source_context}) Error loading/processing health records from {actual_file_path}: {e}", exc_info=True)
        return pd.DataFrame() # Return empty DataFrame on critical failure


@st.cache_data(ttl=app_config.CACHE_TTL_SECONDS_WEB_REPORTS, show_spinner="Loading IoT environmental data...")
def load_iot_clinic_environment_data(file_path: Optional[str] = None, source_context: str = "FacilityNode") -> pd.DataFrame:
    """Loads and cleans IoT environmental data. Primarily for Facility Node or Cloud."""
    actual_file_path = file_path or app_config.IOT_CLINIC_ENVIRONMENT_CSV
    logger.info(f"({source_context}) Attempting to load IoT data from: {actual_file_path}")
    if not os.path.exists(actual_file_path):
        logger.warning(f"({source_context}) IoT data file not found: {actual_file_path}. Environmental data will be unavailable.")
        return pd.DataFrame()
    try:
        df = pd.read_csv(actual_file_path, low_memory=False)
        df = _clean_column_names(df)
        logger.info(f"({source_context}) Successfully loaded {len(df)} IoT records.")
        if 'timestamp' in df.columns: df['timestamp'] = pd.to_datetime(df.get('timestamp'), errors='coerce')
        else: logger.error(f"({source_context}) IoT data missing critical 'timestamp' column. Returning empty."); return pd.DataFrame()

        numeric_iot_cols = [
            'avg_co2_ppm', 'max_co2_ppm', 'avg_pm25', 'voc_index', 'avg_temp_celsius',
            'avg_humidity_rh', 'avg_noise_db', 'waiting_room_occupancy',
            'patient_throughput_per_hour', 'sanitizer_dispenses_per_hour'
        ]
        for col in numeric_iot_cols:
            if col in df.columns: df[col] = _convert_to_numeric(df.get(col), np.nan) # Default to NaN for sensor readings
            else: df[col] = np.nan # Ensure column exists

        string_iot_cols = ['clinic_id', 'room_name', 'zone_id']
        common_na_values_iot = ['', 'nan', 'None', 'N/A', '#N/A', 'np.nan', 'NaT']
        for col in string_iot_cols:
            if col in df.columns:
                df[col] = df[col].fillna("Unknown").astype(str).str.strip()
                df[col] = df[col].replace(common_na_values_iot, "Unknown", regex=False)
            else: df[col] = "Unknown" # Ensure column exists
        logger.info(f"({source_context}) IoT data cleaning complete for {len(df)} records.")
        return df
    except Exception as e:
        logger.error(f"({source_context}) Error loading/processing IoT data from {actual_file_path}: {e}", exc_info=True)
        return pd.DataFrame()


@st.cache_data(ttl=app_config.CACHE_TTL_SECONDS_WEB_REPORTS, hash_funcs={gpd.GeoDataFrame: hash_geodataframe}, show_spinner="Loading zone geographic/attribute data...")
def load_zone_data(attributes_path: Optional[str] = None, geometries_path: Optional[str] = None, source_context: str = "FacilityNode") -> Optional[gpd.GeoDataFrame]:
    """Loads, cleans, and merges zone attributes and geometries. For Facility Node/Cloud or Setup."""
    attr_path = attributes_path or app_config.ZONE_ATTRIBUTES_CSV
    geom_path = geometries_path or app_config.ZONE_GEOMETRIES_GEOJSON
    logger.info(f"({source_context}) Loading zone attributes from {attr_path} and geometries from {geom_path}")
    
    error_messages = []
    if not os.path.exists(attr_path): error_messages.append(f"Attributes file '{os.path.basename(attr_path)}' not found.")
    if not os.path.exists(geom_path): error_messages.append(f"Geometries file '{os.path.basename(geom_path)}' not found.")
    if error_messages: logger.error(f"({source_context}) {' '.join(error_messages)}"); return None

    try:
        zone_attributes_df = pd.read_csv(attr_path); zone_attributes_df = _clean_column_names(zone_attributes_df)
        zone_geometries_gdf = gpd.read_file(geom_path); zone_geometries_gdf = _clean_column_names(zone_geometries_gdf)

        for df_check, name in [(zone_attributes_df, "attributes"), (zone_geometries_gdf, "geometries")]:
            if 'zone_id' not in df_check.columns:
                logger.error(f"({source_context}) Missing 'zone_id' in zone {name}. Cannot merge."); return None
            df_check['zone_id'] = df_check['zone_id'].astype(str).str.strip()

        # Standardize 'name' column (for display) before merge. Prefer 'name', then 'zone_display_name', then default.
        if 'zone_display_name' in zone_attributes_df.columns and 'name' not in zone_attributes_df.columns:
            zone_attributes_df.rename(columns={'zone_display_name': 'name'}, inplace=True)
        elif 'name' not in zone_attributes_df.columns and 'zone_id' in zone_attributes_df.columns:
            zone_attributes_df['name'] = "Zone " + zone_attributes_df['zone_id']
        if 'name' not in zone_geometries_gdf.columns and 'zone_id' in zone_geometries_gdf.columns: # If geoms also have a name field.
             geom_name_col = next((c for c in ['name_geom','name'] if c in zone_geometries_gdf.columns), None)
             if geom_name_col: zone_geometries_gdf.rename(columns={geom_name_col:'name_geom_src'}, inplace=True)

        merged_gdf = zone_geometries_gdf.merge(zone_attributes_df, on="zone_id", how="left", suffixes=('_geom', '_attr'))

        # Resolve column name conflicts from merge, prioritizing _attr version for non-geometry fields
        for col_attr in zone_attributes_df.columns:
            if col_attr == 'zone_id': continue
            if f"{col_attr}_attr" in merged_gdf.columns: # if _attr exists, it came from attributes_df
                merged_gdf[col_attr] = merged_gdf[f"{col_attr}_attr"].fillna(merged_gdf.get(f"{col_attr}_geom")) # Fill from geom if attr is NaN
                merged_gdf.drop(columns=[f"{col_attr}_attr", f"{col_attr}_geom"], errors='ignore', inplace=True)
            elif col_attr not in merged_gdf.columns and f"{col_attr}_geom" in merged_gdf.columns : # Came only from geom
                merged_gdf.rename(columns={f"{col_attr}_geom":col_attr}, inplace=True)

        # Ensure 'geometry' column is correctly named and set
        current_geom_col_name = merged_gdf.geometry.name if hasattr(merged_gdf, 'geometry') else None
        if 'geometry' not in merged_gdf.columns and current_geom_col_name and current_geom_col_name in merged_gdf.columns:
            merged_gdf = merged_gdf.rename_geometry('geometry')
        elif 'geometry' not in merged_gdf.columns and 'geometry_geom' in merged_gdf.columns: # Check common suffix
             merged_gdf = merged_gdf.rename_geometry('geometry', col_name='geometry_geom')
        elif 'geometry' not in merged_gdf.columns :
            logger.error(f"({source_context}) No 'geometry' column identifiable in merged GDF. Columns: {merged_gdf.columns.tolist()}"); return None

        # CRS handling
        if merged_gdf.crs is None: merged_gdf = merged_gdf.set_crs(app_config.DEFAULT_CRS_STANDARD, allow_override=True)
        elif merged_gdf.crs.to_string().upper() != app_config.DEFAULT_CRS_STANDARD.upper(): merged_gdf = merged_gdf.to_crs(app_config.DEFAULT_CRS_STANDARD)

        # Ensure essential columns for system function are present.
        default_zone_cols_map = {
            'name': f"Unnamed Zone", 'population': 0.0,
            'num_clinics': 0.0, 'socio_economic_index': 0.5, # LMIC average might be lower, adjust as per context
            'avg_travel_time_clinic_min': 30.0 # Placeholder, highly context-specific
        }
        for col, default_val in default_zone_cols_map.items():
            if col not in merged_gdf.columns: merged_gdf[col] = default_val
            elif col in ['population', 'num_clinics', 'socio_economic_index', 'avg_travel_time_clinic_min']:
                 merged_gdf[col] = _convert_to_numeric(merged_gdf.get(col), default_val)
            elif col == 'name' : merged_gdf[col] = merged_gdf.get(col,"Unknown").astype(str).fillna("Zone "+merged_gdf['zone_id'])

        logger.info(f"({source_context}) Successfully loaded and merged zone data: {len(merged_gdf)} zones.")
        return merged_gdf
    except Exception as e:
        logger.error(f"({source_context}) Error loading/merging zone data: {e}", exc_info=True); return None


# --- III. Data Enrichment and Aggregation Functions ---
# These functions typically run at Facility Node or Cloud backend.

def enrich_zone_geodata_with_health_aggregates(
    zone_gdf: gpd.GeoDataFrame,
    health_df: pd.DataFrame,
    iot_df: Optional[pd.DataFrame] = None,
    source_context: str = "FacilityNode"
) -> gpd.GeoDataFrame:
    """
    Enriches GeoDataFrame with aggregated health and IoT metrics.
    Designed for Facility Node / Cloud tier processing.
    """
    logger.info(f"({source_context}) Starting zone GeoDataFrame enrichment.")
    if not isinstance(zone_gdf, gpd.GeoDataFrame) or zone_gdf.empty or 'zone_id' not in zone_gdf.columns:
        logger.warning(f"({source_context}) Invalid or empty zone_gdf for enrichment. Returning as is or empty.")
        return zone_gdf if isinstance(zone_gdf, gpd.GeoDataFrame) else gpd.GeoDataFrame(columns=['zone_id', 'name', 'geometry', 'population'], crs=app_config.DEFAULT_CRS_STANDARD)

    enriched_gdf = zone_gdf.copy()
    if 'population' not in enriched_gdf.columns: enriched_gdf['population'] = 0.0
    enriched_gdf['population'] = _convert_to_numeric(enriched_gdf['population'], 0.0)

    # Initialize all expected aggregate columns to prevent KeyErrors later
    agg_cols_to_initialize = [
        'total_population_health_data', 'avg_risk_score', 'total_patient_encounters',
        'active_tb_cases', 'active_malaria_cases', 'hiv_positive_cases', 'pneumonia_cases',
        'total_referrals_made', 'successful_referrals',
        'avg_test_turnaround_critical', 'perc_critical_tests_tat_met',
        'prevalence_per_1000', 'total_active_key_infections',
        'avg_daily_steps_zone', 'zone_avg_co2', 'facility_coverage_score'
    ]
    for col in agg_cols_to_initialize: enriched_gdf[col] = 0.0

    if health_df is not None and not health_df.empty and 'zone_id' in health_df.columns:
        health_df_agg = health_df.copy()
        health_df_agg['zone_id'] = health_df_agg['zone_id'].astype(str).str.strip()

        # Patient counts and risk
        enriched_gdf = _robust_merge_agg(enriched_gdf, health_df_agg.groupby('zone_id')['patient_id'].nunique().reset_index(), 'total_population_health_data')
        enriched_gdf = _robust_merge_agg(enriched_gdf, health_df_agg.groupby('zone_id')['ai_risk_score'].mean().reset_index(), 'avg_risk_score', default_fill_value=np.nan)
        enriched_gdf = _robust_merge_agg(enriched_gdf, health_df_agg.groupby('zone_id')['encounter_id'].nunique().reset_index(), 'total_patient_encounters')

        # Specific conditions (using KEY_CONDITIONS_FOR_ACTION from new config for relevance)
        for condition_name in app_config.KEY_CONDITIONS_FOR_ACTION:
            col_name = f"active_{condition_name.lower().replace(' ', '_').replace('-', '_').replace('(severe)','')}_cases" # Generate column name
            if col_name not in enriched_gdf.columns: enriched_gdf[col_name] = 0.0 # Ensure col exists
            condition_filter = health_df_agg.get('condition', pd.Series(dtype=str)).str.contains(condition_name, case=False, na=False)
            enriched_gdf = _robust_merge_agg(enriched_gdf, health_df_agg[condition_filter].groupby('zone_id')['patient_id'].nunique().reset_index(), col_name)
        
        # Sum for total_active_key_infections using only the actionable list
        actionable_condition_cols = [f"active_{c.lower().replace(' ', '_').replace('-', '_').replace('(severe)','')}_cases" for c in app_config.KEY_CONDITIONS_FOR_ACTION if f"active_{c.lower().replace(' ', '_').replace('-', '_').replace('(severe)','')}_cases" in enriched_gdf.columns]
        if actionable_condition_cols : enriched_gdf['total_active_key_infections'] = enriched_gdf[actionable_condition_cols].sum(axis=1)


        # Referrals
        if 'referral_status' in health_df_agg.columns:
            made_referrals = health_df_agg[health_df_agg['referral_status'].notna() & (~health_df_agg['referral_status'].isin(['N/A', 'Unknown', 'Unknown']))]
            enriched_gdf = _robust_merge_agg(enriched_gdf, made_referrals.groupby('zone_id')['encounter_id'].nunique().reset_index(), 'total_referrals_made')
            if 'referral_outcome' in health_df_agg.columns:
                successful_outcomes = ['Completed', 'Service Provided', 'Attended Consult', 'Attended Followup', 'Attended']
                successful_refs = health_df_agg[health_df_agg['referral_outcome'].isin(successful_outcomes)]
                enriched_gdf = _robust_merge_agg(enriched_gdf, successful_refs.groupby('zone_id')['encounter_id'].nunique().reset_index(), 'successful_referrals')

        # Critical Test TAT
        critical_test_keys_from_config = app_config.CRITICAL_TESTS_LIST # From new config
        if critical_test_keys_from_config and 'test_type' in health_df_agg.columns and 'test_turnaround_days' in health_df_agg.columns:
            tat_df = health_df_agg[
                (health_df_agg['test_type'].isin(critical_test_keys_from_config)) &
                (health_df_agg['test_turnaround_days'].notna()) &
                (~health_df_agg.get('test_result',pd.Series(dtype=str)).isin(['Pending', 'Rejected Sample', 'Unknown', 'Indeterminate', 'Unknown']))
            ].copy()
            if not tat_df.empty:
                enriched_gdf = _robust_merge_agg(enriched_gdf, tat_df.groupby('zone_id')['test_turnaround_days'].mean().reset_index(), 'avg_test_turnaround_critical', default_fill_value=np.nan)
                def _check_tat_met_enriched(row_tat): # check TAT against specific test target or general
                    test_cfg = app_config.KEY_TEST_TYPES_FOR_ANALYSIS.get(row_tat['test_type'])
                    target_days = test_cfg['target_tat_days'] if test_cfg and 'target_tat_days' in test_cfg else app_config.TARGET_TEST_TURNAROUND_DAYS # General fallback not ideal here, config should be complete for critical
                    return row_tat['test_turnaround_days'] <= target_days if pd.notna(row_tat['test_turnaround_days']) and pd.notna(target_days) else False
                tat_df['tat_met_flag'] = tat_df.apply(_check_tat_met_enriched, axis=1)
                perc_met_agg_df = tat_df.groupby('zone_id')['tat_met_flag'].mean().reset_index()
                perc_met_agg_df.iloc[:, 1] *= 100 # Convert mean (proportion) to percentage
                enriched_gdf = _robust_merge_agg(enriched_gdf, perc_met_agg_df, 'perc_critical_tests_tat_met')
        
        # Avg daily steps (from patient PED sync)
        if 'avg_daily_steps' in health_df_agg.columns: # This field comes from patient's wearable via CHW data
            enriched_gdf = _robust_merge_agg(enriched_gdf, health_df_agg.groupby('zone_id')['avg_daily_steps'].mean().reset_index(), 'avg_daily_steps_zone', default_fill_value=np.nan)

    # IoT Data for zone average CO2
    if iot_df is not None and not iot_df.empty and 'zone_id' in iot_df.columns and 'avg_co2_ppm' in iot_df.columns:
        iot_df_agg = iot_df.copy()
        iot_df_agg['zone_id'] = iot_df_agg['zone_id'].astype(str).str.strip()
        enriched_gdf = _robust_merge_agg(enriched_gdf, iot_df_agg.groupby('zone_id')['avg_co2_ppm'].mean().reset_index(), 'zone_avg_co2', default_fill_value=np.nan)

    # Calculated Metrics
    if 'total_active_key_infections' in enriched_gdf.columns and 'population' in enriched_gdf.columns:
         enriched_gdf['prevalence_per_1000'] = enriched_gdf.apply(
             lambda r: (r['total_active_key_infections'] / r['population']) * 1000 if pd.notna(r['population']) and r['population'] > 0 and pd.notna(r['total_active_key_infections']) else 0.0, axis=1).fillna(0.0)

    # Facility Coverage Score - A simplified proxy. LMIC context might need nuanced definition based on travel time, service availability, etc.
    # This placeholder formula rewards more clinics per capita but caps at 100.
    if 'num_clinics' in enriched_gdf.columns and 'population' in enriched_gdf.columns:
        enriched_gdf['facility_coverage_score'] = enriched_gdf.apply(
            lambda r: min(100, (r.get('num_clinics', 0) / r.get('population', 1)) * 20000) if pd.notna(r['population']) and r['population'] > 0 and pd.notna(r['num_clinics']) else 0.0, axis=1).fillna(0.0) # Adjust multiplier as needed
    elif 'facility_coverage_score' not in enriched_gdf.columns:
        enriched_gdf['facility_coverage_score'] = 0.0
    
    # Population density if area can be calculated
    if 'geometry' in enriched_gdf.columns and 'population' in enriched_gdf.columns and enriched_gdf.crs:
        try:
            # Ensure correct projection for area calculation (e.g., a UTM zone or equal-area projection)
            # This requires knowledge of the region. Forcing a common one for example.
            # A better approach would be to pass a target CRS or ensure GDF is already in suitable projection.
            # Using area in degrees if crs is geographic like 4326 then converting, which is rough.
            # It's better to reproject to an equal-area projection first.
            # For simplicity, let's assume area is somewhat meaningful or a placeholder.
            # For EPSG:4326, .area gives area in square degrees. Need conversion. ~111km/degree.
            # Conversion is complex and latitude-dependent. For this demo, will skip precise calculation.
            # enriched_gdf['area_sqkm'] = enriched_gdf.geometry.area / 10**6 # If projected in meters
            # enriched_gdf['population_density'] = enriched_gdf['population'] / enriched_gdf['area_sqkm']
            if 'population_density' not in enriched_gdf.columns: # Only add if not already there
                enriched_gdf['population_density'] = np.nan # Placeholder for density calculation
            logger.info("Population density calculation placeholder used. For accurate density, ensure GDF is in an equal-area projection and calculate area in sqkm.")
        except Exception as e_area:
            logger.warning(f"Could not calculate area/density for GDF enrichment: {e_area}")
            if 'population_density' not in enriched_gdf.columns: enriched_gdf['population_density'] = np.nan

    # Final fillna for all designated aggregate columns after all merges and calculations
    for col in agg_cols_to_initialize:
        if col in enriched_gdf.columns:
            if not pd.api.types.is_numeric_dtype(enriched_gdf[col]): enriched_gdf[col] = pd.to_numeric(enriched_gdf[col], errors='coerce')
            enriched_gdf[col].fillna(0.0, inplace=True) # Numeric aggregates default to 0.0 if still NaN
        else: enriched_gdf[col] = 0.0 # Ensure column exists and is 0.0

    logger.info(f"({source_context}) Zone GeoDataFrame enrichment complete. GDF shape: {enriched_gdf.shape}")
    return enriched_gdf


# --- IV. KPI & Summary Calculation Functions ---
# These generate dictionaries or DataFrames intended for reports or higher-tier dashboard displays.
# Logic for alerts on PEDs would be a simplified, real-time version of alert-generating functions.

def get_overall_kpis(health_df: pd.DataFrame, date_filter_start: Optional[str]=None, date_filter_end: Optional[str]=None, source_context: str = "FacilityNode") -> Dict[str, Any]:
    """
    Calculates high-level KPIs. Runs at Facility Node or Cloud.
    """
    logger.info(f"({source_context}) Calculating overall KPIs.")
    # Initialize KPIs with specific NA types where appropriate (e.g. np.nan for averages)
    kpis: Dict[str, Any] = {
        "total_patients": 0,
        "avg_patient_risk": np.nan,
        "active_tb_cases_current": 0, # Example actionable disease
        "malaria_rdt_positive_rate_period": np.nan, # Example test positivity
        "key_supply_stockout_alerts": 0 # Number of key items critically low
    }
    if health_df is None or health_df.empty: return kpis

    df_kpi = health_df.copy()
    if 'encounter_date' not in df_kpi.columns or df_kpi['encounter_date'].isnull().all():
        logger.warning(f"({source_context}) 'encounter_date' missing or all null in get_overall_kpis. KPIs will be limited.")
        return kpis
    
    # Ensure date conversion and filter by date
    df_kpi['encounter_date'] = pd.to_datetime(df_kpi['encounter_date'], errors='coerce')
    df_kpi.dropna(subset=['encounter_date'], inplace=True)
    if date_filter_start: df_kpi = df_kpi[df_kpi['encounter_date'] >= pd.to_datetime(date_filter_start, errors='coerce')]
    if date_filter_end: df_kpi = df_kpi[df_kpi['encounter_date'] <= pd.to_datetime(date_filter_end, errors='coerce')]
    if df_kpi.empty: logger.info(f"({source_context}) No data after date filtering for overall KPIs."); return kpis

    if 'patient_id' in df_kpi: kpis["total_patients"] = df_kpi['patient_id'].nunique()
    if 'ai_risk_score' in df_kpi and df_kpi['ai_risk_score'].notna().any(): kpis["avg_patient_risk"] = df_kpi['ai_risk_score'].mean()
    
    # Focus on actionable conditions from app_config.KEY_CONDITIONS_FOR_ACTION
    if 'condition' in df_kpi.columns:
        for cond_key in app_config.KEY_CONDITIONS_FOR_ACTION: # Assumes simple string match for condition
             kpi_col_name = f"active_{cond_key.lower().replace(' ', '_').replace('-', '_').replace('(severe)','')}_cases_current"
             kpis[kpi_col_name] = df_kpi[df_kpi['condition'].str.contains(cond_key, case=False, na=False)]['patient_id'].nunique()
        # Example for TB:
        if 'TB' in app_config.KEY_CONDITIONS_FOR_ACTION: kpis["active_tb_cases_current"] = kpis.get("active_tb_cases_current",0)


    # Example test positivity for a critical test (using display names for matching what's likely in the KPI output, but data should use original keys for processing)
    # Config for KEY_TEST_TYPES_FOR_ANALYSIS should map original key to display name & critical status
    # Example: Malaria RDT Positivity
    malaria_rdt_orig_key = "RDT-Malaria" # This is the key from KEY_TEST_TYPES_FOR_ANALYSIS
    if malaria_rdt_orig_key in app_config.KEY_TEST_TYPES_FOR_ANALYSIS and 'test_type' in df_kpi.columns and 'test_result' in df_kpi.columns:
        test_df_mal = df_kpi[(df_kpi['test_type'] == malaria_rdt_orig_key) &
                           (~df_kpi.get('test_result',pd.Series(dtype=str)).isin(["Pending", "Rejected Sample", "Unknown", "Indeterminate", "Unknown"]))]
        if not test_df_mal.empty and len(test_df_mal) > 0:
            kpis["malaria_rdt_positive_rate_period"] = (test_df_mal[test_df_mal['test_result'] == 'Positive'].shape[0] / len(test_df_mal)) * 100
        else: kpis["malaria_rdt_positive_rate_period"] = 0.0 # Or np.nan if preferred for "no data"

    # Key Supply Stockout Alerts (focused list)
    if all(c in df_kpi for c in ['item', 'item_stock_agg_zone', 'consumption_rate_per_day', 'encounter_date']):
        supply_df = df_kpi.sort_values('encounter_date').drop_duplicates(subset=['item', 'zone_id'], keep='last') # Latest stock by item and zone
        supply_df['consumption_rate_per_day'] = supply_df['consumption_rate_per_day'].replace(0, np.nan) # Avoid division by zero if consumption is 0
        supply_df['days_supply'] = supply_df['item_stock_agg_zone'] / supply_df['consumption_rate_per_day']
        supply_df.dropna(subset=['days_supply'], inplace=True)
        
        # Filter for key drugs only
        key_drug_supply_df = supply_df[supply_df['item'].str.contains('|'.join(app_config.KEY_DRUG_SUBSTRINGS_SUPPLY), case=False, na=False)]
        kpis['key_supply_stockout_alerts'] = key_drug_supply_df[key_drug_supply_df['days_supply'] < app_config.CRITICAL_SUPPLY_DAYS_REMAINING]['item'].nunique()

    return kpis


def get_chw_summary(health_df_daily: pd.DataFrame, source_context: str = "FacilityNode/CHWReport") -> Dict[str, Any]:
    """
    Calculates CHW daily summary metrics. Typically run at Facility Node on data from a specific CHW for a day,
    or as a template for on-PED daily summary generation (simpler logic there).
    """
    logger.info(f"({source_context}) Calculating CHW daily summary.")
    # Initialize with np.nan for averages/metrics that might not compute
    summary = {
        "visits_today": 0,
        "avg_patient_risk_visited_today": np.nan,
        "high_ai_prio_followups_today": 0, # Using ai_followup_priority_score
        "patients_critical_spo2_today": 0, # SpO2 < CRITICAL_LOW_THRESHOLD
        "patients_high_fever_today": 0,    # Temp >= HIGH_FEVER_THRESHOLD
        "avg_patient_steps_visited_today": np.nan,
        "patients_fall_detected_today": 0,
        "pending_critical_condition_referrals": 0 # New: CHW specific
    }
    if health_df_daily is None or health_df_daily.empty: return summary

    chw_df = health_df_daily.copy() # Assume health_df_daily is already filtered for the specific CHW and date
    if 'patient_id' in chw_df: summary["visits_today"] = chw_df['patient_id'].nunique()

    if 'ai_risk_score' in chw_df and chw_df['ai_risk_score'].notna().any(): summary["avg_patient_risk_visited_today"] = chw_df['ai_risk_score'].mean()
    if 'ai_followup_priority_score' in chw_df and chw_df['ai_followup_priority_score'].notna().any():
        summary["high_ai_prio_followups_today"] = chw_df[chw_df['ai_followup_priority_score'] >= app_config.FATIGUE_INDEX_HIGH_THRESHOLD]['patient_id'].nunique() # using high fatigue threshold as an example proxy for high priority

    if 'min_spo2_pct' in chw_df and chw_df['min_spo2_pct'].notna().any():
        summary["patients_critical_spo2_today"] = chw_df[chw_df['min_spo2_pct'] < app_config.ALERT_SPO2_CRITICAL_LOW_PCT]['patient_id'].nunique()

    # Use the best available temperature column
    temp_col_chw = next((tc for tc in ['vital_signs_temperature_celsius', 'max_skin_temp_celsius'] if tc in chw_df and chw_df[tc].notna().any()), None)
    if temp_col_chw:
        summary["patients_high_fever_today"] = chw_df[chw_df[temp_col_chw] >= app_config.ALERT_BODY_TEMP_HIGH_FEVER_C]['patient_id'].nunique()

    if 'avg_daily_steps' in chw_df and chw_df['avg_daily_steps'].notna().any(): summary["avg_patient_steps_visited_today"] = chw_df['avg_daily_steps'].mean()
    if 'fall_detected_today' in chw_df and chw_df['fall_detected_today'].notna().any(): summary["patients_fall_detected_today"] = chw_df[chw_df['fall_detected_today'] > 0]['patient_id'].nunique()

    # Pending critical referrals made by this CHW today
    if all(c in chw_df for c in ['condition', 'referral_status', 'referral_reason']):
        # Identifying critical referrals: pending status + condition is in KEY_CONDITIONS_FOR_ACTION or referral_reason indicates urgency
        critical_conditions_for_referral = set(app_config.KEY_CONDITIONS_FOR_ACTION)
        urgent_reason_keywords = ['urgent', 'emergency', 'critical', 'severe']
        
        chw_df['is_critical_referral'] = chw_df.apply(
            lambda row: (
                str(row.get('referral_status', 'Unknown')).lower() == 'pending' and
                (
                    any(cond_key.lower() in str(row.get('condition', 'Unknown')).lower() for cond_key in critical_conditions_for_referral) or
                    any(keyword.lower() in str(row.get('referral_reason', 'Unknown')).lower() for keyword in urgent_reason_keywords)
                )
            ), axis=1
        )
        summary["pending_critical_condition_referrals"] = chw_df[chw_df['is_critical_referral']]['patient_id'].nunique()

    return summary


def get_patient_alerts_for_chw(health_df_daily: pd.DataFrame, source_context: str = "FacilityNode/CHWReport") -> pd.DataFrame:
    """
    Identifies patient alerts for CHW based on daily data.
    Logic here informs rule-based alerts for PED, or batch processing at Facility Node.
    Prioritizes severity and actionable information.
    Returns a DataFrame with patient_id, alert_reason, priority_score, and relevant context.
    """
    logger.info(f"({source_context}) Generating CHW patient alerts.")
    if health_df_daily is None or health_df_daily.empty: return pd.DataFrame()

    df_alerts = health_df_daily.copy()
    alerts_list = []

    # Define relevant columns for alerts and ensure they exist
    # Some might be direct sensor readings, others derived (like AI scores)
    # Note: For PEDs, these checks run continuously. For reports, on daily snapshot.
    cols_for_alerts = { # col_name: default_value (used if missing)
        'patient_id': "Unknown", 'encounter_date': pd.NaT, 'condition': "Unknown",
        'ai_risk_score': np.nan, 'ai_followup_priority_score': np.nan,
        'min_spo2_pct': np.nan, 'vital_signs_temperature_celsius': np.nan, 'max_skin_temp_celsius': np.nan,
        'referral_status': "Unknown", 'referral_reason': "Unknown",
        'fall_detected_today': 0, 'avg_hrv': np.nan, # Example for stress from HRV
        'medication_adherence_self_report': "Unknown"
    }
    for col, default in cols_for_alerts.items():
        if col not in df_alerts.columns: df_alerts[col] = default

    for index, row in df_alerts.iterrows():
        patient_alerts = [] # Store multiple alerts for a single patient row if applicable

        # 1. AI Follow-up Priority Score (if available and high)
        if pd.notna(row['ai_followup_priority_score']) and row['ai_followup_priority_score'] >= app_config.FATIGUE_INDEX_HIGH_THRESHOLD: # Re-using fatigue threshold as proxy
            patient_alerts.append({'reason': "High AI Follow-up Priority", 'score': row['ai_followup_priority_score'], 'details': f"AI Prio: {row['ai_followup_priority_score']:.0f}"})

        # 2. Critical SpO2
        if pd.notna(row['min_spo2_pct']) and row['min_spo2_pct'] < app_config.ALERT_SPO2_CRITICAL_LOW_PCT:
            patient_alerts.append({'reason': "Critical Low SpO2", 'score': 95 + (app_config.ALERT_SPO2_CRITICAL_LOW_PCT - row['min_spo2_pct']), 'details': f"SpO2: {row['min_spo2_pct']:.0f}%"}) # Score higher for lower SpO2

        # 3. High Fever (using best available temp)
        temp_val = row.get('vital_signs_temperature_celsius', row.get('max_skin_temp_celsius', np.nan))
        if pd.notna(temp_val) and temp_val >= app_config.ALERT_BODY_TEMP_HIGH_FEVER_C:
            patient_alerts.append({'reason': "High Fever", 'score': 90 + (temp_val - app_config.ALERT_BODY_TEMP_HIGH_FEVER_C) * 5, 'details': f"Temp: {temp_val:.1f}°C"})

        # 4. Fall Detected
        if pd.notna(row['fall_detected_today']) and row['fall_detected_today'] > 0:
            patient_alerts.append({'reason': "Fall Detected", 'score': 88, 'details': f"Falls: {int(row['fall_detected_today'])}"})

        # 5. High AI Risk Score (if not already covered by AI Follow-up Prio)
        if pd.notna(row['ai_risk_score']) and row['ai_risk_score'] >= app_config.RISK_SCORE_HIGH_THRESHOLD:
            if not any(a['reason'] == "High AI Follow-up Priority" for a in patient_alerts): # Avoid double counting high AI score if follow-up prio used it
                 patient_alerts.append({'reason': "High AI Risk Score", 'score': row['ai_risk_score'], 'details': f"AI Risk: {row['ai_risk_score']:.0f}"})

        # 6. Pending Critical Referral (Simplified for CHW focus)
        critical_conditions_set = set(app_config.KEY_CONDITIONS_FOR_ACTION)
        if str(row.get('referral_status', 'Unknown')).lower() == 'pending' and \
           any(cond_key.lower() in str(row.get('condition', 'Unknown')).lower() for cond_key in critical_conditions_set):
             patient_alerts.append({'reason': f"Pending Critical Referral: {row.get('condition', 'N/A')}", 'score': 80, 'details': f"Ref: {row.get('condition', 'N/A')}"})
        
        # 7. Poor Medication Adherence (Example behavioral flag)
        if str(row.get('medication_adherence_self_report','Unknown')).lower() == 'poor':
             patient_alerts.append({'reason': "Poor Medication Adherence Reported", 'score': 70, 'details': "Adherence: Poor"})


        if patient_alerts:
            # Consolidate: Take the alert with the highest priority score for this patient encounter,
            # or list all unique reasons if scores are similar.
            # For simplicity here, let's take the highest priority one and list its main reason.
            # A PED would show all relevant alerts or use a more complex consolidation.
            top_alert = max(patient_alerts, key=lambda x: x['score'])
            base_info = row[['patient_id', 'encounter_date', 'condition', 'zone_id', 'age', 'gender', 'ai_risk_score', 'ai_followup_priority_score']].to_dict()
            alerts_list.append({
                **base_info,
                'alert_reason_primary': top_alert['reason'],
                'alert_details_summary': top_alert.get('details', ''),
                'priority_score_calculated': top_alert['score']
            })

    if not alerts_list: return pd.DataFrame()
    
    alert_df_final = pd.DataFrame(alerts_list)
    # Clean up display names (e.g., for a table output)
    alert_df_final.rename(columns={
        'alert_reason_primary': 'Alert Reason',
        'alert_details_summary': 'Key Details',
        'priority_score_calculated': 'Priority Score'
    }, inplace=True)
    alert_df_final.sort_values(by='Priority Score', ascending=False, inplace=True)
    return alert_df_final.reset_index(drop=True)


def get_clinic_summary(health_df_period: pd.DataFrame, source_context: str = "FacilityNode/ClinicReport") -> Dict[str, Any]:
    """
    Calculates clinic operational and testing summary. Runs at Facility Node.
    PEDs have simpler, real-time status, not this level of historical aggregation.
    """
    logger.info(f"({source_context}) Calculating clinic summary.")
    summary: Dict[str, Any] = {
        "overall_avg_test_turnaround_conclusive_days": np.nan,
        "perc_critical_tests_tat_met": 0.0,
        "total_pending_critical_tests_patients": 0, # unique patients
        "sample_rejection_rate_perc": 0.0,
        "key_drug_stockouts_count": 0, # count of distinct key drugs stocked out
        "test_summary_details": {} # detailed per-test stats
    }
    if health_df_period is None or health_df_period.empty: return summary
    
    df_cs = health_df_period.copy()

    # Standardize necessary columns
    for col in ['test_type', 'test_result', 'sample_status', 'item', 'zone_id']:
        df_cs[col] = df_cs.get(col, pd.Series(dtype=str)).fillna("Unknown").astype(str)
    df_cs['encounter_date'] = pd.to_datetime(df_cs.get('encounter_date'), errors='coerce')
    for num_col in ['test_turnaround_days', 'item_stock_agg_zone', 'consumption_rate_per_day']:
        df_cs[num_col] = _convert_to_numeric(df_cs.get(num_col), np.nan)

    # 1. Overall Avg TAT for conclusive tests
    conclusive_tests_df = df_cs[
        (~df_cs['test_result'].isin(["Pending", "Rejected Sample", "Unknown", "Indeterminate", "Unknown"])) &
        (df_cs['test_turnaround_days'].notna())
    ].copy()
    if not conclusive_tests_df.empty and conclusive_tests_df['test_turnaround_days'].notna().any():
        summary["overall_avg_test_turnaround_conclusive_days"] = conclusive_tests_df['test_turnaround_days'].mean()

    # 2. % Critical Tests meeting TAT
    critical_test_keys = app_config.CRITICAL_TESTS_LIST # From config
    if critical_test_keys: # Only proceed if critical tests are defined
        critical_conclusive_df = conclusive_tests_df[conclusive_tests_df['test_type'].isin(critical_test_keys)].copy()
        if not critical_conclusive_df.empty:
            def _check_tat_met_cs(row_cs):
                test_config_cs = app_config.KEY_TEST_TYPES_FOR_ANALYSIS.get(row_cs['test_type'])
                target_days_cs = test_config_cs['target_tat_days'] if test_config_cs and 'target_tat_days' in test_config_cs else app_config.TARGET_TEST_TURNAROUND_DAYS # General target
                return pd.notna(row_cs['test_turnaround_days']) and pd.notna(target_days_cs) and row_cs['test_turnaround_days'] <= target_days_cs
            
            critical_conclusive_df['tat_met'] = critical_conclusive_df.apply(_check_tat_met_cs, axis=1)
            if not critical_conclusive_df['tat_met'].empty:
                 summary["perc_critical_tests_tat_met"] = (critical_conclusive_df['tat_met'].mean() * 100)

    # 3. Total Pending Critical Tests (Unique Patients)
    if critical_test_keys and 'patient_id' in df_cs.columns:
        pending_critical_df = df_cs[
            (df_cs['test_type'].isin(critical_test_keys)) &
            (df_cs['test_result'] == "Pending")
        ]
        summary["total_pending_critical_tests_patients"] = pending_critical_df['patient_id'].nunique()

    # 4. Sample Rejection Rate
    processed_samples_df = df_cs[~df_cs['sample_status'].isin(["Pending", "Unknown", "Unknown"])]
    if not processed_samples_df.empty:
        rejected_count = processed_samples_df[processed_samples_df['sample_status'] == 'Rejected'].shape[0]
        summary["sample_rejection_rate_perc"] = (rejected_count / len(processed_samples_df)) * 100
    
    # 5. Key Drug Stockouts
    if all(c in df_cs for c in ['item', 'item_stock_agg_zone', 'consumption_rate_per_day', 'encounter_date']) and app_config.KEY_DRUG_SUBSTRINGS_SUPPLY:
        key_drugs_list = app_config.KEY_DRUG_SUBSTRINGS_SUPPLY # Use the more focused list
        # Filter for key drugs using '|'.join for OR condition in str.contains
        drug_supply_df = df_cs[df_cs['item'].str.contains('|'.join(key_drugs_list), case=False, na=False)].copy()
        if not drug_supply_df.empty:
            drug_supply_df['encounter_date'] = pd.to_datetime(drug_supply_df['encounter_date'], errors='coerce') # Ensure datetime for sort
            drug_supply_df.dropna(subset=['encounter_date'], inplace=True)
            if not drug_supply_df.empty:
                 # Get latest stock info per item (assuming clinic-wide stock if no zone in iot_clinic_df)
                latest_drug_supply_df = drug_supply_df.sort_values('encounter_date').drop_duplicates(subset=['item'], keep='last')
                latest_drug_supply_df['consumption_rate_per_day'] = latest_drug_supply_df['consumption_rate_per_day'].replace(0, np.nan)
                latest_drug_supply_df['days_of_supply'] = latest_drug_supply_df['item_stock_agg_zone'] / latest_drug_supply_df['consumption_rate_per_day']
                summary["key_drug_stockouts_count"] = latest_drug_supply_df[latest_drug_supply_df['days_of_supply'] < app_config.CRITICAL_SUPPLY_DAYS_REMAINING]['item'].nunique()

    # 6. Detailed Per-Test Statistics (still useful for clinic managers)
    test_summary_details = {}
    all_test_types_configured = app_config.KEY_TEST_TYPES_FOR_ANALYSIS
    for test_orig_key, test_props in all_test_types_configured.items():
        display_name = test_props.get("display_name", test_orig_key)
        # Assumes test_type in df_cs holds the original key.
        test_specific_df = df_cs[df_cs['test_type'] == test_orig_key].copy()
        
        current_test_stats = { "positive_rate_perc": np.nan, "avg_tat_days": np.nan, "perc_met_tat_target": 0.0, "pending_count_patients": 0, "rejected_count_patients": 0, "total_conclusive_tests": 0 }
        if test_specific_df.empty: test_summary_details[display_name] = current_test_stats; continue

        conclusive_spec_df = test_specific_df[ (~test_specific_df['test_result'].isin(["Pending", "Rejected Sample", "Unknown", "Indeterminate","Unknown"])) & (test_specific_df['test_turnaround_days'].notna()) ].copy()
        current_test_stats["total_conclusive_tests"] = len(conclusive_spec_df)
        if not conclusive_spec_df.empty:
            current_test_stats["positive_rate_perc"] = (conclusive_spec_df[conclusive_spec_df['test_result'] == 'Positive'].shape[0] / len(conclusive_spec_df)) * 100 if len(conclusive_spec_df) > 0 else 0.0
            if conclusive_spec_df['test_turnaround_days'].notna().any(): current_test_stats["avg_tat_days"] = conclusive_spec_df['test_turnaround_days'].mean()
            
            target_tat_current_test = test_props.get("target_tat_days", app_config.TARGET_TEST_TURNAROUND_DAYS) # fallback to general target
            conclusive_spec_df['tat_met_specific'] = conclusive_spec_df['test_turnaround_days'] <= target_tat_current_test if pd.notna(target_tat_current_test) else False
            if not conclusive_spec_df['tat_met_specific'].empty: current_test_stats["perc_met_tat_target"] = conclusive_spec_df['tat_met_specific'].mean() * 100
        
        current_test_stats["pending_count_patients"] = test_specific_df[test_specific_df['test_result'] == "Pending"]['patient_id'].nunique() if 'patient_id' in test_specific_df.columns else 0
        current_test_stats["rejected_count_patients"] = test_specific_df[test_specific_df['sample_status'] == 'Rejected']['patient_id'].nunique() if 'patient_id' in test_specific_df.columns else 0
        test_summary_details[display_name] = current_test_stats
    summary["test_summary_details"] = test_summary_details
    
    return summary


def get_clinic_environmental_summary(iot_df_period: pd.DataFrame, source_context: str = "FacilityNode/ClinicReport") -> Dict[str, Any]:
    """
    Calculates summary of clinic environmental conditions. For Facility Node reporting.
    PEDs would show immediate, localized ambient conditions from their own/nearby sensors.
    """
    logger.info(f"({source_context}) Calculating clinic environmental summary.")
    summary = {
        "avg_co2_overall_ppm": np.nan, "rooms_co2_very_high_alert_latest_count": 0, # Alert if > VERY_HIGH
        "avg_pm25_overall_ugm3": np.nan, "rooms_pm25_very_high_alert_latest_count": 0,
        "avg_waiting_room_occupancy_overall_persons": np.nan, "waiting_room_high_occupancy_alert_latest_flag": False,
        "avg_noise_overall_dba": np.nan, "rooms_noise_high_alert_latest_count": 0
    }
    if iot_df_period is None or iot_df_period.empty or 'timestamp' not in iot_df_period.columns or not pd.api.types.is_datetime64_any_dtype(iot_df_period['timestamp']):
        return summary

    df_iot = iot_df_period.copy()
    numeric_cols_env = ['avg_co2_ppm', 'avg_pm25', 'waiting_room_occupancy', 'avg_noise_db']
    for col in numeric_cols_env:
        if col in df_iot.columns: df_iot[col] = _convert_to_numeric(df_iot.get(col), np.nan)
        else: df_iot[col] = np.nan # Ensure column exists

    if df_iot['avg_co2_ppm'].notna().any(): summary["avg_co2_overall_ppm"] = df_iot['avg_co2_ppm'].mean()
    if df_iot['avg_pm25'].notna().any(): summary["avg_pm25_overall_ugm3"] = df_iot['avg_pm25'].mean()
    # Focus on 'waiting_room_occupancy' for occupancy KPIs
    if 'waiting_room_occupancy' in df_iot and df_iot['waiting_room_occupancy'].notna().any():
        summary["avg_waiting_room_occupancy_overall_persons"] = df_iot['waiting_room_occupancy'].mean()

    if df_iot['avg_noise_db'].notna().any(): summary["avg_noise_overall_dba"] = df_iot['avg_noise_db'].mean()

    # Latest readings for alert counts
    if all(c in df_iot for c in ['clinic_id', 'room_name', 'timestamp']): # Requires these to identify unique rooms' latest
        latest_room_readings = df_iot.sort_values('timestamp').drop_duplicates(subset=['clinic_id', 'room_name'], keep='last')
        if not latest_room_readings.empty:
            if 'avg_co2_ppm' in latest_room_readings and latest_room_readings['avg_co2_ppm'].notna().any():
                summary["rooms_co2_very_high_alert_latest_count"] = latest_room_readings[latest_room_readings['avg_co2_ppm'] > app_config.ALERT_AMBIENT_CO2_VERY_HIGH_PPM].shape[0]
            if 'avg_pm25' in latest_room_readings and latest_room_readings['avg_pm25'].notna().any():
                summary["rooms_pm25_very_high_alert_latest_count"] = latest_room_readings[latest_room_readings['avg_pm25'] > app_config.ALERT_AMBIENT_PM25_VERY_HIGH_UGM3].shape[0]
            if 'waiting_room_occupancy' in latest_room_readings and latest_room_readings['waiting_room_occupancy'].notna().any():
                 summary["waiting_room_high_occupancy_alert_latest_flag"] = (latest_room_readings['waiting_room_occupancy'] > app_config.TARGET_CLINIC_WAITING_ROOM_OCCUPANCY_MAX).any()
            if 'avg_noise_db' in latest_room_readings and latest_room_readings['avg_noise_db'].notna().any():
                summary["rooms_noise_high_alert_latest_count"] = latest_room_readings[latest_room_readings['avg_noise_db'] > app_config.ALERT_AMBIENT_NOISE_HIGH_DBA].shape[0]
    return summary


def get_patient_alerts_for_clinic(health_df_period: pd.DataFrame, source_context: str = "FacilityNode/ClinicReport") -> pd.DataFrame:
    """
    Identifies patient cases needing review at clinic level from period data.
    This version uses `get_patient_alerts_for_chw` as a base and potentially adds clinic-specific rules.
    Primarily for Facility Node review lists or DHO awareness.
    """
    logger.info(f"({source_context}) Generating clinic patient alerts (based on CHW alert logic plus potential clinic rules).")
    # For now, clinic alerts can reuse CHW alert logic on the period data.
    # Clinic-specific additions could include:
    # - Unusually long pending lab results for critical tests.
    # - Patients with repeated high-risk flags over multiple visits within the period.
    # - Alerts based on specific diagnostic codes indicating severe acute illness.
    # - Lack of follow-up for critical referrals made TO this clinic.
    # The `health_df_period` here is the full data available to the clinic.
    
    # Using a similar structure to get_patient_alerts_for_chw but applying to a clinic's full dataset for the period.
    # Key is the `RISK_SCORE_HIGH_THRESHOLD` for clinic-level concern.
    chw_style_alerts_df = get_patient_alerts_for_chw(health_df_period, source_context=f"{source_context}-BaseAlerts")
    
    # Example of an additional clinic-specific rule: Identify patients with multiple (e.g., >2) high-risk encounters in the period
    if not chw_style_alerts_df.empty and 'patient_id' in chw_style_alerts_df.columns and 'encounter_date' in chw_style_alerts_df.columns and 'AI Risk Score' in chw_style_alerts_df.columns:
        high_risk_encounter_counts = health_df_period[
            health_df_period.get('ai_risk_score',0) >= app_config.RISK_SCORE_HIGH_THRESHOLD
        ].groupby('patient_id')['encounter_date'].nunique()
        
        repeated_high_risk_patients = high_risk_encounter_counts[high_risk_encounter_counts > 2].index.tolist()
        
        if repeated_high_risk_patients:
            # Add/Update alerts for these patients, perhaps increasing priority or adding a specific reason
            for patient_id in repeated_high_risk_patients:
                if patient_id in chw_style_alerts_df['patient_id'].values:
                    idx = chw_style_alerts_df[chw_style_alerts_df['patient_id'] == patient_id].index[0]
                    chw_style_alerts_df.loc[idx, 'Alert Reason'] += "; Repeated High Risk Encounters"
                    chw_style_alerts_df.loc[idx, 'Priority Score'] = max(95, chw_style_alerts_df.loc[idx, 'Priority Score']) # Boost priority
                else:
                    # Add a new alert row if not already alerted by other CHW logic
                    # This requires fetching some latest info for that patient from health_df_period
                    latest_record = health_df_period[health_df_period['patient_id'] == patient_id].sort_values('encounter_date', ascending=False).iloc[0]
                    new_alert_clinic = {
                        'patient_id': patient_id,
                        'encounter_date': latest_record['encounter_date'],
                        'condition': latest_record.get('condition', 'Unknown'),
                        'zone_id': latest_record.get('zone_id', 'Unknown'),
                        'age': latest_record.get('age', np.nan),
                        'gender': latest_record.get('gender', 'Unknown'),
                        'ai_risk_score': latest_record.get('ai_risk_score', np.nan),
                        'ai_followup_priority_score': latest_record.get('ai_followup_priority_score', np.nan),
                        'Alert Reason': "Repeated High Risk Encounters",
                        'Key Details': f"Multiple high-risk visits (count: {high_risk_encounter_counts.get(patient_id, 0)})",
                        'Priority Score': 95
                    }
                    # Concatenate ensuring all columns align. If new_alert_clinic is missing columns present in chw_style_alerts_df, they'll be NaN
                    chw_style_alerts_df = pd.concat([chw_style_alerts_df, pd.DataFrame([new_alert_clinic])], ignore_index=True)
            
            # Re-sort after potential additions/modifications
            chw_style_alerts_df.sort_values(by='Priority Score', ascending=False, inplace=True)
            chw_style_alerts_df.reset_index(drop=True, inplace=True)

    return chw_style_alerts_df


def get_district_summary_kpis(enriched_zone_gdf: gpd.GeoDataFrame, source_context: str = "FacilityNode/DHOReport") -> Dict[str, Any]:
    """
    Calculates district-level summary KPIs from enriched zonal GeoDataFrame.
    For DHO reports/dashboards at Facility Node or Cloud.
    """
    logger.info(f"({source_context}) Calculating district summary KPIs from enriched GDF.")
    kpis: Dict[str, Any] = {
        "total_population_district": 0,
        "population_weighted_avg_ai_risk_score": np.nan,
        "zones_meeting_high_risk_criteria_count": 0, # Zones > DISTRICT_ZONE_HIGH_RISK_AVG_SCORE
        "district_avg_facility_coverage_score": np.nan, # Pop-weighted
        "district_total_active_tb_cases": 0, # Sum from zones
        "district_total_active_malaria_cases": 0, # Sum from zones
        "district_overall_key_disease_prevalence_per_1000": np.nan, # From total key infections & total pop
        "district_population_weighted_avg_steps": np.nan, # For worker wellness insight
        "district_avg_clinic_co2_ppm": np.nan # Avg of zonal clinic CO2 averages
    }
    if not isinstance(enriched_zone_gdf, gpd.GeoDataFrame) or enriched_zone_gdf.empty: return kpis
    
    gdf_dist = enriched_zone_gdf.copy()
    
    # Ensure numeric types for weighted average and sums
    cols_to_num = [
        'population', 'avg_risk_score', 'active_tb_cases', 'active_malaria_cases',
        'total_active_key_infections', 'facility_coverage_score', 'avg_daily_steps_zone', 'zone_avg_co2'
    ]
    for col in cols_to_num:
        if col in gdf_dist.columns: gdf_dist[col] = _convert_to_numeric(gdf_dist.get(col, 0.0), 0.0) # Default to 0 for sums
        else: gdf_dist[col] = 0.0

    if 'population' in gdf_dist.columns: kpis["total_population_district"] = gdf_dist['population'].sum()

    total_pop_for_weighting = kpis["total_population_district"]
    if total_pop_for_weighting is not None and total_pop_for_weighting > 0:
        for metric_col, kpi_key in [
            ('avg_risk_score', 'population_weighted_avg_ai_risk_score'),
            ('facility_coverage_score', 'district_avg_facility_coverage_score'),
            ('avg_daily_steps_zone', 'district_population_weighted_avg_steps')
        ]:
            if metric_col in gdf_dist.columns and gdf_dist[metric_col].notna().any():
                valid_weights = gdf_dist.loc[gdf_dist[metric_col].notna(), 'population']
                valid_values = gdf_dist.loc[gdf_dist[metric_col].notna(), metric_col]
                if not valid_weights.empty and valid_weights.sum() > 0: # Ensure sum of weights > 0
                    kpis[kpi_key] = np.average(valid_values, weights=valid_weights)
                elif not valid_values.empty: kpis[kpi_key] = valid_values.mean() # Unweighted if all weights are zero but values exist
                else: kpis[kpi_key] = np.nan
            else: kpis[kpi_key] = np.nan # Metric col missing or all NaN
        
        if 'total_active_key_infections' in gdf_dist.columns:
            kpis["district_overall_key_disease_prevalence_per_1000"] = \
                (gdf_dist['total_active_key_infections'].sum() / total_pop_for_weighting) * 1000 if total_pop_for_weighting > 0 else 0.0
    else: # If total population is 0 or NaN, fall back to simple means for averages
        logger.warning(f"({source_context}) Total district population is {total_pop_for_weighting}. Using unweighted averages for some district KPIs.")
        for metric_col, kpi_key in [
            ('avg_risk_score', 'population_weighted_avg_ai_risk_score'),
            ('facility_coverage_score', 'district_avg_facility_coverage_score'),
            ('avg_daily_steps_zone', 'district_population_weighted_avg_steps')
        ]:
            kpis[kpi_key] = gdf_dist[metric_col].mean() if metric_col in gdf_dist and gdf_dist[metric_col].notna().any() else np.nan
        kpis["district_overall_key_disease_prevalence_per_1000"] = np.nan # Cannot calculate without population

    if 'avg_risk_score' in gdf_dist.columns:
        kpis["zones_meeting_high_risk_criteria_count"] = gdf_dist[gdf_dist['avg_risk_score'] >= app_config.DISTRICT_ZONE_HIGH_RISK_AVG_SCORE].shape[0]
    
    if 'active_tb_cases' in gdf_dist: kpis["district_total_active_tb_cases"] = int(gdf_dist['active_tb_cases'].sum())
    if 'active_malaria_cases' in gdf_dist: kpis["district_total_active_malaria_cases"] = int(gdf_dist['active_malaria_cases'].sum())
    
    if 'zone_avg_co2' in gdf_dist and gdf_dist['zone_avg_co2'].notna().any():
        kpis["district_avg_clinic_co2_ppm"] = gdf_dist[gdf_dist['zone_avg_co2'] > 0]['zone_avg_co2'].mean() # Exclude zones with 0 CO2 (likely no IoT) from avg
    return kpis


def get_trend_data(df: pd.DataFrame, value_col: str, date_col: str = 'encounter_date', period: str = 'D', agg_func: str = 'mean', filter_col: Optional[str] = None, filter_val: Optional[Any] = None, source_context: str = "FacilityNode/Report") -> pd.Series:
    """
    Generates time series trend data. Runs at Facility Node or Cloud.
    Not for real-time PED display; PEDs show current status or very short-term micro-trends.
    """
    logger.debug(f"({source_context}) Generating trend data for '{value_col}' by '{period}'.")
    if not isinstance(df, pd.DataFrame) or df.empty or date_col not in df.columns or value_col not in df.columns:
        logger.debug(f"Trend data: Input DataFrame invalid or missing columns '{date_col}' or '{value_col}'.")
        return pd.Series(dtype='float64')

    trend_df_work = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(trend_df_work[date_col]):
        trend_df_work[date_col] = pd.to_datetime(trend_df_work[date_col], errors='coerce')
    trend_df_work.dropna(subset=[date_col], inplace=True)

    # For aggregation functions other than 'nunique' or 'count', drop rows where value_col is NaN.
    if agg_func not in ['nunique', 'count'] and value_col in trend_df_work.columns:
        trend_df_work.dropna(subset=[value_col], inplace=True)

    if trend_df_work.empty: logger.debug("Trend data: DataFrame empty after date/NA handling."); return pd.Series(dtype='float64')

    if filter_col and filter_col in trend_df_work.columns and filter_val is not None:
        trend_df_work = trend_df_work[trend_df_work[filter_col] == filter_val]
        if trend_df_work.empty: logger.debug(f"Trend data: DataFrame empty after applying filter: {filter_col}=={filter_val}."); return pd.Series(dtype='float64')

    trend_df_work.set_index(date_col, inplace=True)
    
    # Ensure value_col is numeric for relevant aggregations
    if agg_func in ['mean', 'sum', 'median', 'std', 'var'] and value_col in trend_df_work.columns and not pd.api.types.is_numeric_dtype(trend_df_work[value_col]):
        trend_df_work[value_col] = _convert_to_numeric(trend_df_work[value_col], np.nan)
        trend_df_work.dropna(subset=[value_col], inplace=True) # Drop rows where conversion failed
        if trend_df_work.empty: logger.debug("Trend data: DataFrame empty after numeric conversion and NA drop for aggregation."); return pd.Series(dtype='float64')

    try:
        resampled_data = trend_df_work.groupby(pd.Grouper(freq=period)) # Use Grouper directly
        if agg_func == 'nunique': trend_series = resampled_data[value_col].nunique()
        elif agg_func == 'sum': trend_series = resampled_data[value_col].sum()
        elif agg_func == 'median': trend_series = resampled_data[value_col].median()
        elif agg_func == 'count': trend_series = resampled_data[value_col].count() # True count of non-NA if not dropped
        elif agg_func == 'size': trend_series = resampled_data.size() # Counts all rows in group including NA
        else: trend_series = resampled_data[value_col].mean() # Default to mean
    except Exception as e:
        logger.error(f"({source_context}) Error during trend resampling for {value_col} (agg: {agg_func}): {e}", exc_info=True)
        return pd.Series(dtype='float64')

    return trend_series


def get_supply_forecast_data(
    health_df: pd.DataFrame,
    forecast_days_out: int = 30,
    item_filter_list: Optional[List[str]] = None,
    source_context: str = "FacilityNode"
) -> pd.DataFrame:
    """
    Generates a *simple linear* supply forecast. Runs at Facility Node/Cloud.
    For PEDs, only current personal kit levels and simplified "days remaining" (if synced from hub) are relevant.
    The AI-driven forecast is in ai_analytics_engine.py.
    """
    logger.info(f"({source_context}) Generating simple linear supply forecast.")
    default_cols_supply = ['item', 'date', 'current_stock_at_forecast_start', 'base_consumption_rate_per_day',
                           'forecasted_stock_level', 'forecasted_days_of_supply',
                           'estimated_stockout_date_linear', 'lower_ci_days_supply', 'upper_ci_days_supply',
                           'initial_days_supply_at_forecast_start']
    
    required_cols = ['item', 'encounter_date', 'item_stock_agg_zone', 'consumption_rate_per_day']
    if health_df is None or health_df.empty or not all(c in health_df.columns for c in required_cols):
        logger.warning(f"({source_context}) Missing required columns for supply forecast. Required: {required_cols}")
        return pd.DataFrame(columns=default_cols_supply)

    df_sup_fc = health_df.copy()
    df_sup_fc['encounter_date'] = pd.to_datetime(df_sup_fc['encounter_date'], errors='coerce')
    df_sup_fc.dropna(subset=['encounter_date'], inplace=True)
    for col in ['item_stock_agg_zone', 'consumption_rate_per_day']:
        df_sup_fc[col] = _convert_to_numeric(df_sup_fc.get(col), 0.0) # Default stock/rate to 0 if not numeric
    if df_sup_fc.empty: return pd.DataFrame(columns=default_cols_supply)

    # Get the latest status for each item (potentially per zone if zone is part of item identity)
    latest_supply_status_df = df_sup_fc.sort_values('encounter_date').drop_duplicates(subset=['item'], keep='last') # Simplified: clinic-wide items

    if item_filter_list:
        latest_supply_status_df = latest_supply_status_df[latest_supply_status_df['item'].isin(item_filter_list)]
    if latest_supply_status_df.empty: return pd.DataFrame(columns=default_cols_supply)

    forecast_records = []
    consumption_rate_std_factor = 0.20 # Factor for simple CI estimation (e.g., 20% variation)

    for _, row_item_status in latest_supply_status_df.iterrows():
        item_name = row_item_status['item']
        current_stock = row_item_status.get('item_stock_agg_zone', 0.0)
        base_cons_rate = row_item_status.get('consumption_rate_per_day', 0.0)
        last_known_date = row_item_status['encounter_date']

        if pd.isna(current_stock) or current_stock < 0: current_stock = 0.0
        if pd.isna(base_cons_rate) or base_cons_rate <= 0: base_cons_rate = 0.001 # Avoid division by zero, assume minimal if zero

        forecast_dates = pd.date_range(start=last_known_date + pd.Timedelta(days=1), periods=forecast_days_out, freq='D')
        
        initial_days_supply_val = current_stock / base_cons_rate if base_cons_rate > 0 else np.inf if current_stock > 0 else 0
        est_stockout_date_val = last_known_date + pd.to_timedelta(initial_days_supply_val, unit='D') if np.isfinite(initial_days_supply_val) else pd.NaT

        for i, forecast_date in enumerate(forecast_dates):
            days_elapsed = i + 1
            forecasted_stock = max(0, current_stock - (base_cons_rate * days_elapsed))
            forecasted_days = forecasted_stock / base_cons_rate if base_cons_rate > 0 else np.inf if forecasted_stock > 0 else 0

            # Simple CI based on consumption rate variation
            cons_rate_lower_bound = max(0.001, base_cons_rate * (1 - consumption_rate_std_factor))
            cons_rate_upper_bound = base_cons_rate * (1 + consumption_rate_std_factor)
            
            stock_if_high_cons = max(0, current_stock - (cons_rate_upper_bound * days_elapsed))
            days_if_high_cons = stock_if_high_cons / cons_rate_upper_bound if cons_rate_upper_bound > 0 else np.inf if stock_if_high_cons > 0 else 0
            
            stock_if_low_cons = max(0, current_stock - (cons_rate_lower_bound * days_elapsed))
            days_if_low_cons = stock_if_low_cons / cons_rate_lower_bound if cons_rate_lower_bound > 0 else np.inf if stock_if_low_cons > 0 else 0

            forecast_records.append({
                'item': item_name,
                'date': forecast_date,
                'current_stock_at_forecast_start': current_stock,
                'base_consumption_rate_per_day': base_cons_rate,
                'forecasted_stock_level': forecasted_stock,
                'forecasted_days_of_supply': forecasted_days,
                'estimated_stockout_date_linear': est_stockout_date_val,
                'lower_ci_days_supply': days_if_high_cons, # Lower days of supply if consumption is higher
                'upper_ci_days_supply': days_if_low_cons,   # Upper days of supply if consumption is lower
                'initial_days_supply_at_forecast_start': initial_days_supply_val
            })
    if not forecast_records: return pd.DataFrame(columns=default_cols_supply)
    return pd.DataFrame(forecast_records)
