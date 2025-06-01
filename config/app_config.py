# test/config/app_config.py
# Redesigned for "Sentinel Health Co-Pilot" - LMIC Edge-First System

import os
import pandas as pd # Retained for potential use in higher-tier processing

# --- I. Core System & Directory Configuration ---
# BASE_DIR calculation assumes this config file's location relative to project root.
# Adjust if project structure changes.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ASSETS_DIR: For logos, pre-loaded UI pictograms, offline map tiles, JIT guidance media.
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
# LOCAL_DATA_DIR: For on-device SQLite DBs, logs, batched data on PED/Hubs.
LOCAL_DATA_DIR_PED = os.path.join(BASE_DIR, "local_data_ped") # Example path if running locally
# FACILITY_NODE_DATA_DIR: For aggregated data, reports at Facility Node.
FACILITY_NODE_DATA_DIR = os.path.join(BASE_DIR, "facility_node_data")

# APP_BRANDING - Relevant for all tiers
APP_NAME = "Sentinel Health Co-Pilot" # New system name
APP_VERSION = "3.0.0-alpha" # Reflects major redesign
APP_LOGO_SMALL = os.path.join(ASSETS_DIR, "sentinel_logo_small.png") # For PED UI, compact views
APP_LOGO_LARGE = os.path.join(ASSETS_DIR, "sentinel_logo_large.png") # For reports, web dashboards
# NOTE: STYLE_CSS_PATH is now primarily for web-based DHO dashboards (Tier 2/3) or HTML reports.
#       PED UI styling is native or uses ultra-lightweight framework specific rules.
STYLE_CSS_PATH_WEB = os.path.join(ASSETS_DIR, "style_web_reports.css")

# ORGANIZATIONAL & SUPPORT - For identification and help channels
ORGANIZATION_NAME = "LMIC Health Futures Initiative"
APP_FOOTER_TEXT = f"© {pd.Timestamp('now').year} {ORGANIZATION_NAME}. For Field Operations & Public Health Response."
SUPPORT_CONTACT_INFO = "support@lmic-health-futures.org" # Email or local support number protocol

# --- II. LMIC-Specific Health & Operational Thresholds (Critical for Edge AI & Alerts) ---
# These need to be carefully validated for local context and can be part of configuration pushed to devices.

# A. VITAL SIGN ALERT THRESHOLDS (for individual worker & patient monitoring)
ALERT_SPO2_CRITICAL_LOW_PCT = 90       # Immediate critical alert, potential escalation
ALERT_SPO2_WARNING_LOW_PCT = 94        # Warning alert, advise check/rest
ALERT_BODY_TEMP_FEVER_C = 38.0         # Fever warning
ALERT_BODY_TEMP_HIGH_FEVER_C = 39.5    # High fever critical alert
ALERT_HR_TACHYCARDIA_ADULT_BPM = 100   # Resting Tachycardia
ALERT_HR_BRADYCARDIA_ADULT_BPM = 50    # Resting Bradycardia
# Heat Stress specific (can be combined with ambient)
HEAT_STRESS_BODY_TEMP_TARGET_C = 37.5  # Target to stay below for worker
HEAT_STRESS_RISK_BODY_TEMP_C = 38.5    # Elevated risk

# B. AMBIENT ENVIRONMENT ALERT THRESHOLDS (for PED using phone/connected sensors)
ALERT_AMBIENT_CO2_HIGH_PPM = 1500      # Elevated CO2, recommend ventilation check
ALERT_AMBIENT_CO2_VERY_HIGH_PPM = 2500 # Very High CO2, recommend immediate ventilation/limit occupancy
ALERT_AMBIENT_PM25_HIGH_UGM3 = 35      # Unhealthy PM2.5 (WHO interim target 1)
ALERT_AMBIENT_PM25_VERY_HIGH_UGM3 = 50 # Very Unhealthy PM2.5
ALERT_AMBIENT_NOISE_HIGH_DBA = 85      # Sustained high noise (hearing risk / communication issue)
ALERT_AMBIENT_HEAT_INDEX_RISK_C = 32   # Caution level for heat index
ALERT_AMBIENT_HEAT_INDEX_DANGER_C = 41 # Danger level for heat index

# C. WORKER FATIGUE & STRESS THRESHOLDS (derived by Edge AI)
# These are conceptual; actual values depend on model features.
FATIGUE_INDEX_MODERATE_THRESHOLD = 60  # Scale 0-100 from Edge AI model
FATIGUE_INDEX_HIGH_THRESHOLD = 80
STRESS_HRV_LOW_THRESHOLD_MS = 20       # Example for RMSSD or SDNN, model specific

# D. CLINIC/FACILITY SPECIFIC THRESHOLDS (for Tier 1/2 alerts, less PED direct)
TARGET_CLINIC_WAITING_ROOM_OCCUPANCY_MAX = 10 # Max persons for safety/infection control
TARGET_CLINIC_PATIENT_THROUGHPUT_MIN_PER_HOUR = 5 # For operational monitoring

# E. RISK SCORE & INTERVENTION THRESHOLDS (used across tiers)
# Simplified risk categories for clarity on PED.
RISK_SCORE_LOW_THRESHOLD = 40          # Color Green
RISK_SCORE_MODERATE_THRESHOLD = 60     # Color Yellow
RISK_SCORE_HIGH_THRESHOLD = 75         # Color Red
# DHO-level specific thresholds for triggering zonal reviews.
DISTRICT_ZONE_HIGH_RISK_AVG_SCORE = 70
DISTRICT_INTERVENTION_FACILITY_COVERAGE_LOW_PCT = 60
DISTRICT_INTERVENTION_TB_BURDEN_HIGH_ABS = 10 # Absolute cases, more direct than rate for intervention
DISTRICT_INTERVENTION_KEY_DISEASE_PREVALENCE_HIGH_PERCENTILE = 0.80 # Top 20% prevalence zones

# F. SUPPLY CHAIN THRESHOLDS
CRITICAL_SUPPLY_DAYS_REMAINING = 7     # For PED personal kit & Facility Node stock
LOW_SUPPLY_DAYS_REMAINING = 14

# --- III. Edge Device (PED) & Application Configuration ---
# A. UI & UX
EDGE_APP_DEFAULT_LANGUAGE = "en"       # Can be overridden by user profile
EDGE_APP_SUPPORTED_LANGUAGES = ["en", "sw", "fr"] # Example: English, Swahili, French
# PICTOGRAM_MAPPINGS: This would be a dictionary mapping actions/items to asset names
# e.g., "hydrate": "pict_water_drop.png", "task_visit": "pict_house_call.png"
EDGE_APP_PICTOGRAM_CONFIG_FILE = os.path.join(ASSETS_DIR, "pictogram_map.json")
# HAPTIC_ALERT_PATTERNS: Define patterns for none/low/medium/high severity alerts
# e.g., "high_severity": [1000, 500, 1000, 500], "medium_severity": [500, 200, 500] (on, off ms)
EDGE_APP_HAPTIC_CONFIG_FILE = os.path.join(ASSETS_DIR, "haptic_patterns.json")
AUDIO_ALERT_FILES_DIR = os.path.join(ASSETS_DIR, "audio_alerts") # Contains language-specific pre-recorded alerts

# B. Edge AI Models & Processing
# For pre-trained models bundled with the app. Names or relative paths within app package.
EDGE_MODEL_VITALS_DETERIORATION = "vitals_deterioration_v1.tflite"
EDGE_MODEL_FATIGUE_INDEX = "fatigue_index_v1.tflite"
EDGE_MODEL_PERSONALIZED_ANOMALY = "anomaly_detection_base.tflite" # Base for personalization
PERSONALIZED_BASELINE_WINDOW_DAYS = 7    # How many days of data to establish/update personal baseline
EDGE_PROCESSING_INTERVAL_SECONDS = 60    # How often to run main Edge AI loop on new sensor data
# ON_DEVICE_MODEL_UPDATE_CONFIG: Parameters for simple local model calibration if used.
# e.g. "min_datapoints_for_recal": 100, "recal_frequency_hours": 24

# C. Data Management & Synchronization
PED_SQLITE_DB_NAME = "sentinel_ped.db"
PED_MAX_LOCAL_LOG_SIZE_MB = 50         # For sensor/app logs before rollover
# SYNC_PROTOCOL_PRIORITY: Order for attempting sync
# e.g., ["BLUETOOTH_GROUP", "WIFI_DIRECT_FACILITY", "QR_PACKET", "SD_CARD", "SMS_CRITICAL", "CELLULAR_BATCHED"]
EDGE_SYNC_PROTOCOL_PRIORITY = ["BLUETOOTH_PEER", "WIFI_DIRECT_HUB", "QR_PACKET_SHARE", "SD_CARD_TRANSFER"]
QR_PACKET_MAX_SIZE_BYTES = 256        # Max data per single QR code in a sequence
SMS_COMPRESSION_SCHEME = "BASE85_ZLIB" # Example
# OPPORTUNISTIC_SYNC_MIN_BATTERY_PCT = 20 # Don't attempt energy intensive sync if battery low

# --- IV. Data Input Configuration (Minimum Viable, High-Value) ---
# This section defines expected data fields and their basic types/properties.
# This informs data capture forms, Edge AI feature expectations, and FHIR mapping.

# DEMOGRAPHICS_FIELDS (for worker and simplified patient)
# e.g., {'age': {'type': 'integer', 'range': [0,120]}, 'sex': {'type': 'enum', 'values': ['Male','Female','Other','Unknown']}}
# JOB_CONTEXT_FIELDS (for worker)
# SENSOR_STREAMS_CONFIG (defines which sensors map to which streams, sample rates if configurable)
#   e.g., "HRV": {"source_sensor": "PPG", "required": True, "processing": "RMSSD_5min_window"}
# AMBIENT_SIGNALS_CONFIG
# BEHAVIORAL_EVENT_TYPES (derived from sensor patterns or user input)
#   e.g., "prolonged_inactivity", "fall_detected", "rapid_ui_error_rate"
# PSYCHOMETRIC_INPUT_CONFIG (for emoji/slider checks)
# INCIDENT_LOGGING_TYPES (pre-defined icons/categories for quick logging)
# Note: Detailed structure for these *_FIELDS/*_CONFIG would be in separate JSON/YAML
# or more complex Python dicts if needed for validation schemas etc. This app_config
# would just point to them or hold high-level keys. For brevity, not fully detailed here.

# --- V. Supervisor Hub & Facility Node Configuration ---
# A. Aggregation & Reporting
HUB_DEVICE_SQLITE_DB_NAME = "sentinel_hub.db"
FACILITY_NODE_DB_TYPE = "POSTGRESQL" # Or "SQLITE" for very light nodes
# FACILITY_NODE_DB_CONNECTION_PARAMS: Handled via environment variables for security usually
FHIR_SERVER_ENDPOINT_LOCAL = "http://localhost:8080/fhir" # If Facility Node runs a FHIR server
DEFAULT_REPORT_INTERVAL_HOURS = 24 # For DHO summary reports from Facility Node
# CACHE_TTL_SECONDS_WEB: Caching for any web dashboards served from Facility Node or Cloud.
CACHE_TTL_SECONDS_WEB_REPORTS = 3600

# B. Higher-Level Alerting & Communication
ESCALATION_PROTOCOL_FILE = os.path.join(ASSETS_DIR, "escalation_protocols.json") # Defines supervisor/clinic contact paths
# INTERVENTION_CRITERIA_CONFIG: (Potentially links to criteria defined in section II.E or more complex rules)

# --- VI. Key Data Semantics & Definitions (Mappings for Consistency) ---
# (Existing KEY_TEST_TYPES_FOR_ANALYSIS, KEY_CONDITIONS_FOR_TRENDS, KEY_DRUG_SUBSTRINGS_SUPPLY
#  are still relevant for backend processing, analytics at Facility/Cloud tier,
#  and for populating choice lists or interpreting text inputs.
#  Their direct use on PED is limited to matching predefined strings if necessary.)

KEY_TEST_TYPES_FOR_ANALYSIS = { # Still relevant for Facility/Cloud analytics, and matching string inputs
    "Sputum-AFB": {"disease_group": "TB", "target_tat_days": 2, "critical": True, "display_name": "TB Sputum (AFB)"},
    # ... (rest of the original KEY_TEST_TYPES_FOR_ANALYSIS) ...
    "BP Check": {"disease_group": "Hypertension", "target_tat_days": 0, "critical": False, "display_name": "BP Check"},
}
CRITICAL_TESTS_LIST = [k for k, p in KEY_TEST_TYPES_FOR_ANALYSIS.items() if p.get("critical")]

# Target Turnaround Times: Useful for Facility Node monitoring performance against goals
TARGET_OVERALL_TESTS_MEETING_TAT_PCT_FACILITY = 85
TARGET_SAMPLE_REJECTION_RATE_PCT_FACILITY = 5
OVERDUE_PENDING_TEST_DAYS_GENERAL_FALLBACK = 7 # General fallback for Facility Node logic

KEY_CONDITIONS_FOR_ACTION = ['TB', 'Malaria', 'HIV-Positive', 'Pneumonia', 'Severe Dehydration', 'Heat Stroke', 'Sepsis', 'Diarrheal Diseases (Severe)'] # Focused list for action
KEY_DRUG_SUBSTRINGS_SUPPLY = ['TB-Regimen', 'ACT', 'ARV-Regimen', 'ORS', 'Amoxicillin', 'Paracetamol', 'Penicillin'] # More focused for essential meds

# --- VII. Legacy/Dashboarding Configuration (for DHO Web Views - Tier 2/3) ---
# These settings from the original app primarily apply to any web-based summary dashboards.
WEB_DASHBOARD_DEFAULT_DATE_RANGE_DAYS_VIEW = 1
WEB_DASHBOARD_DEFAULT_DATE_RANGE_DAYS_TREND = 30
WEB_PLOT_DEFAULT_HEIGHT = 400
WEB_PLOT_COMPACT_HEIGHT = 320
WEB_MAP_DEFAULT_HEIGHT = 600
MAP_DEFAULT_CENTER_LAT = app_config.TIJUANA_CENTER_LAT if hasattr(app_config, 'TIJUANA_CENTER_LAT') else 0.0 # Fallback example
MAP_DEFAULT_CENTER_LON = app_config.TIJUANA_CENTER_LON if hasattr(app_config, 'TIJUANA_CENTER_LON') else 0.0
MAP_DEFAULT_ZOOM = app_config.TIJUANA_DEFAULT_ZOOM if hasattr(app_config, 'TIJUANA_DEFAULT_ZOOM') else 2
MAPBOX_STYLE_WEB = app_config.MAPBOX_STYLE if hasattr(app_config, 'MAPBOX_STYLE') else "carto-positron"
DEFAULT_CRS_STANDARD = "EPSG:4326" # Standard CRS for interoperability

# LOGGING (for Python backend components at Tier 2/3, and potentially for PED native logs)
LOG_LEVEL = "INFO" # Can be overridden by ENV VAR
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - [%(module)s.%(funcName)s:%(lineno)d] - %(message)s'
LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

# COLORS (for visual consistency in UIs where colors are used - PED and Web)
# Simplified, high-contrast for PEDs. Can be more nuanced for web.
COLOR_RISK_HIGH = "#D32F2F"      # Strong Red
COLOR_RISK_MODERATE = "#FBC02D"  # Strong Yellow/Amber
COLOR_RISK_LOW = "#388E3C"       # Strong Green
COLOR_RISK_NEUTRAL = "#757575"   # Grey
COLOR_ACTION_PRIMARY = "#1976D2"   # Strong Blue for primary actions/buttons
COLOR_ACTION_SECONDARY = "#546E7A" # Blue-Grey for secondary actions

# Store existing DISEASES_COLORS for reference if used by higher tiers,
# but PED might use a simplified, more iconic system for diseases.
LEGACY_DISEASE_COLORS_WEB = {
    "TB": "#EF4444", "Malaria": "#F59E0B", "HIV-Positive": "#8B5CF6",
    # ... (rest of original DISEASE_COLORS) ...
    "Other": "#6B7280",
}

# --- End of Configuration ---
# Note: For actual PED deployment, many of these configs would be:
# 1. Bundled into the app package.
# 2. Remotely configurable/updatable via secure sync from Facility Node or Cloud (if connected).
# 3. Sensitive items like API keys/DB creds (for Facility Node/Cloud) NEVER hardcoded, always ENV VARS or secrets management.
