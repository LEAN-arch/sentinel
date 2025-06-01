# test/config/app_config.py
# Redesigned for "Sentinel Health Co-Pilot" - LMIC Edge-First System
# (With NameError fix for self-referential variable definition)

import os
import pandas as pd # Retained for potential use in higher-tier processing

# --- I. Core System & Directory Configuration ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ASSETS_DIR = os.path.join(BASE_DIR, "assets")
LOCAL_DATA_DIR_PED = os.path.join(BASE_DIR, "local_data_ped")
FACILITY_NODE_DATA_DIR = os.path.join(BASE_DIR, "facility_node_data")

# Paths to primary data sources (used by loading functions for simulation/Tier2-3)
HEALTH_RECORDS_CSV = os.path.join(FACILITY_NODE_DATA_DIR, "health_records_expanded.csv") # Or DATA_SOURCES_DIR from original
ZONE_ATTRIBUTES_CSV = os.path.join(FACILITY_NODE_DATA_DIR, "zone_attributes.csv")
ZONE_GEOMETRIES_GEOJSON = os.path.join(FACILITY_NODE_DATA_DIR, "zone_geometries.geojson")
IOT_CLINIC_ENVIRONMENT_CSV = os.path.join(FACILITY_NODE_DATA_DIR, "iot_clinic_environment.csv")


APP_NAME = "Sentinel Health Co-Pilot"
APP_VERSION = "3.0.0-alpha"
APP_LOGO_SMALL = os.path.join(ASSETS_DIR, "sentinel_logo_small.png") # For PED UI, compact views
APP_LOGO_LARGE = os.path.join(ASSETS_DIR, "sentinel_logo_large.png") # For reports, web dashboards
STYLE_CSS_PATH_WEB = os.path.join(ASSETS_DIR, "style_web_reports.css")

ORGANIZATION_NAME = "LMIC Health Futures Initiative"
APP_FOOTER_TEXT = f"© {pd.Timestamp('now').year} {ORGANIZATION_NAME}. For Field Operations & Public Health Response."
SUPPORT_CONTACT_INFO = "support@lmic-health-futures.org"

# --- II. LMIC-Specific Health & Operational Thresholds ---
ALERT_SPO2_CRITICAL_LOW_PCT = 90
ALERT_SPO2_WARNING_LOW_PCT = 94
ALERT_BODY_TEMP_FEVER_C = 38.0
ALERT_BODY_TEMP_HIGH_FEVER_C = 39.5
ALERT_HR_TACHYCARDIA_ADULT_BPM = 100
ALERT_HR_BRADYCARDIA_ADULT_BPM = 50
HEAT_STRESS_BODY_TEMP_TARGET_C = 37.5
HEAT_STRESS_RISK_BODY_TEMP_C = 38.5

ALERT_AMBIENT_CO2_HIGH_PPM = 1500
ALERT_AMBIENT_CO2_VERY_HIGH_PPM = 2500
ALERT_AMBIENT_PM25_HIGH_UGM3 = 35
ALERT_AMBIENT_PM25_VERY_HIGH_UGM3 = 50
ALERT_AMBIENT_NOISE_HIGH_DBA = 85
ALERT_AMBIENT_HEAT_INDEX_RISK_C = 32
ALERT_AMBIENT_HEAT_INDEX_DANGER_C = 41

FATIGUE_INDEX_MODERATE_THRESHOLD = 60
FATIGUE_INDEX_HIGH_THRESHOLD = 80
STRESS_HRV_LOW_THRESHOLD_MS = 20

TARGET_CLINIC_WAITING_ROOM_OCCUPANCY_MAX = 10
TARGET_CLINIC_PATIENT_THROUGHPUT_MIN_PER_HOUR = 5

RISK_SCORE_LOW_THRESHOLD = 40
RISK_SCORE_MODERATE_THRESHOLD = 60
RISK_SCORE_HIGH_THRESHOLD = 75
DISTRICT_ZONE_HIGH_RISK_AVG_SCORE = 70 # For DHO KPI on high-risk zones
DISTRICT_INTERVENTION_FACILITY_COVERAGE_LOW_PCT = 60
DISTRICT_INTERVENTION_TB_BURDEN_HIGH_ABS = 10
DISTRICT_INTERVENTION_KEY_DISEASE_PREVALENCE_HIGH_PERCENTILE = 0.80

CRITICAL_SUPPLY_DAYS_REMAINING = 7
LOW_SUPPLY_DAYS_REMAINING = 14

# --- III. Edge Device (PED) & Application Configuration ---
EDGE_APP_DEFAULT_LANGUAGE = "en"
EDGE_APP_SUPPORTED_LANGUAGES = ["en", "sw", "fr"]
EDGE_APP_PICTOGRAM_CONFIG_FILE = os.path.join(ASSETS_DIR, "pictogram_map.json")
EDGE_APP_HAPTIC_CONFIG_FILE = os.path.join(ASSETS_DIR, "haptic_patterns.json")
AUDIO_ALERT_FILES_DIR = os.path.join(ASSETS_DIR, "audio_alerts")

EDGE_MODEL_VITALS_DETERIORATION = "vitals_deterioration_v1.tflite"
EDGE_MODEL_FATIGUE_INDEX = "fatigue_index_v1.tflite"
EDGE_MODEL_PERSONALIZED_ANOMALY = "anomaly_detection_base.tflite"
PERSONALIZED_BASELINE_WINDOW_DAYS = 7
EDGE_PROCESSING_INTERVAL_SECONDS = 60

PED_SQLITE_DB_NAME = "sentinel_ped.db"
PED_MAX_LOCAL_LOG_SIZE_MB = 50
EDGE_SYNC_PROTOCOL_PRIORITY = ["BLUETOOTH_PEER", "WIFI_DIRECT_HUB", "QR_PACKET_SHARE", "SD_CARD_TRANSFER"]
QR_PACKET_MAX_SIZE_BYTES = 256
SMS_COMPRESSION_SCHEME = "BASE85_ZLIB"

# --- IV. Data Input Configuration (Conceptual - details would be in schemas) ---
# (Placeholders - actual definitions would be in separate JSON/YAML schemas linked here if complex)
# DEMOGRAPHICS_FIELDS_SCHEMA_PATH = os.path.join(ASSETS_DIR, "schemas/demographics_schema.json")
# SENSOR_STREAMS_SCHEMA_PATH = os.path.join(ASSETS_DIR, "schemas/sensor_streams_schema.json")

# --- V. Supervisor Hub & Facility Node Configuration ---
HUB_DEVICE_SQLITE_DB_NAME = "sentinel_hub.db"
FACILITY_NODE_DB_TYPE = "POSTGRESQL"
FHIR_SERVER_ENDPOINT_LOCAL = "http://localhost:8080/fhir" # Example
DEFAULT_REPORT_INTERVAL_HOURS = 24
CACHE_TTL_SECONDS_WEB_REPORTS = 3600 # For cached data in Streamlit web reports

ESCALATION_PROTOCOL_FILE = os.path.join(ASSETS_DIR, "escalation_protocols.json")

# --- VI. Key Data Semantics & Definitions ---
KEY_TEST_TYPES_FOR_ANALYSIS = {
    "Sputum-AFB": {"disease_group": "TB", "target_tat_days": 2, "critical": True, "display_name": "TB Sputum (AFB)"},
    "Sputum-GeneXpert": {"disease_group": "TB", "target_tat_days": 1, "critical": True, "display_name": "TB GeneXpert"},
    "RDT-Malaria": {"disease_group": "Malaria", "target_tat_days": 0.5, "critical": True, "display_name": "Malaria RDT"},
    "Microscopy-Malaria": {"disease_group": "Malaria", "target_tat_days": 1, "critical": False, "display_name": "Malaria Microscopy"},
    "HIV-Rapid": {"disease_group": "HIV", "target_tat_days": 0.25, "critical": True, "display_name": "HIV Rapid Test"},
    "HIV-ViralLoad": {"disease_group": "HIV", "target_tat_days": 7, "critical": True, "display_name": "HIV Viral Load"},
    "BP Check": {"disease_group": "Hypertension", "target_tat_days": 0, "critical": False, "display_name": "BP Check"},
    # Add other relevant tests from the original full list if needed for higher-tier analytics
}
CRITICAL_TESTS_LIST = [k for k, p in KEY_TEST_TYPES_FOR_ANALYSIS.items() if p.get("critical")]

TARGET_TEST_TURNAROUND_DAYS = 2 # General target if specific test TAT not defined
TARGET_OVERALL_TESTS_MEETING_TAT_PCT_FACILITY = 85
TARGET_SAMPLE_REJECTION_RATE_PCT_FACILITY = 5
OVERDUE_PENDING_TEST_DAYS_GENERAL_FALLBACK = 7

KEY_CONDITIONS_FOR_ACTION = ['TB', 'Malaria', 'HIV-Positive', 'Pneumonia', 'Severe Dehydration', 'Heat Stroke', 'Sepsis', 'Diarrheal Diseases (Severe)']
KEY_DRUG_SUBSTRINGS_SUPPLY = ['TB-Regimen', 'ACT', 'ARV-Regimen', 'ORS', 'Amoxicillin', 'Paracetamol', 'Penicillin', 'Iron-Folate'] # Focused list

# --- VII. Legacy/Web Dashboarding Configuration (for Tiers 2/3) ---
# Specific map center/zoom (e.g., Tijuana) for legacy or general web map default IF NEEDED.
# If not needed, these TIJUANA_ constants can be removed and MAP_DEFAULT_ constants set directly.
TIJUANA_CENTER_LAT = 32.5149    # DEFINED BEFORE USE
TIJUANA_CENTER_LON = -117.0382   # DEFINED BEFORE USE
TIJUANA_DEFAULT_ZOOM = 10       # DEFINED BEFORE USE

WEB_DASHBOARD_DEFAULT_DATE_RANGE_DAYS_VIEW = 1
WEB_DASHBOARD_DEFAULT_DATE_RANGE_DAYS_TREND = 30
WEB_PLOT_DEFAULT_HEIGHT = 400
WEB_PLOT_COMPACT_HEIGHT = 320
WEB_MAP_DEFAULT_HEIGHT = 600

# CORRECTED DEFINITIONS: Direct reference to already defined variables in this module
MAP_DEFAULT_CENTER_LAT = TIJUANA_CENTER_LAT
MAP_DEFAULT_CENTER_LON = TIJUANA_CENTER_LON
MAP_DEFAULT_ZOOM = TIJUANA_DEFAULT_ZOOM

MAPBOX_STYLE_WEB = "carto-positron" # Default open style for web maps
DEFAULT_CRS_STANDARD = "EPSG:4326"

LOG_LEVEL = "INFO"
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - [%(module)s.%(funcName)s:%(lineno)d] - %(message)s'
LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

# --- VIII. Sentinel System Color Palette (for consistency in UIs) ---
COLOR_RISK_HIGH = "#D32F2F"
COLOR_RISK_MODERATE = "#FBC02D"
COLOR_RISK_LOW = "#388E3C"      # Also for "good", "acceptable"
COLOR_RISK_NEUTRAL = "#757575"  # Also for "no_data"
COLOR_ACTION_PRIMARY = "#1976D2" # Main buttons, active elements
COLOR_ACTION_SECONDARY = "#546E7A" # Secondary buttons, less prominent actions
COLOR_POSITIVE_DELTA = "#27AE60"
COLOR_NEGATIVE_DELTA = "#C0392B"

# Legacy disease colors primarily for web reports/dashboards if needed for continuity
LEGACY_DISEASE_COLORS_WEB = {
    "TB": "#EF4444", "Malaria": "#F59E0B", "HIV-Positive": "#8B5CF6",
    "Pneumonia": "#3B82F6", "Anemia": "#10B981", "STI": "#EC4899", "Dengue": "#6366F1",
    "Hypertension": "#F97316", "Diabetes": "#0EA5E9", "Wellness Visit": "#84CC16",
    "Other": "#6B7280",
    # Add others from original full list if still used by higher-tier analytics needing these specific colors
}

# --- End of Configuration ---
