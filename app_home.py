# test/app_home.py
# Refined for "Sentinel Health Co-Pilot" - LMIC Edge-First System Demonstrator

import streamlit as st
import os
# pandas is not directly used for display logic here, but app_config uses it.
from config import app_config # Uses the fully redesigned app_config
import logging

# --- Page Configuration (Reflects new system identity) ---
st.set_page_config(
    page_title=f"{app_config.APP_NAME} - System Overview", # Updated title
    page_icon=app_config.APP_LOGO_SMALL if os.path.exists(app_config.APP_LOGO_SMALL) else "🌍", # Smaller logo for general use
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={ # Updated help/about links using new config values
        'Get Help': f"mailto:{app_config.SUPPORT_CONTACT_INFO}?subject=Help Request - {app_config.APP_NAME}",
        'Report a bug': f"mailto:{app_config.SUPPORT_CONTACT_INFO}?subject=Bug Report - {app_config.APP_NAME} v{app_config.APP_VERSION}",
        'About': f"""
        ### {app_config.APP_NAME}
        **Version:** {app_config.APP_VERSION}
        An Edge-First Health Intelligence & Action Co-Pilot for LMIC Environments.
        {app_config.APP_FOOTER_TEXT}
        This platform (Sentinel Health Co-Pilot) prioritizes offline-first operations,
        actionable insights for frontline workers, and resilient data systems.
        The interfaces demonstrated here primarily represent supervisor/manager/DHO views
        (typically at Facility or Cloud Nodes), not the native Personal Edge Device (PED) UI.
        """
    }
)

# --- Logging Setup ---
# BasicConfig should ideally be done once if multiple modules use logging.
# If other modules also call basicConfig, it might lead to duplicate handlers or issues.
# For a single entry point app, this is okay.
logging.basicConfig(
    level=getattr(logging, app_config.LOG_LEVEL.upper(), logging.INFO),
    format=app_config.LOG_FORMAT,
    datefmt=app_config.LOG_DATE_FORMAT,
    handlers=[logging.StreamHandler()] # Ensure logs go to console for Streamlit Cloud / dev
)
logger = logging.getLogger(__name__)

# --- CSS Loading (for Web Views - Tier 2/3) ---
@st.cache_resource # Function to load CSS
def load_web_css(css_file_path: str):
    if os.path.exists(css_file_path):
        try:
            with open(css_file_path, encoding="utf-8") as f: # Specify encoding
                st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
            logger.info(f"Web CSS loaded successfully from {css_file_path}")
        except Exception as e_css:
            logger.error(f"Error reading web CSS file {css_file_path}: {e_css}")
    else:
        logger.warning(f"Web CSS file not found: {css_file_path}. Default Streamlit styles will apply.")

# Load the CSS intended for web reports/dashboards
load_web_css(app_config.STYLE_CSS_PATH_WEB)


# --- App Header (Reflects New Branding & System) ---
header_cols_home = st.columns([0.15, 0.85])
with header_cols_home[0]:
    # Use the larger logo for the main landing page if available
    main_page_logo_to_use = app_config.APP_LOGO_LARGE if os.path.exists(app_config.APP_LOGO_LARGE) else app_config.APP_LOGO_SMALL
    if os.path.exists(main_page_logo_to_use):
        st.image(main_page_logo_to_use, width=100) # Adjust width as needed
    else:
        st.markdown("🌍", unsafe_allow_html=True) # Fallback icon

with header_cols_home[1]:
    st.title(app_config.APP_NAME)
    st.caption(f"Version {app_config.APP_VERSION}  |  Transforming Data into Lifesaving Action at the Edge.")
st.markdown("---")

# --- Introduction to Sentinel Health Co-Pilot ---
st.markdown(f"""
    #### Welcome to the {app_config.APP_NAME} System Overview
    
    The **Sentinel Health Co-Pilot** is an **edge-first health intelligence system** redesigned for maximum clinical and
    operational actionability in resource-limited, high-risk environments. It bridges advanced technology with
    real-world field utility by converting wearable, IoT, and contextual data into life-saving, workflow-integrated
    decisions, even with minimal or no internet connectivity.

    **Core Principles:**
    - **Offline-First Operations:** On-device Edge AI on Personal Edge Devices (PEDs) ensures functionality without continuous connectivity.
    - **Action-Oriented Intelligence:** Every insight aims to trigger a clear, targeted response relevant to frontline workflows.
    - **Human-Centered Design:** Interfaces are optimized for low-literacy, high-stress users, prioritizing immediate understanding.
    - **Resilience & Scalability:** Modular design allows scaling from individual PEDs to facility and regional views, with robust, flexible data synchronization.

    👈 **The sidebar provides navigation to simulated views for different operational tiers.**
    These views primarily represent what supervisors, clinic managers, or District Health Officers (DHOs) might see
    at a **Facility Node (Tier 2)** or **Regional/Cloud Node (Tier 3)**.
    The primary interface for frontline workers (e.g., CHWs) is a **native mobile/wearable application on their Personal Edge Device (PED)**,
    which is designed for their specific high-stress, low-resource context and is not fully replicated here.
""")
st.info("💡 **Note:** This web application serves as a demonstrator for the system's data processing capabilities and higher-level reporting views.")


st.subheader("Simulated Role-Specific Views (Facility/Regional Level)")

# --- Updated Expander Descriptions for clarity on target user & system tier ---

with st.expander("🧑‍⚕️ **CHW Operations Summary & Field Support View (Supervisor/Hub Level)**", expanded=False):
    st.markdown("""
    This view simulates how a CHW Supervisor or a Hub coordinator might access summarized data from CHW Personal Edge Devices (PEDs).
    - **Focus (Tier 1-2):** Team performance monitoring, targeted support for CHWs, localized outbreak signal detection based on aggregated CHW reports.
    - **Key Data Points:** CHW activity summaries (visits, tasks completed), patient alert escalations, critical supply needs for CHW kits, early epidemiological signals from specific zones.
    - **Objective:** Enable supervisors to manage CHW teams effectively, provide timely support, identify emerging health issues quickly, and coordinate local responses.
    *The CHW's primary tool is their offline-first native app on their PED, providing real-time alerts & task management.*
    """)
    if st.button("Go to CHW Summary View", key="nav_chw_home_sentinel", type="primary"):
        st.switch_page("pages/1_chw_dashboard.py") # This page would need significant redesign to reflect supervisor view

with st.expander("🏥 **Clinic Operations & Environmental Safety View (Facility Node Level)**", expanded=False):
    st.markdown("""
    Simulates a dashboard for Clinic Managers at a Facility Node (Tier 2), providing insights into service efficiency, care quality, resource management, and environmental conditions.
    - **Focus (Tier 2):** Optimizing clinic workflows, ensuring quality patient care, managing supplies and testing backlogs, monitoring clinic environment for safety and infection control.
    - **Key Data Points:** Clinic performance KPIs (e.g., test TAT, patient throughput), supply stock forecasts, IoT sensor data summaries (CO2, PM2.5, occupancy), clinic-level epidemiological trends, flagged patient cases for review.
    - **Objective:** Enhance operational efficiency, support clinical decision-making, maintain resource availability, and ensure a safe clinic environment.
    """)
    if st.button("Go to Clinic View", key="nav_clinic_home_sentinel", type="primary"):
        st.switch_page("pages/2_clinic_dashboard.py") # This page would be a web dashboard for clinic manager

with st.expander("🗺️ **District Health Strategic Overview (DHO at Facility/Regional Node Level)**", expanded=False):
    st.markdown("""
    Presents a strategic dashboard for District Health Officers (DHOs), typically accessed at a Facility Node (Tier 2) or a Regional/Cloud Node (Tier 3).
    - **Focus (Tier 2-3):** Population health insights, resource allocation across zones, monitoring environmental well-being, and planning targeted interventions.
    - **Key Data Points:** District-wide health KPIs, interactive maps for zonal comparisons (risk, disease burden, resources), trend analyses, intervention planning tools based on aggregated data.
    - **Objective:** Support evidence-based strategic planning, public health interventions, program monitoring, and policy development for the district.
    """)
    if st.button("Go to District Overview", key="nav_dho_home_sentinel", type="primary"):
        st.switch_page("pages/3_district_dashboard.py") # This page is a DHO-level web dashboard

with st.expander("📊 **Population Health Analytics Deep Dive (Epidemiologist/Analyst View - Tier 3)**", expanded=True):
    st.markdown("""
    A view designed for detailed epidemiological and health systems analysis, typically used by analysts or program managers at a Regional/Cloud Node (Tier 3) with access to more comprehensive, aggregated datasets.
    - **Focus (Tier 3):** In-depth analysis of demographic patterns, SDOH impacts, clinical trends, health system performance, and equity across broader populations.
    - **Key Data Points:** Stratified disease burden, AI risk distributions by various factors, aggregated test positivity trends, comorbidity analysis, referral pathway performance, health equity metrics.
    - **Objective:** Provide robust analytical capabilities to understand population health dynamics, evaluate interventions, identify areas for research, and inform large-scale public health strategy.
    """)
    if st.button("Go to Population Analytics", key="nav_pop_analytics_home_sentinel", type="primary"):
        st.switch_page("pages/4_population_dashboard.py") # Renamed for clarity, DHO/analyst tool

st.markdown("---")
st.subheader(f"{app_config.APP_NAME} - Key Capabilities Reimagined")
col1_cap, col2_cap, col3_cap = st.columns(3)
with col1_cap:
    st.markdown("##### 🛡️ **Frontline Worker Safety & Support**")
    st.markdown("<small>Real-time vitals/environmental monitoring, fatigue detection, and safety nudges on Personal Edge Devices (PEDs).</small>", unsafe_allow_html=True)
    st.markdown("##### 🌍 **Offline-First Edge AI**")
    st.markdown("<small>On-device intelligence for alerts, prioritization, and guidance with zero reliance on continuous connectivity.</small>", unsafe_allow_html=True)
with col2_cap:
    st.markdown("##### ⚡ **Actionable, Contextual Insights**")
    st.markdown("<small>From raw data to clear, role-specific recommendations that integrate into field workflows.</small>", unsafe_allow_html=True)
    st.markdown("##### 🤝 **Human-Centered & Accessible UX**")
    st.markdown("<small>Pictogram-based UIs, voice/tap commands, and local language support for low-literacy, high-stress users on PEDs.</small>", unsafe_allow_html=True)
with col3_cap:
    st.markdown("##### 📡 **Resilient Data Synchronization**")
    st.markdown("<small>Flexible data sharing (Bluetooth, QR, SD card, SMS, opportunistic IP) across PEDs, Hubs, and Nodes.</small>", unsafe_allow_html=True)
    st.markdown("##### 🌱 **Scalable & Interoperable Architecture**")
    st.markdown("<small>Modular design from personal to national levels, with FHIR/HL7 compliance for system integration.</small>", unsafe_allow_html=True)


with st.expander("📜 **Glossary of Terms** - Definitions for Sentinel Co-Pilot context", expanded=False):
    st.markdown("""
    - Understand terminology specific to the Sentinel Health Co-Pilot system, including Edge AI, PEDs, and LMIC-focused metrics.
    - Clarify technical definitions and operational terms.
    """)
    if st.button("Go to Sentinel Glossary", key="nav_glossary_home_sentinel", type="secondary"):
        st.switch_page("pages/5_Glossary.py") # Glossary page also needs update for new terms


# --- Sidebar Content ---
st.sidebar.header(f"{app_config.APP_NAME} System") # Shorter title
if os.path.exists(app_config.APP_LOGO_SMALL):
    st.sidebar.image(app_config.APP_LOGO_SMALL, width=180) # Slightly smaller for sidebar consistency
st.sidebar.caption(f"Version {app_config.APP_VERSION}")
st.sidebar.markdown("---")
st.sidebar.markdown("### System Overview")
st.sidebar.info(
    "This is the central overview of the Sentinel Health Co-Pilot system demonstrator. "
    "Select a simulated higher-tier view from the navigation."
)
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Support & Info:**<br/>{app_config.ORGANIZATION_NAME}<br/>"
                    f"Contact: [{app_config.SUPPORT_CONTACT_INFO}](mailto:{app_config.SUPPORT_CONTACT_INFO})", unsafe_allow_html=True)
st.sidebar.markdown("---")
st.sidebar.caption(app_config.APP_FOOTER_TEXT)


logger.info(f"Application home page ({app_config.APP_NAME}) loaded successfully.")
