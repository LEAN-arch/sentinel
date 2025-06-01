/* test/assets/style_web_reports.css */
/* Styles for Sentinel Health Co-Pilot Web Dashboards & Reports (Tiers 2/3) */

/* Define Sentinel Color Variables (Based on Conceptual app_config) */
/* In a real setup, these might be injected or aligned with app_config.py definitions */
:root {
    --sentinel-risk-high-color: #D32F2F; /* Strong Red */
    --sentinel-risk-moderate-color: #FBC02D; /* Strong Yellow/Amber */
    --sentinel-risk-low-color: #388E3C; /* Strong Green (maps to "good" or "acceptable") */
    --sentinel-risk-neutral-color: #757575; /* Grey */
    --sentinel-accent-blue-bright: #4D7BF3; /* Brighter Accent Blue for H1 underline */
    --sentinel-text-dark: #343a40;
    --sentinel-text-headings-dark-blue: #1A2557;
    --sentinel-text-headings-slate-blue: #2C3E50;
    --sentinel-text-link-blue: #3498DB;
    --sentinel-background-light-grey: #f8f9fa;
    --sentinel-background-white: #ffffff;
    --sentinel-border-light: #dee2e6;
    --sentinel-border-medium: #ced4da;
    --sentinel-positive-color: #27AE60; /* Green for positive delta */
    --sentinel-negative-color: #C0392B; /* Red for negative delta */
}

/* ----- Base Styles & Typography ----- */
body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif, "Apple Color Emoji", "Segoe UI Emoji", "Noto Color Emoji";
    background-color: var(--sentinel-background-light-grey);
    color: var(--sentinel-text-dark);
    line-height: 1.65;
    font-size: 15px; /* Base font size for reports */
}

.stApp { /* Main Streamlit app container */
    /* App-wide padding can be useful for reports */
    padding: 1rem 1.5rem;
}

/* ----- Headings & Titles ----- */
h1, h2, h3, h4, h5, h6 {
    color: var(--sentinel-text-headings-dark-blue);
    font-weight: 700;
    letter-spacing: -0.02em;
}
h1 { /* Page Titles */
    font-size: 2.2rem; /* Slightly smaller for reports */
    border-bottom: 3px solid var(--sentinel-accent-blue-bright);
    padding-bottom: 0.6rem;
    margin-bottom: 1.8rem;
}
h2 { /* Major Section Titles */
    font-size: 1.7rem;
    margin-top: 2.5rem;
    margin-bottom: 1.2rem;
    color: var(--sentinel-text-headings-slate-blue);
    border-bottom: 2px solid var(--sentinel-border-light);
    padding-bottom: 0.4rem;
}
h3 { /* Sub-section Titles / Chart Titles (if not part of Plotly theme) */
    font-size: 1.35rem;
    color: var(--sentinel-text-link-blue);
    margin-top: 1.8rem;
    margin-bottom: 0.9rem;
}

/* ----- KPI Card Styling (for render_web_kpi_card) ----- */
.kpi-card {
    background-color: var(--sentinel-background-white);
    border-radius: 10px; /* Softer corners */
    padding: 1.4rem;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.06); /* Softer shadow */
    border-left: 6px solid var(--sentinel-risk-neutral-color); /* Default neutral accent */
    margin-bottom: 1.2rem;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    height: 100%; /* For consistent height in st.columns */
    transition: box-shadow 0.2s ease-in-out;
}
.kpi-card:hover {
    box-shadow: 0 6px 18px rgba(0, 0, 0, 0.09);
}

/* KPI Card Status Variants - map to new status_level names */
.kpi-card.status-high-risk, .kpi-card.status-high-concern { border-left-color: var(--sentinel-risk-high-color); }
.kpi-card.status-high-risk .kpi-icon, .kpi-card.status-high-concern .kpi-icon { color: var(--sentinel-risk-high-color); }

.kpi-card.status-moderate-risk, .kpi-card.status-moderate-concern { border-left-color: var(--sentinel-risk-moderate-color); }
.kpi-card.status-moderate-risk .kpi-icon, .kpi-card.status-moderate-concern .kpi-icon { color: var(--sentinel-risk-moderate-color); }

.kpi-card.status-low-risk, .kpi-card.status-acceptable, .kpi-card.status-good-performance { border-left-color: var(--sentinel-risk-low-color); }
.kpi-card.status-low-risk .kpi-icon, .kpi-card.status-acceptable .kpi-icon, .kpi-card.status-good-performance .kpi-icon { color: var(--sentinel-risk-low-color); }

.kpi-card.status-neutral, .kpi-card.status-no-data { border-left-color: var(--sentinel-risk-neutral-color); }
.kpi-card.status-neutral .kpi-icon, .kpi-card.status-no-data .kpi-icon { color: var(--sentinel-risk-neutral-color); }


.kpi-card-header {
    display: flex;
    align-items: flex-start;
    margin-bottom: 0.8rem;
}
.kpi-icon {
    font-size: 2.2rem; /* Slightly smaller for cleaner report cards */
    margin-right: 1rem;
    color: var(--sentinel-risk-neutral-color); /* Default icon color */
    flex-shrink: 0;
    line-height: 1;
}
.kpi-title { /* This is an h3.kpi-title */
    font-size: 0.9rem; /* Optimized for card context */
    color: #566573; /* Medium-Dark Gray */
    margin-bottom: 0.3rem;
    font-weight: 600; /* Semibold */
    line-height: 1.3;
    margin-top: 0;
    padding-top: 0.05em; /* Fine tune vertical alignment with icon */
}
.kpi-body {
    text-align: left;
}
.kpi-value {
    font-size: 2.0rem; /* Prominent value */
    font-weight: 700;
    color: var(--sentinel-text-headings-slate-blue);
    margin-bottom: 0.3rem;
    line-height: 1.1;
    word-wrap: break-word;
}
.kpi-units {
    font-size: 0.85rem;
    color: #6c757d;
    margin-left: 0.3em;
    font-weight: 500;
}
.kpi-delta {
    font-size: 0.85rem;
    font-weight: 500;
    display: inline-flex;
    align-items: center;
}
.kpi-delta.positive { color: var(--sentinel-positive-color); }
.kpi-delta.positive::before { content: "▲ "; margin-right: 0.25em; font-size: 0.85em; }
.kpi-delta.negative { color: var(--sentinel-negative-color); }
.kpi-delta.negative::before { content: "▼ "; margin-right: 0.25em; font-size: 0.85em; }
.kpi-delta.neutral { color: var(--sentinel-risk-neutral-color); } /* If delta is neutral itself */


/* ----- Traffic Light Indicator Styling (for render_web_traffic_light_indicator) ----- */
.traffic-light-indicator {
    display: flex;
    align-items: center;
    padding: 0.75rem 1rem;
    border-radius: 8px;
    background-color: var(--sentinel-background-white); /* Lighter background for indicator */
    margin-bottom: 0.75rem;
    border: 1px solid var(--sentinel-border-light);
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
.traffic-light-dot {
    width: 1rem;
    height: 1rem;
    border-radius: 50%;
    margin-right: 0.9rem;
    flex-shrink: 0;
    border: 1px solid rgba(0,0,0,0.08);
}
/* Statuses for traffic light dots (map to new status_level names) */
.traffic-light-dot.status-high-risk, .traffic-light-dot.status-high-concern  { background-color: var(--sentinel-risk-high-color); }
.traffic_light-dot.status-moderate-risk, .traffic-light-dot.status-moderate-concern { background-color: var(--sentinel-risk-moderate-color); }
.traffic-light-dot.status-low-risk, .traffic-light-dot.status-acceptable, .traffic-light-dot.status-good-performance { background-color: var(--sentinel-risk-low-color); }
.traffic-light-dot.status-neutral, .traffic-light-dot.status-no-data { background-color: var(--sentinel-risk-neutral-color); }

.traffic-light-message {
    font-size: 0.9rem;
    color: var(--sentinel-text-dark);
    font-weight: 500;
}
.traffic-light-details {
    font-size: 0.8rem;
    color: #6c757d; /* Lighter grey for details */
    margin-left: auto;
    padding-left: 0.75rem;
    font-style: italic; /* Italicize details for distinction */
}

/* ----- Streamlit Component Overrides & Enhancements (for Web Reports) ----- */
/* Sidebar (less aggressive styling, assumes Sentinel theme from Plotly might influence it too) */
section[data-testid="stSidebar"] {
    background-color: var(--sentinel-background-white); /* Cleaner sidebar */
    border-right: 1px solid var(--sentinel-border-medium);
    padding-top: 0.8rem;
}
section[data-testid="stSidebar"] h1 { /* Sidebar Title */
    font-size: 1.4rem; text-align: center; margin-bottom: 1rem;
    padding-bottom: 0.5rem; border-bottom-color: var(--sentinel-border-light);
    color: var(--sentinel-text-headings-slate-blue);
}
section[data-testid="stSidebar"] .stImage > img {
    display: block; margin-left: auto; margin-right: auto; margin-bottom: 0.8rem;
}
section[data-testid="stSidebar"] .stMarkdown p, section[data-testid="stSidebar"] .stCaption {
    font-size: 0.85rem; color: #566573;
}

/* Streamlit Metric (st.metric) - If used directly as an alternative to custom KPIs */
div[data-testid="stMetric"] {
    background-color: var(--sentinel-background-white);
    border-radius: 8px; padding: 1rem 1.2rem;
    box-shadow: 0 2px 6px rgba(0,0,0,0.05);
    border-left: 5px solid var(--sentinel-text-link-blue); /* Default accent for st.metric */
    margin-bottom: 1rem;
}
div[data-testid="stMetric"] > div:first-child { /* Label */
    font-size: 0.9rem; color: #566573; font-weight: 600; margin-bottom: 0.3rem;
}
div[data-testid="stMetricValue"] { font-size: 1.9rem; font-weight: 700; color: var(--sentinel-text-headings-slate-blue); }
div[data-testid="stMetricDelta"] { font-size: 0.85rem; font-weight: 500; padding-top: 0.2rem; }
/* Assuming Streamlit uses .positive and .negative classes within stMetricDelta */
div[data-testid="stMetricDelta"] .positive { color: var(--sentinel-positive-color); }
div[data-testid="stMetricDelta"] .negative { color: var(--sentinel-negative-color); }


/* Expander styling */
.styled-expander, div[data-testid="stExpander"] { /* Target both potential custom and Streamlit expander */
    border: 1px solid var(--sentinel-border-light);
    border-radius: 8px; margin-bottom: 1.2rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04); overflow: hidden;
}
.styled-expander summary, div[data-testid="stExpander"] summary {
    font-weight: 600; color: var(--sentinel-text-headings-slate-blue);
    padding: 0.9rem 1.2rem; background-color: var(--sentinel-background-light-grey);
    border-bottom: 1px solid var(--sentinel-border-light);
    transition: background-color 0.2s; cursor: pointer;
}
.styled-expander summary:hover, div[data-testid="stExpander"] summary:hover {
    background-color: #e9ecef; /* Slightly darker hover */
}
.styled-expander > div, div[data-testid="stExpander"] > div[data-testid="stExpanderDetails"] { /* Expander content area */
    padding: 1.2rem; background-color: var(--sentinel-background-white);
}


/* Tabs styling */
div[data-testid="stTabs"] button {
    font-weight: 600; color: #566573;
    padding: 0.7rem 1.1rem; border-radius: 6px 6px 0 0;
    transition: all 0.2s ease; border-bottom: 2px solid transparent; margin-right: 1px;
    background-color: var(--sentinel-background-light-grey); /* Tab itself lighter */
}
div[data-testid="stTabs"] button:hover {
    background-color: #e2e6ea; color: var(--sentinel-text-headings-slate-blue);
    border-bottom-color: var(--sentinel-border-medium);
}
div[data-testid="stTabs"] button[aria-selected="true"] {
    color: var(--sentinel-text-link-blue); border-bottom: 2px solid var(--sentinel-text-link-blue);
    background-color: var(--sentinel-background-white); /* Active tab content area bg */
}
div[data-testid="stTabs"] div[data-testid^="stVerticalBlock"] { /* Tab content pane */
    border: 1px solid var(--sentinel-border-light); border-top: none;
    padding: 1.5rem; border-radius: 0 0 8px 8px;
    background-color: var(--sentinel-background-white);
}

/* DataFrame (st.dataframe) Styling */
.stDataFrame {
    border: 1px solid var(--sentinel-border-light);
    border-radius: 6px; box-shadow: 0 1px 2px rgba(0,0,0,0.03); font-size: 0.88rem;
}
.stDataFrame thead th {
    background-color: #f1f3f5; color: #495057; font-weight: 600;
    text-transform: uppercase; font-size: 0.8rem; letter-spacing: 0.02em;
    border-bottom: 2px solid var(--sentinel-border-medium); padding: 0.6rem 0.5rem;
}
.stDataFrame tbody td {
    padding: 0.5rem 0.5rem; border-bottom: 1px solid #e9ecef;
}
.stDataFrame tbody tr:nth-of-type(odd) { background-color: rgba(248,249,250, 0.4); }
.stDataFrame tbody tr:hover { background-color: rgba(222, 226, 230, 0.3); }


/* General Interactive Elements */
a { color: var(--sentinel-text-link-blue); text-decoration: none; font-weight: 500; }
a:hover { text-decoration: underline; color: #2980B9; /* Darker blue on hover */ }

hr { border: none; border-top: 1px solid var(--sentinel-border-medium); margin-top: 1.5rem; margin-bottom: 1.5rem; }

.stButton>button {
    border-radius: 5px; padding: 0.45rem 0.9rem; font-weight: 500;
    transition: background-color 0.15s ease, transform 0.1s ease;
    /* Assuming button primary/secondary distinction would be done by Streamlit theme or custom classes */
}
.stButton>button:hover { transform: translateY(-1px); box-shadow: 0 2px 4px rgba(0,0,0,0.08); }
.stButton>button:active { transform: translateY(0px); box-shadow: inset 0 1px 2px rgba(0,0,0,0.1); }

/* Progress Bar */
div[data-testid="stProgress"] > div {
    background-color: var(--sentinel-text-link-blue); /* Consistent with link blue */
    border-radius: 3px;
}

/* Custom KPI box styles from original (ensure these classes are used in Python if creating raw HTML KPIs) */
/* Styling for the container of the custom markdown KPI */
.custom-markdown-kpi-box {
    background-color: var(--sentinel-background-white);
    border-radius: 10px; /* Aligned with .kpi-card */
    padding: 1.4rem;     /* Consistent padding */
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.06);
    border-left: 6px solid var(--sentinel-risk-neutral-color); /* Default neutral grey accent */
    margin-bottom: 1.2rem;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    height: 100%;
    text-align: left;
}
.custom-markdown-kpi-box.highlight-red-edge { /* Specific for Population dashboard top condition */
    border-left-color: var(--sentinel-risk-high-color) !important;
}
.custom-kpi-label-top-condition { /* Label for the Top Condition KPI */
    font-size: 0.85rem;
    color: #566573;
    font-weight: 600;
    margin-bottom: 0.4rem;
    line-height: 1.3;
}
.custom-kpi-value-large { /* Value for the Top Condition KPI */
    font-size: 1.8rem; /* Slightly smaller than .kpi-value */
    font-weight: 700;
    color: var(--sentinel-text-headings-slate-blue);
    line-height: 1.2;
    margin-bottom: 0.2rem;
    word-wrap: break-word;
    overflow-wrap: break-word;
}
.custom-kpi-subtext-small { /* Subtext for count in Top Condition KPI */
    font-size: 0.8rem;
    color: #7F8C8D;
    margin-top: 0.1rem;
}
