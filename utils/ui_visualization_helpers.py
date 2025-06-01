# test/utils/ui_visualization_helpers.py
# Redesigned for "Sentinel Health Co-Pilot" - LMIC Edge-First System
# This module now provides:
#   1. Styling helpers for simplified HTML-based components (e.g., reports, web dashboards for Tiers 2/3).
#   2. Plotly chart generation functions, primarily for Tier 2 (Facility Node web views) and Tier 3 (Cloud dashboards).
#      Their direct use on Personal Edge Devices (PEDs) is highly limited or non-existent.
#      PEDs will use native UI elements (pictograms, full-screen alerts, haptics, audio).
#   3. Theme settings reflecting LMIC priorities (high contrast, clear fonts for web).

import streamlit as st # Kept for potential use in higher-tier Streamlit apps.
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
import logging
import plotly.io as pio
# Assuming app_config is in the PYTHONPATH or project root.
from config import app_config # Using the new, redesigned app_config
import html
import geopandas as gpd # For type hints only, plots can accept pd.DataFrame for non-geo.
import os
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)

# --- Mapbox Token Handling (Essential for Map Visualizations in Tiers 2/3) ---
MAPBOX_TOKEN_SET_FLAG = False # Renamed to avoid conflict if app_config also has a similar name
try:
    _MAPBOX_ACCESS_TOKEN_ENV = os.getenv("MAPBOX_ACCESS_TOKEN")
    if _MAPBOX_ACCESS_TOKEN_ENV and _MAPBOX_ACCESS_TOKEN_ENV.strip() and \
       "YOUR_MAPBOX_ACCESS_TOKEN" not in _MAPBOX_ACCESS_TOKEN_ENV and \
       len(_MAPBOX_ACCESS_TOKEN_ENV) > 20:
        px.set_mapbox_access_token(_MAPBOX_ACCESS_TOKEN_ENV)
        MAPBOX_TOKEN_SET_FLAG = True
        logger.info("Mapbox access token found and set for Plotly Express.")
    else:
        log_msg_token = "MAPBOX_ACCESS_TOKEN env var not found or is placeholder."
        logger.warning(f"{log_msg_token} Tier 2/3 maps requiring a token may default to open styles.")
except Exception as e_token_setup:
    logger.error(f"Error setting Mapbox token during UI helper init: {e_token_setup}")

# --- I. Core Theming and Color Utilities (for Plotly Charts & Web Components) ---

def _get_theme_color(index: Any = 0, fallback_color: str = app_config.COLOR_ACTION_PRIMARY, color_type: str = "general") -> str:
    """
    Safely retrieves a color from various theme configurations or Plotly's default.
    Prioritizes specific types (disease, risk), then Plotly's active colorway, then fallback.
    Uses redesigned app_config for risk/action colors.
    """
    try:
        if color_type == "risk_high": return app_config.COLOR_RISK_HIGH
        if color_type == "risk_moderate": return app_config.COLOR_RISK_MODERATE
        if color_type == "risk_low": return app_config.COLOR_RISK_LOW
        if color_type == "action_primary": return app_config.COLOR_ACTION_PRIMARY

        # Legacy disease colors still available for web reports from app_config
        if color_type == "disease" and hasattr(app_config, 'LEGACY_DISEASE_COLORS_WEB') and app_config.LEGACY_DISEASE_COLORS_WEB:
            if isinstance(index, str) and index in app_config.LEGACY_DISEASE_COLORS_WEB:
                return app_config.LEGACY_DISEASE_COLORS_WEB[index]
        
        # Fallback to Plotly's default template colorway
        active_template_name = pio.templates.default
        colorway_to_use = px.colors.qualitative.Plotly # Default Plotly colorway

        if active_template_name and active_template_name in pio.templates:
            current_template_layout = pio.templates[active_template_name].layout
            if hasattr(current_template_layout, 'colorway') and current_template_layout.colorway:
                colorway_to_use = current_template_layout.colorway
        
        if colorway_to_use: # Ensure colorway is not None or empty
            # Ensure index is an int; use hash for string indices to get a consistent color
            num_idx_for_color = index if isinstance(index, int) else abs(hash(str(index)))
            return colorway_to_use[num_idx_for_color % len(colorway_to_use)] # Modulo length for safety
            
    except Exception as e_get_color:
        logger.warning(f"Could not retrieve theme color for index/key '{index}', type '{color_type}': {e_get_color}. Using fallback: {fallback_color}")
    return fallback_color


def set_sentinel_plotly_theme_web():
    """
    Sets a custom Plotly theme ('sentinel_web_theme') optimized for clarity in web reports/dashboards (Tiers 2/3).
    Uses high-contrast colors and clear fonts from the redesigned app_config.
    """
    theme_font_family_web = '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif, "Apple Color Emoji", "Segoe UI Emoji", "Noto Color Emoji"'
    theme_text_color_web = app_config.COLOR_RISK_NEUTRAL # Using neutral color for general text
    theme_grid_color_web = "#E0E0E0"  # Lighter grid for less visual clutter
    theme_border_color_web = "#BDBDBD"
    theme_paper_bg_web = "#FFFFFF" # Cleaner white background for reports
    theme_plot_bg_web = "#FAFAFA"   # Very light grey for plot area if different from paper

    # Define a clear, high-contrast colorway using Sentinel action/risk colors first
    sentinel_colorway_list = [
        app_config.COLOR_ACTION_PRIMARY,
        app_config.COLOR_RISK_LOW,      # Green for positive/good trends
        app_config.COLOR_RISK_MODERATE, # Amber for warning
        app_config.COLOR_RISK_HIGH,       # Red for critical
        _get_theme_color(fallback_color="#00ACC1"), # Teal variant
        _get_theme_color(fallback_color="#5E35B1"), # Deep Purple variant
    ]
    # Pad with more distinct colors if needed for charts with many series
    sentinel_colorway_list.extend(px.colors.qualitative.Bold[len(sentinel_colorway_list):])


    layout_settings_web = {
        'font': dict(family=theme_font_family_web, size=11, color=theme_text_color_web), # Slightly smaller base font for dense reports
        'paper_bgcolor': theme_paper_bg_web,
        'plot_bgcolor': theme_plot_bg_web,
        'colorway': sentinel_colorway_list,
        'xaxis': dict(gridcolor=theme_grid_color_web, linecolor=theme_border_color_web, zerolinecolor=theme_grid_color_web, zerolinewidth=1, title_font_size=12, tickfont_size=10, automargin=True, title_standoff=12),
        'yaxis': dict(gridcolor=theme_grid_color_web, linecolor=theme_border_color_web, zerolinecolor=theme_grid_color_web, zerolinewidth=1, title_font_size=12, tickfont_size=10, automargin=True, title_standoff=12),
        'title': dict(
            font=dict(family=theme_font_family_web, size=16, color=app_config.COLOR_ACTION_SECONDARY), # Using secondary action color for titles
            x=0.02, xanchor='left', y=0.96, yanchor='top', pad=dict(t=25, b=10, l=2)
        ),
        'legend': dict(bgcolor='rgba(255,255,255,0.9)', bordercolor=theme_border_color_web, borderwidth=0.5, orientation='h', yanchor='bottom', y=1.01, xanchor='right', x=1, font_size=10),
        'margin': dict(l=60, r=20, t=70, b=50) # Adjust margins for report plots
    }

    # Mapbox configuration for web reports (Tier 2/3)
    mapbox_style_for_web = app_config.MAPBOX_STYLE_WEB if hasattr(app_config, 'MAPBOX_STYLE_WEB') else "carto-positron"
    if not MAPBOX_TOKEN_SET_FLAG and mapbox_style_for_web not in ["open-street-map", "carto-positron", "carto-darkmatter"]: # Limiting non-token styles
        mapbox_style_for_web = "open-street-map"
    layout_settings_web['mapbox'] = dict(
        style=mapbox_style_for_web,
        center=dict(lat=app_config.MAP_DEFAULT_CENTER_LAT, lon=app_config.MAP_DEFAULT_CENTER_LON),
        zoom=app_config.MAP_DEFAULT_ZOOM
    )
    
    sentinel_web_template = go.layout.Template(layout=go.Layout(**layout_settings_web))
    pio.templates["sentinel_web_theme"] = sentinel_web_template
    # Add base "plotly" to ensure all plot types have defaults, then overlay sentinel specifics.
    pio.templates.default = "plotly+sentinel_web_theme"
    logger.info("Plotly theme 'sentinel_web_theme' (for Tiers 2/3 Web) set as default.")

# Apply the theme when this module is imported.
set_sentinel_plotly_theme_web()


# --- II. HTML-Based UI Components (for Streamlit Reports/Web Dashboards - Tiers 2/3) ---
# These use the CSS defined in `style_web_reports.css` (referenced by app_config.STYLE_CSS_PATH_WEB)

def render_web_kpi_card(title: str, value: str, icon: str = "●", status_level: str = "neutral",
                        delta: Optional[str] = None, delta_is_positive: Optional[bool] = None,
                        help_text: Optional[str] = None, units: Optional[str] = ""):
    """
    Renders a KPI card using HTML/CSS for web-based reports.
    Status maps to risk colors. Delta positive maps to green, negative to red.
    On PED, similar info uses native icons/colors.
    """
    status_class_map = { # Maps status levels to CSS classes (from style_web_reports.css)
        "high_risk": "status-high", "critical": "status-high",
        "moderate_risk": "status-moderate", "warning": "status-moderate",
        "low_risk": "status-low", "good": "status-low", # 'low' class from style.css should be green or positive indicator
        "neutral": "status-neutral", "default": "status-neutral"
    }
    css_status_class = status_class_map.get(status_level.lower(), "status-neutral")

    delta_html_content = ""
    if delta is not None and str(delta).strip():
        delta_class = ""
        if delta_is_positive is True: delta_class = "positive"
        elif delta_is_positive is False: delta_class = "negative"
        # Else, neutral, no specific class
        delta_html_content = f'<p class="kpi-delta {delta_class}">{html.escape(str(delta))}</p>'

    tooltip_attr = f'title="{html.escape(str(help_text))}"' if help_text and str(help_text).strip() else ''
    value_units_html = f"{html.escape(str(value))}<span class='kpi-units'>{html.escape(str(units))}</span>" if units else html.escape(str(value))

    # Assumes .kpi-card, .kpi-icon, .kpi-title, .kpi-value, .kpi-delta, .kpi-units are defined in STYLE_CSS_PATH_WEB
    # And status classes like .status-high, .status-moderate, .status-low color the border/icon
    html_render_content = f"""
    <div class="kpi-card {css_status_class}" {tooltip_attr}>
        <div class="kpi-card-header">
            <div class="kpi-icon">{html.escape(str(icon))}</div>
            <h3 class="kpi-title">{html.escape(str(title))}</h3>
        </div>
        <div class="kpi-body">
            <p class="kpi-value">{value_units_html}</p>
            {delta_html_content}
        </div>
    </div>
    """.replace("\n", "") # Minify slightly
    st.markdown(html_render_content, unsafe_allow_html=True)

def render_web_traffic_light_indicator(message: str, status_level: str, details_text: str = ""):
    """
    Renders a "traffic light" style status indicator for web reports.
    Status: "high" (red), "moderate" (yellow), "low" (green), "neutral" (grey).
    On PED, this is a full screen color, haptic, or audio alert.
    """
    status_class_map_tl = {
        "high": "status-high", "critical": "status-high",
        "moderate": "status-moderate", "warning": "status-moderate",
        "low": "status-low", "good": "status-low", "ok": "status-low",
        "neutral": "status-neutral", "unknown": "status-neutral"
    }
    dot_css_class = "status-" + status_level.lower() # Directly use if style.css has .status-high, .status-moderate etc for traffic-light-dot
    # Fallback if not exact match from map (e.g. if only generic .status-high exists in css):
    if not any(s_key == status_level.lower() for s_key in status_class_map_tl.keys()):
         dot_css_class = status_class_map_tl.get(status_level.lower(), "status-neutral")


    details_html_span = f'<span class="traffic-light-details">{html.escape(str(details_text))}</span>' if details_text and str(details_text).strip() else ""
    # Assumes .traffic-light-indicator, .traffic-light-dot, .traffic-light-message, .traffic-light-details are in CSS
    # and .status-high, .status-moderate, .status-low modify background-color of .traffic-light-dot
    html_render_content = f"""
    <div class="traffic-light-indicator">
        <span class="traffic-light-dot {dot_css_class}"></span>
        <span class="traffic-light-message">{html.escape(str(message))}</span>
        {details_html_span}
    </div>
    """.replace("\n", "")
    st.markdown(html_render_content, unsafe_allow_html=True)


# --- III. Plotly Chart Generation Functions (for Tiers 2/3 - Facility/Cloud Web Dashboards & Reports) ---

def _create_empty_plot_figure(title_str: str, height_val: Optional[int], message_str: str = "No data available to display.") -> go.Figure:
    """Helper to create a blank Plotly figure with a message. For Tiers 2/3."""
    fig_empty = go.Figure()
    final_fig_height = height_val if height_val is not None else app_config.WEB_PLOT_DEFAULT_HEIGHT
    fig_empty.update_layout(
        title_text=f"{title_str}: {message_str}",
        height=final_fig_height,
        xaxis={'visible': False},
        yaxis={'visible': False},
        annotations=[dict(text=message_str, xref="paper", yref="paper", showarrow=False, font=dict(size=12, color=_get_theme_color(color_type="action_secondary")))]
    )
    return fig_empty


def plot_layered_choropleth_map_web(
    gdf_data: gpd.GeoDataFrame, value_col_name: str, map_title: str,
    id_col_name: str = 'zone_id', featureidkey_prefix_str: str = 'properties',
    color_scale: str = "Viridis_r", # Defaulting to reverse Viridis for risk-like data
    hover_data_cols: Optional[List[str]] = None,
    facility_points_gdf: Optional[gpd.GeoDataFrame] = None, # For overlaying points like clinics
    facility_size_col_name: Optional[str] = None, facility_hover_col_name: Optional[str] = None,
    facility_marker_color: Optional[str] = None,
    map_height: Optional[int] = None,
    center_override_lat: Optional[float] = None, center_override_lon: Optional[float] = None,
    zoom_override: Optional[int] = None, mapbox_style_override: Optional[str] = None
) -> go.Figure:
    """
    Generates a Plotly Choropleth Map. Primarily for Tier 2/3 web dashboards.
    Uses configuration from app_config for defaults.
    """
    logger.debug(f"Generating choropleth map: {map_title} for value: {value_col_name}")
    final_map_height = map_height if map_height is not None else app_config.WEB_MAP_DEFAULT_HEIGHT

    if not isinstance(gdf_data, gpd.GeoDataFrame) or gdf_data.empty:
        return _create_empty_plot_figure(map_title, final_map_height, "Geographic data unavailable.")
    
    active_geom_col_map = gdf_data.geometry.name if hasattr(gdf_data, 'geometry') and hasattr(gdf_data.geometry, 'name') else 'geometry'
    if active_geom_col_map not in gdf_data.columns or \
       not hasattr(gdf_data[active_geom_col_map], 'is_empty') or gdf_data[active_geom_col_map].is_empty.all() or \
       not hasattr(gdf_data[active_geom_col_map], 'is_valid') or not gdf_data[active_geom_col_map].is_valid.any():
        return _create_empty_plot_figure(map_title, final_map_height, "Invalid or empty geometries in provided data.")

    gdf_for_plot = gdf_data.copy()
    if id_col_name not in gdf_for_plot.columns or value_col_name not in gdf_for_plot.columns:
        return _create_empty_plot_figure(map_title, final_map_height, f"Required columns '{id_col_name}' or '{value_col_name}' missing.")

    if not pd.api.types.is_numeric_dtype(gdf_for_plot[value_col_name]):
        gdf_for_plot[value_col_name] = pd.to_numeric(gdf_for_plot[value_col_name], errors='coerce')
    gdf_for_plot[value_col_name].fillna(0, inplace=True) # Default NaNs to 0 for color scale consistency
    gdf_for_plot[id_col_name] = gdf_for_plot[id_col_name].astype(str) # Ensure ID is string for geojson matching

    # Filter out invalid or empty geometries before creating GeoJSON interface
    gdf_valid_geojson = gdf_for_plot[gdf_for_plot.geometry.is_valid & ~gdf_for_plot.geometry.is_empty].copy()
    if gdf_valid_geojson.empty:
        return _create_empty_plot_figure(map_title, final_map_height, "No valid geometries remaining after filtering for map display.")

    # Mapbox style configuration
    chosen_mapbox_style = mapbox_style_override or pio.templates.default.layout.get('mapbox', {}).get('style', app_config.MAPBOX_STYLE_WEB)
    if not MAPBOX_TOKEN_SET_FLAG and chosen_mapbox_style not in ["open-street-map", "carto-positron", "carto-darkmatter"]: # Stricter list for non-token
        logger.warning(f"Map ('{map_title}'): Style '{chosen_mapbox_style}' likely requires token but not set. Defaulting to 'open-street-map'.")
        chosen_mapbox_style = "open-street-map"
    
    # Hover data configuration
    display_name_col_map = "name" if "name" in gdf_for_plot.columns else id_col_name
    default_hover_data_map = [display_name_col_map, value_col_name, 'population'] # Common useful hover fields
    final_hover_data_list_map = hover_data_cols if hover_data_cols is not None else default_hover_data_map
    # Ensure hover columns exist and are not all NaN.
    hover_data_for_plotly = {
        col: True for col in final_hover_data_list_map 
        if col in gdf_for_plot.columns and col != display_name_col_map and gdf_for_plot[col].notna().any()
    }
    labels_for_plotly_map = {col: str(col).replace('_', ' ').title() for col in [value_col_name] + list(hover_data_for_plotly.keys())}

    try:
        fig_map = px.choropleth_mapbox(
            data_frame=gdf_valid_geojson,
            geojson=gdf_valid_geojson.geometry.__geo_interface__,
            locations=id_col_name,
            featureidkey=f"{featureidkey_prefix_str}.{id_col_name}" if featureidkey_prefix_str and featureidkey_prefix_str.strip() else id_col_name,
            color=value_col_name,
            color_continuous_scale=color_scale,
            opacity=0.7, # Good default for choropleths
            hover_name=display_name_col_map,
            hover_data=hover_data_for_plotly,
            labels=labels_for_plotly_map,
            mapbox_style=chosen_mapbox_style,
            center={"lat": center_override_lat or app_config.MAP_DEFAULT_CENTER_LAT, 
                    "lon": center_override_lon or app_config.MAP_DEFAULT_CENTER_LON},
            zoom=zoom_override if zoom_override is not None else app_config.MAP_DEFAULT_ZOOM
        )
    except Exception as e_px_map:
        logger.error(f"Plotly Mapbox Error for '{map_title}': {e_px_map}", exc_info=True)
        return _create_empty_plot_figure(map_title, final_map_height, f"Map rendering error. Check data and configuration. Details: {str(e_px_map)[:150]}")

    # Overlay facility points if provided
    if isinstance(facility_points_gdf, gpd.GeoDataFrame) and not facility_points_gdf.empty and 'geometry' in facility_points_gdf.columns:
        facility_plot_data = facility_points_gdf[facility_points_gdf.geometry.geom_type == 'Point'].copy()
        if not facility_plot_data.empty:
            facility_hover_text_series = facility_plot_data.get(facility_hover_col_name, pd.Series(["Facility"] * len(facility_plot_data), index=facility_plot_data.index)) if facility_hover_col_name else pd.Series(["Facility"] * len(facility_plot_data), index=facility_plot_data.index)
            
            marker_size_series = pd.Series([8] * len(facility_plot_data), index=facility_plot_data.index) # Default size for facilities
            if facility_size_col_name and facility_size_col_name in facility_plot_data.columns and pd.api.types.is_numeric_dtype(facility_plot_data[facility_size_col_name]):
                raw_sizes = pd.to_numeric(facility_plot_data[facility_size_col_name], errors='coerce').fillna(1) # Fill NaN size with 1
                min_px_size, max_px_size = 5, 15 # Pixel size range
                min_val_size, max_val_size = raw_sizes.min(), raw_sizes.max()
                if max_val_size > min_val_size and max_val_size > 0 : # Avoid division by zero or no variation
                     marker_size_series = min_px_size + ((raw_sizes - min_val_size) * (max_px_size - min_px_size) / (max_val_size - min_val_size))
                elif raw_sizes.notna().any() and raw_sizes.max() > 0: # If all same value but positive
                     marker_size_series = pd.Series([(min_px_size + max_px_size) / 2] * len(facility_plot_data), index=facility_plot_data.index)
            
            final_facility_marker_color = facility_marker_color if facility_marker_color else _get_theme_color(color_type="action_secondary") # Use a secondary action color

            fig_map.add_trace(go.Scattermapbox(
                lon=facility_plot_data.geometry.x, lat=facility_plot_data.geometry.y,
                mode='markers',
                marker=go.scattermapbox.Marker(
                    size=marker_size_series, sizemin=4, color=final_facility_marker_color,
                    opacity=0.85, allowoverlap=False # Avoid overlap for clarity
                ),
                text=facility_hover_text_series, hoverinfo='text', name='Facilities/Points'
            ))
    
    fig_map.update_layout(
        title_text=map_title, height=final_map_height,
        margin={"r":15,"t":50,"l":15,"b":15}, # Adjusted margins for map
        legend=dict(yanchor="top", y=0.97, xanchor="left", x=0.02, bgcolor='rgba(250,250,250,0.75)')
    )
    return fig_map

# Annotated Line Chart, Bar Chart, Donut Chart, Heatmap functions (plot_annotated_line_chart, etc.)
# would largely remain the same as the "last good version" provided for their Plotly Express logic,
# with these key adaptations for the Sentinel system (Tiers 2/3):
#   - Default heights would use app_config.WEB_PLOT_DEFAULT_HEIGHT / WEB_PLOT_COMPACT_HEIGHT.
#   - Color logic would use _get_theme_color with Sentinel/LMIC appropriate fallbacks or types.
#   - Font sizes and margins will align with set_sentinel_plotly_theme_web.
#   - Any data validation or `_create_empty_plot_figure` calls should use the updated helper.
#   - Tick formatting (y_is_count logic) would be preserved but ensure it's clear and not overly dense.

# Example structure for plot_annotated_line_chart_web (Tier 2/3):
def plot_annotated_line_chart_web(
    data_series_input: pd.Series, chart_title: str, y_axis_label: str = "Value",
    line_color: Optional[str] = None,
    target_ref_line: Optional[float] = None, target_ref_label: Optional[str] = None,
    show_conf_interval: bool = False, lower_ci_series: Optional[pd.Series] = None, upper_ci_series: Optional[pd.Series] = None,
    chart_height: Optional[int] = None, show_anomalies_option: bool = False, # Anomalies less critical for simple reports
    date_display_format: str = "%d-%b-%y", # More compact date format
    y_axis_is_count: bool = False # Helps with tick formatting
) -> go.Figure:
    final_chart_height = chart_height if chart_height is not None else app_config.WEB_PLOT_COMPACT_HEIGHT # Compact for web reports
    if not isinstance(data_series_input, pd.Series) or data_series_input.empty:
        return _create_empty_plot_figure(chart_title, final_chart_height)
    
    data_series_clean = pd.to_numeric(data_series_input, errors='coerce')
    if data_series_clean.isnull().all():
        return _create_empty_plot_figure(chart_title, final_chart_height, "All data non-numeric or became NaN.")

    fig_line = go.Figure()
    chosen_line_color = line_color if line_color else _get_theme_color(0) # Default theme color

    y_hover_format_str = 'd' if y_axis_is_count else ',.1f' # Adjusted default, fewer decimals
    hovertemplate_line = f'<b>Date</b>: %{{x|{date_display_format}}}<br><b>{y_axis_label}</b>: %{{customdata:{y_hover_format_str}}}<extra></extra>'

    fig_line.add_trace(go.Scatter(
        x=data_series_clean.index, y=data_series_clean.values,
        mode="lines+markers", name=y_axis_label,
        line=dict(color=chosen_line_color, width=2.2), marker=dict(size=5),
        customdata=data_series_clean.values, hovertemplate=hovertemplate_line
    ))

    # Confidence Interval plotting (simplified from original)
    if show_conf_interval and isinstance(lower_ci_series, pd.Series) and isinstance(upper_ci_series, pd.Series) and \
       not lower_ci_series.empty and not upper_ci_series.empty:
        common_idx = data_series_clean.index.intersection(lower_ci_series.index).intersection(upper_ci_series.index)
        if not common_idx.empty:
            ls_ci = pd.to_numeric(lower_ci_series.reindex(common_idx), errors='coerce')
            us_ci = pd.to_numeric(upper_ci_series.reindex(common_idx), errors='coerce')
            valid_ci_data = ls_ci.notna() & us_ci.notna() & (us_ci >= ls_ci)
            if valid_ci_data.any():
                x_ci_vals = common_idx[valid_ci_data]
                y_upper_vals = us_ci[valid_ci_data]
                y_lower_vals = ls_ci[valid_ci_data]
                fill_color_val = f"rgba({','.join(str(int(c,16)) for c in (chosen_line_color[1:3], chosen_line_color[3:5], chosen_line_color[5:7]))},0.1)" if chosen_line_color.startswith('#') and len(chosen_line_color)==7 else "rgba(100,100,100,0.1)"
                fig_line.add_trace(go.Scatter(x=list(x_ci_vals) + list(x_ci_vals[::-1]),
                                              y=list(y_upper_vals.values) + list(y_lower_vals.values[::-1]),
                                              fill="toself", fillcolor=fill_color_val,
                                              line=dict(width=0), name="CI", hoverinfo='skip'))
    
    # Target line
    if target_ref_line is not None:
        target_display_label = target_ref_label if target_ref_label else f"Target: {target_ref_line:,.2f}"
        fig_line.add_hline(y=target_ref_line, line_dash="dash", line_color=_get_theme_color(color_type="risk_high", fallback_color="#E57373"), line_width=1.2,
                           annotation_text=target_display_label, annotation_position="bottom right", annotation_font_size=9) # Smaller font

    # Simplified Anomaly Detection (can be removed for pure reports if too busy)
    if show_anomalies_option and len(data_series_clean.dropna()) > 7 and data_series_clean.nunique() > 2: # Adjusted conditions
        q1_anom = data_series_clean.quantile(0.25); q3_anom = data_series_clean.quantile(0.75); iqr_anom = q3_anom - q1_anom
        if pd.notna(iqr_anom) and iqr_anom > 1e-7: # Ensure IQR is meaningful
            upper_bound_anom = q3_anom + 1.5 * iqr_anom; lower_bound_anom = q1_anom - 1.5 * iqr_anom
            anomalies_data = data_series_clean[(data_series_clean < lower_bound_anom) | (data_series_clean > upper_bound_anom)]
            if not anomalies_data.empty:
                fig_line.add_trace(go.Scatter(
                    x=anomalies_data.index, y=anomalies_data.values, mode='markers',
                    marker=dict(color=_get_theme_color(color_type="risk_moderate", fallback_color="#FFB74D"), size=7, symbol='circle-open', line=dict(width=1.5)),
                    name='Anomaly', customdata=anomalies_data.values, hovertemplate=(f'<b>Anomaly Date</b>: %{{x|{date_display_format}}}<br><b>Value</b>: %{{customdata:{y_hover_format_str}}}<extra></extra>')
                ))

    final_x_axis_label = data_series_clean.index.name if data_series_clean.index.name and str(data_series_clean.index.name).strip() else "Date/Time"
    yaxis_line_config = dict(title_text=y_axis_label, rangemode='tozero' if y_axis_is_count and data_series_clean.min() >= 0 else 'normal')
    if y_axis_is_count: # Integer ticks for counts
        yaxis_line_config['tickformat'] = 'd'
        max_val_line = data_series_clean.max(); min_val_line = data_series_clean.min()
        if pd.notna(max_val_line) and max_val_line > 0: # Auto-adjust dtick for count y-axes for better readability
             if max_val_line <= 10: yaxis_line_config['dtick'] = 1
             elif max_val_line <= 50: yaxis_line_config['dtick'] = 5
             # For larger values, Plotly's auto nticks usually works well.
    
    fig_line.update_layout(title_text=chart_title, xaxis_title=final_x_axis_label, yaxis=yaxis_line_config,
                           height=final_chart_height, hovermode="x unified", legend=dict(traceorder='normal'))
    return fig_line


# ... Implement plot_bar_chart_web, plot_donut_chart_web, plot_heatmap_web similarly,
#     adjusting defaults, colors, and any specific logic (like y_is_count tick formatting)
#     to suit the 'sentinel_web_theme' and report/dashboard context for Tiers 2/3.
#     Ensure they use _create_empty_plot_figure and _get_theme_color as updated.
# For brevity, these will follow the same pattern as plot_annotated_line_chart_web.

# Example stubs for other plot functions to indicate pattern:
def plot_bar_chart_web(*args, **kwargs) -> go.Figure:
    # Implement using principles from plot_annotated_line_chart_web:
    # - Use app_config for WEB_PLOT heights
    # - Use _get_theme_color for sentinel colors
    # - Call _create_empty_plot_figure for empty/invalid data
    # - Adapt y_is_count for tick formatting
    # - Adjust text_format and hover_templates for clarity
    logger.debug("plot_bar_chart_web called (stub for Tier 2/3 reports)")
    # Basic pass-through for now for type hinting, real implementation would be robust.
    height = kwargs.get('chart_height', kwargs.get('height', app_config.WEB_PLOT_DEFAULT_HEIGHT))
    if not args or not isinstance(args[0], pd.DataFrame) or args[0].empty:
        return _create_empty_plot_figure(kwargs.get('chart_title', kwargs.get('title','Bar Chart')), height)
    # Actual robust implementation as per `plot_bar_chart` in the originally provided UI helpers, adapted for web.
    df_input_bar, x_col_bar, y_col_bar, title_bar = args[0], args[1], args[2], args[3] # Simplistic unpacking
    return _create_empty_plot_figure(title_bar,height, "Bar chart plot_bar_chart_web needs full porting")


def plot_donut_chart_web(*args, **kwargs) -> go.Figure:
    logger.debug("plot_donut_chart_web called (stub for Tier 2/3 reports)")
    height = kwargs.get('chart_height', kwargs.get('height', app_config.WEB_PLOT_COMPACT_HEIGHT))
    if not args or not isinstance(args[0], pd.DataFrame) or args[0].empty:
         return _create_empty_plot_figure(kwargs.get('chart_title', kwargs.get('title','Donut Chart')), height)
    return _create_empty_plot_figure(kwargs.get('chart_title', kwargs.get('title','Donut Chart')),height,"Donut plot_donut_chart_web needs full porting")

def plot_heatmap_web(*args, **kwargs) -> go.Figure:
    logger.debug("plot_heatmap_web called (stub for Tier 2/3 reports)")
    height = kwargs.get('chart_height', kwargs.get('height', app_config.WEB_PLOT_DEFAULT_HEIGHT))
    if not args or not isinstance(args[0], pd.DataFrame) or args[0].empty:
         return _create_empty_plot_figure(kwargs.get('chart_title', kwargs.get('title','Heatmap')), height)
    return _create_empty_plot_figure(kwargs.get('chart_title', kwargs.get('title','Heatmap')), height, "Heatmap plot_heatmap_web needs full porting")
