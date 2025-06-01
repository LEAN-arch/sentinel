# test/tests/test_ui_visualization_helpers.py
# Pytest tests for the refactored UI visualization helpers in utils.ui_visualization_helpers.py
# Aligned with "Sentinel Health Co-Pilot" redesign (primarily for Tiers 2/3 web views).

import pytest
import pandas as pd
import geopandas as gpd
import numpy as np
import plotly.graph_objects as go
from unittest.mock import patch # For mocking st.markdown

# Functions to be tested (import the _web suffixed ones)
from utils.ui_visualization_helpers import (
    set_sentinel_plotly_theme_web, # Ensure theme is set before tests that check layout
    _get_theme_color,
    render_web_kpi_card,
    render_web_traffic_light_indicator,
    _create_empty_plot_figure,
    plot_layered_choropleth_map_web,
    plot_annotated_line_chart_web,
    plot_bar_chart_web,
    plot_donut_chart_web,
    plot_heatmap_web,
    # Import the MAPBOX_TOKEN_SET_FLAG for mocking if needed for map tests
    MAPBOX_TOKEN_SET_FLAG as vis_helper_mapbox_token_flag 
)
from config import app_config # The NEW, redesigned app_config

# Fixtures from conftest.py
# Assume: sample_series_data, sample_bar_df, sample_donut_df, sample_heatmap_df, sample_choropleth_gdf
#         are available and suitable.
#         sample_choropleth_gdf needs 'name' and 'zone_id' for the map function.

# Apply the theme once for all tests in this module
@pytest.fixture(scope="module", autouse=True)
def apply_theme_for_tests():
    set_sentinel_plotly_theme_web()

# --- Tests for Core Theming and Color Utilities ---

def test_get_theme_color_specifics():
    assert _get_theme_color(color_type="risk_high") == app_config.COLOR_RISK_HIGH
    assert _get_theme_color(color_type="action_primary") == app_config.COLOR_ACTION_PRIMARY
    # Test fallback
    assert _get_theme_color(index=999, fallback_color="#123456", color_type="unknown") == "#123456"
    # Test legacy disease color if defined
    if app_config.LEGACY_DISEASE_COLORS_WEB and "TB" in app_config.LEGACY_DISEASE_COLORS_WEB:
        assert _get_theme_color(index="TB", color_type="disease") == app_config.LEGACY_DISEASE_COLORS_WEB["TB"]
    # Test getting a color from the default colorway
    # This assumes the sentinel_web_theme has been set and has a colorway
    first_colorway_color = go.layout.Template(pio.templates.default).layout.colorway[0]
    assert _get_theme_color(index=0, color_type="general") == first_colorway_color


# --- Tests for HTML Component Renderers (using mocking) ---

@patch('utils.ui_visualization_helpers.st.markdown') # Mock st.markdown within the target module
def test_render_web_kpi_card(mock_st_markdown):
    render_web_kpi_card(title="Test KPI", value="123", icon="💡", status_level="HIGH_RISK", units="units")
    mock_st_markdown.assert_called_once()
    args, kwargs = mock_st_markdown.call_args
    html_output = args[0]
    assert 'class="kpi-card status-high"' in html_output # Check for correct status class (was status-high_risk before)
    assert '<h3 class="kpi-title">Test KPI</h3>' in html_output
    assert '<p class="kpi-value">123<span class=\'kpi-units\'>units</span></p>' in html_output # Check value and units

    mock_st_markdown.reset_mock()
    render_web_kpi_card(title="Delta KPI", value="10", delta="+5", delta_is_positive=True)
    args_delta, _ = mock_st_markdown.call_args
    assert 'class="kpi-delta positive">+5</p>' in args_delta[0]

@patch('utils.ui_visualization_helpers.st.markdown')
def test_render_web_traffic_light_indicator(mock_st_markdown):
    render_web_traffic_light_indicator(message="System Status", status_level="LOW", details_text="All clear.")
    mock_st_markdown.assert_called_once()
    args, _ = mock_st_markdown.call_args
    html_output = args[0]
    assert 'class="traffic-light-dot status-low"' in html_output # Ensure correct class
    assert '<span class="traffic-light-message">System Status</span>' in html_output
    assert '<span class="traffic-light-details">All clear.</span>' in html_output


# --- Tests for Plotting Functions (primarily ensuring they run and return Figures) ---

def test_create_empty_plot_figure():
    fig = _create_empty_plot_figure("Empty Test", 300, "No data here.")
    assert isinstance(fig, go.Figure)
    assert fig.layout.title.text == "Empty Test: No data here."
    assert fig.layout.xaxis.visible is False

# For each plotting function:
# 1. Test with valid sample data -> returns go.Figure, check basic title.
# 2. Test with empty DataFrame -> returns an "empty" go.Figure from _create_empty_plot_figure.
# 3. Test with DataFrame missing required columns -> returns an "empty" go.Figure.

def test_plot_annotated_line_chart_web(sample_series_data):
    fig = plot_annotated_line_chart_web(sample_series_data, "Test Line Chart")
    assert isinstance(fig, go.Figure)
    assert fig.layout.title.text == "Test Line Chart"
    
    # Test with empty series
    fig_empty = plot_annotated_line_chart_web(pd.Series(dtype=float), "Empty Line Chart")
    assert "Empty Line Chart: No data available" in fig_empty.layout.title.text

def test_plot_bar_chart_web(sample_bar_df):
    fig = plot_bar_chart_web(sample_bar_df, x_col_bar='category', y_col_bar='value', title_bar="Test Bar Chart") # use specific arg names
    assert isinstance(fig, go.Figure)
    assert fig.layout.title.text == "Test Bar Chart"

    fig_empty = plot_bar_chart_web(pd.DataFrame(), x_col_bar='category', y_col_bar='value', title_bar="Empty Bar")
    assert "Empty Bar: No data available" in fig_empty.layout.title.text
    
    fig_missing_col = plot_bar_chart_web(sample_bar_df, x_col_bar='non_existent', y_col_bar='value', title_bar="Missing Col Bar")
    assert "Missing Col Bar: No data available" in fig_missing_col.layout.title.text


def test_plot_donut_chart_web(sample_donut_df):
    fig = plot_donut_chart_web(sample_donut_df, labels_col='status', values_col='count', title="Test Donut Chart")
    assert isinstance(fig, go.Figure)
    assert fig.layout.title.text == "Test Donut Chart"

    fig_empty = plot_donut_chart_web(pd.DataFrame(), labels_col='status', values_col='count', title="Empty Donut")
    assert "Empty Donut: No data available" in fig_empty.layout.title.text


def test_plot_heatmap_web(sample_heatmap_df):
    fig = plot_heatmap_web(sample_heatmap_df, title="Test Heatmap")
    assert isinstance(fig, go.Figure)
    assert fig.layout.title.text == "Test Heatmap"

    fig_empty = plot_heatmap_web(pd.DataFrame(), title="Empty Heatmap")
    assert "Empty Heatmap: Invalid data for Heatmap." in fig_empty.layout.title.text


def test_plot_layered_choropleth_map_web(sample_choropleth_gdf): # Ensure sample_choropleth_gdf has 'name' and 'zone_id'
    """
    Test choropleth map generation. `sample_choropleth_gdf` fixture is crucial.
    It should be a GeoDataFrame with 'zone_id', 'name', 'geometry', and a value column (e.g., 'risk_score').
    """
    if sample_choropleth_gdf.empty or 'risk_score' not in sample_choropleth_gdf.columns or \
       'zone_id' not in sample_choropleth_gdf.columns or 'name' not in sample_choropleth_gdf.columns:
        pytest.skip("Sample choropleth GDF not configured correctly for this test.")

    fig = plot_layered_choropleth_map_web(
        gdf_data=sample_choropleth_gdf,
        value_col_name='risk_score',
        map_title="Test Choropleth Map",
        id_col_name='zone_id' # GDF must have 'zone_id'
    )
    assert isinstance(fig, go.Figure)
    assert fig.layout.title.text == "Test Choropleth Map"
    assert fig.layout.mapbox.style is not None # A style should be set by the theme or function default

    # Test fallback for empty or invalid GDF
    fig_empty = plot_layered_choropleth_map_web(gpd.GeoDataFrame(), value_col_name='val', map_title="Empty Map")
    assert "Empty Map: Geographic data unavailable" in fig_empty.layout.title.text


@patch('utils.ui_visualization_helpers.MAPBOX_TOKEN_SET_FLAG', False) # Mock the global flag in that module
def test_plot_layered_choropleth_map_web_no_token_default_style(sample_choropleth_gdf, monkeypatch):
    """
    Test that map defaults to a non-token style if token is not set and a token-requiring style was asked.
    """
    if sample_choropleth_gdf.empty or 'risk_score' not in sample_choropleth_gdf.columns:
        pytest.skip("Sample choropleth GDF not configured for no-token test.")

    # Temporarily override app_config's MAPBOX_STYLE_WEB to be a token-requiring one for this test
    # monkeypatch.setattr(app_config, 'MAPBOX_STYLE_WEB', 'streamlit', raising=False) # Example for Streamlit style (if it needs token)
    # More directly, test against a known private style
    # For the test to be meaningful, ui_visualization_helpers should try to use a private style first
    
    # Scenario: Theme asks for 'streets' (token required), but token is False
    # We patch the global vis_helper_mapbox_token_flag to False
    # and request a style that would normally require a token.
    # Note: The set_sentinel_plotly_theme_web already has logic to pick a default style based on token flag.
    # This test confirms the map plotting function ALSO respects it or falls back.
    
    # Test the mapbox_style argument directly
    fig = plot_layered_choropleth_map_web(
        gdf_data=sample_choropleth_gdf,
        value_col_name='risk_score',
        map_title="No Token Map Test",
        id_col_name='zone_id',
        mapbox_style_override='satellite-streets-v11' # A style that typically needs a token
    )
    # Expect it to fall back to 'open-street-map' as per logic in plot_layered_choropleth_map_web
    assert fig.layout.mapbox.style == 'open-street-map'
