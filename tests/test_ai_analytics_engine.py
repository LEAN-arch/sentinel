# test/tests/test_ai_analytics_engine.py
# Pytest tests for the refactored AI simulation logic in utils.ai_analytics_engine.py
# Aligned with "Sentinel Health Co-Pilot" redesign.

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Functions/Classes to be tested
from utils.ai_analytics_engine import (
    RiskPredictionModel,
    FollowUpPrioritizer,
    SupplyForecastingModel,
    apply_ai_models
)
# Import app_config to use its thresholds and keys in test assertions
from config import app_config # The NEW, redesigned app_config

# Fixtures from conftest.py (using their NEW Sentinel-aligned names)
# from ..conftest import sample_health_records_df_main # if using relative imports
# We'll assume fixtures are available in the pytest environment.


# --- Tests for RiskPredictionModel ---

@pytest.fixture(scope="module")
def risk_model():
    return RiskPredictionModel()

def test_risk_model_get_condition_base_score(risk_model):
    assert risk_model._get_condition_base_score("TB") == risk_model.condition_base_scores.get("TB", 0)
    assert risk_model._get_condition_base_score("Severe Dehydration; Pneumonia") > risk_model.condition_base_scores.get("Severe Dehydration", 0) # Should pick up one or combine
    assert risk_model._get_condition_base_score("Unknown Condition") == 0.0
    assert risk_model._get_condition_base_score(None) == 0.0
    # Test a key condition from the new config
    if app_config.KEY_CONDITIONS_FOR_ACTION:
        key_cond = app_config.KEY_CONDITIONS_FOR_ACTION[0]
        assert risk_model._get_condition_base_score(key_cond) == risk_model.condition_base_scores.get(key_cond, 0)


def test_risk_model_predict_risk_score_specific_factors(risk_model):
    # Test SpO2 impact
    features_low_spo2 = pd.Series({'min_spo2_pct': app_config.ALERT_SPO2_CRITICAL_LOW_PCT - 1, 'condition': 'Wellness Visit'})
    score_low_spo2 = risk_model.predict_risk_score(features_low_spo2)
    features_ok_spo2 = pd.Series({'min_spo2_pct': 98, 'condition': 'Wellness Visit'})
    score_ok_spo2 = risk_model.predict_risk_score(features_ok_spo2)
    assert score_low_spo2 > score_ok_spo2 + risk_model.base_risk_factors['min_spo2_pct']['factor_low'] * risk_model.base_risk_factors['min_spo2_pct']['weight'] - 5 # Account for small variations

    # Test high temperature impact
    features_high_temp = pd.Series({'vital_signs_temperature_celsius': app_config.ALERT_BODY_TEMP_HIGH_FEVER_C + 0.5, 'condition': 'Wellness Visit'})
    score_high_temp = risk_model.predict_risk_score(features_high_temp)
    assert score_high_temp > score_ok_spo2 + risk_model.base_risk_factors['vital_signs_temperature_celsius']['factor_super_high'] * risk_model.base_risk_factors['vital_signs_temperature_celsius']['weight'] - 5

    # Test fall detected
    features_fall = pd.Series({'fall_detected_today': 1, 'condition': 'Wellness Visit'})
    score_fall = risk_model.predict_risk_score(features_fall)
    assert score_fall > score_ok_spo2 + risk_model.base_risk_factors['fall_detected_today']['factor_true'] * risk_model.base_risk_factors['fall_detected_today']['weight'] - 5

    # Test chronic condition flag
    features_chronic = pd.Series({'chronic_condition_flag': 1, 'condition': 'Wellness Visit'})
    score_chronic = risk_model.predict_risk_score(features_chronic)
    assert score_chronic > score_ok_spo2 + risk_model.CHRONIC_CONDITION_FLAG_RISK_POINTS - 5

    # Test psychometric input
    features_psych_distress = pd.Series({'rapid_psychometric_distress_score': 8, 'condition': 'Wellness Visit'}) # Assuming 0-10 scale, higher is more distress
    score_psych_distress = risk_model.predict_risk_score(features_psych_distress)
    assert score_psych_distress > score_ok_spo2 + (8 * 1.0) - 5 # Simpler multiplier assumption

def test_risk_model_predict_bulk_risk_scores(risk_model, sample_health_records_df_main):
    if sample_health_records_df_main.empty:
        pytest.skip("Sample health records DF is empty, skipping bulk risk score test.")
    
    risk_scores = risk_model.predict_bulk_risk_scores(sample_health_records_df_main.copy()) # Pass copy
    assert isinstance(risk_scores, pd.Series)
    assert len(risk_scores) == len(sample_health_records_df_main)
    assert risk_scores.notna().all() # Should not produce NaNs after clipping
    assert risk_scores.min() >= 0
    assert risk_scores.max() <= 100

    # Test with df missing an expected column (should use default)
    df_missing_col = sample_health_records_df_main.copy().drop(columns=['min_spo2_pct'], errors='ignore')
    risk_scores_missing_col = risk_model.predict_bulk_risk_scores(df_missing_col)
    assert len(risk_scores_missing_col) == len(df_missing_col) # Should still process


# --- Tests for FollowUpPrioritizer ---

@pytest.fixture(scope="module")
def priority_model():
    return FollowUpPrioritizer()

def test_priority_model_critical_vitals_logic(priority_model):
    assert priority_model._has_active_critical_vitals_alert(pd.Series({'min_spo2_pct': app_config.ALERT_SPO2_CRITICAL_LOW_PCT - 2})) is True
    assert priority_model._has_active_critical_vitals_alert(pd.Series({'vital_signs_temperature_celsius': app_config.ALERT_BODY_TEMP_HIGH_FEVER_C + 0.1})) is True
    assert priority_model._has_active_critical_vitals_alert(pd.Series({'fall_detected_today': 1})) is True
    assert priority_model._has_active_critical_vitals_alert(pd.Series({'min_spo2_pct': 98, 'vital_signs_temperature_celsius': 37.0})) is False

def test_priority_model_calculate_priority_score(priority_model):
    # Base case
    features_base = pd.Series({'ai_risk_score': 50})
    base_score = priority_model.calculate_priority_score(features_base, days_task_overdue=0)
    assert np.isclose(base_score, 50 * priority_model.priority_weights['base_ai_risk_score_contribution_pct'])

    # With critical vitals
    features_crit_vitals = pd.Series({'ai_risk_score': 50, 'min_spo2_pct': app_config.ALERT_SPO2_CRITICAL_LOW_PCT - 1})
    crit_vitals_score = priority_model.calculate_priority_score(features_crit_vitals)
    assert crit_vitals_score > base_score + priority_model.priority_weights['critical_vital_alert_points'] - 5 # allow for float issues

    # With urgent task (e.g., critical condition referral pending)
    features_urgent_task = pd.Series({'ai_risk_score': 50, 'referral_status': 'Pending', 'condition': app_config.KEY_CONDITIONS_FOR_ACTION[0]})
    urgent_task_score = priority_model.calculate_priority_score(features_urgent_task)
    assert urgent_task_score > base_score + priority_model.priority_weights['pending_urgent_task_points'] - 5
    
    # With task overdue
    overdue_score = priority_model.calculate_priority_score(features_base, days_task_overdue=5)
    assert overdue_score > base_score + (5 * priority_model.priority_weights['task_overdue_factor_per_day']) - 5


def test_priority_model_generate_followup_priorities(priority_model, sample_health_records_df_main):
    if sample_health_records_df_main.empty:
        pytest.skip("Sample health records DF is empty, skipping bulk priority score test.")

    # Ensure ai_risk_score is present for the prioritizer to use (apply RiskPredictionModel first)
    risk_model_temp = RiskPredictionModel()
    health_df_with_risk = sample_health_records_df_main.copy()
    health_df_with_risk['ai_risk_score'] = risk_model_temp.predict_bulk_risk_scores(health_df_with_risk)
    # Add a sample 'days_task_overdue' column for testing that aspect
    health_df_with_risk['days_task_overdue'] = np.random.randint(0, 5, size=len(health_df_with_risk))

    priority_scores = priority_model.generate_followup_priorities(health_df_with_risk)
    assert isinstance(priority_scores, pd.Series)
    assert len(priority_scores) == len(health_df_with_risk)
    assert priority_scores.notna().all()
    assert priority_scores.min() >= 0
    assert priority_scores.max() <= 100


# --- Tests for SupplyForecastingModel (AI-Simulated) ---

@pytest.fixture(scope="module")
def supply_forecast_model_ai():
    return SupplyForecastingModel()

def test_supply_model_ai_predict_daily_consumption(supply_forecast_model_ai):
    # Test for a known item with specific params if defined, or default
    item_to_test_supply = app_config.KEY_DRUG_SUBSTRINGS_SUPPLY[0] if app_config.KEY_DRUG_SUBSTRINGS_SUPPLY else "TestItem"
    base_cons = 10.0
    test_date = pd.Timestamp('2023-03-15') # March (month 3)
    pred_cons_day1 = supply_forecast_model_ai._predict_daily_consumption_ai(base_cons, item_to_test_supply, test_date, 1)
    assert pred_cons_day1 > 0

    # Check if seasonality is applied (coeffs for month 3 vs month 7 for e.g. ACT)
    test_date_alt_season = pd.Timestamp('2023-07-15') # July (month 7)
    pred_cons_alt_season = supply_forecast_model_ai._predict_daily_consumption_ai(base_cons, "ACT Tablets", test_date_alt_season, 1)
    if "ACT Tablets" in supply_forecast_model_ai.item_params: # Only if ACT has distinct params
        act_params = supply_forecast_model_ai._get_item_params("ACT Tablets")
        if act_params["coeffs"][2] != act_params["coeffs"][6]: # If March and July coeffs differ
            assert not np.isclose(pred_cons_day1, pred_cons_alt_season) or item_to_test_supply != "ACT Tablets"

def test_supply_model_ai_forecast_supply_levels_advanced(supply_forecast_model_ai):
    current_supply_data = pd.DataFrame({
        'item': [app_config.KEY_DRUG_SUBSTRINGS_SUPPLY[0] if app_config.KEY_DRUG_SUBSTRINGS_SUPPLY else "ItemA", "ItemB"],
        'current_stock': [100.0, 50.0],
        'avg_daily_consumption_historical': [10.0, 2.0],
        'last_stock_update_date': pd.to_datetime(['2023-10-01', '2023-10-01'])
    })
    forecast_horizon = 7 # Test short horizon
    forecast_df = supply_forecast_model_ai.forecast_supply_levels_advanced(current_supply_data, forecast_days_out=forecast_horizon)
    
    assert isinstance(forecast_df, pd.DataFrame)
    if not forecast_df.empty:
        assert len(forecast_df['item'].unique()) <= 2
        assert len(forecast_df) <= 2 * forecast_horizon
        expected_cols = ['item', 'forecast_date', 'forecasted_stock_level', 'forecasted_days_of_supply', 'predicted_daily_consumption', 'estimated_stockout_date_ai']
        for col in expected_cols:
            assert col in forecast_df.columns
        # Check stock depletion
        for item_name_check in current_supply_data['item'].unique():
            item_fc = forecast_df[forecast_df['item'] == item_name_check]
            if not item_fc.empty:
                 assert item_fc['forecasted_stock_level'].iloc[-1] < item_fc['forecasted_stock_level'].iloc[0] or item_fc['forecasted_stock_level'].iloc[0] == 0


# --- Tests for Central apply_ai_models Function ---

def test_apply_ai_models_adds_columns(sample_health_records_df_main):
    if sample_health_records_df_main.empty:
        pytest.skip("Sample health records DF is empty, skipping apply_ai_models test.")
    
    enriched_df, _ = apply_ai_models(sample_health_records_df_main.copy()) # Pass copy
    
    assert 'ai_risk_score' in enriched_df.columns
    assert 'ai_followup_priority_score' in enriched_df.columns
    assert len(enriched_df) == len(sample_health_records_df_main)
    if not enriched_df.empty:
        assert enriched_df['ai_risk_score'].notna().all()
        assert enriched_df['ai_followup_priority_score'].notna().all()

def test_apply_ai_models_with_supply_forecast(sample_health_records_df_main):
    if sample_health_records_df_main.empty:
        pytest.skip("Sample health records DF is empty.")

    # Create a dummy current_supply_status_df for testing the supply part
    mock_supply_status = pd.DataFrame({
        'item': [app_config.KEY_DRUG_SUBSTRINGS_SUPPLY[0]] if app_config.KEY_DRUG_SUBSTRINGS_SUPPLY else ['TestItem'],
        'current_stock': [150.0],
        'avg_daily_consumption_historical': [5.0],
        'last_stock_update_date': [pd.Timestamp('2023-10-01')]
    })
    
    _, supply_fc_df = apply_ai_models(sample_health_records_df_main.copy(), current_supply_status_df=mock_supply_status)
    
    assert isinstance(supply_fc_df, pd.DataFrame)
    # If mock_supply_status was not empty and forecasting worked, supply_fc_df should not be empty
    if not mock_supply_status.empty:
        assert not supply_fc_df.empty
        assert 'estimated_stockout_date_ai' in supply_fc_df.columns
    else: # This case should probably mean supply_fc_df is None or empty df as per logic
        assert supply_fc_df is None or supply_fc_df.empty


def test_apply_ai_models_empty_input():
    empty_df = pd.DataFrame()
    enriched_df, supply_df = apply_ai_models(empty_df)
    assert enriched_df.empty
    assert supply_df is None # Or empty df depending on impl.
