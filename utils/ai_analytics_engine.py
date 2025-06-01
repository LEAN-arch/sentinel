# test/utils/ai_analytics_engine.py
# Redesigned for "Sentinel Health Co-Pilot" - LMIC Edge-First System
# This module simulates the core AI/Analytics logic that would run primarily on:
#   - Personal Edge Devices (PEDs) using lightweight models (e.g., TinyML, TensorFlow Lite).
#   - Supervisor Hubs for team-level aggregation.
#   - Facility Nodes/Cloud for more complex model training and population analytics.
# The Python classes here serve as reference implementations, for backend simulation,
# and for generating training data or baseline logic for edge models.

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
import logging
# Assuming app_config is in the PYTHONPATH or project root.
from config import app_config # Using the new, redesigned app_config

logger = logging.getLogger(__name__)

class RiskPredictionModel:
    """
    SIMULATES a pre-trained patient/worker risk prediction model, adaptable for Edge deployment.
    Uses rule-based logic with weights and factors for core features.
    Actual Edge model would be optimized (e.g., quantized TFLite model derived from this logic or richer data).
    """
    def __init__(self):
        # Factors based on new app_config thresholds and lean data inputs
        self.base_risk_factors = {
            'age': { # Demographics
                'weight': 0.5, 'threshold_high': app_config.RISK_SCORE_MODERATE_THRESHOLD, 'factor_high': 10, # Older age (contextualized)
                'threshold_low': 18, 'factor_low': -2 # Very young might have specific vulnerabilities (not neg risk)
            },
            'min_spo2_pct': { # Sensor stream
                'weight': 2.5, # Increased weight for critical vital
                'threshold_low': app_config.ALERT_SPO2_CRITICAL_LOW_PCT, 'factor_low': 30,
                'mid_threshold_low': app_config.ALERT_SPO2_WARNING_LOW_PCT, 'factor_mid_low': 15
            },
            'vital_signs_temperature_celsius': { # Sensor stream / manual
                'weight': 2.0,
                'threshold_high': app_config.ALERT_BODY_TEMP_FEVER_C, 'factor_high': 15,
                'super_high_threshold': app_config.ALERT_BODY_TEMP_HIGH_FEVER_C, 'factor_super_high': 25
            },
            'max_skin_temp_celsius': { # Alternative for temp if body temp not available
                'weight': 1.8,
                'threshold_high': app_config.HEAT_STRESS_RISK_BODY_TEMP_C, 'factor_high': 10,
                'super_high_threshold': app_config.ALERT_BODY_TEMP_HIGH_FEVER_C - 0.5, 'factor_super_high': 20 # Skin temp slightly less direct
            },
            'stress_level_score': { # Psychometric input or derived from HRV
                'weight': 0.8, 'threshold_high': app_config.FATIGUE_INDEX_MODERATE_THRESHOLD, 'factor_high': 10, # Assuming stress score aligned with fatigue index for now
                'super_high_threshold': app_config.FATIGUE_INDEX_HIGH_THRESHOLD, 'factor_super_high': 15
            },
            'avg_hrv': { # Sensor stream (if available and processed into a numeric index like RMSSD)
                'weight': 1.2, 'threshold_low': app_config.STRESS_HRV_LOW_THRESHOLD_MS, 'factor_low': 15 # Low HRV indicates stress/fatigue
            },
            'tb_contact_traced': { # Contextual / CHW Input (Less for worker self-risk)
                'weight': 1.0, 'is_flag': True, 'flag_value': 1, 'factor_true': 12 # Example: for patient risk
            },
            'fall_detected_today': { # Sensor stream
                'weight': 2.0, 'is_flag': True, 'flag_value': 1, 'factor_true': 20
            },
            # Worker-specific contextual factors could be added here (e.g. 'shift_intensity_high': {'weight': 0.5, 'factor': 5})
            # Ambient factors could also contribute (e.g. prolonged high heat index exposure)
            'ambient_heat_index_c': { # Ambient sensor input
                 'weight': 0.7, 'threshold_high': app_config.ALERT_AMBIENT_HEAT_INDEX_RISK_C, 'factor_high': 8,
                 'super_high_threshold': app_config.ALERT_AMBIENT_HEAT_INDEX_DANGER_C, 'factor_super_high': 15
            },
             'ppe_non_compliant': { # Job Context / Input
                'weight': 1.0, 'is_flag': True, 'flag_value': 1, 'factor_true': 10 # Example if PPE not worn
            }
        }
        # More focused, actionable conditions from the new config
        self.condition_base_scores = {cond: 25 for cond in app_config.KEY_CONDITIONS_FOR_ACTION} # Assign a base score
        self.condition_base_scores.update({ # Refine scores for specific conditions if needed
            "Sepsis": 40, "Severe Dehydration": 35, "Heat Stroke": 38,
            "TB": 30, "HIV-Positive": 22, "Pneumonia": 28, "Malaria": 20,
            "Wellness Visit": -10, "Follow-up Health": -5 # For patient context
        })
        # Chronic condition flag contribution
        self.CHRONIC_CONDITION_FLAG_RISK_POINTS = 15

        logger.info("Simulated RiskPredictionModel (Edge Optimized Logic) initialized.")

    def _get_condition_base_score(self, condition_str: Optional[str]) -> float:
        if pd.isna(condition_str) or str(condition_str).lower() == "unknown": return 0.0
        base_score = 0.0
        # LMIC context: condition might be a simple string, not multiple semi-colon separated formal diagnoses.
        # CHW might input "bad cough + fever" which maps to a "Respiratory Distress" like condition key.
        # For this simulation, we'll keep simple string matching from KEY_CONDITIONS_FOR_ACTION.
        condition_input_lower = str(condition_str).lower()
        for known_cond, score_val in self.condition_base_scores.items():
            if known_cond.lower() in condition_input_lower: # Direct match or partial for flexibility
                base_score = max(base_score, score_val) # Take highest base if multiple (simple approach)
                # More sophisticated: sum or weighted sum if co-morbidities fully parsed.
                # For PED, simpler "dominant condition" score might be better.
        return base_score

    def predict_risk_score(self, features: pd.Series) -> float:
        """Predicts risk score for a single entity (patient or worker) based on features."""
        calculated_risk = 0.0
        
        # 1. Condition-based score (primarily for patients, could be adapted for worker's self-reported illness)
        calculated_risk += self._get_condition_base_score(features.get('condition'))

        # 2. Chronic condition flag (simple Y/N for LMIC)
        if features.get('chronic_condition_flag') == 1 or str(features.get('chronic_condition_flag')).lower() == 'yes':
            calculated_risk += self.CHRONIC_CONDITION_FLAG_RISK_POINTS

        # 3. Factor-based scoring from `base_risk_factors`
        for feature_key, params in self.base_risk_factors.items():
            if feature_key in features and pd.notna(features[feature_key]):
                value = features[feature_key]; weight = params.get('weight', 1.0)

                if params.get('is_flag'):
                    if value == params.get('flag_value', 1): calculated_risk += params.get('factor_true', 0) * weight
                # (Removed is_value_match for string; prefer direct feature flags like 'ppe_non_compliant' or condition scoring)
                else: # Numeric with thresholds
                    # Order matters: check super_high, then high, then mid_low, then low
                    if 'super_high_threshold' in params and value >= params['super_high_threshold']:
                        calculated_risk += params.get('factor_super_high', 0) * weight
                    elif 'threshold_high' in params and value >= params['threshold_high']:
                        calculated_risk += params.get('factor_high', 0) * weight
                    
                    # For factors where low values are bad (e.g. SpO2, HRV)
                    if 'threshold_low' in params and value < params['threshold_low']:
                         calculated_risk += params.get('factor_low', 0) * weight
                    elif 'mid_threshold_low' in params and value < params['mid_threshold_low']: # Mutually exclusive with above if ordered
                         calculated_risk += params.get('factor_mid_low', 0) * weight
        
        # Adherence only if relevant (patient context, less for worker self-risk model unless it's adherence to safety protocols)
        adherence = features.get('medication_adherence_self_report', "Unknown")
        if str(adherence).lower() == 'poor': calculated_risk += 10 # Reduced impact, more specific features should capture acute risk
        elif str(adherence).lower() == 'fair': calculated_risk += 5

        # Behavioral inputs (psychometric fatigue/distress) - direct addition to score
        if pd.notna(features.get('rapid_psychometric_fatigue_score')): # Assuming 0-10 scale for a slider
            calculated_risk += features.get('rapid_psychometric_fatigue_score',0) * 1.5 # Weighted contribution
        if pd.notna(features.get('rapid_psychometric_distress_emoji_score')): # e.g., sad_emoji = 10, neutral = 5, happy = 0
            calculated_risk += features.get('rapid_psychometric_distress_emoji_score',0)

        # Ensure risk score is within 0-100
        return float(np.clip(calculated_risk, 0, 100))

    def predict_bulk_risk_scores(self, data_df: pd.DataFrame) -> pd.Series:
        """Applies risk prediction to a DataFrame."""
        if data_df.empty: return pd.Series(dtype='float64')
        
        # Ensure all features the model expects are present in data_df, filling with neutral defaults if not.
        # This is crucial for consistent model application.
        temp_df_for_pred = data_df.copy()
        all_expected_features = list(self.base_risk_factors.keys()) + ['condition', 'chronic_condition_flag', 'medication_adherence_self_report', 'rapid_psychometric_fatigue_score', 'rapid_psychometric_distress_emoji_score']
        for feature_name in all_expected_features:
            if feature_name not in temp_df_for_pred.columns:
                # Determine appropriate default: 0 for flags/scores, np.nan for sensor readings, "Unknown" for strings
                if any(f_part in feature_name for f_part in ['_flag', '_score', '_today', '_compliant']): default = 0
                elif any(f_part in feature_name for f_part in ['spo2','temp','hrv','heat_index','age']): default = np.nan
                else: default = "Unknown"
                temp_df_for_pred[feature_name] = default
        
        return temp_df_for_pred.apply(lambda row: self.predict_risk_score(row), axis=1)


class FollowUpPrioritizer:
    """
    SIMULATES prioritization logic for follow-up tasks or escalations. Adaptable for Edge.
    This moves beyond just patient follow-up to contextual task prioritization for workers.
    """
    def __init__(self):
        self.priority_weights = {
            'base_ai_risk_score_contribution_pct': 0.35, # Worker's or Patient's calculated AI Risk
            'critical_vital_alert_points': 35,         # If active critical vital sign alert
            'pending_urgent_task_points': 25,          # e.g., pending critical referral for patient, or urgent safety task for worker
            'acute_condition_severity_points': 20,     # Points for specific highly severe conditions/symptoms
            'contextual_hazard_points': 15,            # e.g., worker in very high heat index zone
            'task_overdue_factor_per_day': 0.5          # Points added per day a task is overdue (max cap)
        }
        logger.info("Simulated FollowUpPrioritizer (Edge Optimized Logic) initialized.")

    def _has_active_critical_vitals_alert(self, features: pd.Series) -> bool:
        # Re-check conditions that would trigger immediate high-severity alerts
        if pd.notna(features.get('min_spo2_pct')) and features['min_spo2_pct'] < app_config.ALERT_SPO2_CRITICAL_LOW_PCT: return True
        temp_val = features.get('vital_signs_temperature_celsius', features.get('max_skin_temp_celsius', np.nan))
        if pd.notna(temp_val) and temp_val >= app_config.ALERT_BODY_TEMP_HIGH_FEVER_C: return True
        if pd.notna(features.get('fall_detected_today')) and features['fall_detected_today'] > 0: return True
        # Could add rules for very low HRV or extreme HR if data is reliable
        return False

    def _is_pending_urgent_task(self, features: pd.Series) -> bool: # Could be patient or worker task
        # Patient context (from CHW data input)
        if str(features.get('referral_status', 'Unknown')).lower() == 'pending':
            # Using the actionable condition list for "critical"
            if any(cond_key.lower() in str(features.get('condition', '')).lower() for cond_key in app_config.KEY_CONDITIONS_FOR_ACTION):
                return True
        # Worker context (example - e.g., assigned an urgent equipment check or safety protocol)
        if str(features.get('worker_task_priority', 'Normal')).lower() == 'urgent':
            return True
        return False
        
    def _has_acute_condition_severity(self, features: pd.Series) -> bool: # Primarily for patients
        condition_str = str(features.get('condition','')).lower()
        # Sepsis, Severe Dehydration, Heat Stroke already have high base scores.
        # This could add points if, e.g., "Pneumonia" is present *with* very low SpO2
        if ("pneumonia" in condition_str and pd.notna(features.get('min_spo2_pct')) and features.get('min_spo2_pct') < app_config.ALERT_SPO2_WARNING_LOW_PCT):
            return True
        # Other combinations, e.g., certain symptoms reported + specific vital deviation
        return False

    def _contextual_hazard_present(self, features: pd.Series) -> bool: # Primarily for workers
        if pd.notna(features.get('ambient_heat_index_c')) and features['ambient_heat_index_c'] >= app_config.ALERT_AMBIENT_HEAT_INDEX_DANGER_C: return True
        if pd.notna(features.get('ambient_co2_ppm')) and features['ambient_co2_ppm'] >= app_config.ALERT_AMBIENT_CO2_VERY_HIGH_PPM: return True
        # Add PM2.5, noise etc. from new app_config
        if pd.notna(features.get('ambient_pm25_ugm3')) and features['ambient_pm25_ugm3'] >= app_config.ALERT_AMBIENT_PM25_VERY_HIGH_UGM3: return True
        return False

    def calculate_priority_score(self, features: pd.Series, days_task_overdue: int = 0) -> float:
        """Calculates a priority score (0-100) for a given set of features and task status."""
        priority_score = 0.0

        # 1. Contribution from base AI Risk Score
        ai_risk_value = features.get('ai_risk_score', 0.0) # From RiskPredictionModel
        if pd.notna(ai_risk_value):
            priority_score += ai_risk_value * self.priority_weights['base_ai_risk_score_contribution_pct']
        
        # 2. Critical Vital Alerts
        if self._has_active_critical_vitals_alert(features):
            priority_score += self.priority_weights['critical_vital_alert_points']

        # 3. Pending Urgent Task
        if self._is_pending_urgent_task(features):
            priority_score += self.priority_weights['pending_urgent_task_points']

        # 4. Acute Condition Severity (for patients)
        if self._has_acute_condition_severity(features):
            priority_score += self.priority_weights['acute_condition_severity_points']
            
        # 5. Contextual Hazard (for workers)
        if self._contextual_hazard_present(features):
            priority_score += self.priority_weights['contextual_hazard_points']
        
        # 6. Task Overdue
        priority_score += min(days_task_overdue, 30) * self.priority_weights['task_overdue_factor_per_day'] # Cap overdue bonus

        return float(np.clip(priority_score, 0, 100))

    def generate_followup_priorities(self, data_df: pd.DataFrame) -> pd.Series:
        """Applies prioritization logic to a DataFrame."""
        if data_df.empty: return pd.Series(dtype='float64')
        
        # 'days_since_last_contact' logic from previous simulation might be complex for PED.
        # Simpler: If task has 'due_date' or 'assigned_date', calculate overdue days.
        # For this simulation, let's assume a 'days_task_overdue' column might be pre-calculated or defaulted to 0.
        if 'days_task_overdue' not in data_df.columns: # Ensure column exists
            temp_data_df = data_df.copy() # Work on a copy
            temp_data_df['days_task_overdue'] = 0 # Default if not provided
        else:
            temp_data_df = data_df

        return temp_data_df.apply(lambda row: self.calculate_priority_score(row, row.get('days_task_overdue',0)), axis=1)


class SupplyForecastingModel:
    """
    SIMULATES an AI-driven supply forecasting model for critical items.
    On PED, this might be very simplified (e.g., worker's personal kit based on typical daily use).
    At Facility Node, a more complex model could run. This sim uses item-specific seasonal patterns.
    """
    def __init__(self):
        # Example parameters: Seasonal coefficients (monthly), trend factor, noise.
        # These would be learned from historical data in a real system.
        self.item_params = {} # Initialize empty
        for item_key in app_config.KEY_DRUG_SUBSTRINGS_SUPPLY: # Use key drugs from config
            # Assign generic but slightly varied params for simulation if not detailed
            self.item_params[item_key] = {
                "coeffs": np.random.uniform(0.8, 1.2, 12).tolist(), # Monthly seasonal adjustment
                "trend": np.random.uniform(0.001, 0.015),           # Slight positive trend per 30 days
                "noise_std": np.random.uniform(0.05, 0.15)         # Random daily variation
            }
        # Override for specific items if known patterns exist (example)
        if "ACT Tablets" in self.item_params:
            self.item_params["ACT Tablets"]["coeffs"] = [0.7,0.7,0.8,0.9,1.1,1.3,1.4,1.2,1.0,0.8,0.7,0.7] # Malaria season example
        logger.info("Simulated AI SupplyForecastingModel initialized with dynamic item parameters.")

    def _get_item_params(self, item_name: str) -> Dict:
        # Find best match in self.item_params using KEY_DRUG_SUBSTRINGS_SUPPLY
        for key_substring in self.item_params.keys(): # Keys are now from KEY_DRUG_SUBSTRINGS_SUPPLY
            if key_substring.lower() in item_name.lower():
                return self.item_params[key_substring]
        # Fallback if no specific substring match - very generic
        return {"coeffs": [1.0]*12, "trend": 0.001, "noise_std": 0.1}

    def _predict_daily_consumption_ai(self, base_avg_daily_consumption: float, item_name: str, forecast_date: pd.Timestamp, days_since_forecast_start: int) -> float:
        if pd.isna(base_avg_daily_consumption) or base_avg_daily_consumption <= 0: return 0.001 # Min consumption if positive base
        
        item_specific_params = self._get_item_params(item_name)
        monthly_seasonality_factor = item_specific_params["coeffs"][forecast_date.month - 1]
        trend_effect = (1 + item_specific_params["trend"] / 30)**days_since_forecast_start # Compounding trend
        random_noise_factor = np.random.normal(1.0, item_specific_params["noise_std"])
        
        predicted_consumption = base_avg_daily_consumption * monthly_seasonality_factor * trend_effect * random_noise_factor
        return max(0.001, predicted_consumption) # Ensure consumption is not zero or negative

    def forecast_supply_levels_advanced(
        self,
        current_supply_levels_df: pd.DataFrame, # Expects columns: item, current_stock, avg_daily_consumption_historical, last_stock_update_date
        forecast_days_out: int = 7, # Shorter horizon for PED/field utility
        item_filter_list: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Forecasts supply levels using the AI simulation.
        Input DF should represent current state (e.g., from a sync or last known good state).
        """
        logger.info(f"Performing AI-simulated supply forecast for {forecast_days_out} days.")
        output_cols = ['item', 'forecast_date', 'forecasted_stock_level', 'forecasted_days_of_supply', 'predicted_daily_consumption', 'estimated_stockout_date_ai']
        
        if not isinstance(current_supply_levels_df, pd.DataFrame) or current_supply_levels_df.empty:
            logger.warning("Input for AI supply forecast is empty.")
            return pd.DataFrame(columns=output_cols)
        
        required_cols = ['item', 'current_stock', 'avg_daily_consumption_historical', 'last_stock_update_date']
        if not all(col in current_supply_levels_df.columns for col in required_cols):
            logger.error(f"AI Supply Forecast: Input DF missing one or more required columns: {required_cols}")
            return pd.DataFrame(columns=output_cols)

        df_to_forecast = current_supply_levels_df.copy()
        if item_filter_list:
            df_to_forecast = df_to_forecast[df_to_forecast['item'].isin(item_filter_list)]
        if df_to_forecast.empty: return pd.DataFrame(columns=output_cols)

        df_to_forecast['last_stock_update_date'] = pd.to_datetime(df_to_forecast['last_stock_update_date'], errors='coerce')
        df_to_forecast.dropna(subset=['last_stock_update_date'], inplace=True)

        all_forecasts_list = []
        for _, item_row in df_to_forecast.iterrows():
            item_name_fc = item_row['item']
            stock_at_start_fc = float(item_row.get('current_stock', 0))
            base_cons_fc = float(item_row.get('avg_daily_consumption_historical', 0.001)) # Use a tiny minimum
            last_date_fc = item_row['last_stock_update_date']

            if pd.isna(stock_at_start_fc) or stock_at_start_fc < 0: stock_at_start_fc = 0.0
            if pd.isna(base_cons_fc) or base_cons_fc <=0: base_cons_fc = 0.001

            running_stock_level = stock_at_start_fc
            calculated_stockout_date = pd.NaT

            for day_offset in range(forecast_days_out):
                current_forecast_date = last_date_fc + pd.Timedelta(days=day_offset + 1)
                predicted_daily_use = self._predict_daily_consumption_ai(base_cons_fc, item_name_fc, current_forecast_date, day_offset + 1)
                
                stock_before_consumption_today = running_stock_level
                running_stock_level -= predicted_daily_use
                running_stock_level = max(0, running_stock_level) # Cannot go below zero

                days_supply_remaining = (running_stock_level / predicted_daily_use) if predicted_daily_use > 0.0001 else (np.inf if running_stock_level > 0 else 0)

                if pd.isna(calculated_stockout_date) and stock_before_consumption_today > 0 and running_stock_level <= 0:
                    # Estimate fraction of day it stocked out
                    fraction_of_day_stocked_out = (stock_before_consumption_today / predicted_daily_use) if predicted_daily_use > 0.0001 else 0.0
                    calculated_stockout_date = last_date_fc + pd.Timedelta(days=day_offset + fraction_of_day_stocked_out)

                all_forecasts_list.append({
                    'item': item_name_fc,
                    'forecast_date': current_forecast_date,
                    'forecasted_stock_level': running_stock_level,
                    'forecasted_days_of_supply': days_supply_remaining,
                    'predicted_daily_consumption': predicted_daily_use,
                    'estimated_stockout_date_ai': calculated_stockout_date # Propagates NaT until stockout
                })
            
            # If not stocked out within forecast period, make a final estimate based on average predicted use in period
            if pd.isna(calculated_stockout_date) and stock_at_start_fc > 0:
                period_forecasts_for_item = [f for f in all_forecasts_list if f['item'] == item_name_fc and f['forecast_date'] > last_date_fc] # get forecasts for this item in current loop
                if period_forecasts_for_item:
                     avg_predicted_consumption_in_period = pd.Series([f['predicted_daily_consumption'] for f in period_forecasts_for_item]).mean()
                     if avg_predicted_consumption_in_period > 0.0001:
                         days_to_final_stockout = stock_at_start_fc / avg_predicted_consumption_in_period
                         final_est_stockout = last_date_fc + pd.to_timedelta(days_to_final_stockout, unit='D')
                         # Update all NaT stockout dates for this item with this final estimate
                         for record in all_forecasts_list:
                             if record['item'] == item_name_fc and pd.isna(record['estimated_stockout_date_ai']):
                                 record['estimated_stockout_date_ai'] = final_est_stockout


        if not all_forecasts_list: return pd.DataFrame(columns=output_cols)
        forecast_df_final = pd.DataFrame(all_forecasts_list)
        forecast_df_final['estimated_stockout_date_ai'] = pd.to_datetime(forecast_df_final['estimated_stockout_date_ai'], errors='coerce')
        return forecast_df_final


# --- Central AI Application Function ---
def apply_ai_models(
    health_df: pd.DataFrame,
    current_supply_status_df: Optional[pd.DataFrame] = None, # Optional: For integrated supply forecasting trigger
    source_context: str = "FacilityNode/BatchProcessing"
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Applies relevant simulated AI models to the health data.
    In an Edge scenario, individual models (risk, priority) would run closer to data capture.
    Supply forecast is more complex and likely on Hub/Facility or with highly simplified rules on PED.
    Returns tuple: (enriched_health_df, optional_supply_forecast_df)
    """
    logger.info(f"({source_context}) Applying AI models to health data (rows: {len(health_df) if health_df is not None else 0}).")
    if health_df is None or health_df.empty:
        logger.warning(f"({source_context}) Input health_df to apply_ai_models is empty. No AI processing done.")
        return pd.DataFrame(columns=health_df.columns if health_df is not None else []), None

    df_enriched = health_df.copy()

    # 1. Patient/Worker Risk Scoring (applies to each row which represents an encounter or state)
    risk_model = RiskPredictionModel()
    df_enriched['ai_risk_score'] = risk_model.predict_bulk_risk_scores(df_enriched)
    logger.info(f"({source_context}) Applied simulated AI risk scoring.")

    # 2. Follow-up/Task Prioritization
    # This model might need historical context (e.g., days_task_overdue) which needs to be
    # passed or calculated before calling. Here, it's simplified assuming relevant fields are in df_enriched.
    priority_model = FollowUpPrioritizer()
    df_enriched['ai_followup_priority_score'] = priority_model.generate_followup_priorities(df_enriched)
    logger.info(f"({source_context}) Applied simulated AI follow-up/task prioritization.")
    
    # 3. Supply Forecasting (Example call if current_supply_status_df is provided)
    # On PED, this would be much simpler. This is for Hub/Facility sim.
    supply_forecast_df_final = None
    if current_supply_status_df is not None and not current_supply_status_df.empty:
        supply_model = SupplyForecastingModel()
        # Assuming current_supply_status_df has ['item', 'current_stock', 'avg_daily_consumption_historical', 'last_stock_update_date']
        supply_forecast_df_final = supply_model.forecast_supply_levels_advanced(
            current_supply_status_df,
            forecast_days_out=app_config.EDGE_APP_DEFAULT_LANGUAGE # Using this as an example, should be meaningful
        )
        logger.info(f"({source_context}) Generated AI-simulated supply forecast.")
    else:
        logger.info(f"({source_context}) No current supply status provided; skipping AI supply forecast.")

    logger.info(f"({source_context}) AI model application complete. Enriched DF rows: {len(df_enriched)}")
    return df_enriched, supply_forecast_df_final
