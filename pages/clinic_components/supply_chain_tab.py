# test/pages/clinic_components/supply_chain_tab.py
# Redesigned for "Sentinel Health Co-Pilot" - LMIC Edge-First System
# This module calculates and prepares supply forecast data for medical items at a clinic.
# This data is intended for:
#   1. Display on a simplified web dashboard/report for clinic managers at a Facility Node (Tier 2)
#      to monitor stock levels and anticipate shortages.
#   2. Informing procurement and supply chain logistics decisions.
# Direct plotting is removed; a separate UI component would visualize this data.

import pandas as pd
import numpy as np
import logging
# Assuming app_config is in the PYTHONPATH or project root.
from config import app_config # Uses new, redesigned app_config
from utils.core_data_processing import get_supply_forecast_data # For simple linear forecast
from utils.ai_analytics_engine import SupplyForecastingModel # For AI-simulated forecast
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

def prepare_clinic_supply_forecast_data(
    clinic_historical_health_df: pd.DataFrame, # Full historical data for consumption rates and latest stock
    reporting_period_str: str,
    forecast_days_out: int = 30, # Default forecast horizon for clinic planning
    use_ai_forecast_model: bool = False,
    items_to_forecast: Optional[List[str]] = None # If None, attempts to forecast KEY_DRUG_SUBSTRINGS_SUPPLY
) -> Dict[str, Any]:
    """
    Prepares supply forecast data for selected medical items.

    Args:
        clinic_historical_health_df: DataFrame of health records containing item usage,
                                     stock levels, and consumption rates for the clinic.
                                     Expected columns: 'item', 'encounter_date',
                                     'item_stock_agg_zone' (latest stock), 'consumption_rate_per_day'.
        reporting_period_str: String describing the current reporting context.
        forecast_days_out: Number of days into the future to forecast.
        use_ai_forecast_model: Boolean flag to use the AI simulation model.
        items_to_forecast: Optional list of specific item names to forecast. If None,
                           defaults to items matching KEY_DRUG_SUBSTRINGS_SUPPLY from app_config.

    Returns:
        Dict[str, Any]: A dictionary containing the forecast data.
        Example: {
            "reporting_period": "Current Outlook",
            "forecast_model_used": "AI-Simulated" | "Simple Linear",
            "forecast_data_df": pd.DataFrame(...), // Columns like: item, date, forecast_stock, forecast_days, est_stockout_date, [CI_lower, CI_upper]
            "forecast_items_summary_list": List[Dict], // Summary per item: item_name, current_stock, est_stockout_date
            "data_availability_notes": []
        }
    """
    logger.info(f"Preparing clinic supply forecast data. Model: {'AI' if use_ai_forecast_model else 'Linear'}. Horizon: {forecast_days_out} days.")

    supply_forecast_output: Dict[str, Any] = {
        "reporting_period": reporting_period_str,
        "forecast_model_used": "AI-Simulated" if use_ai_forecast_model else "Simple Linear",
        "forecast_data_df": None, # This will hold the main DataFrame with daily forecasts
        "forecast_items_summary_list": [], # Quick summary for overview
        "data_availability_notes": []
    }

    required_cols = ['item', 'encounter_date', 'item_stock_agg_zone', 'consumption_rate_per_day']
    if clinic_historical_health_df is None or clinic_historical_health_df.empty or \
       not all(c in clinic_historical_health_df.columns for c in required_cols):
        msg = f"Historical health data is missing one or more required columns for supply forecasts: {required_cols}"
        logger.error(msg)
        supply_forecast_output["data_availability_notes"].append(msg)
        return supply_forecast_output

    df_hist_supply = clinic_historical_health_df.copy()
    df_hist_supply['encounter_date'] = pd.to_datetime(df_hist_supply['encounter_date'], errors='coerce')
    df_hist_supply.dropna(subset=['encounter_date', 'item'], inplace=True) # Essential for getting latest status
    
    # Determine which items to focus on
    if items_to_forecast:
        item_filter_for_model = items_to_forecast
    elif app_config.KEY_DRUG_SUBSTRINGS_SUPPLY:
        # Get unique items from historical data that match any key drug substring
        all_items_in_data = df_hist_supply['item'].dropna().unique()
        item_filter_for_model = [
            item for item in all_items_in_data
            if any(sub.lower() in str(item).lower() for sub in app_config.KEY_DRUG_SUBSTRINGS_SUPPLY)
        ]
        if not item_filter_for_model:
             item_filter_for_model = all_items_in_data[:5] # Fallback to top 5 if no key drugs match
             supply_forecast_output["data_availability_notes"].append("No key drugs matched; forecasting for top available items if any.")
    else: # No specific items and no key drug list in config
        item_filter_for_model = df_hist_supply['item'].dropna().unique()[:5].tolist() # Fallback to first 5 unique items
        if not item_filter_for_model:
            supply_forecast_output["data_availability_notes"].append("No items found in historical data to forecast.")
            return supply_forecast_output

    if not item_filter_for_model: # If after all filtering, no items are selected
        supply_forecast_output["data_availability_notes"].append("No items selected or available for forecasting.")
        return supply_forecast_output


    forecast_df: Optional[pd.DataFrame] = None
    if use_ai_forecast_model:
        logger.info(f"Using AI Supply Forecasting Model for items: {item_filter_for_model}")
        ai_supply_model = SupplyForecastingModel()
        # AI model expects a DataFrame with current status: item, current_stock, avg_daily_consumption_historical, last_stock_update_date
        latest_item_status_df = df_hist_supply.sort_values('encounter_date').drop_duplicates(subset=['item'], keep='last')
        latest_item_status_df = latest_item_status_df.rename(columns={
            'item_stock_agg_zone': 'current_stock',
            'consumption_rate_per_day': 'avg_daily_consumption_historical',
            'encounter_date': 'last_stock_update_date'
        })[['item', 'current_stock', 'avg_daily_consumption_historical', 'last_stock_update_date']]
        
        # Filter this prepared df for the items we want to forecast
        latest_item_status_df_filtered = latest_item_status_df[latest_item_status_df['item'].isin(item_filter_for_model)]

        if not latest_item_status_df_filtered.empty:
            forecast_df = ai_supply_model.forecast_supply_levels_advanced(
                latest_item_status_df_filtered,
                forecast_days_out=forecast_days_out
            )
            # Expected columns from AI model: 'item', 'forecast_date', 'forecasted_stock_level',
            # 'forecasted_days_of_supply', 'predicted_daily_consumption', 'estimated_stockout_date_ai'
            if forecast_df is not None and not forecast_df.empty :
                forecast_df.rename(columns={'forecast_date':'date', 'estimated_stockout_date_ai':'estimated_stockout_date'}, inplace=True, errors='ignore')

        else:
            supply_forecast_output["data_availability_notes"].append(f"No current status data found for AI forecasting selected items: {item_filter_for_model}")

    else: # Use simple linear forecast
        logger.info(f"Using Simple Linear Supply Forecasting for items: {item_filter_for_model}")
        forecast_df = get_supply_forecast_data(
            health_df=df_hist_supply, # Uses original historical df
            forecast_days_out=forecast_days_out,
            item_filter_list=item_filter_for_model, # Pass the specific items
            source_context="ClinicSupplyTab/LinearFcst"
        )
        # Expected columns from linear model: item, date, current_stock_at_forecast_start, base_consumption_rate_per_day,
        #                                   forecasted_stock_level, forecasted_days_of_supply,
        #                                   estimated_stockout_date_linear, lower_ci_days_supply, upper_ci_days_supply
        #                                   initial_days_supply_at_forecast_start
        if forecast_df is not None and not forecast_df.empty:
            forecast_df.rename(columns={'estimated_stockout_date_linear':'estimated_stockout_date',
                                        'current_stock_at_forecast_start': 'initial_stock',
                                        'base_consumption_rate_per_day': 'initial_consumption_rate'}, inplace=True, errors='ignore')


    if forecast_df is not None and not forecast_df.empty:
        supply_forecast_output["forecast_data_df"] = forecast_df.sort_values(by=['item', 'date']).reset_index(drop=True)
        
        # Create a summary list for quick overview (e.g., for a summary table)
        # This requires initial stock, consumption rate, and estimated stockout date for each item.
        # The forecast_df might have these repeated or only the linear one has all upfront.
        # For simplicity, we derive from the first row of each item's forecast.
        item_summary_data = []
        for item_name_sum in forecast_df['item'].unique():
            item_fc_df_sum = forecast_df[forecast_df['item'] == item_name_sum]
            if not item_fc_df_sum.empty:
                first_row = item_fc_df_sum.iloc[0]
                current_s = first_row.get('initial_stock', first_row.get('current_stock', 0)) # current_stock_at_forecast_start or current_stock
                base_c = first_row.get('initial_consumption_rate', first_row.get('consumption_rate', 0)) # base_consumption_rate_per_day or consumption_rate
                est_so_date = pd.to_datetime(first_row.get('estimated_stockout_date', pd.NaT), errors='coerce')
                init_dos = first_row.get('initial_days_supply_at_forecast_start', current_s/base_c if base_c >0 else np.inf)

                item_summary_data.append({
                    "item_name": item_name_sum,
                    "current_stock_level": current_s,
                    "avg_daily_consumption_used_for_fcst": base_c,
                    "initial_days_of_supply": round(init_dos,1) if np.isfinite(init_dos) else "Inf.",
                    "estimated_stockout_date": est_so_date.strftime('%Y-%m-%d') if pd.notna(est_so_date) else "N/A (or > forecast horizon)"
                })
        supply_forecast_output["forecast_items_summary_list"] = item_summary_data
        
    elif not supply_forecast_output["data_availability_notes"]: # if forecast is empty but no specific notes yet
        supply_forecast_output["data_availability_notes"].append("Supply forecast could not be generated. Check data quality or model logs.")

    logger.info("Clinic supply forecast data preparation complete.")
    return supply_forecast_output


# --- Example Usage (for testing or integration into a reporting script) ---
if __name__ == '__main__':
    logger.setLevel(logging.DEBUG)
    logger.info("Running Clinic Supply Chain Tab component data preparation simulation...")

    # Create sample historical data for supplies
    dates_supply = pd.to_datetime(pd.date_range(start='2023-09-01', end='2023-10-05', freq='D'))
    items_supply = app_config.KEY_DRUG_SUBSTRINGS_SUPPLY[:3] + ["Gauze Rolls", "Syringes 5ml"]
    
    records_supply = []
    for item_s in items_supply:
        current_stock_item = np.random.randint(50, 200)
        base_cons_item = np.random.uniform(0.5, 5.0)
        for dt_s in dates_supply:
            stock_on_date = max(0, current_stock_item - base_cons_item * (dt_s - dates_supply.min()).days * np.random.uniform(0.8,1.2))
            cons_on_date = base_cons_item * np.random.uniform(0.7, 1.3)
            records_supply.append({
                'encounter_date': dt_s, # Using encounter_date as the date of stock record
                'item': item_s,
                'item_stock_agg_zone': stock_on_date, # Simulating aggregated stock for the item
                'consumption_rate_per_day': cons_on_date,
                'zone_id': 'ClinicMainZone' # Assuming clinic-wide stock tracking for this test
            })
    sample_supply_historical_df = pd.DataFrame(records_supply)

    reporting_str_supply = "October 2023 Forecast"
    
    # Test Linear Forecast
    linear_forecast_output = prepare_clinic_supply_forecast_data(
        clinic_historical_health_df=sample_supply_historical_df,
        reporting_period_str=reporting_str_supply,
        use_ai_forecast_model=False,
        items_to_forecast=items_supply[:2] # Forecast for first 2 items
    )
    print(f"\n--- Prepared Clinic Supply Forecast Data (Linear Model) for: {reporting_str_supply} ---")
    print(f"Model Used: {linear_forecast_output['forecast_model_used']}")
    print("Items Summary:")
    for item_sum in linear_forecast_output.get("forecast_items_summary_list", []): print(f"  - {item_sum}")
    if linear_forecast_output["forecast_data_df"] is not None:
        print("\nDaily Forecast Data (showing first few rows per item):")
        print(linear_forecast_output["forecast_data_df"].groupby('item').head(2).to_string())
    if linear_forecast_output["data_availability_notes"]: print(f"Notes: {linear_forecast_output['data_availability_notes']}")

    # Test AI-Simulated Forecast
    ai_forecast_output = prepare_clinic_supply_forecast_data(
        clinic_historical_health_df=sample_supply_historical_df,
        reporting_period_str=reporting_str_supply,
        use_ai_forecast_model=True,
        # items_to_forecast=None # Let it pick from KEY_DRUG_SUBSTRINGS_SUPPLY
    )
    print(f"\n--- Prepared Clinic Supply Forecast Data (AI-Simulated Model) for: {reporting_str_supply} ---")
    print(f"Model Used: {ai_forecast_output['forecast_model_used']}")
    print("Items Summary:")
    for item_sum in ai_forecast_output.get("forecast_items_summary_list", []): print(f"  - {item_sum}")
    if ai_forecast_output["forecast_data_df"] is not None:
        print("\nDaily Forecast Data (showing first few rows per item):")
        print(ai_forecast_output["forecast_data_df"].groupby('item').head(2).to_string())
    if ai_forecast_output["data_availability_notes"]: print(f"Notes: {ai_forecast_output['data_availability_notes']}")
