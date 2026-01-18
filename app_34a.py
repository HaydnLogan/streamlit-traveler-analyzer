# v34a - 01.18.26 RESTORED Origin Filtering to main area with proper defaults
# Changes from v34:
# - Restored Origin Filtering section to main area (was moved to sidebar in v34)
# - Changed default: "Filter Origins" checkbox now enabled by default (value=True)
# - Epic and Anchor origins remain preselected by default
# - Fixed origin filtering logic to match app_33f2 (exact column name matching)

# v34 - 01.18.26 INTEGRATED Nested Swing Detector + Look Back Days + Fixed 'Input @ start' bug
# Changes from v33f2:
# - Fixed 'Input @ start' column bug: Now correctly uses the day start, not 2 days prior
# - Added 'Look Back Days' setting (default 20) under Day Start Configuration 
# - Updated 'Full Range (single)' to export to one tab like 'Full Range (multiple)'
# - Changed default Window Radius from 60 to 600 for both single and multiple modes
# - Updated naming convention to include look back days: "asset + RawTrav for + date + (lookback days) + radius + window"
# - Integrated Nested Swing Detector with new 'Turns' column showing zone-based swing distances
# - 'Turns' column placed after 'bg HOD/LOD' and before 'Star'

# v33f - 12.27.25 FIXED datetime formatting, single mode highlights, added HOD/LOD zones
# Previous changes preserved in git history

import streamlit as st
import pandas as pd
import datetime as dt
import io
from typing import Optional, List, Dict, Tuple
from pandas import ExcelWriter
from openpyxl import load_workbook
from openpyxl.styles import Font
from openpyxl.utils import get_column_letter
from custom_range_calculator_0813 import apply_custom_ranges_advanced, apply_full_range_advanced
from hod_lod_processor import (
    process_hod_lod_mode, 
    is_complete_trading_day, 
    get_trading_day_bounds
)
from nested_swing_detector import analyze_swings, parse_timestamp_naive

# Configure pandas to handle large datasets
pd.set_option("styler.render.max_elements", 2000000)

# Import functions
from a_helpers import (
    clean_timestamp, process_feed, get_input_at_day_start, apply_excel_highlighting,
    GROUP_1A_TRAVELERS, GROUP_1B_TRAVELERS, GROUP_2A_TRAVELERS, GROUP_2B_TRAVELERS,
    get_input_value, highlight_traveler_report, get_input_at_time, highlight_custom_traveler_report,
    generate_master_traveler_list, EPIC_ORIGINS, ANCHOR_ORIGINS   
)

# Model imports
from model_g_manager import run_model_g_detection

try:
    from models.models_a_today import run_a_model_detection_today
except ImportError:
    def run_a_model_detection_today(df):
        st.warning("Model A detection not available in this environment")

try:
    from models.mod_b_05pg1 import run_b_model_detection
except ImportError:
    def run_b_model_detection(df):
        st.warning("Model B detection not available in this environment")

try:
    from models.mod_c_04gpr3 import run_c_model_detection
except ImportError:
    def run_c_model_detection(df, run_c01=True, run_c02=True, run_c04=True):
        st.warning("Model C detection not available in this environment")

try:
    from models.mod_x_03g import run_x_model_detection
except ImportError:
    def run_x_model_detection(df):
        st.warning("Model X detection not available in this environment")

try:
    from models.simple_mega_report2 import run_simple_single_line_analysis
except ImportError:
    def run_simple_single_line_analysis(df):
        st.warning("Single Line Mega Report not available in this environment")


# === Helper Functions for Full Range (multiple) ===

def get_trading_day_from_datetime(dt_obj):
    """
    Given a datetime (e.g., 1800 on 7-Dec), return the trading day date.
    The 1800 open on 7-Dec is the start of 8-Dec trading day.
    If time is before 1800 (6PM), it's the same day. If >= 1800, it's the next day.
    """
    if dt_obj.hour >= 18:
        return (dt_obj + dt.timedelta(days=1)).date()
    else:
        return dt_obj.date()

def format_tab_name(dt_obj):
    """
    Format datetime to tab name like '07Dec 1800' or '08Dec 0315'
    """
    return dt_obj.strftime("%d%b %H%M")

def combine_and_format_groups(grp_1a_df, grp_1b_df):
    """
    Combine Grp_1a and Grp_1b DataFrames, sort by Output (desc) then Arrival (asc),
    and add Star and Reason columns.
    """
    # Combine the groups
    combined = pd.concat([grp_1a_df, grp_1b_df], ignore_index=True)
    
    # Sort by Output (desc) then Arrival (asc)
    if not combined.empty and 'Output' in combined.columns and 'Arrival' in combined.columns:
        combined = combined.sort_values(['Output', 'Arrival'], ascending=[False, True])
    elif not combined.empty and 'Output' in combined.columns:
        combined = combined.sort_values(['Output'], ascending=[False])
    
    # Add Star and Reason columns if they don't exist
    if 'Star' not in combined.columns:
        combined['Star'] = pd.NA
    if 'Reason' not in combined.columns:
        combined['Reason'] = pd.NA
    
    return combined

def get_hod_lod_for_trading_day(feed_df, trading_day_start):
    """
    Find the High of Day (HOD) and Low of Day (LOD) for a trading day.
    
    Args:
        feed_df: DataFrame with 'time', 'high', 'low' columns
        trading_day_start: datetime of 18:00 on the previous day (start of trading day)
    
    Returns:
        tuple: (hod_value, lod_value) or (None, None) if not found
    """
    if feed_df is None or feed_df.empty:
        return None, None
    
    # Trading day runs from 18:00 to 16:45 next day
    trading_day_end = trading_day_start + dt.timedelta(hours=22, minutes=45)
    
    # Filter to trading day hours
    day_data = feed_df[
        (feed_df['time'] >= trading_day_start) & 
        (feed_df['time'] <= trading_day_end)
    ]
    
    if day_data.empty:
        return None, None
    
    # Get HOD and LOD
    hod = day_data['high'].max() if 'high' in day_data.columns else None
    lod = day_data['low'].min() if 'low' in day_data.columns else None
    
    return hod, lod

def calculate_hod_lod_zone(output_value, hod, lod, feed_prefix='sm'):
    """
    Calculate which HOD/LOD zone an output value falls into.
    
    Args:
        output_value: The output price to check
        hod: High of day value
        lod: Low of day value
        feed_prefix: 'sm' or 'bg'
    
    Returns:
        str: Zone label like 'sm HOD 0 to 9' or None if not in any zone
    """
    if pd.isna(output_value) or hod is None or lod is None:
        return None
    
    # HOD zones (going down from HOD)
    # Zone 0-9: HOD to HOD-9
    # Zone 09-18: HOD-9 to HOD-18
    # Zone 18-27: HOD-18 to HOD-27
    if hod - 9 <= output_value <= hod:
        return f"{feed_prefix} HOD 0 to 9"
    elif hod - 18 <= output_value < hod - 9:
        return f"{feed_prefix} HOD 09 to 18"
    elif hod - 27 <= output_value < hod - 18:
        return f"{feed_prefix} HOD 18 to 27"
    
    # LOD zones (going up from LOD)
    # Zone 0-9: LOD to LOD+9
    # Zone 09-18: LOD+9 to LOD+18
    # Zone 18-27: LOD+18 to LOD+27
    elif lod <= output_value <= lod + 9:
        return f"{feed_prefix} LOD 0 to 9"
    elif lod + 9 < output_value <= lod + 18:
        return f"{feed_prefix} LOD 09 to 18"
    elif lod + 18 < output_value <= lod + 27:
        return f"{feed_prefix} LOD 18 to 27"
    
    return None

def add_hod_lod_columns(df, small_df, big_df, trading_day_start):
    """
    Add HOD/LOD zone columns to the dataframe.
    
    Args:
        df: Traveler report dataframe
        small_df: Small feed dataframe
        big_df: Big feed dataframe
        trading_day_start: datetime of 18:00 on previous day
    
    Returns:
        DataFrame with added 'sm HOD/LOD' and 'bg HOD/LOD' columns
    """
    if df is None or df.empty:
        return df
    
    df = df.copy()
    
    # Get HOD/LOD for each feed
    sm_hod, sm_lod = get_hod_lod_for_trading_day(small_df, trading_day_start)
    bg_hod, bg_lod = get_hod_lod_for_trading_day(big_df, trading_day_start)
    
    # Calculate zones for each row
    df['sm HOD/LOD'] = df['Output'].apply(
        lambda x: calculate_hod_lod_zone(x, sm_hod, sm_lod, 'sm')
    )
    df['bg HOD/LOD'] = df['Output'].apply(
        lambda x: calculate_hod_lod_zone(x, bg_hod, bg_lod, 'bg')
    )
    
    # Reorder columns to put HOD/LOD before Star and Reason
    cols = df.columns.tolist()
    
    # Find positions
    star_idx = cols.index('Star') if 'Star' in cols else len(cols)
    
    # Remove HOD/LOD from current position
    cols.remove('sm HOD/LOD')
    cols.remove('bg HOD/LOD')
    
    # Insert before Star
    cols.insert(star_idx, 'bg HOD/LOD')
    cols.insert(star_idx, 'sm HOD/LOD')
    
    df = df[cols]
    
    return df

def calculate_swing_turns_column(df, feed_df, trading_day_start, lookback_days=20):
    """
    Calculate the 'Turns' column using Nested Swing Detector.
    
    For each output in the dataframe, determines which swing zones it falls into
    and formats as per specification:
    - For outputs in zone 1 (0-9 points from swing origin): show swing_size - 9
    - For outputs in zone 2 (9-18 points): show swing_size - 18
    - For outputs in zone 3 (18-27 points): show swing_size - 27
    - Positive values for upswings, negative for downswings
    - Multiple swings separated by commas, first swing listed first
    
    Args:
        df: Traveler report dataframe with 'Output' column
        feed_df: Feed dataframe with time/high/low for swing detection
        trading_day_start: datetime of 18:00 on previous day (start of trading day)
        lookback_days: Number of days to look back for swing analysis
    
    Returns:
        DataFrame with added 'Turns' column
    """
    if df is None or df.empty or feed_df is None or feed_df.empty:
        if df is not None:
            df['Turns'] = pd.NA
        return df
    
    df = df.copy()
    
    # Define time range for swing analysis (lookback_days before trading_day_start to end of trading day)
    start_time = trading_day_start - dt.timedelta(days=lookback_days)
    end_time = trading_day_start + dt.timedelta(hours=22, minutes=45)  # Trading day ends at 16:45 next day
    
    try:
        # Run swing analysis
        nested_swings, major_points = analyze_swings(
            feed_df,
            start_time,
            end_time,
            min_swing_size=60,
            pullback_tolerance=30,
            min_time_separation_minutes=90
        )
        
        # For each output, check which swing zones it falls into
        turns_data = []
        
        for output_val in df['Output']:
            if pd.isna(output_val):
                turns_data.append(None)
                continue
            
            swing_labels = []
            
            # Check each swing to see if output falls in any of its zones
            for swing in nested_swings:
                from_price = swing['from_price']
                swing_size = swing['swing_size']
                direction = swing['direction']
                
                # Determine if output is within 27 points of the swing origin
                if direction == 'Up':
                    # Upswing: origin is at bottom, price moves up
                    # Zones: [from_price, from_price+9], [from_price+9, from_price+18], [from_price+18, from_price+27]
                    if from_price <= output_val <= from_price + 27:
                        distance_from_origin = output_val - from_price
                        
                        if distance_from_origin <= 9:
                            # Zone 1: swing_size - 9
                            label_val = int(swing_size - 9)
                            swing_labels.append((swing['from_time'], label_val))
                        elif distance_from_origin <= 18:
                            # Zone 2: swing_size - 18
                            label_val = int(swing_size - 18)
                            swing_labels.append((swing['from_time'], label_val))
                        elif distance_from_origin <= 27:
                            # Zone 3: swing_size - 27
                            label_val = int(swing_size - 27)
                            swing_labels.append((swing['from_time'], label_val))
                
                elif direction == 'Down':
                    # Downswing: origin is at top, price moves down
                    # Zones: [from_price-9, from_price], [from_price-18, from_price-9], [from_price-27, from_price-18]
                    if from_price - 27 <= output_val <= from_price:
                        distance_from_origin = from_price - output_val
                        
                        if distance_from_origin <= 9:
                            # Zone 1: -(swing_size - 9)
                            label_val = -int(swing_size - 9)
                            swing_labels.append((swing['from_time'], label_val))
                        elif distance_from_origin <= 18:
                            # Zone 2: -(swing_size - 18)
                            label_val = -int(swing_size - 18)
                            swing_labels.append((swing['from_time'], label_val))
                        elif distance_from_origin <= 27:
                            # Zone 3: -(swing_size - 27)
                            label_val = -int(swing_size - 27)
                            swing_labels.append((swing['from_time'], label_val))
            
            # Sort by time (first swing first), then format as comma-separated string
            if swing_labels:
                swing_labels.sort(key=lambda x: x[0])  # Sort by time
                turns_str = ", ".join([str(val) for _, val in swing_labels])
                turns_data.append(turns_str)
            else:
                turns_data.append(None)
        
        # Add Turns column to dataframe
        df['Turns'] = turns_data
        
    except Exception as e:
        st.warning(f"Error calculating Turns column: {e}")
        df['Turns'] = pd.NA
    
    # Reorder columns to put Turns after 'bg HOD/LOD' and before 'Star'
    cols = df.columns.tolist()
    
    if 'Turns' in cols:
        cols.remove('Turns')
        
        # Find insertion point (after 'bg HOD/LOD', before 'Star')
        if 'bg HOD/LOD' in cols:
            insert_idx = cols.index('bg HOD/LOD') + 1
        elif 'Star' in cols:
            insert_idx = cols.index('Star')
        else:
            insert_idx = len(cols)
        
        cols.insert(insert_idx, 'Turns')
        df = df[cols]
    
    return df

def apply_excel_highlighting_with_hodlod(workbook, worksheet, df, is_custom_ranges=False):
    """
    Apply highlighting to Excel export including HOD/LOD zones.
    Enhanced version of apply_excel_highlighting with HOD/LOD support.
    """
    
    # Base formats
    header_format = workbook.add_format({
        'bold': True, 'text_wrap': True, 'valign': 'top',
        'fg_color': '#D7E4BC', 'border': 1
    })
    
    # Date formats
    date_fmt = workbook.add_format({'num_format': 'mm/dd/yyyy hh:mm'})
    day_zero_format = workbook.add_format({'fg_color': '#FFFF00'})
    day_zero_date_fmt = workbook.add_format({'fg_color': '#FFFF00', 'num_format': 'mm/dd/yyyy hh:mm'})
    
    # Origin color formats
    spain_saturn_format = workbook.add_format({'fg_color': '#39FF14'})
    jupiter_format = workbook.add_format({'fg_color': '#d1ecf1'})
    kepler_format = workbook.add_format({'fg_color': '#ff4d00'})
    trinidad_tobago_format = workbook.add_format({'fg_color': '#f0cb59'})
    wasp_format = workbook.add_format({'fg_color': '#C0C0C0'})
    macedonia_format = workbook.add_format({'fg_color': '#e022d7'})
    
    # M# formats
    m0_format = workbook.add_format({'fg_color': '#E6E6FA', 'bold': True})
    m40_format = workbook.add_format({'fg_color': '#D3D3D3', 'bold': True})
    m54_format = workbook.add_format({'fg_color': '#ADD8E6', 'bold': True})
    
    # HOD/LOD zone formats
    hod_0to9_format = workbook.add_format({'fg_color': '#FF5050'})      # Red
    hod_09to18_format = workbook.add_format({'fg_color': '#FF9999'})    # Light Red
    hod_18to27_format = workbook.add_format({'fg_color': '#FCD5B4'})    # Orange
    lod_0to9_format = workbook.add_format({'fg_color': '#00B0F0'})      # Blue
    lod_09to18_format = workbook.add_format({'fg_color': '#66FFFF'})    # Sky Blue
    lod_18to27_format = workbook.add_format({'fg_color': '#FCD5B4'})    # Orange
    
    # Header row
    for col_num, value in enumerate(df.columns.values):
        worksheet.write(0, col_num, value, header_format)
    
    # Set Arrival column format
    arrival_col_idx = df.columns.get_loc('Arrival') if 'Arrival' in df.columns else None
    if arrival_col_idx is not None:
        worksheet.set_column(arrival_col_idx, arrival_col_idx, 19, date_fmt)
    
    # Helper: safe datetime conversion
    def _as_pydt(x):
        if pd.isna(x):
            return None
        if isinstance(x, dt.datetime):
            return x
        try:
            return pd.to_datetime(x).to_pydatetime()
        except Exception:
            return None
    
    # Row formatting
    for row_idx, (_, row) in enumerate(df.iterrows(), start=1):
        # Origin highlighting
        if 'Origin' in df.columns:
            origin_col = df.columns.get_loc('Origin')
            origin = str(row.get('Origin', '')).lower()
            origin_fmt = None
            
            if origin in ['spain', 'saturn']:
                origin_fmt = spain_saturn_format
            elif origin == 'jupiter':
                origin_fmt = jupiter_format
            elif origin in ['kepler-62', 'kepler-44']:
                origin_fmt = kepler_format
            elif origin in ['trinidad', 'tobago']:
                origin_fmt = trinidad_tobago_format
            elif 'wasp' in origin:
                origin_fmt = wasp_format
            elif 'macedonia' in origin:
                origin_fmt = macedonia_format
            
            if origin_fmt:
                worksheet.write(row_idx, origin_col, row['Origin'], origin_fmt)
        
        # M# highlighting
        if 'M #' in df.columns:
            m_col = df.columns.get_loc('M #')
            m_val = row.get('M #')
            m_fmt = None
            
            if m_val == 0:
                m_fmt = m0_format
            elif m_val in [40, -40]:
                m_fmt = m40_format
            elif m_val in [54, -54]:
                m_fmt = m54_format
            
            if m_fmt:
                worksheet.write(row_idx, m_col, m_val, m_fmt)
        
        # Day column highlighting (Day[0] = yellow)
        if 'Day' in df.columns:
            day_col = df.columns.get_loc('Day')
            day_val = str(row.get('Day', ''))
            if '[0]' in day_val:
                worksheet.write(row_idx, day_col, row['Day'], day_zero_format)
        
        # Arrival datetime with proper formatting
        if 'Arrival' in df.columns:
            arrival_col = df.columns.get_loc('Arrival')
            arrival_val = row.get('Arrival')
            pydt = _as_pydt(arrival_val)
            
            # Check if Day[0] for yellow background
            day_val = str(row.get('Day', ''))
            if '[0]' in day_val:
                if pydt:
                    worksheet.write_datetime(row_idx, arrival_col, pydt, day_zero_date_fmt)
                else:
                    worksheet.write(row_idx, arrival_col, arrival_val, day_zero_format)
            else:
                if pydt:
                    worksheet.write_datetime(row_idx, arrival_col, pydt, date_fmt)
        
        # sm HOD/LOD zone highlighting
        if 'sm HOD/LOD' in df.columns:
            sm_col = df.columns.get_loc('sm HOD/LOD')
            sm_zone = row.get('sm HOD/LOD')
            sm_fmt = None
            
            if pd.notna(sm_zone):
                if 'HOD 0 to 9' in str(sm_zone):
                    sm_fmt = hod_0to9_format
                elif 'HOD 09 to 18' in str(sm_zone):
                    sm_fmt = hod_09to18_format
                elif 'HOD 18 to 27' in str(sm_zone):
                    sm_fmt = hod_18to27_format
                elif 'LOD 0 to 9' in str(sm_zone):
                    sm_fmt = lod_0to9_format
                elif 'LOD 09 to 18' in str(sm_zone):
                    sm_fmt = lod_09to18_format
                elif 'LOD 18 to 27' in str(sm_zone):
                    sm_fmt = lod_18to27_format
                
                if sm_fmt:
                    worksheet.write(row_idx, sm_col, sm_zone, sm_fmt)
        
        # bg HOD/LOD zone highlighting
        if 'bg HOD/LOD' in df.columns:
            bg_col = df.columns.get_loc('bg HOD/LOD')
            bg_zone = row.get('bg HOD/LOD')
            bg_fmt = None
            
            if pd.notna(bg_zone):
                if 'HOD 0 to 9' in str(bg_zone):
                    bg_fmt = hod_0to9_format
                elif 'HOD 09 to 18' in str(bg_zone):
                    bg_fmt = hod_09to18_format
                elif 'HOD 18 to 27' in str(bg_zone):
                    bg_fmt = hod_18to27_format
                elif 'LOD 0 to 9' in str(bg_zone):
                    bg_fmt = lod_0to9_format
                elif 'LOD 09 to 18' in str(bg_zone):
                    bg_fmt = lod_09to18_format
                elif 'LOD 18 to 27' in str(bg_zone):
                    bg_fmt = lod_18to27_format
                
                if bg_fmt:
                    worksheet.write(row_idx, bg_col, bg_zone, bg_fmt)


def create_multi_day_excel(all_day_reports, asset_id, trading_day_base, window_radius, lookback_days):
    """
    Create Excel file with multiple days of reports with highlighting.
    Each day has 4 tabs (1800, 0315, 0900, 1230).
    Uses xlsxwriter for highlighting support.
    
    Args:
        all_day_reports: List of tuples (tab_name, dataframe)
        asset_id: Asset identifier (e.g., 'nq', 'es')
        trading_day_base: The first trading day for filename
        window_radius: Window radius value
        lookback_days: Number of look back days
    
    Returns:
        BytesIO buffer containing the Excel file
    """
    excel_buffer = io.BytesIO()
    
    with pd.ExcelWriter(excel_buffer, engine='xlsxwriter', datetime_format='mm/dd/yyyy hh:mm') as writer:
        workbook = writer.book
        
        for tab_name, df in all_day_reports:
            if df is None or df.empty:
                continue
            
            # Write dataframe to sheet
            df.to_excel(writer, sheet_name=tab_name, index=False)
            worksheet = writer.sheets[tab_name]
            
            # Apply highlighting
            apply_excel_highlighting_with_hodlod(workbook, worksheet, df, is_custom_ranges=False)
            
            # Auto-filter on headers
            if len(df) > 0:
                worksheet.autofilter(0, 0, len(df), len(df.columns) - 1)
            
            # Freeze top row
            worksheet.freeze_panes(1, 0)
    
    excel_buffer.seek(0)
    return excel_buffer.getvalue()


# === Unified Export Helper ===
def render_unified_export(traveler_reports, report_time, asset_id="", window_radius=None, lookback_days=None):
    """
    Unified export for traveler reports with optional window radius for single-tab export.
    
    Args:
        traveler_reports: Dict of report dataframes
        report_time: Report datetime
        asset_id: Asset ID
        window_radius: Optional window radius (for Full Range single mode)
        lookback_days: Optional lookback days value
    """
    if not traveler_reports:
        return

    st.markdown("---")
    st.markdown("### 📥 Unified Excel Download")

    # Add asset_id to filename if provided
    asset_prefix = f"{asset_id.lower()}_" if asset_id else ""
    report_datetime_str = report_time.strftime("%d-%b-%y")

    # If window_radius is provided, export as single tab with updated naming
    if window_radius is not None:
        # Combine all reports into single dataframe
        combined_df = pd.concat([df for df in traveler_reports.values() if not df.empty], ignore_index=True)
        
        # Sort by Output (desc) then Arrival (asc)
        if not combined_df.empty:
            if 'Output' in combined_df.columns and 'Arrival' in combined_df.columns:
                combined_df = combined_df.sort_values(['Output', 'Arrival'], ascending=[False, True])
            elif 'Output' in combined_df.columns:
                combined_df = combined_df.sort_values(['Output'], ascending=[False])
        
        # Create single-tab Excel export
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine="xlsxwriter", datetime_format="mm/dd/yyyy hh:mm") as writer:
            workbook = writer.book
            
            # Write to single sheet
            sheet_name = "Full_Range"
            combined_df.to_excel(writer, sheet_name=sheet_name, index=False)
            worksheet = writer.sheets[sheet_name]
            
            # Apply highlighting
            apply_excel_highlighting_with_hodlod(workbook, worksheet, combined_df, is_custom_ranges=False)
            
            # Auto-filter and freeze panes
            if len(combined_df) > 0:
                worksheet.autofilter(0, 0, len(combined_df), len(combined_df.columns) - 1)
            worksheet.freeze_panes(1, 0)
        
        excel_buffer.seek(0)
        
        # Build filename with new naming convention
        # Format: asset + RawTrav for + date + (lookback days) + radius + window
        lookback_str = f"({lookback_days}days)" if lookback_days else ""
        filename = f"{asset_prefix}RawTrav_for_{report_datetime_str}_{lookback_str}_{int(window_radius)}_radius_window.xlsx"
        
        st.download_button(
            "📥 Download Full Range Report",
            data=excel_buffer.getvalue(),
            file_name=filename,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            help=f"Single tab Excel file with {len(combined_df)} entries"
        )
    else:
        # Original multi-sheet export
        def _coerce_arrival_datetime(df: pd.DataFrame) -> pd.DataFrame:
            df = df.copy()
            if "Arrival_datetime" in df.columns:
                df["Arrival"] = pd.to_datetime(df["Arrival_datetime"], errors="coerce")
            elif "Arrival" in df.columns:
                df["Arrival"] = pd.to_datetime(df["Arrival"], errors="coerce", infer_datetime_format=True)
            return df

        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine="xlsxwriter", datetime_format="mm/dd/yyyy hh:mm") as writer:
            workbook = writer.book
            header_fmt = workbook.add_format({
                "bold": True, "text_wrap": True, "valign": "top",
                "fg_color": "#D7E4BC", "border": 1
            })
            date_fmt = workbook.add_format({"num_format": "mm/dd/yyyy hh:mm"})

            for group_name, group_data in traveler_reports.items():
                if not isinstance(group_data, pd.DataFrame) or group_data.empty:
                    continue

                sheet_name = group_name.replace(" ", "_").replace("-", "_")[:31]
                export_data = group_data.drop(columns=["Group"], errors="ignore").copy()
                export_data = _coerce_arrival_datetime(export_data)

                export_data.to_excel(writer, sheet_name=sheet_name, index=False)
                ws = writer.sheets[sheet_name]

                # headers
                for c, name in enumerate(export_data.columns):
                    ws.write(0, c, name, header_fmt)

                # make Arrival display as a date in Excel
                if "Arrival" in export_data.columns:
                    a_idx = export_data.columns.get_loc("Arrival")
                    ws.set_column(a_idx, a_idx, 18, date_fmt)

                # conditional coloring
                try:
                    apply_excel_highlighting(workbook, ws, export_data, False)
                except Exception as e:
                    st.warning(f"Highlighting skipped for '{sheet_name}': {e}")

        excel_buffer.seek(0)
        total_entries = sum(len(df) for df in traveler_reports.values() if isinstance(df, pd.DataFrame))
        num_groups = sum(1 for v in traveler_reports.values() if isinstance(v, pd.DataFrame) and not v.empty)

        st.download_button(
            "📥 Download Excel Report",
            data=excel_buffer.getvalue(),
            file_name=f"{asset_prefix}traveler_report_{report_datetime_str}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            help=f"Excel file contains {num_groups} sheets with {total_entries} total entries"
        )


# Streamlit interface
st.set_page_config(layout="wide")
st.header("🧬 Traveler Report Generator v34a | Origin Filtering + Nested Swing Detector")

# Asset ID Selector (above file uploads)
st.markdown("### Asset Selection")
asset_id = st.selectbox(
    "Select Asset ID",
    options=["NQ", "ES", "YM", "RTY"],
    index=0,
    help="Select the asset/instrument for this analysis"
)
st.info(f"Selected Asset: **{asset_id}**")

# File uploads - 7 files in 3 rows
st.markdown("---")
st.markdown("### File Uploads")

# Row 1: Small feeds (3m, 5m, 15m)
col1, col2, col3 = st.columns(3)
with col1:
    small_3m_file = st.file_uploader("Upload 3m small", type="csv", key="small_3m")
with col2:
    small_5m_file = st.file_uploader("Upload 5m small", type="csv", key="small_5m")
with col3:
    small_15m_file = st.file_uploader("Upload 15m small", type="csv", key="small_15m")

# Row 2: Big feeds (3m, 5m, 15m)
col4, col5, col6 = st.columns(3)
with col4:
    big_3m_file = st.file_uploader("Upload 3m big", type="csv", key="big_3m")
with col5:
    big_5m_file = st.file_uploader("Upload 5m big", type="csv", key="big_5m")
with col6:
    big_15m_file = st.file_uploader("Upload 15m big", type="csv", key="big_15m")

# Row 3: Measurement file + Bypass option
col7, col8, col9 = st.columns(3)
with col7:
    measurement_file = st.file_uploader("Upload measurement file", type="xlsx", key="measurement")
with col8:
    bypass_traveler_file = st.file_uploader("Upload Final Traveler Report (Bypass)", type=["xlsx", "csv"], key="bypass",
        help="Optional: Upload a pre-generated traveler report to skip data processing")

# === Origin Filtering Section ===
st.markdown("---")
st.markdown("### 🌍 Origin Filtering")

# Extract all origins from column names (following swing tool pattern)
# Origins are embedded in column names like "Spain H", "Saturn L", "Trinidad C"
all_origins = set()
files_checked = 0
files_with_origins = 0

for file_obj, name in [(small_3m_file, '3m small'), (small_5m_file, '5m small'), (small_15m_file, '15m small'), 
                        (big_3m_file, '3m big'), (big_5m_file, '5m big'), (big_15m_file, '15m big')]:
    if file_obj is not None:
        files_checked += 1
        try:
            temp_df = pd.read_csv(file_obj)
            
            # Extract origins from column names ending with ' H' (High columns)
            # Example: "Spain H" -> "Spain", "WASP-12b H" -> "WASP-12b"
            origins_in_file = [col[:-2] for col in temp_df.columns if col.endswith(' H')]
            
            if len(origins_in_file) > 0:
                all_origins.update(origins_in_file)
                files_with_origins += 1
            
            file_obj.seek(0)  # Reset file pointer
        except Exception as e:
            st.warning(f"Could not read origins from {name}: {e}")

# Categorize origins (case-insensitive matching)
EPIC_NAMES = {name.lower() for name in EPIC_ORIGINS}
ANCHOR_NAMES = {name.lower() for name in ANCHOR_ORIGINS}

epic_origins = [o for o in all_origins if o.lower() in EPIC_NAMES]
anchor_origins = [o for o in all_origins if o.lower() in ANCHOR_NAMES]
other_origins = [o for o in all_origins if o.lower() not in EPIC_NAMES and o.lower() not in ANCHOR_NAMES]

# Sort alphabetically
epic_origins.sort()
anchor_origins.sort()
other_origins.sort()

if all_origins:
    st.success(f"✅ Detected {len(all_origins)} origins from {files_with_origins} file(s): {len(epic_origins)} Epic, {len(anchor_origins)} Anchor, {len(other_origins)} Other")
    
    # Debug expander to show detected origins
    with st.expander("🔍 Show detected origins", expanded=False):
        if epic_origins:
            st.write(f"**Epic Origins ({len(epic_origins)}):** {', '.join(epic_origins)}")
        if anchor_origins:
            st.write(f"**Anchor Origins ({len(anchor_origins)}):** {', '.join(anchor_origins)}")
        if other_origins:
            st.write(f"**Other Origins ({len(other_origins)}):** {', '.join(other_origins)}")
            
elif files_checked > 0:
    st.warning(f"⚠️ Checked {files_checked} file(s) but found no columns ending with ' H' (High columns). Origin filtering will be disabled.")
    
    # Debug expander
    with st.expander("🔍 Troubleshooting - Show column extraction", expanded=False):
        st.write("Looking for column names ending with ' H' to extract origin names...")
        st.write("Example: 'Spain H' -> 'Spain', 'Saturn H' -> 'Saturn'")
        st.write("")
        for file_obj, name in [(small_3m_file, '3m small'), (small_5m_file, '5m small'), (small_15m_file, '15m small'), 
                                (big_3m_file, '3m big'), (big_5m_file, '5m big'), (big_15m_file, '15m big')]:
            if file_obj is not None:
                try:
                    temp_df = pd.read_csv(file_obj)
                    h_cols = [col for col in temp_df.columns if col.endswith(' H')]
                    if h_cols:
                        extracted = [col[:-2] for col in h_cols]
                        st.write(f"**{name}:** Found {len(h_cols)} High columns")
                        st.write(f"  Extracted origins: {', '.join(extracted[:10])}" + (" ..." if len(extracted) > 10 else ""))
                    else:
                        st.write(f"**{name}:** No columns ending with ' H' found")
                        st.write(f"  Total columns: {len(temp_df.columns)}")
                    file_obj.seek(0)
                except Exception as e:
                    st.write(f"**{name}:** Error reading file: {e}")
else:
    st.info("Upload feed files to enable origin filtering")

col_filter1, col_filter2 = st.columns([2, 1])
with col_filter1:
    filter_origins = st.checkbox(
        "Filter Origins (Faster processing, smaller results)", 
        value=True,  # Changed from False to True - now enabled by default
        key="filter_origins_main",
        disabled=(len(all_origins) == 0)
    )
    if len(all_origins) > 0:
        st.caption("Process only selected origins. All priority origins (Epic + Anchor) selected by default.")
    else:
        st.caption("Origin filtering unavailable - no origins detected in uploaded files.")

allowed_origins = None
if filter_origins and all_origins:
    # Epic Origins
    if epic_origins:
        st.markdown("**Epic Origins (Priority):**")
        epic_selections = {}
        cols = st.columns(min(len(epic_origins), 4))
        for idx, origin in enumerate(epic_origins):
            with cols[idx % len(cols)]:
                epic_selections[origin] = st.checkbox(origin, value=True, key=f"epic_main_{origin}")
    
    # Anchor Origins
    if anchor_origins:
        st.markdown("**Anchor Origins (Priority):**")
        anchor_selections = {}
        cols = st.columns(min(len(anchor_origins), 5))
        for idx, origin in enumerate(anchor_origins):
            with cols[idx % len(cols)]:
                anchor_selections[origin] = st.checkbox(origin, value=True, key=f"anchor_main_{origin}")
    
    # Other Origins
    if other_origins:
        st.markdown("**Other Origins (Optional):**")
        other_selections = {}
        cols = st.columns(min(len(other_origins), 4))
        for idx, origin in enumerate(other_origins):
            with cols[idx % len(cols)]:
                other_selections[origin] = st.checkbox(origin, value=False, key=f"other_main_{origin}")
    
    # Build allowed origins set (use exact names from CSV, case-sensitive)
    allowed_origins = set()
    if epic_origins:
        allowed_origins.update([o for o, selected in epic_selections.items() if selected])
    if anchor_origins:
        allowed_origins.update([o for o, selected in anchor_selections.items() if selected])
    if other_origins:
        allowed_origins.update([o for o, selected in other_selections.items() if selected])
    
    epic_count = sum(1 for o, s in (epic_selections.items() if epic_origins else []) if s)
    anchor_count = sum(1 for o, s in (anchor_selections.items() if anchor_origins else []) if s)
    other_count = sum(1 for o, s in (other_selections.items() if other_origins else []) if s)
    
    st.info(f"Processing {len(allowed_origins)} origins ({epic_count} Epic + {anchor_count} Anchor + {other_count} Other)")
elif filter_origins and not all_origins:
    st.warning("Origin filtering enabled but no origins detected. Processing will continue with all data.")
else:
    if all_origins:
        st.info(f"Processing ALL {len(all_origins)} origins (may take longer with large results)")
    else:
        st.info("Processing all data (no origin filtering available)")

st.markdown("---")

# === SIDEBAR SETTINGS ===
st.sidebar.title("⚙️ Settings & Configuration")

# Model Detection Toggles
st.sidebar.markdown("### 🔬 Model Detection")
run_g_models = st.sidebar.checkbox("🟢 Run Model G Detection", value=False)

if run_g_models:
    st.sidebar.markdown("**G Model Sub-detections:**")
    run_g05_g06 = st.sidebar.checkbox("   • G.05 & G.06", value=True)
    run_g08 = st.sidebar.checkbox("   • G.08", value=True)
    run_g09 = st.sidebar.checkbox("   • G.09", value=True)
    run_g10 = st.sidebar.checkbox("   • G.10", value=False)
    run_g11 = st.sidebar.checkbox("   • G.11", value=False)
    
    st.sidebar.markdown("**Global Output Spread Filter:**")
    output_spread_filter = st.sidebar.number_input(
        "Output Spread (pts)",
        value=3.0,
        min_value=0.0,
        step=0.5,
        help="Maximum point spread between outputs in a group"
    )
    
    show_rejected_groups = st.sidebar.checkbox(
        "Show Rejected Groups",
        value=False,
        help="Display groups that failed output spread filter"
    )
    
    if run_g10:
        st.sidebar.markdown("**G.10 Groups:**")
        g10_group_0 = st.sidebar.checkbox("   • Group 0", value=True)
        g10_group_1 = st.sidebar.checkbox("   • Group 1", value=True)
        g10_group_2 = st.sidebar.checkbox("   • Group 2", value=True)
        g10_group_3 = st.sidebar.checkbox("   • Group 3", value=True)
        g10_group_4 = st.sidebar.checkbox("   • Group 4", value=True)
    else:
        g10_group_0 = g10_group_1 = g10_group_2 = g10_group_3 = g10_group_4 = False
    
    if run_g11:
        st.sidebar.markdown("**G.11 Groups:**")
        g11_group_0 = st.sidebar.checkbox("   • Group 0 (same feed)", value=True, key="g11_g0")
        g11_group_1 = st.sidebar.checkbox("   • Group 1 (same feed)", value=True, key="g11_g1")
        g11_group_2 = st.sidebar.checkbox("   • Group 2 (same feed)", value=True, key="g11_g2")
        g11_group_3 = st.sidebar.checkbox("   • Group 3 (same feed)", value=True, key="g11_g3")
        g11_group_4 = st.sidebar.checkbox("   • Group 4 (same feed)", value=True, key="g11_g4")
        
        st.sidebar.markdown("**G.11 Display Options:**")
        g11_display_recipes = st.sidebar.checkbox("Display G.11 Recipe Groups", value=True)
        g11_display_others = st.sidebar.checkbox("Display G.11 Other Groups", value=True)
    else:
        g11_group_0 = g11_group_1 = g11_group_2 = g11_group_3 = g11_group_4 = False
        g11_display_recipes = g11_display_others = False
else:
    run_g05_g06 = run_g08 = run_g09 = run_g10 = run_g11 = False
    output_spread_filter = 3.0
    show_rejected_groups = False
    g10_group_0 = g10_group_1 = g10_group_2 = g10_group_3 = g10_group_4 = False
    g11_group_0 = g11_group_1 = g11_group_2 = g11_group_3 = g11_group_4 = False
    g11_display_recipes = g11_display_others = False

run_a_models = st.sidebar.checkbox("🔵 Run Model A Detection", value=False)
run_b_models = st.sidebar.checkbox("🟡 Run Model B Detection", value=False)

run_c_models = st.sidebar.checkbox("🟠 Run Model C Detection", value=False)
if run_c_models:
    st.sidebar.markdown("**C Model Sub-detections:**")
    run_c01 = st.sidebar.checkbox("   • C.01", value=True)
    run_c02 = st.sidebar.checkbox("   • C.02", value=True)
    run_c04 = st.sidebar.checkbox("   • C.04", value=True)
else:
    run_c01 = run_c02 = run_c04 = False

run_x_models = st.sidebar.checkbox("🟣 Run Model X Detection", value=False)
run_single_line = st.sidebar.checkbox("📊 Simple Single Line Mega Report", value=False)

# Performance toggles
fast_mode = st.sidebar.checkbox("⚡ Fast Mode (skip dataframe displays)", value=False)
minimal_display = st.sidebar.checkbox("📉 Minimal Display (summaries only)", value=False)
filter_future_data = st.sidebar.checkbox("🔮 Filter Future Data", value=False, 
    help="Remove entries with arrival times after report time")

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Processing Mode")
processing_mode = st.sidebar.radio(
    "Select Processing Mode",
    ["Full Range (single)", "Full Range (multiple)", "Custom Ranges", "HOD/LOD Mode"],
    help="Choose how to process the data"
)

# Full Range (single) options
if processing_mode == "Full Range (single)":
    st.sidebar.markdown("**Full Range (single) Settings:**")
    window_radius = st.sidebar.number_input("Window Radius", value=600.0, min_value=1.0, step=1.0,
        help="Full range window radius around center value")
    input_value_at_start = st.sidebar.number_input("Input Value @ Start (optional)", value=0.0,
        help="Leave at 0 to auto-calculate from feed data")
    run_g_on_full = st.sidebar.checkbox("Run Model G on Full Range", value=False)

# Full Range (multiple) options
if processing_mode == "Full Range (multiple)":
    st.sidebar.markdown("**Full Range (multiple) Settings:**")
    st.sidebar.info("Processes up to 30 days with 4 grab times per day (1800, 0315, 0900, 1230)")
    
    num_days_multi = st.sidebar.number_input(
        "Number of Days", 
        value=1, 
        min_value=1, 
        max_value=30,
        help="Number of trading days to process"
    )
    
    multi_start_date = st.sidebar.date_input(
        "Starting Date (1800 open)", 
        value=dt.date.today() - dt.timedelta(days=1),
        help="The date of the first 1800 open (e.g., 7-Dec for 8-Dec trading day)"
    )
    
    window_radius_multi = st.sidebar.number_input(
        "Window Radius", 
        value=600.0, 
        min_value=1.0, 
        step=1.0,
        help="Full range window radius around center value"
    )
    
    input_value_at_start_multi = st.sidebar.number_input(
        "Input Value @ Start (optional)", 
        value=0.0,
        help="Leave at 0 to auto-calculate from feed data"
    )

# Custom Ranges options
if processing_mode == "Custom Ranges":
    st.sidebar.markdown("**Custom Range Settings:**")
    st.sidebar.info("Define up to 4 custom ranges (2 High, 2 Low)")
    
    use_high1 = st.sidebar.checkbox("Enable High Range 1", value=False)
    high1 = st.sidebar.number_input("High Range 1 Center", value=0.0, step=0.1,
        help="Range will be [value-24, value]") if use_high1 else 0.0
    
    use_high2 = st.sidebar.checkbox("Enable High Range 2", value=False)
    high2 = st.sidebar.number_input("High Range 2 Center", value=0.0, step=0.1,
        help="Range will be [value-24, value]") if use_high2 else 0.0
    
    use_low1 = st.sidebar.checkbox("Enable Low Range 1", value=False)
    low1 = st.sidebar.number_input("Low Range 1 Center", value=0.0, step=0.1,
        help="Range will be [value, value+24]") if use_low1 else 0.0
    
    use_low2 = st.sidebar.checkbox("Enable Low Range 2", value=False)
    low2 = st.sidebar.number_input("Low Range 2 Center", value=0.0, step=0.1,
        help="Range will be [value, value+24]") if use_low2 else 0.0
    
    run_g_on_custom = st.sidebar.checkbox("Run Model G on Custom Ranges", value=False)

# HOD/LOD options
if processing_mode == "HOD/LOD Mode":
    st.sidebar.markdown("**HOD/LOD Settings:**")
    hod_lod_num_days = st.sidebar.number_input("Number of Days to Analyze", value=1, min_value=1, max_value=30,
        help="Number of complete trading days to analyze")
    include_partial_day = st.sidebar.checkbox("Include Partial Current Day", value=True,
        help="Include today's session even if incomplete")
    run_g_on_hod_lod = st.sidebar.checkbox("Run Model G on HOD/LOD", value=False)

# Processing Optimization
st.sidebar.markdown("---")
st.sidebar.markdown("### ⚡ Processing Optimization")
separate_timeframes = st.sidebar.checkbox(
    "Process Timeframes Separately",
    value=True,
    help="Process each timeframe independently for better performance (recommended for Custom Ranges and Full Range)"
)

# Day start time configuration + Look Back Days
st.sidebar.markdown("---")
st.sidebar.markdown("### 🕐 Day Start Configuration")
day_start_hour = st.sidebar.slider("Day Start Hour", 0, 23, 18, help="Hour when trading day starts (default: 18 = 6 PM)")
lookback_days = st.sidebar.number_input("Look Back Days", value=20, min_value=1, max_value=365,
    help="Look back period for swing analysis and data processing")

# === Report Time Selection (Main Area) ===
# Only show for single-report modes (not Full Range (multiple))
st.markdown("---")
if processing_mode != "Full Range (multiple)":
    st.markdown("### 📅 Report Time & Date")
    report_mode = st.radio("Select Report Time & Date", ["Most Current", "Choose a time"])
    if report_mode == "Choose a time":
        selected_date = st.date_input("Select Report Date", value=dt.date.today())
        selected_time = st.time_input("Select Report Time", value=dt.time(18, 0))
        report_time = dt.datetime.combine(selected_date, selected_time)
    else:
        # Use current time as default for "Most Current" mode
        report_time = dt.datetime.now()
else:
    # For Full Range (multiple), datetime is set in sidebar - use placeholder here
    st.info("ℹ️ **Full Range (multiple) mode:** Date range configured in sidebar settings above. No need to select a datetime here.")
    report_time = dt.datetime.now()  # Placeholder, not used in multi mode

st.markdown("---")

# Run button
if st.button("🚀 Process Data"):
    # Check required files
    if not bypass_traveler_file:
        has_small = any([small_3m_file, small_5m_file, small_15m_file])
        has_big = any([big_3m_file, big_5m_file, big_15m_file])
        has_measurement = measurement_file is not None
        
        if not (has_small or has_big) or not has_measurement:
            st.error("⚠️ Please upload at least one feed file (small or big) and a measurement file, or upload a Final Traveler Report")
            st.stop()
    
    # === Load bypass traveler file if provided ===
    bypass_df = None
    if bypass_traveler_file:
        try:
            if bypass_traveler_file.name.endswith('.csv'):
                bypass_df = pd.read_csv(bypass_traveler_file)
            else:
                bypass_df = pd.read_excel(bypass_traveler_file)
            st.success(f"✅ Loaded bypass traveler report with {len(bypass_df)} entries")
        except Exception as e:
            st.error(f"❌ Error loading bypass file: {e}")
            st.stop()
    
    # === Load and process feeds ===
    small_feeds_dict = {}
    big_feeds_dict = {}
    
    if not bypass_traveler_file:
        # Process small feeds
        for file, timeframe in [(small_3m_file, '3m'), (small_5m_file, '5m'), (small_15m_file, '15m')]:
            if file:
                try:
                    df = pd.read_csv(file)
                    
                    # Apply origin filtering by keeping only columns for selected origins
                    if allowed_origins is not None:
                        # Identify base columns (non-origin columns) to always keep
                        base_cols = ['time', 'open', 'high', 'low', 'close', 'Volume']
                        base_cols += [col for col in df.columns if col.startswith('RSI-')]
                        
                        # Identify origin columns to keep (those belonging to allowed origins)
                        origin_cols = []
                        for origin in allowed_origins:
                            # Keep columns like "Spain H", "Spain L", "Spain C" if "Spain" is allowed
                            for suffix in [' H', ' L', ' C']:
                                col_name = origin + suffix
                                if col_name in df.columns:
                                    origin_cols.append(col_name)
                            # Also check for bracket variations like "WASP-12b H[1]"
                            for col in df.columns:
                                if col.startswith(origin + ' ') and col[len(origin)+1] in ['H', 'L', 'C']:
                                    if col not in origin_cols:  # Avoid duplicates
                                        origin_cols.append(col)
                        
                        # Keep only base columns and selected origin columns
                        cols_to_keep = [col for col in df.columns if col in base_cols or col in origin_cols]
                        df = df[cols_to_keep].copy()
                        
                        if len(origin_cols) == 0:
                            st.warning(f"⚠️ No origin columns found for selected origins in {timeframe} small feed")
                            continue
                        else:
                            st.info(f"📊 {timeframe} small: Keeping {len(origin_cols)} origin columns for {len(allowed_origins)} origins")
                    
                    # Simple processing like original app - just clean timestamps
                    df['time'] = pd.to_datetime(df['time'].str.replace(r'[-+]\d{2}:\d{2}$', '', regex=True))
                    small_feeds_dict[timeframe] = df
                    st.success(f"✅ Loaded {timeframe} small feed ({len(df)} rows)")
                except Exception as e:
                    st.error(f"❌ Error loading {timeframe} small: {e}")
        
        # Process big feeds
        for file, timeframe in [(big_3m_file, '3m'), (big_5m_file, '5m'), (big_15m_file, '15m')]:
            if file:
                try:
                    df = pd.read_csv(file)
                    
                    # Apply origin filtering by keeping only columns for selected origins
                    if allowed_origins is not None:
                        # Identify base columns (non-origin columns) to always keep
                        base_cols = ['time', 'open', 'high', 'low', 'close', 'Volume']
                        base_cols += [col for col in df.columns if col.startswith('RSI-')]
                        
                        # Identify origin columns to keep (those belonging to allowed origins)
                        origin_cols = []
                        for origin in allowed_origins:
                            # Keep columns like "Spain H", "Spain L", "Spain C" if "Spain" is allowed
                            for suffix in [' H', ' L', ' C']:
                                col_name = origin + suffix
                                if col_name in df.columns:
                                    origin_cols.append(col_name)
                            # Also check for bracket variations like "WASP-12b H[1]"
                            for col in df.columns:
                                if col.startswith(origin + ' ') and col[len(origin)+1] in ['H', 'L', 'C']:
                                    if col not in origin_cols:  # Avoid duplicates
                                        origin_cols.append(col)
                        
                        # Keep only base columns and selected origin columns
                        cols_to_keep = [col for col in df.columns if col in base_cols or col in origin_cols]
                        df = df[cols_to_keep].copy()
                        
                        if len(origin_cols) == 0:
                            st.warning(f"⚠️ No origin columns found for selected origins in {timeframe} big feed")
                            continue
                        else:
                            st.info(f"📊 {timeframe} big: Keeping {len(origin_cols)} origin columns for {len(allowed_origins)} origins")
                    
                    # Simple processing like original app - just clean timestamps
                    df['time'] = pd.to_datetime(df['time'].str.replace(r'[-+]\d{2}:\d{2}$', '', regex=True))
                    big_feeds_dict[timeframe] = df
                    st.success(f"✅ Loaded {timeframe} big feed ({len(df)} rows)")
                except Exception as e:
                    st.error(f"❌ Error loading {timeframe} big: {e}")
        
        # Combine feeds if not processing separately
        if not separate_timeframes:
            if small_feeds_dict:
                small_df = pd.concat(small_feeds_dict.values(), ignore_index=True)
                st.success(f"✅ Combined small feeds: {len(small_df)} rows")
            else:
                small_df = None
            
            if big_feeds_dict:
                big_df = pd.concat(big_feeds_dict.values(), ignore_index=True)
                st.success(f"✅ Combined big feeds: {len(big_df)} rows")
            else:
                big_df = None
        else:
            small_df = None
            big_df = None
        
        # Load measurement file
        try:
            measurements_df = pd.read_excel(measurement_file)
            st.success(f"✅ Loaded measurement file with {len(measurements_df)} rows")
        except Exception as e:
            st.error(f"❌ Error loading measurement file: {e}")
            st.stop()
    
    # === PROCESSING MODE LOGIC ===
    
    use_full_range_single = (processing_mode == "Full Range (single)")
    use_full_range_multi = (processing_mode == "Full Range (multiple)")
    use_custom_ranges = (processing_mode == "Custom Ranges")
    use_hod_lod = (processing_mode == "HOD/LOD Mode")
    
    traveler_reports = {}
    
    if bypass_df is not None:
        st.markdown("---")
        st.markdown("### 📊 Bypass Mode: Using Uploaded Traveler Report")
        
        # Split into groups for display
        traveler_reports["Grp 1a"] = bypass_df[bypass_df['M #'].isin(GROUP_1A_TRAVELERS)].copy()
        traveler_reports["Grp 1b"] = bypass_df[bypass_df['M #'].isin(GROUP_1B_TRAVELERS)].copy()
        traveler_reports["Grp 2a"] = bypass_df[bypass_df['M #'].isin(GROUP_2A_TRAVELERS)].copy()
        traveler_reports["Grp 2b"] = bypass_df[bypass_df['M #'].isin(GROUP_2B_TRAVELERS)].copy()
        
        for gname, gdf in traveler_reports.items():
            if not gdf.empty:
                st.markdown(f"#### {gname}")
                st.info(f"{len(gdf)} entries")
                if not fast_mode:
                    st.dataframe(gdf, use_container_width=True)
    
    else:
        # ---------- 1) FULL RANGE (SINGLE) MODE ----------
        if use_full_range_single:
            st.markdown("---")
            st.markdown("### Full Range (single) Processing Mode")
            
            # Determine input value
            input_val = input_value_at_start if input_value_at_start != 0.0 else None
            
            if separate_timeframes and (small_feeds_dict or big_feeds_dict):
                # Process each timeframe separately and combine results
                st.info("Processing timeframes separately for optimal performance...")
                all_timeframe_results = []
                
                timeframes = sorted(set(list(small_feeds_dict.keys()) + list(big_feeds_dict.keys())))
                
                for tf in timeframes:
                    st.text(f"Processing {tf} timeframe...")
                    tf_small = small_feeds_dict.get(tf, pd.DataFrame())
                    tf_big = big_feeds_dict.get(tf, pd.DataFrame())
                    
                    if tf_small.empty and tf_big.empty:
                        continue
                    
                    # Process this timeframe
                    tf_result = apply_full_range_advanced(
                        measurements_df,
                        tf_small if not tf_small.empty else pd.DataFrame(),
                        report_time,
                        window_radius,
                        day_start_hour=day_start_hour,
                        input_value_at_start=input_val,
                        big_df=tf_big if not tf_big.empty else pd.DataFrame(),
                        run_model_g=False  # Run G models only on combined data
                    )
                    
                    if tf_result is not None and not tf_result.empty:
                        all_timeframe_results.append(tf_result)
                
                # Combine all timeframe results
                if all_timeframe_results:
                    final_df_filtered = pd.concat(all_timeframe_results, ignore_index=True)
                else:
                    final_df_filtered = pd.DataFrame()
            else:
                # Combined processing (original logic)
                if small_df is None and big_df is None:
                    st.error("No feed data available for combined processing. Please disable 'Process Timeframes Separately' or ensure files are uploaded.")
                    final_df_filtered = pd.DataFrame()
                else:
                    final_df_filtered = apply_full_range_advanced(
                        measurements_df,
                        small_df if small_df is not None else pd.DataFrame(),
                        report_time,
                        window_radius,
                        day_start_hour=day_start_hour,
                        input_value_at_start=input_val,
                        big_df=big_df if big_df is not None else pd.DataFrame(),
                        run_model_g=run_g_on_full
                    )
            
            if final_df_filtered is None or final_df_filtered.empty:
                st.warning("No entries found in full range processing")
                traveler_reports = {}
            else:
                # Calculate trading_day_start for HOD/LOD
                # If report_time is 1800, it's the trading day start
                # Otherwise, find the 1800 on the previous day
                if report_time.hour == 18 and report_time.minute == 0:
                    trading_day_start = report_time
                else:
                    # Find the 1800 on the day that starts this trading day
                    # If time is >= 1800, trading day starts today at 1800
                    # If time < 1800, trading day started yesterday at 1800
                    if report_time.hour >= 18:
                        trading_day_start = report_time.replace(hour=18, minute=0, second=0, microsecond=0)
                    else:
                        trading_day_start = (report_time - dt.timedelta(days=1)).replace(hour=18, minute=0, second=0, microsecond=0)
                
                # Get feeds for HOD/LOD calculation
                sm_feed_for_hod = None
                bg_feed_for_hod = None
                
                # Select first available non-empty small feed
                if small_feeds_dict:
                    for tf in ['15m', '5m', '3m']:
                        df = small_feeds_dict.get(tf)
                        if df is not None and not df.empty:
                            sm_feed_for_hod = df
                            break
                elif small_df is not None and not small_df.empty:
                    sm_feed_for_hod = small_df
                
                # Select first available non-empty big feed
                if big_feeds_dict:
                    for tf in ['15m', '5m', '3m']:
                        df = big_feeds_dict.get(tf)
                        if df is not None and not df.empty:
                            bg_feed_for_hod = df
                            break
                elif big_df is not None and not big_df.empty:
                    bg_feed_for_hod = big_df
                
                # Add HOD/LOD columns
                final_df_filtered = add_hod_lod_columns(final_df_filtered, sm_feed_for_hod, bg_feed_for_hod, trading_day_start)
                
                # Add Turns column using Nested Swing Detector
                # Use first available feed (prefer 15m if available)
                feed_for_swings = None
                if small_feeds_dict:
                    for tf in ['15m', '5m', '3m']:
                        df = small_feeds_dict.get(tf)
                        if df is not None and not df.empty:
                            feed_for_swings = df
                            break
                elif sm_feed_for_hod is not None:
                    feed_for_swings = sm_feed_for_hod
                
                if feed_for_swings is not None:
                    final_df_filtered = calculate_swing_turns_column(
                        final_df_filtered,
                        feed_for_swings,
                        trading_day_start,
                        lookback_days=lookback_days
                    )
                
                # Sort by Output (desc) then Arrival (asc)
                if not final_df_filtered.empty:
                    sort_cols = []
                    if 'Output' in final_df_filtered.columns:
                        sort_cols.append('Output')
                    if 'Arrival' in final_df_filtered.columns:
                        sort_cols.append('Arrival')
                    
                    if sort_cols:
                        ascending_list = [False if col == 'Output' else True for col in sort_cols]
                        final_df_filtered = final_df_filtered.sort_values(sort_cols, ascending=ascending_list)
                
                # Assign group labels
                final_df_filtered['Group'] = final_df_filtered['M #'].apply(
                    lambda m_num: 
                        '1a' if m_num in GROUP_1A_TRAVELERS else
                        '1b' if m_num in GROUP_1B_TRAVELERS else
                        '2a' if m_num in GROUP_2A_TRAVELERS else
                        '2b' if m_num in GROUP_2B_TRAVELERS else
                        'Other'
                )
                
                # Create single combined report
                traveler_reports = {"Full Range": final_df_filtered}
                
                if not minimal_display:
                    st.markdown("#### Full Range Report")
                    st.info(f"{len(final_df_filtered)} total entries")
                    if not fast_mode:
                        st.dataframe(final_df_filtered, use_container_width=True)
                
                st.success("Full range processing complete.")
        
        # ---------- 2) FULL RANGE (MULTIPLE) MODE ----------
        elif use_full_range_multi:
            st.markdown("---")
            st.markdown("### Full Range (multiple) Processing Mode")
            st.info(f"Processing {num_days_multi} days with 4 grab times per day (1800, 0315, 0900, 1230)")
            
            # Determine input value
            input_val = input_value_at_start_multi if input_value_at_start_multi != 0.0 else None
            
            # Generate list of all datetimes to process
            # Note: First grab (1800) is on the selected date (not previous day as before)
            # FIXED BUG: Now correctly uses the selected date for 1800 grab
            grab_times = [(18, 0), (3, 15), (9, 0), (12, 30)]  # (hour, minute) tuples
            all_datetimes = []
            
            for day_offset in range(num_days_multi):
                current_date = multi_start_date + dt.timedelta(days=day_offset)
                
                for idx, (hour, minute) in enumerate(grab_times):
                    # All grabs use the current_date as the base
                    # 1800 grab is on current_date
                    # 0315, 0900, 1230 grabs are on the next day (trading day)
                    if idx == 0:  # 1800 grab - starts the trading day
                        grab_date = current_date
                    else:  # 0315, 0900, 1230 grabs - next calendar day
                        grab_date = current_date + dt.timedelta(days=1)
                    
                    grab_dt = dt.datetime.combine(grab_date, dt.time(hour, minute))
                    all_datetimes.append(grab_dt)
            
            # Process each datetime
            all_day_reports = []
            
            # Start timer
            import time
            start_time = time.time()
            
            progress_bar = st.progress(0)
            progress_text = st.empty()
            
            for idx, grab_dt in enumerate(all_datetimes):
                progress_text.text(f"Processing {format_tab_name(grab_dt)}...")
                
                # Process this grab time
                if separate_timeframes and (small_feeds_dict or big_feeds_dict):
                    all_timeframe_results = []
                    timeframes = sorted(set(list(small_feeds_dict.keys()) + list(big_feeds_dict.keys())))
                    
                    for tf in timeframes:
                        tf_small = small_feeds_dict.get(tf, pd.DataFrame())
                        tf_big = big_feeds_dict.get(tf, pd.DataFrame())
                        
                        if tf_small.empty and tf_big.empty:
                            continue
                        
                        tf_result = apply_full_range_advanced(
                            measurements_df,
                            tf_small if not tf_small.empty else pd.DataFrame(),
                            grab_dt,
                            window_radius_multi,
                            day_start_hour=day_start_hour,
                            input_value_at_start=input_val,
                            big_df=tf_big if not tf_big.empty else pd.DataFrame(),
                            run_model_g=False
                        )
                        
                        if tf_result is not None and not tf_result.empty:
                            all_timeframe_results.append(tf_result)
                    
                    if all_timeframe_results:
                        final_df = pd.concat(all_timeframe_results, ignore_index=True)
                    else:
                        final_df = pd.DataFrame()
                else:
                    if small_df is None and big_df is None:
                        st.error("No feed data available")
                        final_df = pd.DataFrame()
                    else:
                        final_df = apply_full_range_advanced(
                            measurements_df,
                            small_df if small_df is not None else pd.DataFrame(),
                            grab_dt,
                            window_radius_multi,
                            day_start_hour=day_start_hour,
                            input_value_at_start=input_val,
                            big_df=big_df if big_df is not None else pd.DataFrame(),
                            run_model_g=False
                        )
                
                # Split into Grp_1a and Grp_1b, combine and format
                if final_df is not None and not final_df.empty:
                    grp_1a = final_df[final_df['M #'].isin(GROUP_1A_TRAVELERS)].copy()
                    grp_1b = final_df[final_df['M #'].isin(GROUP_1B_TRAVELERS)].copy()
                    
                    combined = combine_and_format_groups(grp_1a, grp_1b)
                    
                    # Calculate trading_day_start for HOD/LOD
                    # If grab_dt is 1800, it's already the trading day start
                    # If grab_dt is 0315, 0900, 1230, the trading day started at 1800 previous day
                    if grab_dt.hour == 18:
                        trading_day_start = grab_dt
                    else:
                        # Find the 1800 on the previous day
                        trading_day_start = grab_dt.replace(hour=18, minute=0) - dt.timedelta(days=1)
                    
                    # Get feeds for HOD/LOD calculation
                    # Use the first available timeframe's feed data
                    sm_feed_for_hod = None
                    bg_feed_for_hod = None
                    
                    # Select first available non-empty small feed
                    if small_feeds_dict:
                        for tf in ['15m', '5m', '3m']:
                            df = small_feeds_dict.get(tf)
                            if df is not None and not df.empty:
                                sm_feed_for_hod = df
                                break
                    
                    # Select first available non-empty big feed
                    if big_feeds_dict:
                        for tf in ['15m', '5m', '3m']:
                            df = big_feeds_dict.get(tf)
                            if df is not None and not df.empty:
                                bg_feed_for_hod = df
                                break
                    
                    # Add HOD/LOD columns
                    combined = add_hod_lod_columns(combined, sm_feed_for_hod, bg_feed_for_hod, trading_day_start)
                    
                    # Add Turns column using Nested Swing Detector
                    if sm_feed_for_hod is not None:
                        combined = calculate_swing_turns_column(
                            combined,
                            sm_feed_for_hod,
                            trading_day_start,
                            lookback_days=lookback_days
                        )
                    
                    tab_name = format_tab_name(grab_dt)
                    all_day_reports.append((tab_name, combined))
                
                # Update progress
                progress_bar.progress((idx + 1) / len(all_datetimes))
            
            progress_text.text("Processing complete!")
            
            # Calculate and display elapsed time
            elapsed_time = time.time() - start_time
            st.info(f"⏱️ Processing completed in {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")
            
            # Create Excel file with all tabs
            if all_day_reports:
                st.success(f"✅ Generated {len(all_day_reports)} report tabs")
                
                # Determine trading day for filename
                trading_day = get_trading_day_from_datetime(all_datetimes[0])
                trading_day_str = trading_day.strftime("%d-%b")
                
                # Create Excel file
                excel_buffer = create_multi_day_excel(
                    all_day_reports, 
                    asset_id, 
                    trading_day,
                    window_radius_multi,
                    lookback_days
                )
                
                # Updated naming convention: asset + RawTrav for + date + (lookback days) + radius + window
                filename = f"{asset_id.lower()}_RawTrav_for_{trading_day_str}_({lookback_days}days)_{int(window_radius_multi)}_radius_window.xlsx"
                
                st.download_button(
                    label="📥 Download Multi-Day Report",
                    data=excel_buffer,
                    file_name=filename,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
                
                # Display summary
                st.markdown("### Report Summary")
                for tab_name, df in all_day_reports:
                    st.text(f"{tab_name}: {len(df)} entries")
            else:
                st.warning("No data generated for the selected date range")
        
        # ---------- 3) CUSTOM RANGES MODE ----------
        elif use_custom_ranges:
            st.markdown("---")
            st.markdown("### Custom Ranges Processing Mode")
            
            enabled_ranges = []
            if use_high1 and high1 != 0.0:
                enabled_ranges.append(("High", high1))
            if use_high2 and high2 != 0.0:
                enabled_ranges.append(("High", high2))
            if use_low1 and low1 != 0.0:
                enabled_ranges.append(("Low", low1))
            if use_low2 and low2 != 0.0:
                enabled_ranges.append(("Low", low2))
            
            if not enabled_ranges:
                st.error("Please enable and configure at least one custom range")
            else:
                st.info(f"Processing {len(enabled_ranges)} custom ranges")
                
                if separate_timeframes and (small_feeds_dict or big_feeds_dict):
                    all_timeframe_results = []
                    timeframes = sorted(set(list(small_feeds_dict.keys()) + list(big_feeds_dict.keys())))
                    
                    for tf in timeframes:
                        st.text(f"Processing {tf} timeframe...")
                        tf_small = small_feeds_dict.get(tf, pd.DataFrame())
                        tf_big = big_feeds_dict.get(tf, pd.DataFrame())
                        
                        if tf_small.empty and tf_big.empty:
                            continue
                        
                        tf_result = apply_custom_ranges_advanced(
                            measurements_df,
                            tf_small if not tf_small.empty else pd.DataFrame(),
                            report_time,
                            enabled_ranges,
                            day_start_hour=day_start_hour,
                            big_df=tf_big if not tf_big.empty else pd.DataFrame(),
                            run_model_g=False
                        )
                        
                        if tf_result is not None and not tf_result.empty:
                            all_timeframe_results.append(tf_result)
                    
                    if all_timeframe_results:
                        final_df_filtered = pd.concat(all_timeframe_results, ignore_index=True)
                    else:
                        final_df_filtered = pd.DataFrame()
                else:
                    if small_df is None and big_df is None:
                        st.error("No feed data available")
                        final_df_filtered = pd.DataFrame()
                    else:
                        final_df_filtered = apply_custom_ranges_advanced(
                            measurements_df,
                            small_df if small_df is not None else pd.DataFrame(),
                            report_time,
                            enabled_ranges,
                            day_start_hour=day_start_hour,
                            big_df=big_df if big_df is not None else pd.DataFrame(),
                            run_model_g=run_g_on_custom
                        )
                
                if final_df_filtered is None or final_df_filtered.empty:
                    st.warning("No entries found in custom ranges")
                    traveler_reports = {}
                else:
                    # Sort by Range, Group, Output (desc), Arrival (asc)
                    sort_cols = []
                    if 'Range' in final_df_filtered.columns:
                        sort_cols.append('Range')
                    if 'Group' in final_df_filtered.columns:
                        sort_cols.append('Group')
                    if 'Output' in final_df_filtered.columns:
                        sort_cols.append('Output')
                    if 'Arrival' in final_df_filtered.columns:
                        sort_cols.append('Arrival')
                    
                    if sort_cols:
                        # Define sort order: Range and Group ascending, Output descending, Arrival ascending
                        ascending_list = []
                        for col in sort_cols:
                            if col in ['Range', 'Group']:
                                ascending_list.append(True)
                            elif col == 'Output':
                                ascending_list.append(False)
                            else:  # Arrival
                                ascending_list.append(True)
                        
                        final_df_filtered = final_df_filtered.sort_values(sort_cols, ascending=ascending_list)
                    
                    # Assign group labels
                    final_df_filtered['Group'] = final_df_filtered['M #'].apply(
                        lambda m_num: 
                            '1a' if m_num in GROUP_1A_TRAVELERS else
                            '1b' if m_num in GROUP_1B_TRAVELERS else
                            '2a' if m_num in GROUP_2A_TRAVELERS else
                            '2b' if m_num in GROUP_2B_TRAVELERS else
                            'Other'
                    )
                    
                    traveler_reports = {"Custom Ranges": final_df_filtered}
                    
                    if not minimal_display:
                        st.markdown("#### Custom Ranges Report")
                        st.info(f"{len(final_df_filtered)} total entries")
                        if not fast_mode:
                            st.dataframe(final_df_filtered, use_container_width=True)
                    
                    st.success("Custom ranges processing complete.")
        
        # ---------- 4) HOD/LOD MODE ----------
        elif use_hod_lod:
            st.markdown("---")
            st.markdown("### HOD/LOD Processing Mode")
            
            if small_df is None:
                st.error("HOD/LOD mode requires at least one small feed file")
            else:
                # Process HOD/LOD
                results = process_hod_lod_mode(
                    small_df,
                    measurements_df,
                    hod_lod_num_days,
                    include_partial_day,
                    day_start_hour
                )
                
                if results:
                    st.success(f"✅ Processed {len(results)} trading days")
                    
                    # Build traveler reports from HOD/LOD results
                    for day_label, day_data in results.items():
                        st.markdown(f"#### {day_label}")
                        
                        if 'hod_travelers' in day_data and not day_data['hod_travelers'].empty:
                            st.markdown("**HOD Travelers:**")
                            st.dataframe(day_data['hod_travelers'], use_container_width=True)
                        
                        if 'lod_travelers' in day_data and not day_data['lod_travelers'].empty:
                            st.markdown("**LOD Travelers:**")
                            st.dataframe(day_data['lod_travelers'], use_container_width=True)
                    
                    # Create download for HOD/LOD results
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        for day_label, day_data in results.items():
                            if 'hod_travelers' in day_data and not day_data['hod_travelers'].empty:
                                day_data['hod_travelers'].to_excel(writer, sheet_name=f"{day_label}_HOD", index=False)
                            if 'lod_travelers' in day_data and not day_data['lod_travelers'].empty:
                                day_data['lod_travelers'].to_excel(writer, sheet_name=f"{day_label}_LOD", index=False)
                    
                    output.seek(0)
                    st.download_button(
                        label="📥 Download HOD/LOD Report",
                        data=output,
                        file_name=f"{asset_id.lower()}_hod_lod_report_{dt.datetime.now().strftime('%d-%b-%y_%H-%M')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                else:
                    st.warning("No HOD/LOD data generated")
    
    # === MODEL DETECTION (runs on traveler_reports) ===
    if traveler_reports and not bypass_df:
        st.markdown("---")
        st.markdown("### 🔍 Model Detection")
        
        # Combine all reports for model detection
        combined_for_models = pd.concat([df for df in traveler_reports.values() if not df.empty], ignore_index=True)
        
        if filter_future_data:
            if 'Arrival' in combined_for_models.columns:
                combined_for_models['Arrival'] = pd.to_datetime(combined_for_models['Arrival'], errors='coerce')
                before_filter = len(combined_for_models)
                combined_for_models = combined_for_models[combined_for_models['Arrival'] <= report_time]
                after_filter = len(combined_for_models)
                st.info(f"🔮 Filtered future data: {before_filter} → {after_filter} entries ({before_filter - after_filter} removed)")
        
        # Run Model G
        if run_g_models and not combined_for_models.empty:
            st.markdown("#### 🟢 Model G Detection")
            
            g_results = run_model_g_detection(
                combined_for_models,
                run_g05_g06=run_g05_g06,
                run_g08=run_g08,
                run_g09=run_g09,
                run_g10=run_g10,
                run_g11=run_g11,
                output_spread_filter=output_spread_filter,
                show_rejected_groups=show_rejected_groups,
                g10_group_0=g10_group_0,
                g10_group_1=g10_group_1,
                g10_group_2=g10_group_2,
                g10_group_3=g10_group_3,
                g10_group_4=g10_group_4,
                g11_group_0=g11_group_0,
                g11_group_1=g11_group_1,
                g11_group_2=g11_group_2,
                g11_group_3=g11_group_3,
                g11_group_4=g11_group_4,
                g11_display_recipes=g11_display_recipes,
                g11_display_others=g11_display_others
            )
            
            if g_results:
                for model_name, model_df in g_results.items():
                    if not model_df.empty:
                        st.markdown(f"**{model_name}:** {len(model_df)} detections")
                        if not fast_mode:
                            st.dataframe(model_df, use_container_width=True)
        
        # Run Model A
        if run_a_models and not combined_for_models.empty:
            st.markdown("#### 🔵 Model A Detection")
            a_result = run_a_model_detection_today(combined_for_models)
            if a_result is not None and not a_result.empty:
                st.dataframe(a_result, use_container_width=True)
        
        # Run Model B
        if run_b_models and not combined_for_models.empty:
            st.markdown("#### 🟡 Model B Detection")
            b_result = run_b_model_detection(combined_for_models)
            if b_result is not None and not b_result.empty:
                st.dataframe(b_result, use_container_width=True)
        
        # Run Model C
        if run_c_models and not combined_for_models.empty:
            st.markdown("#### 🟠 Model C Detection")
            c_result = run_c_model_detection(combined_for_models, run_c01, run_c02, run_c04)
            if c_result is not None and not c_result.empty:
                st.dataframe(c_result, use_container_width=True)
        
        # Run Model X
        if run_x_models and not combined_for_models.empty:
            st.markdown("#### 🟣 Model X Detection")
            x_result = run_x_model_detection(combined_for_models)
            if x_result is not None and not x_result.empty:
                st.dataframe(x_result, use_container_width=True)
        
        # Run Simple Single Line Mega Report
        if run_single_line and not combined_for_models.empty:
            st.markdown("#### 📊 Simple Single Line Mega Report")
            single_line_result = run_simple_single_line_analysis(combined_for_models)
            if single_line_result is not None and not single_line_result.empty:
                st.dataframe(single_line_result, use_container_width=True)
    
    # === EXPORT (only for non-multi modes) ===
    if traveler_reports and not use_full_range_multi:
        # Determine window radius and lookback_days to pass (only for Full Range single mode)
        wr_to_pass = window_radius if use_full_range_single else None
        lb_to_pass = lookback_days if use_full_range_single else None
        render_unified_export(traveler_reports, report_time, asset_id, window_radius=wr_to_pass, lookback_days=lb_to_pass)

st.markdown("---")
st.caption("🌌 Traveler Report Generator v34a | Origin Filtering in Main Area + Nested Swing Detector + Look Back Days")
