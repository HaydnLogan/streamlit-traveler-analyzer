import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
from typing import List, Dict, Tuple
import io
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Market Swing Analysis 03b", layout="wide")
st.header("📈 Market Swing Analysis Tool 03b")

# File upload - supports both CSV and Excel
uploaded_file = st.file_uploader("Upload OHLC file", type=['csv', 'xlsx', 'xls'])

def parse_timestamp_naive(timestamp_str):
    """Parse timestamp and return naive datetime (remove timezone info)"""
    try:
        # Parse the timestamp
        if isinstance(timestamp_str, str):
            # Handle ISO format with timezone - remove timezone part
            if 'T' in timestamp_str and ('-' in timestamp_str[-6:] or '+' in timestamp_str[-6:]):
                # Find timezone offset and remove it
                for tz_sep in ['+', '-']:
                    if tz_sep in timestamp_str[-6:]:
                        timestamp_str = timestamp_str[:timestamp_str.rfind(tz_sep)]
                        break
            
            # Parse without timezone
            return pd.to_datetime(timestamp_str, infer_datetime_format=True)
        else:
            return pd.to_datetime(timestamp_str)
    except:
        return pd.to_datetime(timestamp_str, errors='coerce')

def categorize_range(range_val):
    """Categorize daily range"""
    if range_val < 100:
        return "< 100"
    elif range_val < 150:
        return "100-150"
    elif range_val < 200:
        return "150-200"
    elif range_val < 250:
        return "200-250"
    elif range_val < 350:
        return "250-350"
    elif range_val < 500:
        return "350-500"
    elif range_val < 1000:
        return "500-1000"
    else:
        return "1000+"

def categorize_swing(move_size):
    """Categorize swing moves"""
    if move_size < 60:
        return "30-60"
    elif move_size < 100:
        return "60-100"
    elif move_size < 150:
        return "100-150"
    elif move_size < 200:
        return "150-200"
    else:
        return "200+"

def get_trading_day(timestamp, day_start_hour=18):
    """Get trading day based on start hour (default 18:00)"""
    if timestamp.hour < day_start_hour:
        # Before 18:00, belongs to current calendar day's session
        return timestamp.date()
    else:
        # After 18:00, belongs to next day's session
        return (timestamp + dt.timedelta(days=1)).date()

def detect_swings(df, swing_threshold=30, drawdown_limit=25):
    """
    Detect swings by tracking actual price progression bar by bar
    Only record moves that actually happen in chronological order
    FIXED: Eliminates duplicate from/to times and ensures chronological order
    """
    swings = []
    
    if len(df) == 0:
        return swings
    
    # Start from the first bar's range
    current_extreme_high = df.iloc[0]['high']
    current_extreme_low = df.iloc[0]['low']
    extreme_high_time = df.iloc[0]['time']
    extreme_low_time = df.iloc[0]['time']
    
    # Track what we're currently measuring from and the starting point
    measuring_from = 'low'  # Start by measuring moves up from the low
    swing_start_price = current_extreme_low
    swing_start_time = extreme_low_time
    
    for idx, row in df.iterrows():
        bar_high = row['high']
        bar_low = row['low']
        bar_time = row['time']
        
        if measuring_from == 'low':
            # We're tracking upward moves from the current extreme low
            
            # Update our extreme low if we go lower (and reset our starting point)
            if bar_low < current_extreme_low:
                current_extreme_low = bar_low
                extreme_low_time = bar_time
                # Reset starting point for the swing
                swing_start_price = current_extreme_low
                swing_start_time = extreme_low_time
            
            # Update our extreme high as we move up
            if bar_high > current_extreme_high:
                current_extreme_high = bar_high
                extreme_high_time = bar_time
            
            # Check if we've made a significant upward move
            upward_move = current_extreme_high - current_extreme_low
            if upward_move >= swing_threshold:
                # Now check for drawdown to confirm this as a swing high
                drawdown = current_extreme_high - bar_low
                if drawdown >= drawdown_limit:
                    # FIXED: Only record if from and to are different AND chronologically correct
                    if (swing_start_time < extreme_high_time and 
                        swing_start_price != current_extreme_high):
                        
                        # Record the upward swing
                        swings.append({
                            'type': 'high',
                            'swing_price': current_extreme_high,
                            'swing_time': extreme_high_time,
                            'move_size': upward_move,
                            'category': categorize_swing(upward_move),
                            'from_price': swing_start_price,
                            'from_time': swing_start_time,
                            'to_price': current_extreme_high,
                            'to_time': extreme_high_time,
                            'direction': 'up'
                        })
                    
                    # Now start measuring downward moves from this high
                    measuring_from = 'high'
                    current_extreme_low = bar_low
                    extreme_low_time = bar_time
                    # Set new starting point for next swing (confirmed high becomes start)
                    swing_start_price = current_extreme_high
                    swing_start_time = extreme_high_time
        
        else:  # measuring_from == 'high'
            # We're tracking downward moves from the current extreme high
            
            # Update our extreme high if we go higher (and reset our starting point)
            if bar_high > current_extreme_high:
                current_extreme_high = bar_high
                extreme_high_time = bar_time
                # Reset starting point for the swing
                swing_start_price = current_extreme_high
                swing_start_time = extreme_high_time
            
            # Update our extreme low as we move down
            if bar_low < current_extreme_low:
                current_extreme_low = bar_low
                extreme_low_time = bar_time
            
            # Check if we've made a significant downward move
            downward_move = current_extreme_high - current_extreme_low
            if downward_move >= swing_threshold:
                # Now check for bounce to confirm this as a swing low
                bounce = bar_high - current_extreme_low
                if bounce >= drawdown_limit:
                    # FIXED: Only record if from and to are different AND chronologically correct
                    if (swing_start_time < extreme_low_time and 
                        swing_start_price != current_extreme_low):
                        
                        # Record the downward swing
                        swings.append({
                            'type': 'low',
                            'swing_price': current_extreme_low,
                            'swing_time': extreme_low_time,
                            'move_size': downward_move,
                            'category': categorize_swing(downward_move),
                            'from_price': swing_start_price,
                            'from_time': swing_start_time,
                            'to_price': current_extreme_low,
                            'to_time': extreme_low_time,
                            'direction': 'down'
                        })
                    
                    # Now start measuring upward moves from this low
                    measuring_from = 'low'
                    current_extreme_high = bar_high
                    extreme_high_time = bar_time
                    # Set new starting point for next swing (confirmed low becomes start)
                    swing_start_price = current_extreme_low
                    swing_start_time = extreme_low_time
    
    return swings

def analyze_daily_data(df, day_start_hour=18):
    """Analyze daily market structure based on trading day definition"""
    daily_stats = []
    
    # Add trading day column
    df['trading_day'] = df['time'].apply(lambda x: get_trading_day(x, day_start_hour))
    
    for trading_day, day_data in df.groupby('trading_day'):
        # Sort by time
        day_data = day_data.sort_values('time')
        
        # Basic daily stats
        daily_high = day_data['high'].max()
        daily_low = day_data['low'].min()
        daily_range = daily_high - daily_low
        
        # Find exact times
        high_time = day_data[day_data['high'] == daily_high]['time'].iloc[0]
        low_time = day_data[day_data['low'] == daily_low]['time'].iloc[0]
        
        # Session start and end times
        session_start = day_data['time'].min()
        session_end = day_data['time'].max()
        
        # Detect swings for this day
        swings = detect_swings(day_data.reset_index(drop=True))
        
        # Get NY session swings (starting at or after 8 AM)
        ny_session_time = dt.datetime.combine(trading_day, dt.time(8, 0))
        ny_swings = []
        for swing in swings:
            swing_start = swing['from_time']
            if swing_start.time() >= dt.time(8, 0):
                ny_swings.append(swing)
        
        # Get first 3 NY swings
        ny_swings = sorted(ny_swings, key=lambda x: x['from_time'])[:3]
        
        # Get top 3 swings
        swing_moves = [s['move_size'] for s in swings]
        swing_moves.sort(reverse=True)
        top_3_swings = swing_moves[:3]
        
        # Count swings by category
        swing_categories = [s['category'] for s in swings]
        category_counts = {
            '30-60': swing_categories.count('30-60'),
            '60-100': swing_categories.count('60-100'),
            '100-150': swing_categories.count('100-150'),
            '150-200': swing_categories.count('150-200'),
            '200+': swing_categories.count('200+')
        }
        
        daily_stats.append({
            'trading_day': trading_day,
            'session_start': session_start,
            'session_end': session_end,
            'daily_high': daily_high,
            'daily_high_time': high_time,
            'daily_low': daily_low,
            'daily_low_time': low_time,
            'daily_range': daily_range,
            'range_category': categorize_range(daily_range),
            'ny_swings_count': len(ny_swings),
            'ny_1': ny_swings[0]['move_size'] if len(ny_swings) > 0 else 'none',
            'ny_2': ny_swings[1]['move_size'] if len(ny_swings) > 1 else 'none',
            'ny_3': ny_swings[2]['move_size'] if len(ny_swings) > 2 else 'none',
            'total_swings': len(swings),
            'top_1_swing': top_3_swings[0] if len(top_3_swings) > 0 else 0,
            'top_2_swing': top_3_swings[1] if len(top_3_swings) > 1 else 0,
            'top_3_swing': top_3_swings[2] if len(top_3_swings) > 2 else 0,
            'swings_30_60': category_counts['30-60'],
            'swings_60_100': category_counts['60-100'],
            'swings_100_150': category_counts['100-150'],
            'swings_150_200': category_counts['150-200'],
            'swings_200_plus': category_counts['200+'],
            'all_swings': swings,
            'ny_swings': ny_swings
        })
    
    return daily_stats

if uploaded_file is not None:
    try:
        # Determine file type and read accordingly
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        if file_extension == 'csv':
            df = pd.read_csv(uploaded_file)
            sheet_name = None
        else:  # Excel file
            # First, let user select sheet
            excel_file = pd.ExcelFile(uploaded_file)
            sheet_names = excel_file.sheet_names
            
            if len(sheet_names) > 1:
                sheet_name = st.selectbox("Select Excel Sheet", sheet_names)
            else:
                sheet_name = sheet_names[0]
            
            df = pd.read_excel(uploaded_file, sheet_name=sheet_name)
        
        # Display file info
        sheet_info = f" (Sheet: {sheet_name})" if sheet_name else ""
        st.success(f"✅ File loaded{sheet_info}: {len(df)} rows")
        
        # Show column mapping options
        st.subheader("📋 Column Mapping")
        cols = df.columns.tolist()
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            time_col = st.selectbox("Time Column", cols, index=0 if 'time' in cols else 0)
        with col2:
            open_col = st.selectbox("Open Column", cols, index=next((i for i, col in enumerate(cols) if 'open' in col.lower()), 1))
        with col3:
            high_col = st.selectbox("High Column", cols, index=next((i for i, col in enumerate(cols) if 'high' in col.lower()), 2))
        with col4:
            low_col = st.selectbox("Low Column", cols, index=next((i for i, col in enumerate(cols) if 'low' in col.lower()), 3))
        with col5:
            close_col = st.selectbox("Close Column", cols, index=next((i for i, col in enumerate(cols) if 'close' in col.lower()), 4))
        
        # Parameters
        st.subheader("⚙️ Analysis Parameters")
        param_col1, param_col2, param_col3 = st.columns(3)
        with param_col1:
            swing_threshold = st.number_input("Swing Threshold (units)", min_value=1, value=30)
        with param_col2:
            drawdown_limit = st.number_input("Drawdown Limit (units)", min_value=1, value=25)
        with param_col3:
            day_start_hour = st.number_input("Trading Day Start Hour (24hr format)", min_value=0, max_value=23, value=18)
        
        if st.button("🚀 Run Analysis"):
            # Standardize column names
            analysis_df = df[[time_col, open_col, high_col, low_col, close_col]].copy()
            analysis_df.columns = ['time', 'open', 'high', 'low', 'close']
            
            # Parse timestamps (naive/local time)
            analysis_df['time'] = analysis_df['time'].apply(parse_timestamp_naive)
            analysis_df = analysis_df.dropna(subset=['time']).sort_values('time')
            
            # Convert price columns to numeric
            for col in ['open', 'high', 'low', 'close']:
                analysis_df[col] = pd.to_numeric(analysis_df[col], errors='coerce')
            
            # Run daily analysis
            with st.spinner("Analyzing market structure..."):
                daily_stats = analyze_daily_data(analysis_df, day_start_hour)
            
            if daily_stats:
                st.success(f"✅ Analysis complete! Found data for {len(daily_stats)} trading days")
                
                # Create summary DataFrame
                summary_df = pd.DataFrame(daily_stats)
                display_summary = summary_df.drop(['all_swings', 'ny_swings'], axis=1).copy()
                
                # Format datetime columns for display
                datetime_cols = ['session_start', 'session_end', 'daily_high_time', 'daily_low_time']
                for col in datetime_cols:
                    if col in display_summary.columns:
                        display_summary[col] = display_summary[col].dt.strftime('%Y-%m-%d %H:%M:%S')
                
                # Display NY Results first
                st.subheader("🗽 Daily NY Results")
                ny_summary = summary_df.drop(['all_swings', 'ny_swings'], axis=1).copy()
                
                # Format datetime columns for display
                datetime_cols = ['session_start', 'session_end', 'daily_high_time', 'daily_low_time']
                for col in datetime_cols:
                    if col in ny_summary.columns:
                        ny_summary[col] = ny_summary[col].dt.strftime('%Y-%m-%d %H:%M:%S')
                
                st.dataframe(ny_summary, use_container_width=True)
                
                # Create detailed NY swings DataFrame
                detailed_ny_swings = []
                for day_stat in daily_stats:
                    trading_day = day_stat['trading_day']
                    daily_high_time = day_stat['daily_high_time']
                    daily_low_time = day_stat['daily_low_time']
                    daily_high = day_stat['daily_high']
                    daily_low = day_stat['daily_low']
                    
                    # Collect all relevant swings with their IDs
                    swing_records = []
                    
                    # Add HOD and LOD
                    swing_records.append({
                        'swing_id': 'HOD',
                        'time': daily_high_time,
                        'price': daily_high,
                        'swing_type': 'high',
                        'direction': 'n/a',
                        'from_datetime': 'n/a',
                        'from_price': 'n/a',
                        'to_datetime': daily_high_time.strftime('%Y-%m-%d %H:%M'),
                        'to_price': daily_high,
                        'move_size': 'n/a',
                        'category': 'n/a'
                    })
                    
                    swing_records.append({
                        'swing_id': 'LOD',
                        'time': daily_low_time,
                        'price': daily_low,
                        'swing_type': 'low',
                        'direction': 'n/a',
                        'from_datetime': 'n/a',
                        'from_price': 'n/a',
                        'to_datetime': daily_low_time.strftime('%Y-%m-%d %H:%M'),
                        'to_price': daily_low,
                        'move_size': 'n/a',
                        'category': 'n/a'
                    })
                    
                    # Add NY swings with IDs
                    ny_swings = day_stat.get('ny_swings', [])
                    for i, ny_swing in enumerate(ny_swings):
                        ny_id = f'NY {i+1}'
                        
                        # Check if this NY swing shares start/end points with HOD/LOD
                        swing_id = ny_id
                        
                        # Check if NY swing start matches HOD/LOD
                        if (ny_swing['from_price'] == daily_high and 
                            abs((ny_swing['from_time'] - daily_high_time).total_seconds()) < 60):
                            swing_id = f'HOD, {ny_id}'
                        elif (ny_swing['from_price'] == daily_low and 
                              abs((ny_swing['from_time'] - daily_low_time).total_seconds()) < 60):
                            swing_id = f'LOD, {ny_id}'
                        
                        # Check if NY swing end matches HOD/LOD
                        if (ny_swing['to_price'] == daily_high and 
                            abs((ny_swing['to_time'] - daily_high_time).total_seconds()) < 60):
                            if 'HOD' not in swing_id:
                                swing_id = f'HOD, {ny_id}' if swing_id == ny_id else f'{swing_id}, HOD'
                        elif (ny_swing['to_price'] == daily_low and 
                              abs((ny_swing['to_time'] - daily_low_time).total_seconds()) < 60):
                            if 'LOD' not in swing_id:
                                swing_id = f'LOD, {ny_id}' if swing_id == ny_id else f'{swing_id}, LOD'
                        
                        swing_records.append({
                            'swing_id': swing_id,
                            'time': ny_swing['from_time'],
                            'price': ny_swing['from_price'],
                            'swing_type': ny_swing['type'],
                            'direction': ny_swing['direction'],
                            'from_datetime': ny_swing['from_time'].strftime('%Y-%m-%d %H:%M'),
                            'from_price': ny_swing['from_price'],
                            'to_datetime': ny_swing['to_time'].strftime('%Y-%m-%d %H:%M'),
                            'to_price': ny_swing['to_price'],
                            'move_size': ny_swing['move_size'],
                            'category': ny_swing['category']
                        })
                    
                    # Sort chronologically by time
                    swing_records.sort(key=lambda x: x['time'])
                    
                    # Add to detailed list
                    for record in swing_records:
                        detailed_ny_swings.append({
                            'trading_day': trading_day,
                            'swing_id': record['swing_id'],
                            'swing_type': record['swing_type'],
                            'direction': record['direction'],
                            'from_datetime': record['from_datetime'],
                            'from_price': record['from_price'],
                            'to_datetime': record['to_datetime'],
                            'to_price': record['to_price'],
                            'move_size': record['move_size'],
                            'category': record['category']
                        })
                
                if detailed_ny_swings:
                    detailed_ny_df = pd.DataFrame(detailed_ny_swings)
                    st.subheader("🎯 Detailed NY Swing Analysis")
                    st.dataframe(detailed_ny_df, use_container_width=True)
                
                # Display regular Daily Analysis Results
                st.subheader("📊 Daily Analysis Results")
                regular_summary = display_summary.drop(['ny_swings_count', 'ny_1', 'ny_2', 'ny_3'], axis=1, errors='ignore').copy()
                st.dataframe(regular_summary, use_container_width=True)
                
                # Create detailed swings DataFrame
                detailed_swings = []
                for day_stat in daily_stats:
                    for swing in day_stat['all_swings']:
                        detailed_swings.append({
                            'trading_day': day_stat['trading_day'],
                            'swing_type': swing['type'],
                            'direction': swing['direction'],
                            'from_datetime': swing['from_time'].strftime('%Y-%m-%d %H:%M:%S'),
                            'from_price': swing['from_price'],
                            'to_datetime': swing['to_time'].strftime('%Y-%m-%d %H:%M:%S'),
                            'to_price': swing['to_price'],
                            'move_size': swing['move_size'],
                            'category': swing['category']
                        })
                
                if detailed_swings:
                    detailed_df = pd.DataFrame(detailed_swings)
                    st.subheader("🎯 Detailed Swing Analysis")
                    st.dataframe(detailed_df, use_container_width=True)
                else:
                    # Ensure detailed_df exists so export buttons don't crash
                    detailed_df = pd.DataFrame(columns=[
                        'trading_day','swing_type','direction',
                        'from_datetime','from_price','to_datetime','to_price',
                        'move_size','category'
                    ])

                
                # Summary statistics
                st.subheader("📈 Summary Statistics")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    avg_range = regular_summary['daily_range'].mean()
                    st.metric("Avg Daily Range", f"{avg_range:.1f}")
                
                with col2:
                    avg_swings = regular_summary['total_swings'].mean()
                    st.metric("Avg Swings/Day", f"{avg_swings:.1f}")
                
                with col3:
                    total_days = len(regular_summary)
                    st.metric("Total Trading Days", total_days)
                
                with col4:
                    avg_ny_swings = ny_summary['ny_swings_count'].mean()
                    st.metric("Avg NY Swings/Day", f"{avg_ny_swings:.1f}")
                
                # Range category distribution
                st.subheader("📊 Range Category Distribution")
                range_dist = regular_summary['range_category'].value_counts()
                fig_range = go.Figure(data=[go.Bar(x=range_dist.index, y=range_dist.values)])
                fig_range.update_layout(title="Daily Range Categories", xaxis_title="Range Category", yaxis_title="Number of Days")
                st.plotly_chart(fig_range, use_container_width=True)
                
                # Swing category distribution
                st.subheader("🎯 Swing Category Analysis")
                swing_cols = ['swings_30_60', 'swings_60_100', 'swings_100_150', 'swings_150_200', 'swings_200_plus']
                swing_totals = regular_summary[swing_cols].sum()
                swing_totals.index = ['30-60', '60-100', '100-150', '150-200', '200+']
                
                fig_swings = go.Figure(data=[go.Bar(x=swing_totals.index, y=swing_totals.values)])
                fig_swings.update_layout(title="Swing Size Distribution", xaxis_title="Swing Category", yaxis_title="Total Swings")
                st.plotly_chart(fig_swings, use_container_width=True)
                
                # Export options
                st.subheader("💾 Export Data")
                
                # 1) NY Summary (keep true datetimes for export)
                ny_export_summary = ny_summary.copy()
                for col in ['session_start', 'session_end', 'daily_high_time', 'daily_low_time']:
                    if col in ny_export_summary.columns:
                        ny_export_summary[col] = pd.to_datetime(ny_export_summary[col], errors='coerce')
                
                # 2) Regular Summary (keep true datetimes for export)
                export_summary = regular_summary.copy()
                for col in ['session_start', 'session_end', 'daily_high_time', 'daily_low_time']:
                    if col in export_summary.columns:
                        export_summary[col] = pd.to_datetime(export_summary[col], errors='coerce')
                
                # 3) Build detailed swings for BOTH export (datetimes) and display (strings)
                detailed_swings_export = []
                detailed_swings_display = []
                for day_stat in daily_stats:
                    for s in day_stat.get('all_swings', []):
                        # --- export (datetimes) ---
                        detailed_swings_export.append({
                            'trading_day': day_stat['trading_day'],
                            'swing_type': s['type'],
                            'direction': s['direction'],
                            'from_datetime': s['from_time'],   # datetime
                            'from_price': s['from_price'],
                            'to_datetime': s['to_time'],       # datetime
                            'to_price': s['to_price'],
                            'move_size': s['move_size'],
                            'category': s['category']
                        })
                        # --- display (strings) ---
                        detailed_swings_display.append({
                            'trading_day': day_stat['trading_day'],
                            'swing_type': s['type'],
                            'direction': s['direction'],
                            'from_datetime': pd.to_datetime(s['from_time']).strftime('%Y-%m-%d %H:%M'),
                            'from_price': s['from_price'],
                            'to_datetime': pd.to_datetime(s['to_time']).strftime('%Y-%m-%d %H:%M'),
                            'to_price': s['to_price'],
                            'move_size': s['move_size'],
                            'category': s['category']
                        })
                
                detailed_df_export = pd.DataFrame(detailed_swings_export)
                detailed_df_display = pd.DataFrame(detailed_swings_display)
                
                # 4) NY detailed swings for export
                detailed_ny_export = []
                for record in detailed_ny_swings:
                    export_record = record.copy()
                    # Convert string datetimes back to datetime objects for export
                    if export_record['from_datetime'] != 'n/a':
                        export_record['from_datetime'] = pd.to_datetime(export_record['from_datetime'])
                    if export_record['to_datetime'] != 'n/a':
                        export_record['to_datetime'] = pd.to_datetime(export_record['to_datetime'])
                    detailed_ny_export.append(export_record)
                
                detailed_ny_df_export = pd.DataFrame(detailed_ny_export)
                
                # 5) Single Excel writer with datetime format for ALL datetime cols
                excel_buffer = io.BytesIO()
                with pd.ExcelWriter(
                    excel_buffer,
                    engine='xlsxwriter',
                    datetime_format='yyyy-mm-dd hh:mm'
                ) as writer:
                    # NY Results first
                    ny_export_summary.to_excel(writer, sheet_name='Daily NY Results', index=False)
                    
                    if not detailed_ny_df_export.empty:
                        detailed_ny_df_export.to_excel(writer, sheet_name='Detailed NY Swings', index=False)
                    
                    # Regular results
                    export_summary.to_excel(writer, sheet_name='Daily Analysis', index=False)
                    
                    if not detailed_df_export.empty:
                        detailed_df_export.to_excel(writer, sheet_name='Detailed Swings', index=False)
                
                excel_buffer.seek(0)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.download_button(
                        label="📘 Download Excel Report",
                        data=excel_buffer.getvalue(),
                        file_name=f"swing_analysis_{dt.datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                
                with col2:
                    st.download_button(
                        label="🗽 Download NY Results CSV",
                        data=ny_summary.to_csv(index=False),
                        file_name=f"ny_results_{dt.datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv"
                    )
                
                with col3:
                    st.download_button(
                        label="📄 Download All Swings CSV",
                        data=detailed_df_display.to_csv(index=False),
                        file_name=f"swing_details_{dt.datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv"
                    )
                    # name='Detailed Swings', index=False)
                
                excel_buffer.seek(0)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.download_button(
                        label="📘 Download Excel Report",
                        data=excel_buffer.getvalue(),
                        file_name=f"swing_analysis_{dt.datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                
                with col2:
                    st.download_button(
                        label="📄 Download Summary CSV",
                        data=display_summary.to_csv(index=False),
                        file_name=f"swing_summary_{dt.datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv"
                    )
                
                with col3:
                    st.download_button(
                        label="📄 Download Swings CSV",
                        data=detailed_df_display.to_csv(index=False),
                        file_name=f"swing_details_{dt.datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv"
                    )

                
                # Sample visualization
                if len(daily_stats) > 0 and st.checkbox("📈 Show Trading Day Visualization"):
                    sample_date = st.selectbox("Select Trading Day to Visualize", [str(d['trading_day']) for d in daily_stats])
                    sample_day = next(d for d in daily_stats if str(d['trading_day']) == sample_date)
                    
                    # Get the day's data
                    day_data = analysis_df[analysis_df['time'].apply(lambda x: get_trading_day(x, day_start_hour)) == sample_day['trading_day']]
                    day_data = day_data.sort_values('time')
                    
                    # Create candlestick chart
                    fig = make_subplots(rows=1, cols=1, subplot_titles=[f"Market Structure - Trading Day {sample_date}"])
                    
                    # Add candlestick
                    fig.add_trace(go.Candlestick(
                        x=day_data['time'],
                        open=day_data['open'],
                        high=day_data['high'],
                        low=day_data['low'],
                        close=day_data['close'],
                        name="OHLC"
                    ))
                    
                    # Add swing points with from/to lines
                    for i, swing in enumerate(sample_day['all_swings']):
                        color = 'red' if swing['type'] == 'high' else 'green'
                        
                        # Add swing line from start to end
                        fig.add_trace(go.Scatter(
                            x=[swing['from_time'], swing['to_time']],
                            y=[swing['from_price'], swing['to_price']],
                            mode='lines+markers',
                            line=dict(color=color, width=2),
                            marker=dict(color=color, size=8, symbol='diamond'),
                            name=f"{swing['category']} {swing['direction']} ({swing['move_size']:.1f})",
                            showlegend=True
                        ))
                        
                        # Add text annotation
                        mid_time = swing['from_time'] + (swing['to_time'] - swing['from_time']) / 2
                        mid_price = (swing['from_price'] + swing['to_price']) / 2
                        
                        fig.add_annotation(
                            x=mid_time,
                            y=mid_price,
                            text=f"{swing['move_size']:.1f}",
                            showarrow=False,
                            font=dict(color=color, size=10),
                            bgcolor="white",
                            bordercolor=color,
                            borderwidth=1
                        )
                    
                    fig.update_layout(
                        title=f"Trading Day Range: {sample_day['daily_range']:.1f} ({sample_day['range_category']}) | Swings: {sample_day['total_swings']}",
                        xaxis_title="Time",
                        yaxis_title="Price",
                        height=700,
                        xaxis_rangeslider_visible=False
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show swing details for selected day
                    if sample_day['all_swings']:
                        st.subheader(f"🎯 Swing Details for {sample_date}")
                        swing_details = detailed_df_export[
                            detailed_df_export['trading_day'] == sample_day['trading_day']
                        ]
                        st.dataframe(swing_details, use_container_width=True)

            
            else:
                st.error("No data found to analyze")
    
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        st.text("Please ensure your file has the correct format with time, open, high, low, close columns")

else:
    st.info("👆 Upload a CSV or Excel file with OHLC data to begin analysis")
    
    # Show expected format
    st.subheader("📋 Expected File Format")
    sample_data = {
        'time': ['2025-06-15T18:00:00-04:00', '2025-06-15T18:15:00-04:00', '2025-06-15T18:30:00-04:00'],
        'open': [21784, 21821.25, 21835],
        'high': [21850.75, 21842, 21851.50],
        'low': [21722, 21815, 21798.25],
        'close': [21821.25, 21835, 21834.75]
    }
    st.table(pd.DataFrame(sample_data))
    
    st.info("""
    **Key Features:**
    - ✅ **CSV and Excel Support**: Upload either file type, select Excel sheets
    - ✅ **Naive Time Handling**: Timezone offsets are removed (e.g., -04:00 stripped)
    - ✅ **Custom Trading Day**: Define when your trading day starts (default: 18:00)
    - ✅ **Swing From/To Tracking**: See exact datetime and price ranges for each swing
    - ✅ **Enhanced Visualization**: View swing movements with from/to lines
    - ✅ **Detailed Export**: Get comprehensive Excel reports with multiple sheets
    - 🔧 **FIXED**: Eliminates duplicate from/to times and prices - no more impossible swings!
    """)
