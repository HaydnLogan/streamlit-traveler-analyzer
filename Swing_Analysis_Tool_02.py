import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
from typing import List, Dict, Tuple
import io
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Market Swing Analysis", layout="wide")
st.header("📈 Market Swing Analysis Tool")

# File upload - now supports both CSV and Excel
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
    """
    swings = []
    
    if len(df) == 0:
        return swings
    
    # Start from the first bar's range
    current_extreme_high = df.iloc[0]['high']
    current_extreme_low = df.iloc[0]['low']
    extreme_high_time = df.iloc[0]['time']
    extreme_low_time = df.iloc[0]['time']
    
    # Track what we're currently measuring from
    measuring_from = 'low'  # Start by measuring moves up from the low
    
    for idx, row in df.iterrows():
        bar_high = row['high']
        bar_low = row['low']
        bar_time = row['time']
        
        if measuring_from == 'low':
            # We're tracking upward moves from the current extreme low
            
            # Update our extreme low if we go lower
            if bar_low < current_extreme_low:
                current_extreme_low = bar_low
                extreme_low_time = bar_time
            
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
                    # Record the upward swing
                    swings.append({
                        'type': 'high',
                        'swing_price': current_extreme_high,
                        'swing_time': extreme_high_time,
                        'move_size': upward_move,
                        'category': categorize_swing(upward_move),
                        'from_price': current_extreme_low,
                        'from_time': extreme_low_time,
                        'to_price': current_extreme_high,
                        'to_time': extreme_high_time,
                        'direction': 'up'
                    })
                    
                    # Now start measuring downward moves from this high
                    measuring_from = 'high'
                    current_extreme_low = bar_low
                    extreme_low_time = bar_time
        
        else:  # measuring_from == 'high'
            # We're tracking downward moves from the current extreme high
            
            # Update our extreme high if we go higher
            if bar_high > current_extreme_high:
                current_extreme_high = bar_high
                extreme_high_time = bar_time
            
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
                    # Record the downward swing
                    swings.append({
                        'type': 'low',
                        'swing_price': current_extreme_low,
                        'swing_time': extreme_low_time,
                        'move_size': downward_move,
                        'category': categorize_swing(downward_move),
                        'from_price': current_extreme_high,
                        'from_time': extreme_high_time,
                        'to_price': current_extreme_low,
                        'to_time': extreme_low_time,
                        'direction': 'down'
                    })
                    
                    # Now start measuring upward moves from this low
                    measuring_from = 'low'
                    current_extreme_high = bar_high
                    extreme_high_time = bar_time
    
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
            'total_swings': len(swings),
            'top_1_swing': top_3_swings[0] if len(top_3_swings) > 0 else 0,
            'top_2_swing': top_3_swings[1] if len(top_3_swings) > 1 else 0,
            'top_3_swing': top_3_swings[2] if len(top_3_swings) > 2 else 0,
            'swings_30_60': category_counts['30-60'],
            'swings_60_100': category_counts['60-100'],
            'swings_100_150': category_counts['100-150'],
            'swings_150_200': category_counts['150-200'],
            'swings_200_plus': category_counts['200+'],
            'all_swings': swings
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
                display_summary = summary_df.drop('all_swings', axis=1).copy()
                
                # Format datetime columns for display
                datetime_cols = ['session_start', 'session_end', 'daily_high_time', 'daily_low_time']
                for col in datetime_cols:
                    if col in display_summary.columns:
                        display_summary[col] = display_summary[col].dt.strftime('%Y-%m-%d %H:%M:%S')
                
                # Display results
                st.subheader("📊 Daily Analysis Results")
                st.dataframe(display_summary, use_container_width=True)
                
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
                
                # Summary statistics
                st.subheader("📈 Summary Statistics")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    avg_range = display_summary['daily_range'].mean()
                    st.metric("Avg Daily Range", f"{avg_range:.1f}")
                
                with col2:
                    avg_swings = display_summary['total_swings'].mean()
                    st.metric("Avg Swings/Day", f"{avg_swings:.1f}")
                
                with col3:
                    total_days = len(display_summary)
                    st.metric("Total Trading Days", total_days)
                
                with col4:
                    max_range = display_summary['daily_range'].max()
                    st.metric("Max Daily Range", f"{max_range:.1f}")
                
                # Range category distribution
                st.subheader("📊 Range Category Distribution")
                range_dist = display_summary['range_category'].value_counts()
                fig_range = go.Figure(data=[go.Bar(x=range_dist.index, y=range_dist.values)])
                fig_range.update_layout(title="Daily Range Categories", xaxis_title="Range Category", yaxis_title="Number of Days")
                st.plotly_chart(fig_range, use_container_width=True)
                
                # Swing category distribution
                st.subheader("🎯 Swing Category Analysis")
                swing_cols = ['swings_30_60', 'swings_60_100', 'swings_100_150', 'swings_150_200', 'swings_200_plus']
                swing_totals = display_summary[swing_cols].sum()
                swing_totals.index = ['30-60', '60-100', '100-150', '150-200', '200+']
                
                fig_swings = go.Figure(data=[go.Bar(x=swing_totals.index, y=swing_totals.values)])
                fig_swings.update_layout(title="Swing Size Distribution", xaxis_title="Swing Category", yaxis_title="Total Swings")
                st.plotly_chart(fig_swings, use_container_width=True)
                
                # Export options
                st.subheader("💾 Export Data")
                
                # Excel export
                excel_buffer = io.BytesIO()
                with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                    # Daily summary
                    export_summary = display_summary.copy()
                    export_summary.to_excel(writer, sheet_name='Daily Analysis', index=False)
                    
                    # Detailed swings
                    if detailed_swings:
                        detailed_df.to_excel(writer, sheet_name='Detailed Swings', index=False)
                
                excel_buffer.seek(0)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        label="💾 Download Excel Report",
                        data=excel_buffer.getvalue(),
                        file_name=f"swing_analysis_{dt.datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                
                with col2:
                    csv_data = display_summary.to_csv(index=False)
                    st.download_button(
                        label="💾 Download CSV Report",
                        data=csv_data,
                        file_name=f"swing_analysis_{dt.datetime.now().strftime('%Y%m%d_%H%M')}.csv",
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
                        swing_details = detailed_df[detailed_df['trading_day'] == dt.datetime.strptime(sample_date, '%Y-%m-%d').date()]
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
    """)
