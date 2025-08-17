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

# File upload
uploaded_file = st.file_uploader("Upload OHLC CSV file", type=['csv'])

def parse_timestamp(timestamp_str):
    """Parse various timestamp formats"""
    try:
        # Handle ISO format with timezone
        if 'T' in timestamp_str and ('-' in timestamp_str[-6:] or '+' in timestamp_str[-6:]):
            return pd.to_datetime(timestamp_str, format='%Y-%m-%dT%H:%M:%S%z', utc=True)
        else:
            return pd.to_datetime(timestamp_str, infer_datetime_format=True)
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

def detect_swings(df, swing_threshold=30, drawdown_limit=25):
    """
    Detect swings that move >= swing_threshold before drawdown_limit occurs
    """
    swings = []
    
    if len(df) == 0:
        return swings
    
    # Initialize state
    current_swing_high = df.iloc[0]['high']
    current_swing_low = df.iloc[0]['low'] 
    swing_high_time = df.iloc[0]['time']
    swing_low_time = df.iloc[0]['time']
    looking_for_high = True
    
    for idx, row in df.iterrows():
        if looking_for_high:
            # Looking for swing high
            if row['high'] > current_swing_high:
                current_swing_high = row['high']
                swing_high_time = row['time']
            elif current_swing_high - row['low'] >= drawdown_limit:
                # Found potential swing high
                move_size = current_swing_high - current_swing_low
                if move_size >= swing_threshold:
                    swings.append({
                        'type': 'high',
                        'price': current_swing_high,
                        'time': swing_high_time,
                        'move_size': move_size,
                        'category': categorize_swing(move_size),
                        'start_price': current_swing_low,
                        'start_time': swing_low_time
                    })
                
                # Switch to looking for low
                current_swing_low = row['low']
                swing_low_time = row['time']
                looking_for_high = False
        else:
            # Looking for swing low
            if row['low'] < current_swing_low:
                current_swing_low = row['low']
                swing_low_time = row['time']
            elif row['high'] - current_swing_low >= drawdown_limit:
                # Found potential swing low
                move_size = current_swing_high - current_swing_low
                if move_size >= swing_threshold:
                    swings.append({
                        'type': 'low',
                        'price': current_swing_low,
                        'time': swing_low_time,
                        'move_size': move_size,
                        'category': categorize_swing(move_size),
                        'start_price': current_swing_high,
                        'start_time': swing_high_time
                    })
                
                # Switch to looking for high
                current_swing_high = row['high']
                swing_high_time = row['time']
                looking_for_high = True
    
    return swings

def analyze_daily_data(df):
    """Analyze daily market structure"""
    daily_stats = []
    
    # Group by date
    df['date'] = df['time'].dt.date
    
    for date, day_data in df.groupby('date'):
        # Basic daily stats
        daily_high = day_data['high'].max()
        daily_low = day_data['low'].min()
        daily_range = daily_high - daily_low
        
        # Find exact times
        high_time = day_data[day_data['high'] == daily_high]['time'].iloc[0]
        low_time = day_data[day_data['low'] == daily_low]['time'].iloc[0]
        
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
            'date': date,
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
        # Read the CSV file
        df = pd.read_csv(uploaded_file)
        
        # Display file info
        st.success(f"✅ File loaded: {len(df)} rows")
        
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
        param_col1, param_col2 = st.columns(2)
        with param_col1:
            swing_threshold = st.number_input("Swing Threshold (units)", min_value=1, value=30)
        with param_col2:
            drawdown_limit = st.number_input("Drawdown Limit (units)", min_value=1, value=25)
        
        if st.button("🚀 Run Analysis"):
            # Standardize column names
            analysis_df = df[[time_col, open_col, high_col, low_col, close_col]].copy()
            analysis_df.columns = ['time', 'open', 'high', 'low', 'close']
            
            # Parse timestamps
            analysis_df['time'] = analysis_df['time'].apply(parse_timestamp)
            analysis_df = analysis_df.dropna(subset=['time']).sort_values('time')
            
            # Convert price columns to numeric
            for col in ['open', 'high', 'low', 'close']:
                analysis_df[col] = pd.to_numeric(analysis_df[col], errors='coerce')
            
            # Run daily analysis
            with st.spinner("Analyzing market structure..."):
                daily_stats = analyze_daily_data(analysis_df)
            
            if daily_stats:
                st.success(f"✅ Analysis complete! Found data for {len(daily_stats)} days")
                
                # Create summary DataFrame
                summary_df = pd.DataFrame(daily_stats)
                summary_df = summary_df.drop('all_swings', axis=1)  # Remove detailed swings for display
                
                # Display results
                st.subheader("📊 Daily Analysis Results")
                st.dataframe(summary_df, use_container_width=True)
                
                # Summary statistics
                st.subheader("📈 Summary Statistics")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    avg_range = summary_df['daily_range'].mean()
                    st.metric("Avg Daily Range", f"{avg_range:.1f}")
                
                with col2:
                    avg_swings = summary_df['total_swings'].mean()
                    st.metric("Avg Swings/Day", f"{avg_swings:.1f}")
                
                with col3:
                    total_days = len(summary_df)
                    st.metric("Total Days", total_days)
                
                with col4:
                    max_range = summary_df['daily_range'].max()
                    st.metric("Max Daily Range", f"{max_range:.1f}")
                
                # Range category distribution
                st.subheader("📊 Range Category Distribution")
                range_dist = summary_df['range_category'].value_counts()
                fig_range = go.Figure(data=[go.Bar(x=range_dist.index, y=range_dist.values)])
                fig_range.update_layout(title="Daily Range Categories", xaxis_title="Range Category", yaxis_title="Number of Days")
                st.plotly_chart(fig_range, use_container_width=True)
                
                # Swing category distribution
                st.subheader("🎯 Swing Category Analysis")
                swing_cols = ['swings_30_60', 'swings_60_100', 'swings_100_150', 'swings_150_200', 'swings_200_plus']
                swing_totals = summary_df[swing_cols].sum()
                swing_totals.index = ['30-60', '60-100', '100-150', '150-200', '200+']
                
                fig_swings = go.Figure(data=[go.Bar(x=swing_totals.index, y=swing_totals.values)])
                fig_swings.update_layout(title="Swing Size Distribution", xaxis_title="Swing Category", yaxis_title="Total Swings")
                st.plotly_chart(fig_swings, use_container_width=True)
                
                # Export options
                st.subheader("📥 Export Data")
                
                # Prepare export data
                export_df = summary_df.copy()
                export_df['daily_high_time'] = export_df['daily_high_time'].dt.strftime('%Y-%m-%d %H:%M:%S')
                export_df['daily_low_time'] = export_df['daily_low_time'].dt.strftime('%Y-%m-%d %H:%M:%S')
                
                # Excel export
                excel_buffer = io.BytesIO()
                with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                    export_df.to_excel(writer, sheet_name='Daily Analysis', index=False)
                    
                    # Add detailed swings sheet
                    detailed_swings = []
                    for day_stat in daily_stats:
                        for swing in day_stat['all_swings']:
                            detailed_swings.append({
                                'date': day_stat['date'],
                                'swing_type': swing['type'],
                                'swing_price': swing['price'],
                                'swing_time': swing['time'].strftime('%Y-%m-%d %H:%M:%S'),
                                'move_size': swing['move_size'],
                                'category': swing['category'],
                                'start_price': swing['start_price'],
                                'start_time': swing['start_time'].strftime('%Y-%m-%d %H:%M:%S')
                            })
                    
                    if detailed_swings:
                        detailed_df = pd.DataFrame(detailed_swings)
                        detailed_df.to_excel(writer, sheet_name='Detailed Swings', index=False)
                
                excel_buffer.seek(0)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        label="📥 Download Excel Report",
                        data=excel_buffer.getvalue(),
                        file_name=f"swing_analysis_{dt.datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                
                with col2:
                    csv_data = export_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download CSV Report",
                        data=csv_data,
                        file_name=f"swing_analysis_{dt.datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv"
                    )
                
                # Sample visualization
                if len(daily_stats) > 0 and st.checkbox("📈 Show Sample Day Visualization"):
                    sample_date = st.selectbox("Select Date to Visualize", [str(d['date']) for d in daily_stats])
                    sample_day = next(d for d in daily_stats if str(d['date']) == sample_date)
                    
                    # Get the day's data
                    day_data = analysis_df[analysis_df['time'].dt.date == sample_day['date']]
                    
                    # Create candlestick chart
                    fig = make_subplots(rows=1, cols=1, subplot_titles=[f"Market Structure - {sample_date}"])
                    
                    # Add candlestick
                    fig.add_trace(go.Candlestick(
                        x=day_data['time'],
                        open=day_data['open'],
                        high=day_data['high'],
                        low=day_data['low'],
                        close=day_data['close'],
                        name="OHLC"
                    ))
                    
                    # Add swing points
                    for swing in sample_day['all_swings']:
                        color = 'red' if swing['type'] == 'high' else 'green'
                        fig.add_trace(go.Scatter(
                            x=[swing['time']],
                            y=[swing['price']],
                            mode='markers',
                            marker=dict(color=color, size=10, symbol='diamond'),
                            name=f"{swing['category']} swing",
                            showlegend=False
                        ))
                    
                    fig.update_layout(
                        title=f"Daily Range: {sample_day['daily_range']:.1f} ({sample_day['range_category']}) | Swings: {sample_day['total_swings']}",
                        xaxis_title="Time",
                        yaxis_title="Price",
                        height=600
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            else:
                st.error("No data found to analyze")
    
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        st.text("Please ensure your CSV has the correct format with time, open, high, low, close columns")

else:
    st.info("👆 Upload a CSV file with OHLC data to begin analysis")
    
    # Show expected format
    st.subheader("📋 Expected CSV Format")
    sample_data = {
        'time': ['2025-06-15T18:00:00-04:00', '2025-06-15T18:15:00-04:00', '2025-06-15T18:30:00-04:00'],
        'open': [21784, 21821.25, 21835],
        'high': [21850.75, 21842, 21851.50],
        'low': [21722, 21815, 21798.25],
        'close': [21821.25, 21835, 21834.75]
    }
    st.table(pd.DataFrame(sample_data))
