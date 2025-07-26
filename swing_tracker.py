# Swing High/Low Target Tracker
# Detects swing highs/lows and tracks how far price moves after each swing point

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Target tracking configuration
TARGET_CONFIG = {
    30: {"emoji": "👍", "label": "Target 1"},
    60: {"emoji": "🎯", "label": "Target 2"}, 
    100: {"emoji": "💯", "label": "Target 3"},
    150: {"emoji": "🏆", "label": "Target 4"},
    200: {"emoji": "🚀", "label": "Target 5"}
}

DRAWDOWN_LIMIT = 25  # Points before stop
STOP_CONFIG = {"emoji": "🚫", "label": "Stopped"}

def detect_swing_points(df, length=21):
    """
    Detect swing highs and lows using pivot point logic
    Similar to ta.pivothigh() and ta.pivotlow() in Pine Script
    """
    highs = df['High'].values
    lows = df['Low'].values
    
    swing_highs = []
    swing_lows = []
    
    for i in range(length, len(df) - length):
        # Check for swing high
        is_high = True
        current_high = highs[i]
        for j in range(i - length, i + length + 1):
            if j != i and highs[j] >= current_high:
                is_high = False
                break
        
        if is_high:
            swing_highs.append((i, current_high))
        
        # Check for swing low  
        is_low = True
        current_low = lows[i]
        for j in range(i - length, i + length + 1):
            if j != i and lows[j] <= current_low:
                is_low = False
                break
                
        if is_low:
            swing_lows.append((i, current_low))
    
    return swing_highs, swing_lows

def classify_swing_type(current_price, previous_price, swing_type):
    """Classify swing as HH, LH, HL, or LL"""
    if swing_type == 'high':
        return 'HH' if current_price > previous_price else 'LH'
    else:  # swing_type == 'low'
        return 'LL' if current_price < previous_price else 'HL'

def track_targets_after_swing(df, swing_idx, swing_price, swing_type):
    """
    Track how far price moves after a swing point
    Returns the highest target achieved and whether stopped out
    """
    if swing_idx >= len(df) - 1:
        return None, None, None
    
    # Get price data after the swing point
    future_data = df.iloc[swing_idx + 1:].copy()
    if len(future_data) == 0:
        return None, None, None
    
    targets_hit = []
    stopped_out = False
    max_favorable = 0
    max_adverse = 0
    
    for i, row in future_data.iterrows():
        high_price = row['High']
        low_price = row['Low']
        
        if swing_type == 'high':
            # For swing highs, we expect price to move down
            # Favorable move = price going down
            # Adverse move = price going up (drawdown)
            
            favorable_move = swing_price - low_price
            adverse_move = high_price - swing_price
            
            max_favorable = max(max_favorable, favorable_move)
            max_adverse = max(max_adverse, adverse_move)
            
            # Check for targets (price moving down from swing high)
            for target_points in sorted(TARGET_CONFIG.keys()):
                if favorable_move >= target_points and target_points not in targets_hit:
                    targets_hit.append(target_points)
            
            # Check for stop (price moving up too much)
            if adverse_move >= DRAWDOWN_LIMIT:
                stopped_out = True
                break
                
        else:  # swing_type == 'low'
            # For swing lows, we expect price to move up
            # Favorable move = price going up
            # Adverse move = price going down (drawdown)
            
            favorable_move = high_price - swing_price
            adverse_move = swing_price - low_price
            
            max_favorable = max(max_favorable, favorable_move)
            max_adverse = max(max_adverse, adverse_move)
            
            # Check for targets (price moving up from swing low)
            for target_points in sorted(TARGET_CONFIG.keys()):
                if favorable_move >= target_points and target_points not in targets_hit:
                    targets_hit.append(target_points)
            
            # Check for stop (price moving down too much)
            if adverse_move >= DRAWDOWN_LIMIT:
                stopped_out = True
                break
    
    # Determine final result
    if stopped_out and not targets_hit:
        result = {"emoji": STOP_CONFIG["emoji"], "label": STOP_CONFIG["label"]}
        highest_target = 0
    elif targets_hit:
        highest_target = max(targets_hit)
        result = {
            "emoji": TARGET_CONFIG[highest_target]["emoji"],
            "label": TARGET_CONFIG[highest_target]["label"]
        }
    else:
        result = {"emoji": "⏳", "label": "Pending"}
        highest_target = 0
    
    return result, max_favorable, max_adverse

def analyze_swing_performance(df, length=21):
    """Main function to analyze swing points and their performance"""
    swing_highs, swing_lows = detect_swing_points(df, length)
    
    results = []
    
    # Track previous swing values for HH/LH/HL/LL classification
    prev_high = None
    prev_low = None
    
    # Process swing highs
    for idx, price in swing_highs:
        swing_classification = classify_swing_type(price, prev_high, 'high') if prev_high else 'H'
        
        # Track performance after this swing high
        performance, max_favorable, max_adverse = track_targets_after_swing(df, idx, price, 'high')
        
        results.append({
            'Type': 'Swing High',
            'Index': idx,
            'Timestamp': df.iloc[idx]['Timestamp'] if 'Timestamp' in df.columns else idx,
            'Price': price,
            'Classification': swing_classification,
            'Performance': performance,
            'Max_Favorable': max_favorable,
            'Max_Adverse': max_adverse,
            'Direction': 'Short' # Expecting price to go down from high
        })
        
        prev_high = price
    
    # Process swing lows
    for idx, price in swing_lows:
        swing_classification = classify_swing_type(price, prev_low, 'low') if prev_low else 'L'
        
        # Track performance after this swing low
        performance, max_favorable, max_adverse = track_targets_after_swing(df, idx, price, 'low')
        
        results.append({
            'Type': 'Swing Low',
            'Index': idx,
            'Timestamp': df.iloc[idx]['Timestamp'] if 'Timestamp' in df.columns else idx,
            'Price': price,
            'Classification': swing_classification,
            'Performance': performance,
            'Max_Favorable': max_favorable,
            'Max_Adverse': max_adverse,
            'Direction': 'Long'  # Expecting price to go up from low
        })
        
        prev_low = price
    
    # Sort by index to maintain chronological order
    results.sort(key=lambda x: x['Index'])
    
    return pd.DataFrame(results)

def create_swing_chart(df, swing_results):
    """Create interactive chart showing swing points and their performance"""
    
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.8, 0.2],
        subplot_titles=('Price Chart with Swing Points', 'Performance Summary'),
        vertical_spacing=0.1
    )
    
    # Main price chart
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'], 
            low=df['Low'],
            close=df['Close'],
            name='Price'
        ),
        row=1, col=1
    )
    
    # Add swing points
    for _, swing in swing_results.iterrows():
        if swing['Performance'] and isinstance(swing['Performance'], dict):
            emoji = swing['Performance']['emoji']
            label = swing['Performance']['label']
            
            color = 'red' if swing['Type'] == 'Swing High' else 'green'
            
            fig.add_trace(
                go.Scatter(
                    x=[swing['Index']],
                    y=[swing['Price']],
                    mode='markers+text',
                    marker=dict(size=12, color=color),
                    text=f"{swing['Classification']}<br>{emoji} {label}",
                    textposition='top center' if swing['Type'] == 'Swing High' else 'bottom center',
                    name=f"{swing['Type']} - {label}",
                    showlegend=False
                ),
                row=1, col=1
            )
    
    fig.update_layout(
        title='Swing High/Low Target Tracker',
        xaxis_title='Time',
        yaxis_title='Price',
        height=800
    )
    
    return fig

def show_performance_summary(swing_results):
    """Display performance summary statistics"""
    
    st.subheader("📊 Performance Summary")
    
    if swing_results.empty:
        st.info("No swing points detected. Try adjusting the length parameter.")
        return
    
    # Filter out pending results for statistics
    completed_results = swing_results[
        swing_results['Performance'].apply(
            lambda x: isinstance(x, dict) and x['label'] != 'Pending'
        )
    ].copy()
    
    if completed_results.empty:
        st.info("No completed swing point outcomes yet.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Swing Highs Performance")
        highs = completed_results[completed_results['Type'] == 'Swing High']
        if not highs.empty:
            for _, row in highs.iterrows():
                perf = row['Performance']
                st.markdown(f"• {perf['emoji']} {perf['label']} - {row['Max_Favorable']:.1f} pts")
    
    with col2:
        st.markdown("#### Swing Lows Performance")
        lows = completed_results[completed_results['Type'] == 'Swing Low']
        if not lows.empty:
            for _, row in lows.iterrows():
                perf = row['Performance']
                st.markdown(f"• {perf['emoji']} {perf['label']} - {row['Max_Favorable']:.1f} pts")
    
    # Overall statistics
    st.markdown("#### Overall Statistics")
    
    target_counts = {}
    for target in TARGET_CONFIG.keys():
        count = completed_results[
            completed_results['Performance'].apply(
                lambda x: target in x['label'] if isinstance(x, dict) else False
            )
        ].shape[0]
        target_counts[target] = count
    
    stopped_count = completed_results[
        completed_results['Performance'].apply(
            lambda x: x['label'] == 'Stopped' if isinstance(x, dict) else False
        )
    ].shape[0]
    
    cols = st.columns(len(TARGET_CONFIG) + 1)
    
    for i, (target, config) in enumerate(TARGET_CONFIG.items()):
        with cols[i]:
            st.metric(f"{config['emoji']} {config['label']}", target_counts[target])
    
    with cols[-1]:
        st.metric(f"{STOP_CONFIG['emoji']} {STOP_CONFIG['label']}", stopped_count)

def run_swing_tracker(df):
    """Main function to run swing tracking analysis"""
    
    st.title("🎯 Swing High/Low Target Tracker")
    
    st.markdown("""
    This tool detects swing highs and lows in price data and tracks how far price moves after each swing point.
    
    **Target System:**
    - 👍 Target 1: 30 points
    - 🎯 Target 2: 60 points  
    - 💯 Target 3: 100 points
    - 🏆 Target 4: 150 points
    - 🚀 Target 5: 200+ points
    - 🚫 Stopped: 25 point drawdown
    """)
    
    # Parameters
    st.sidebar.subheader("Parameters")
    length = st.sidebar.slider("Swing Detection Length", 5, 50, 21)
    
    # Analyze swings
    swing_results = analyze_swing_performance(df, length)
    
    if not swing_results.empty:
        # Show chart
        chart = create_swing_chart(df, swing_results)
        st.plotly_chart(chart, use_container_width=True)
        
        # Show performance summary
        show_performance_summary(swing_results)
        
        # Show detailed results
        st.subheader("📋 Detailed Results")
        
        # Format the results for display
        display_results = swing_results.copy()
        display_results['Performance_Text'] = display_results['Performance'].apply(
            lambda x: f"{x['emoji']} {x['label']}" if isinstance(x, dict) else "⏳ Pending"
        )
        
        display_cols = ['Type', 'Classification', 'Price', 'Performance_Text', 'Max_Favorable', 'Max_Adverse']
        st.dataframe(display_results[display_cols].rename(columns={
            'Performance_Text': 'Result',
            'Max_Favorable': 'Max Favorable (pts)',
            'Max_Adverse': 'Max Drawdown (pts)'
        }))
        
    else:
        st.warning("No swing points detected. Try adjusting the length parameter or check your data.")
    
    return swing_results

# Sample usage function for testing
def create_sample_data():
    """Create sample OHLC data for testing"""
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=200, freq='D')
    
    # Generate sample price data with trends and reversals
    base_price = 100
    prices = [base_price]
    
    for i in range(199):
        change = np.random.normal(0, 2)
        # Add some trend bias occasionally
        if i % 30 == 0:
            change += np.random.choice([-5, 5])
        prices.append(max(10, prices[-1] + change))
    
    # Create OHLC data
    df = pd.DataFrame({
        'Timestamp': dates,
        'Open': prices,
        'High': [p + abs(np.random.normal(0, 1)) for p in prices],
        'Low': [p - abs(np.random.normal(0, 1)) for p in prices],
        'Close': [p + np.random.normal(0, 0.5) for p in prices]
    })
    
    return df

if __name__ == "__main__":
    # For testing with sample data
    sample_df = create_sample_data()
    run_swing_tracker(sample_df)
