"""
Pattern Scanner Streamlit App V04
Enhanced with:
- Multiple feed support (big and small feeds)
- Auto-populated start/end times from traveler data
- Open price display per feed
- Enhanced lookforward analysis with turn vs exit detection
- Chronological zone reporting
- Distance tracking from zones
- Processing timer
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os
import time

# Add paths
sys.path.append('/home/claude')
sys.path.append('/mnt/user-data/uploads')

from pattern_scanner_v03 import HaydnPatternScanner, MODELS, FOGZ, WEIGHTS
from nested_swing_detector import analyze_swings, calculate_zone_distances

st.set_page_config(
    page_title="Pattern Scanner V04",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .zone-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #1f77b4;
    }
    .pattern-header {
        font-size: 1.3rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 1rem;
    }
    .success-rate {
        font-size: 1.1rem;
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin: 0.5rem 0;
    }
    .high-success {
        background-color: #d4edda;
        color: #155724;
    }
    .medium-success {
        background-color: #fff3cd;
        color: #856404;
    }
    .low-success {
        background-color: #f8d7da;
        color: #721c24;
    }
    .timer-box {
        background-color: #e8f4f8;
        padding: 0.5rem;
        border-radius: 0.3rem;
        font-weight: bold;
        color: #0066cc;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'scanner' not in st.session_state:
    st.session_state.scanner = None
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'processing_time' not in st.session_state:
    st.session_state.processing_time = None
if 'ohlc_feeds' not in st.session_state:
    st.session_state.ohlc_feeds = {}


def get_most_recent_day0_time(traveler_df: pd.DataFrame) -> str:
    """Extract most recent Day [0] arrival time from traveler report"""
    try:
        day0_df = traveler_df[traveler_df['Day'] == '[0]']
        if len(day0_df) > 0:
            most_recent = pd.to_datetime(day0_df['Arrival']).max()
            # Ensure timezone-naive for consistency
            if hasattr(most_recent, 'tz') and most_recent.tz is not None:
                most_recent = most_recent.tz_localize(None)
            return most_recent.strftime('%Y-%m-%d %H:%M:%S')
    except:
        pass
    return "2025-12-08 23:00:00"


def get_end_time_for_start(start_time_str: str) -> str:
    """
    Calculate end time (16:45 of trading day) based on start time.
    
    Rules:
    - If start is before 16:45 same day → end is 16:45 same day
    - If start is after 16:45 → end is 16:45 next day
    """
    try:
        start_dt = pd.to_datetime(start_time_str)
        
        # Ensure timezone-naive
        if hasattr(start_dt, 'tz') and start_dt.tz is not None:
            start_dt = start_dt.tz_localize(None)
        
        # Trading day ends at 16:45
        end_time_of_day = start_dt.replace(hour=16, minute=45, second=0)
        
        # If start is after 16:45, move to next day's 16:45
        if start_dt.time() > datetime.strptime("16:45", "%H:%M").time():
            end_time_of_day = end_time_of_day + timedelta(days=1)
        
        return end_time_of_day.strftime('%Y-%m-%d %H:%M:%S')
    except:
        return "2025-12-09 16:45:00"


def get_open_prices_at_time(ohlc_feeds: dict, target_time: str) -> dict:
    """Get open price for each feed at or near target time"""
    open_prices = {}
    
    try:
        target_dt = pd.to_datetime(target_time)
        
        # Ensure target is timezone-naive
        if hasattr(target_dt, 'tz') and target_dt.tz is not None:
            target_dt = target_dt.tz_localize(None)
        
        for feed_name, df in ohlc_feeds.items():
            try:
                df_copy = df.copy()
                
                # Debug: Check what we have
                if 'time' not in df_copy.columns:
                    st.error(f"{feed_name}: 'time' column not found. Available columns: {list(df_copy.columns)}")
                    continue
                
                # Convert time column to datetime properly
                if df_copy['time'].dtype == 'object' or df_copy['time'].dtype.name == 'object':
                    df_copy['time'] = pd.to_datetime(df_copy['time'])
                elif not pd.api.types.is_datetime64_any_dtype(df_copy['time']):
                    df_copy['time'] = pd.to_datetime(df_copy['time'])
                
                # Ensure times are timezone-naive for comparison
                if hasattr(df_copy['time'].dtype, 'tz') and df_copy['time'].dtype.tz is not None:
                    df_copy['time'] = df_copy['time'].dt.tz_localize(None)
                
                # Calculate time differences
                df_copy['time_diff'] = (df_copy['time'] - target_dt).abs()
                
                # Find closest bar to target time
                closest_idx = df_copy['time_diff'].idxmin()
                
                open_prices[feed_name] = {
                    'price': float(df_copy.loc[closest_idx, 'open']),
                    'time': pd.to_datetime(df_copy.loc[closest_idx, 'time']).strftime('%Y-%m-%d %H:%M:%S')
                }
            except Exception as e:
                st.warning(f"Could not get open price for {feed_name}: {e}")
                import traceback
                st.code(f"Debug trace:\n{traceback.format_exc()}")
                continue
        
    except Exception as e:
        st.error(f"Error processing open prices: {e}")
    
    return open_prices


def ensure_outputs_directory():
    """Create outputs directory if it doesn't exist"""
    output_dir = '/mnt/user-data/outputs'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir


def main():
    st.markdown('<div class="main-header">🎯 Pattern Scanner V04</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # File uploads
        st.subheader("📤 Data Upload")
        
        traveler_file = st.file_uploader(
            "Traveler Report (Excel)",
            type=['xlsx'],
            help="Upload traveler report Excel file"
        )
        
        st.write("**OHLC Data (CSV):**")
        st.caption("Upload one or more feed CSVs")
        
        # Multiple feed uploads
        big_feed_file = st.file_uploader(
            "Big Feed (NQ 3m/5m/15m)",
            type=['csv'],
            help="Upload NQ OHLC data",
            key="big_feed"
        )
        
        small_feed_file = st.file_uploader(
            "Small Feed (ES/YM/RTY)",
            type=['csv'],
            help="Upload ES/YM/RTY OHLC data",
            key="small_feed"
        )
        
        # Sheet selection
        sheet_name = None
        traveler_df = None
        if traveler_file:
            try:
                xl = pd.ExcelFile(traveler_file)
                sheets = xl.sheet_names
                sheet_name = st.selectbox("Select Sheet", sheets)
                
                # Load traveler data for auto-population
                if sheet_name:
                    traveler_df = pd.read_excel(traveler_file, sheet_name=sheet_name)
                    st.success(f"✅ Loaded {len(traveler_df)} arrivals")
            except Exception as e:
                st.error(f"Error reading Excel: {e}")
        
        st.divider()
        
        # Analysis parameters
        st.subheader("🎛️ Analysis Parameters")
        
        zone_mode = st.radio(
            "Zone Selection Mode",
            ["Automatic (Swing Detection)", "Manual Zone Entry"],
            help="Automatic uses swing detector, Manual lets you specify zones"
        )
        
        if zone_mode == "Automatic (Swing Detection)":
            st.write("**Swing Detection Settings:**")
            min_swing_size = st.slider("Min Swing Size (points)", 30, 150, 60, 10)
            
            zone_width = st.slider(
                "Zone Width (points)", 
                5.0, 30.0, 10.0, 2.5,
                help="Width of the zone around each detected swing point. "
                     "A 10-point zone means ±5 points from center price. "
                     "Travelers within this range are included in pattern analysis."
            )
            
            # Time range with auto-population
            st.write("**Time Range:**")
            
            # Auto-populate start time from most recent Day [0]
            default_start = get_most_recent_day0_time(traveler_df) if traveler_df is not None else "2025-12-08 23:00:00"
            start_time = st.text_input(
                "Start Time (YYYY-MM-DD HH:MM:SS)", 
                default_start,
                help="Auto-populated from most recent Day [0] arrival in traveler report"
            )
            
            # Auto-populate end time to 16:45 of trading day
            default_end = get_end_time_for_start(start_time)
            end_time = st.text_input(
                "End Time (YYYY-MM-DD HH:MM:SS)", 
                default_end,
                help="Auto-populated to 16:45 of the trading day"
            )
            
        else:
            st.write("**Manual Zone Entry:**")
            center_price = st.number_input("Center Price", min_value=0.0, value=25721.25, step=0.25)
            
            zone_width = st.slider(
                "Zone Width (points)", 
                5.0, 50.0, 18.0, 2.5,
                help="Width of the zone. A 10-point zone means ±5 points from center."
            )
        
        match_tolerance = st.slider(
            "Match Tolerance (points)", 
            0.5, 5.0, 3.0, 0.5,
            help="Maximum output spread between travelers to consider them matching"
        )
        
        st.divider()
        
        # Success analysis settings
        st.subheader("📈 Success Analysis")
        
        enable_success = st.checkbox("Enable Lookforward Analysis", value=True,
                                     help="Track zone performance using nested swing detector")
        
        if enable_success:
            lookforward_hours = st.slider(
                "Lookforward Window (hours)", 
                1, 48, 18, 1,
                help="Hours to look forward from zone time to track price action"
            )
            
            success_threshold = st.slider(
                "Success Threshold (points)", 
                10, 100, 30, 5,
                help="Minimum move toward zone to count as 'success'. "
                     "For High zones: price must rally up to zone then reverse down this many points. "
                     "For Low zones: price must drop to zone then bounce up this many points."
            )
        
        # Run button
        st.divider()
        run_analysis = st.button("🚀 Run Analysis", type="primary", use_container_width=True)
    
    # Main content area
    if not traveler_file:
        st.info("👈 Upload traveler report and OHLC data to begin")
        
        # Show quick guide
        with st.expander("📖 Quick Start Guide", expanded=True):
            st.markdown("""
            ### How to Use Pattern Scanner V04
            
            **1. Upload Files:**
            - Traveler Report (Excel) - required
            - Big Feed CSV (NQ 3m/5m/15m) - optional
            - Small Feed CSV (ES/YM/RTY) - optional
            
            **2. Configure Analysis:**
            - **Zone Width**: The ±points around each swing point to search for patterns
            - **Match Tolerance**: Maximum spread between traveler outputs to match
            - **Success Threshold**: Minimum reversal size to count as success
            
            **3. Understanding Lookforward Mode:**
            - Zones are reported in **chronological order** (time sequence)
            - Uses **nested swing detector** to identify turns vs exits
            - Calculates **distance moved** from each zone
            - Shows **turn-to-turn movements** in price action
            
            **4. Features:**
            - 12+ Pattern Types detected
            - 24 Models (23 existing + Model 24)
            - Multi-feed support
            - Auto-populated times from your data
            - Processing timer
            - Excel export
            """)
        
        # Show current configuration
        st.markdown("### 🔧 Current Settings")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Default Zone Width", "10.0 points")
            st.metric("Default Match Tolerance", "3.0 points")
        with col2:
            st.metric("Default Lookforward", "18 hours")
            st.metric("Default Success Threshold", "30 points")
        
        return
    
    if run_analysis and traveler_file and sheet_name:
        # Start timer
        start_time_processing = time.time()
        
        with st.spinner("Loading data..."):
            try:
                # Load traveler data
                traveler_df = pd.read_excel(traveler_file, sheet_name=sheet_name)
                
                # Load OHLC feeds
                ohlc_feeds = {}
                combined_ohlc = None
                
                if big_feed_file:
                    big_df = pd.read_csv(big_feed_file)
                    ohlc_feeds['Big Feed (NQ)'] = big_df
                    combined_ohlc = big_df.copy()
                
                if small_feed_file:
                    small_df = pd.read_csv(small_feed_file)
                    ohlc_feeds['Small Feed (ES/YM/RTY)'] = small_df
                    if combined_ohlc is None:
                        combined_ohlc = small_df.copy()
                    else:
                        combined_ohlc = pd.concat([combined_ohlc, small_df], ignore_index=True)
                
                st.session_state.ohlc_feeds = ohlc_feeds
                
                # Initialize scanner
                scanner = HaydnPatternScanner(traveler_df, combined_ohlc)
                st.session_state.scanner = scanner
                
                st.success(f"✅ Loaded {len(traveler_df)} traveler arrivals")
                if combined_ohlc is not None:
                    st.success(f"✅ Loaded {len(combined_ohlc)} OHLC bars from {len(ohlc_feeds)} feed(s)")
                
            except Exception as e:
                st.error(f"Error loading data: {e}")
                import traceback
                st.code(traceback.format_exc())
                return
        
        # Display open prices at analysis time
        if zone_mode == "Automatic (Swing Detection)" and ohlc_feeds:
            open_prices = get_open_prices_at_time(ohlc_feeds, start_time)
            
            if len(open_prices) > 0:
                st.markdown("### 💵 Open Prices at Analysis Time")
                
                cols = st.columns(len(open_prices))
                for idx, (feed_name, data) in enumerate(open_prices.items()):
                    with cols[idx]:
                        st.metric(
                            feed_name,
                            f"${data['price']:.2f}",
                            f"at {data['time']}"
                        )
        
        # Run analysis
        with st.spinner("Analyzing patterns..."):
            try:
                if zone_mode == "Automatic (Swing Detection)":
                    # Auto-detect zones
                    zones_df = scanner.scan_all_zones(
                        start_time=start_time,
                        end_time=end_time,
                        min_swing_size=min_swing_size,
                        zone_width=zone_width
                    )
                    
                    if len(zones_df) == 0:
                        st.warning("No zones detected by swing detector")
                        return
                    
                    # Lookforward analysis if enabled
                    if enable_success and combined_ohlc is not None:
                        zones_df = add_lookforward_analysis(
                            zones_df,
                            combined_ohlc,
                            lookforward_hours,
                            success_threshold,
                            min_swing_size
                        )
                        
                        # Sort chronologically for lookforward mode
                        zones_df = zones_df.sort_values('zone_time').reset_index(drop=True)
                        zones_df['chronological_rank'] = range(1, len(zones_df) + 1)
                    
                    st.session_state.analysis_results = zones_df
                    
                else:
                    # Manual zone
                    analysis = scanner.analyze_zone(
                        center_price=center_price,
                        zone_width=zone_width,
                        match_tolerance=match_tolerance
                    )
                    
                    if 'error' in analysis:
                        st.error(analysis['error'])
                        return
                    
                    # Convert to DataFrame
                    zones_df = pd.DataFrame([analysis])
                    zones_df['zone_time'] = datetime.now()
                    zones_df['zone_type'] = 'Manual'
                    zones_df['zone_subtype'] = 'Manual'
                    zones_df['rank'] = 1
                    
                    st.session_state.analysis_results = zones_df
                
                # Stop timer
                end_time_processing = time.time()
                processing_time = end_time_processing - start_time_processing
                st.session_state.processing_time = processing_time
                
                # Display processing time
                st.markdown(f'<div class="timer-box">⏱️ Processing completed in {processing_time:.2f} seconds</div>', 
                           unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Error analyzing: {e}")
                import traceback
                st.code(traceback.format_exc())
                return
    
    # Display results
    if st.session_state.analysis_results is not None:
        zones_df = st.session_state.analysis_results
        
        st.markdown("---")
        st.markdown(f"### 📊 Analysis Results ({len(zones_df)} zones)")
        
        # Show summary stats
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Zones", len(zones_df))
        
        with col2:
            avg_score = zones_df['score'].mean()
            st.metric("Avg Score", f"{avg_score:.0f}")
        
        with col3:
            if 'touched_zone' in zones_df.columns:
                touched_pct = (zones_df['touched_zone'].sum() / len(zones_df) * 100)
                st.metric("Touched Rate", f"{touched_pct:.1f}%")
        
        with col4:
            if 'reversal_confirmed' in zones_df.columns:
                success_pct = (zones_df['reversal_confirmed'].sum() / len(zones_df) * 100)
                st.metric("Success Rate", f"{success_pct:.1f}%")
        
        # Export button
        if st.button("📥 Export to Excel"):
            export_results(zones_df)
        
        st.markdown("---")
        
        # Display each zone
        for idx, zone in zones_df.iterrows():
            display_zone_analysis(zone, idx, zones_df)


def add_lookforward_analysis(zones_df: pd.DataFrame,
                             ohlc_df: pd.DataFrame,
                             lookforward_hours: int,
                             success_threshold: float,
                             min_swing_size: float) -> pd.DataFrame:
    """
    Add lookforward analysis using nested swing detector
    
    Identifies:
    - Whether zone was touched
    - Turn vs exit detection
    - Distances moved from zone
    - Reversal confirmation
    """
    
    zones_df = zones_df.copy()
    
    # Prepare OHLC data
    ohlc_df = ohlc_df.copy()
    ohlc_df['time'] = pd.to_datetime(ohlc_df['time'])
    if hasattr(ohlc_df['time'].dtype, 'tz') and ohlc_df['time'].dtype.tz is not None:
        ohlc_df['time'] = ohlc_df['time'].dt.tz_localize(None)
    
    # Add analysis columns
    zones_df['touched_zone'] = False
    zones_df['num_turns'] = 0
    zones_df['max_distance_from_zone'] = 0.0
    zones_df['reversal_confirmed'] = False
    zones_df['early_exit_detected'] = False
    zones_df['turn_to_turn_distances'] = None
    
    for idx, zone in zones_df.iterrows():
        zone_time = pd.to_datetime(zone['zone_time']).replace(tzinfo=None)
        zone_price = zone['center_price']
        zone_type = zone['zone_subtype']  # 'High' or 'Low'
        
        # Run nested swing analysis
        end_time = zone_time + timedelta(hours=lookforward_hours)
        
        try:
            nested_swings, major_points = analyze_swings(
                ohlc_df,
                start_time=zone_time,
                end_time=end_time,
                min_swing_size=min_swing_size,
                pullback_tolerance=30
            )
            
            # Calculate distances from zone
            distances = calculate_zone_distances(
                zone_price=zone_price,
                major_points=major_points,
                zone_time=zone_time,
                ohlc_df=ohlc_df,
                lookforward_hours=lookforward_hours
            )
            
            zones_df.at[idx, 'touched_zone'] = distances['touched_zone']
            zones_df.at[idx, 'num_turns'] = distances['num_turns']
            zones_df.at[idx, 'max_distance_from_zone'] = distances['max_distance']
            zones_df.at[idx, 'early_exit_detected'] = distances['early_exit'] is not None
            zones_df.at[idx, 'turn_to_turn_distances'] = distances['distances']
            
            # Check for reversal confirmation
            if distances['touched_zone'] and distances['max_distance'] >= success_threshold:
                zones_df.at[idx, 'reversal_confirmed'] = True
            
        except Exception as e:
            st.warning(f"Lookforward analysis failed for zone at {zone_price}: {e}")
            continue
    
    return zones_df


def display_zone_analysis(zone, idx, zones_df):
    """Display comprehensive zone analysis"""
    
    # Zone header
    rank_col = 'chronological_rank' if 'chronological_rank' in zone.index else 'rank'
    rank_display = f"#{int(zone[rank_col])}" if rank_col in zone.index else f"#{idx+1}"
    
    zone_type_emoji = "🔴" if zone['zone_subtype'] == 'High' else "🟢"
    
    header = f"{zone_type_emoji} Zone {rank_display}: ${zone['center_price']:.2f}"
    
    if 'zone_time' in zone.index:
        zone_time_str = pd.to_datetime(zone['zone_time']).strftime('%Y-%m-%d %H:%M')
        header += f" @ {zone_time_str}"
    
    with st.expander(header, expanded=(idx < 5)):
        # Metrics row
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Score", f"{zone['score']:.0f}")
        
        with col2:
            st.metric("Arrivals", zone['num_arrivals'])
        
        with col3:
            if 'touched_zone' in zone.index:
                touched_emoji = "✅" if zone['touched_zone'] else "❌"
                st.metric("Touched", touched_emoji)
        
        with col4:
            if 'num_turns' in zone.index:
                st.metric("Turns", int(zone['num_turns']))
        
        with col5:
            if 'reversal_confirmed' in zone.index:
                success_emoji = "✅" if zone['reversal_confirmed'] else "❌"
                st.metric("Reversal", success_emoji)
        
        # Turn-to-turn distances
        if 'turn_to_turn_distances' in zone.index and zone['turn_to_turn_distances']:
            st.markdown("#### 📏 Turn-to-Turn Distances")
            
            distances = zone['turn_to_turn_distances']
            
            if distances and len(distances) > 0:
                for i, dist in enumerate(distances, 1):
                    reversal_icon = "🔄" if dist['is_reversal_to'] else "➡️"
                    
                    from_time = dist['from_time'].strftime('%H:%M')
                    to_time = dist['to_time'].strftime('%H:%M')
                    
                    st.write(
                        f"{reversal_icon} Turn {i}: "
                        f"{dist['from_type']} at ${dist['from_price']:.2f} ({from_time}) → "
                        f"{dist['to_type']} at ${dist['to_price']:.2f} ({to_time}) | "
                        f"**Distance: {dist['distance']:.1f} pts** | "
                        f"From zone: {dist['from_zone_distance']:.1f} pts"
                    )
            else:
                st.info("No significant turns detected in lookforward window")
        
        # Pattern details
        st.markdown("---")
        display_all_patterns_summary(zone['patterns'])
        
        # Show top patterns
        st.markdown("#### 🎯 Key Patterns")
        
        patterns = zone['patterns']
        
        # Epic same origin
        if patterns.get('epic_same_origin'):
            with st.expander(f"⭐ Epic Same Origin ({len(patterns['epic_same_origin'])} matches)", expanded=True):
                for match in patterns['epic_same_origin'][:10]:
                    st.write(f"{match['type']}: M#{match['m1']:+.0f} + M#{match['m2']:+.0f} | Spread: {match['spread']:.2f} | {match['feed']}")
        
        # X0 Sequential Descents
        if patterns.get('x0_sequential_descents'):
            with st.expander(f"📉 X0 Sequential Descents ({len(patterns['x0_sequential_descents'])})", expanded=True):
                for seq in patterns['x0_sequential_descents']:
                    st.write(f"Length: {seq['sequence_length']} | X0p: {seq['x0p_count']} | X0d: {seq['x0d_count']} | Crosses Zero: {seq['crosses_zero']}")
        
        # FOGZ Presence
        if patterns.get('fogz_presence'):
            with st.expander(f"🎯 FOGZ Presence ({len(patterns['fogz_presence'])})", expanded=True):
                for fogz in patterns['fogz_presence']:
                    zero_flag = "✨ M# 0" if fogz['is_zero'] else ""
                    epic_flag = "⭐ EPIC" if fogz['is_epic'] else ""
                    st.write(f"M#{fogz['m']:+.0f} from {fogz['origin']} at {fogz['output']:.2f} {zero_flag} {epic_flag}")
        
        # Constellations
        if patterns.get('constellations'):
            with st.expander(f"🌟 Constellations ({len(patterns['constellations'])})", expanded=True):
                for const in patterns['constellations'][:5]:
                    wild_flag = "✨ BOTH WILD" if const['has_both_wild'] else ""
                    st.write(f"Anchor: M#{const['anchor_m']:+.0f} ({const['anchor_family']}) from {const['anchor_origin']} | Members: {const['member_count']} {wild_flag}")
        
        # Model matches
        model_matches = patterns.get('model_matches', {})
        total_model = sum(len(matches) for matches in model_matches.values())
        if total_model > 0:
            with st.expander(f"📊 Model Matches ({total_model} total)"):
                for model_name, matches in sorted(model_matches.items(), key=lambda x: len(x[1]), reverse=True):
                    if len(matches) > 0:
                        st.write(f"**{model_name}**: {len(matches)} matches")


def display_all_patterns_summary(patterns):
    """Display summary of all pattern counts"""
    
    summary_data = {
        'Pattern Type': [],
        'Count': []
    }
    
    pattern_mapping = {
        'Epic Same Origin': 'epic_same_origin',
        'Trinidad-Tobago Pairs': 'epic_epic_pairs',
        'X0 Sequential Descents': 'x0_sequential_descents',
        'FOGZ Presence': 'fogz_presence',
        'Constellations': 'constellations',
        'Indigo Wild Pairs': 'wild_pairs',
        'Same Origin Tag Descents': 'same_origin_tag_descents',
        'X0 Alignments': 'x0_alignments',
        'Downgrades': 'downgrades',
        'Large M# (80+)': 'large_m_presence',
        'Family Clusters': 'family_clusters',
        'Flip Matches': 'flip_matches'
    }
    
    for display_name, key in pattern_mapping.items():
        count = len(patterns.get(key, []))
        if key == 'family_clusters':
            count = len(patterns.get(key, {}))
        
        if count > 0:  # Only show non-zero
            summary_data['Pattern Type'].append(display_name)
            summary_data['Count'].append(count)
    
    # Add model matches
    model_matches = patterns.get('model_matches', {})
    total_model = sum(len(matches) for matches in model_matches.values())
    if total_model > 0:
        summary_data['Pattern Type'].append('Model Matches (23 models)')
        summary_data['Count'].append(total_model)
    
    if summary_data['Pattern Type']:
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('Count', ascending=False)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
    else:
        st.info("No patterns detected in this zone")


def export_results(zones_df):
    """Export results to Excel"""
    
    try:
        # Ensure outputs directory exists
        output_dir = ensure_outputs_directory()
        output_file = os.path.join(output_dir, 'pattern_scanner_results.xlsx')
        
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            # Summary sheet
            summary_cols = ['center_price', 'score', 'num_arrivals', 'zone_type', 'zone_subtype']
            
            if 'chronological_rank' in zones_df.columns:
                summary_cols.insert(0, 'chronological_rank')
            elif 'rank' in zones_df.columns:
                summary_cols.insert(0, 'rank')
            
            if 'zone_time' in zones_df.columns:
                summary_cols.append('zone_time')
            
            if 'touched_zone' in zones_df.columns:
                summary_cols.extend(['touched_zone', 'num_turns', 'max_distance_from_zone', 'reversal_confirmed'])
            
            available_cols = [col for col in summary_cols if col in zones_df.columns]
            zones_df[available_cols].to_excel(writer, sheet_name='Summary', index=False)
            
            # Pattern counts for each zone
            pattern_counts = []
            for idx, zone in zones_df.iterrows():
                patterns = zone['patterns']
                row = {
                    'Rank': zone.get('chronological_rank', zone.get('rank', idx+1)),
                    'Price': zone['center_price'],
                    'Score': zone['score'],
                    'Epic Same Origin': len(patterns.get('epic_same_origin', [])),
                    'TT Pairs': len(patterns.get('epic_epic_pairs', [])),
                    'X0 Sequential': len(patterns.get('x0_sequential_descents', [])),
                    'FOGZ': len(patterns.get('fogz_presence', [])),
                    'Constellations': len(patterns.get('constellations', [])),
                    'Wild Pairs': len(patterns.get('wild_pairs', [])),
                    'Model Matches': sum(len(m) for m in patterns.get('model_matches', {}).values()),
                    'Downgrades': len(patterns.get('downgrades', []))
                }
                
                if 'touched_zone' in zone.index:
                    row['Touched'] = zone['touched_zone']
                    row['Turns'] = zone.get('num_turns', 0)
                    row['Reversal'] = zone.get('reversal_confirmed', False)
                
                pattern_counts.append(row)
            
            pd.DataFrame(pattern_counts).to_excel(writer, sheet_name='Pattern Counts', index=False)
            
            # Turn-to-turn distances sheet
            if 'turn_to_turn_distances' in zones_df.columns:
                distance_rows = []
                
                for idx, zone in zones_df.iterrows():
                    if zone['turn_to_turn_distances']:
                        rank = zone.get('chronological_rank', zone.get('rank', idx+1))
                        
                        for i, dist in enumerate(zone['turn_to_turn_distances'], 1):
                            distance_rows.append({
                                'Zone_Rank': rank,
                                'Zone_Price': zone['center_price'],
                                'Turn_Number': i,
                                'From_Type': dist['from_type'],
                                'From_Price': dist['from_price'],
                                'From_Time': dist['from_time'],
                                'To_Type': dist['to_type'],
                                'To_Price': dist['to_price'],
                                'To_Time': dist['to_time'],
                                'Distance': dist['distance'],
                                'From_Zone_Distance': dist['from_zone_distance'],
                                'Is_Reversal': dist['is_reversal_to']
                            })
                
                if distance_rows:
                    pd.DataFrame(distance_rows).to_excel(writer, sheet_name='Turn Distances', index=False)
        
        st.success(f"✅ Results exported to: {output_file}")
        
        # Provide download link
        with open(output_file, 'rb') as f:
            st.download_button(
                label="📥 Download Excel File",
                data=f,
                file_name="pattern_scanner_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    
    except Exception as e:
        st.error(f"Error exporting: {e}")
        import traceback
        st.code(traceback.format_exc())


if __name__ == "__main__":
    main()
