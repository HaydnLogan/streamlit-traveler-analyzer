"""
Pattern Scanner Streamlit App
Comprehensive pattern detection and success analysis interface
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add paths
sys.path.append('/home/claude')
sys.path.append('/mnt/user-data/uploads')

from pattern_scanner_v03 import HaydnPatternScanner, MODELS, FOGZ

st.set_page_config(
    page_title="Pattern Scanner",
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
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'scanner' not in st.session_state:
    st.session_state.scanner = None
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'success_tracking' not in st.session_state:
    st.session_state.success_tracking = []

def main():
    st.markdown('<div class="main-header">🎯 Pattern Scanner & Success Analyzer</div>', unsafe_allow_html=True)
    
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
        
        ohlc_file = st.file_uploader(
            "OHLC Data (CSV)",
            type=['csv'],
            help="Upload OHLC price data for swing detection and success analysis"
        )
        
        sheet_name = None
        if traveler_file:
            try:
                xl = pd.ExcelFile(traveler_file)
                sheets = xl.sheet_names
                sheet_name = st.selectbox("Select Sheet", sheets)
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
            zone_width = st.slider("Zone Width (points)", 5.0, 30.0, 10.0, 2.5)
            
            # Time range
            st.write("**Time Range:**")
            start_time = st.text_input("Start Time (YYYY-MM-DD HH:MM:SS)", "2025-12-08 23:00:00")
            end_time = st.text_input("End Time (YYYY-MM-DD HH:MM:SS)", "2025-12-09 21:45:00")
            
        else:
            st.write("**Manual Zone Entry:**")
            center_price = st.number_input("Center Price", min_value=0.0, value=25721.25, step=0.25)
            zone_width = st.slider("Zone Width (points)", 5.0, 50.0, 18.0, 2.5)
        
        match_tolerance = st.slider("Match Tolerance (points)", 0.5, 3.0, 1.0, 0.5)
        
        st.divider()
        
        # Success analysis settings
        st.subheader("📈 Success Analysis")
        
        enable_success = st.checkbox("Enable Success Tracking", value=True,
                                     help="Track pattern performance vs price action")
        
        if enable_success:
            lookforward_hours = st.slider("Lookforward Window (hours)", 1, 24, 4, 1)
            success_threshold = st.slider("Success Threshold (points)", 10, 100, 30, 5,
                                         help="Minimum move toward zone to count as success")
        
        # Run button
        st.divider()
        run_analysis = st.button("🚀 Run Analysis", type="primary", use_container_width=True)
    
    # Main content area
    if not traveler_file:
        st.info("👈 Upload traveler report and OHLC data to begin")
        
        # Show quick guide
        st.markdown("""
        ### Quick Start Guide
        
        1. **Upload Files**: Upload traveler report (Excel) and OHLC data (CSV)
        2. **Select Sheet**: Choose the report time sheet
        3. **Choose Mode**: Automatic swing detection or manual zone entry
        4. **Configure**: Set analysis parameters
        5. **Run**: Click "Run Analysis" to start
        
        ### Features
        
        - **12 Pattern Types**: All patterns from manual analysis
        - **24 Models**: 23 existing + Model 24 (X0 Sequential Descents)
        - **Success Tracking**: Automatic pattern performance analysis
        - **Export**: Download results to Excel
        """)
        return
    
    if run_analysis and traveler_file and ohlc_file and sheet_name:
        with st.spinner("Loading data..."):
            try:
                # Load data
                traveler_df = pd.read_excel(traveler_file, sheet_name=sheet_name)
                ohlc_df = pd.read_csv(ohlc_file)
                
                # Initialize scanner
                scanner = HaydnPatternScanner(traveler_df, ohlc_df)
                st.session_state.scanner = scanner
                
                st.success(f"✅ Loaded {len(traveler_df)} traveler arrivals and {len(ohlc_df)} OHLC bars")
                
            except Exception as e:
                st.error(f"Error loading data: {e}")
                return
        
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
                    
                    # Success analysis if enabled
                    if enable_success:
                        zones_df = add_success_analysis(
                            zones_df, 
                            ohlc_df, 
                            lookforward_hours, 
                            success_threshold
                        )
                    
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
                    zones_df['zone_type'] = 'Manual'
                    zones_df['zone_subtype'] = 'User Defined'
                    zones_df['zone_time'] = datetime.now()
                    zones_df['rank'] = 1
                    
                    # Success analysis if enabled
                    if enable_success and ohlc_file:
                        zones_df = add_success_analysis(
                            zones_df,
                            ohlc_df,
                            lookforward_hours,
                            success_threshold
                        )
                    
                    st.session_state.analysis_results = zones_df
                
                st.success(f"✅ Analysis complete! Found {len(zones_df)} zone(s)")
                
            except Exception as e:
                st.error(f"Error during analysis: {e}")
                import traceback
                st.code(traceback.format_exc())
                return
    
    # Display results
    if st.session_state.analysis_results is not None:
        display_results(st.session_state.analysis_results, enable_success if 'enable_success' in locals() else False)

def add_success_analysis(zones_df, ohlc_df, lookforward_hours, threshold):
    """Add success analysis to zones DataFrame"""
    
    # Parse OHLC times
    ohlc_df = ohlc_df.copy()
    ohlc_df['time'] = pd.to_datetime(ohlc_df['time'], utc=True).dt.tz_localize(None)
    
    success_data = []
    
    for idx, zone in zones_df.iterrows():
        zone_time = pd.to_datetime(zone['zone_time']).replace(tzinfo=None)
        zone_price = zone['center_price']
        zone_type = zone['zone_subtype']  # High or Low
        
        # Get future bars
        future_bars = ohlc_df[
            (ohlc_df['time'] > zone_time) &
            (ohlc_df['time'] <= zone_time + timedelta(hours=lookforward_hours))
        ]
        
        if len(future_bars) == 0:
            success_data.append({
                'touched_zone': False,
                'max_move_toward': 0,
                'max_move_away': 0,
                'success': False,
                'bars_to_touch': None,
                'reversal_confirmed': False
            })
            continue
        
        # Check if zone was touched
        if zone_type == 'Low':
            # For lows, check if price went down to zone
            touched = (future_bars['low'].min() <= zone_price)
            max_toward = zone_price - future_bars['low'].min() if touched else 0
            max_away = future_bars['high'].max() - zone_price
            
            # Find when it was touched
            if touched:
                touch_idx = future_bars[future_bars['low'] <= zone_price].index[0]
                bars_to_touch = future_bars.index.get_loc(touch_idx) + 1
                
                # Check for reversal (did it bounce?)
                after_touch = future_bars.loc[touch_idx:]
                if len(after_touch) > 0:
                    bounce = after_touch['high'].max() - zone_price
                    reversal_confirmed = bounce >= threshold
                else:
                    reversal_confirmed = False
            else:
                bars_to_touch = None
                reversal_confirmed = False
                
        else:  # High
            # For highs, check if price went up to zone
            touched = (future_bars['high'].max() >= zone_price)
            max_toward = future_bars['high'].max() - zone_price if touched else 0
            max_away = zone_price - future_bars['low'].min()
            
            if touched:
                touch_idx = future_bars[future_bars['high'] >= zone_price].index[0]
                bars_to_touch = future_bars.index.get_loc(touch_idx) + 1
                
                # Check for reversal
                after_touch = future_bars.loc[touch_idx:]
                if len(after_touch) > 0:
                    bounce = zone_price - after_touch['low'].min()
                    reversal_confirmed = bounce >= threshold
                else:
                    reversal_confirmed = False
            else:
                bars_to_touch = None
                reversal_confirmed = False
        
        # Determine success
        success = touched and (max_toward >= threshold)
        
        success_data.append({
            'touched_zone': touched,
            'max_move_toward': max_toward,
            'max_move_away': max_away,
            'success': success,
            'bars_to_touch': bars_to_touch,
            'reversal_confirmed': reversal_confirmed
        })
    
    # Add to DataFrame
    for key in success_data[0].keys():
        zones_df[key] = [d[key] for d in success_data]
    
    return zones_df

def display_results(zones_df, show_success=False):
    """Display analysis results"""
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Zones Analyzed", len(zones_df))
    
    with col2:
        avg_score = zones_df['score'].mean()
        st.metric("Avg Score", f"{avg_score:,.0f}")
    
    with col3:
        avg_patterns = zones_df['num_arrivals'].mean()
        st.metric("Avg Arrivals", f"{avg_patterns:.0f}")
    
    with col4:
        if show_success and 'success' in zones_df.columns:
            success_rate = (zones_df['success'].sum() / len(zones_df)) * 100
            st.metric("Success Rate", f"{success_rate:.1f}%")
    
    # Success analysis summary
    if show_success and 'success' in zones_df.columns:
        st.markdown("---")
        st.markdown("### 📊 Success Analysis Summary")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            touched = zones_df['touched_zone'].sum()
            st.info(f"**Zones Touched:** {touched} / {len(zones_df)}")
        
        with col2:
            if zones_df['reversal_confirmed'].sum() > 0:
                reversal_rate = (zones_df['reversal_confirmed'].sum() / touched) * 100 if touched > 0 else 0
                st.success(f"**Reversal Rate:** {reversal_rate:.1f}%")
        
        with col3:
            avg_bars = zones_df[zones_df['bars_to_touch'].notna()]['bars_to_touch'].mean()
            if not pd.isna(avg_bars):
                st.warning(f"**Avg Bars to Touch:** {avg_bars:.1f}")
    
    st.markdown("---")
    
    # Zone selector
    st.markdown("### 🎯 Zones")
    
    zone_options = []
    for idx, row in zones_df.iterrows():
        success_icon = ""
        if show_success and 'success' in zones_df.columns:
            if row['success']:
                success_icon = "✅"
            elif row['touched_zone']:
                success_icon = "⚠️"
            else:
                success_icon = "❌"
        
        label = f"{success_icon} Rank #{int(row['rank'])}: {row['zone_type']} {row['zone_subtype']} @ {row['center_price']:.2f} (Score: {row['score']:,.0f})"
        zone_options.append(label)
    
    selected_zone_label = st.selectbox("Select Zone to View Details", zone_options)
    selected_idx = zone_options.index(selected_zone_label)
    
    # Display selected zone
    display_zone_details(zones_df.iloc[selected_idx], show_success)
    
    # Export button
    st.markdown("---")
    if st.button("📥 Export Results to Excel", use_container_width=True):
        export_results(zones_df)

def display_zone_details(zone, show_success=False):
    """Display detailed analysis for a single zone"""
    
    st.markdown(f"### Zone Details: {zone['center_price']:.2f}")
    
    # Basic info
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.info(f"**Type:** {zone['zone_type']}")
    with col2:
        st.info(f"**Subtype:** {zone['zone_subtype']}")
    with col3:
        st.info(f"**Score:** {zone['score']:,.0f}")
    with col4:
        st.info(f"**Arrivals:** {zone['num_arrivals']}")
    
    # Success metrics
    if show_success and 'success' in zone:
        st.markdown("#### 📈 Performance Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if zone['touched_zone']:
                st.success("✅ **Zone Touched**")
            else:
                st.error("❌ **Not Touched**")
        
        with col2:
            if zone['success']:
                st.success(f"✅ **Success**")
            else:
                st.warning("⚠️ **No Success**")
        
        with col3:
            st.metric("Move Toward", f"{zone['max_move_toward']:.2f} pts")
        
        with col4:
            if zone['reversal_confirmed']:
                st.success("✅ **Reversal**")
            else:
                st.info("ℹ️ **No Reversal**")
    
    st.markdown("---")
    
    # Pattern details
    patterns = zone['patterns']
    
    # Create tabs for pattern categories
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🔥 Epic & FOGZ",
        "🎯 Sequential & X0",
        "🌟 Constellations",
        "📊 Model Matches",
        "⬇️ Downgrades",
        "📋 All Patterns"
    ])
    
    with tab1:
        display_epic_fogz_patterns(patterns)
    
    with tab2:
        display_sequential_patterns(patterns)
    
    with tab3:
        display_constellation_patterns(patterns)
    
    with tab4:
        display_model_matches(patterns)
    
    with tab5:
        display_downgrade_patterns(patterns)
    
    with tab6:
        display_all_patterns_summary(patterns)

def display_epic_fogz_patterns(patterns):
    """Display Epic origin and FOGZ patterns"""
    
    st.markdown("#### 🔥 FOGZ Presence")
    fogz = patterns.get('fogz_presence', [])
    if fogz:
        st.success(f"Found {len(fogz)} FOGZ members")
        
        fogz_df = pd.DataFrame(fogz)
        st.dataframe(
            fogz_df[['m', 'origin', 'output', 'feed', 'is_zero', 'is_epic']],
            use_container_width=True
        )
    else:
        st.info("No FOGZ members found")
    
    st.markdown("---")
    
    st.markdown("#### 🔥 Epic Same Origin")
    epic_same = patterns.get('epic_same_origin', [])
    if epic_same:
        st.success(f"Found {len(epic_same)} matches")
        
        for match in epic_same[:10]:
            with st.expander(f"{match['type']}: M#{match['m1']:+.0f} → M#{match['m2']:+.0f} (spread {match['spread']:.2f})"):
                st.write(f"**Origins:** {match['origin1']} → {match['origin2']}")
                st.write(f"**Families:** {match['family1']} → {match['family2']}")
                st.write(f"**Flip Type:** {match['flip_type']}")
                st.write(f"**Feed:** {match['feed']}")
    else:
        st.info("No Epic same origin matches found")
    
    st.markdown("---")
    
    st.markdown("#### 🔥 Trinidad-Tobago Pairs")
    tt_pairs = patterns.get('epic_epic_pairs', [])
    if tt_pairs:
        st.success(f"Found {len(tt_pairs)} TT pairs")
        
        for match in tt_pairs[:10]:
            st.write(f"M#{match['m1']:+.0f} ({match['origin1']}) → M#{match['m2']:+.0f} ({match['origin2']}) | Spread: {match['spread']:.2f}")
    else:
        st.info("No TT pairs found")

def display_sequential_patterns(patterns):
    """Display sequential and X0 patterns"""
    
    st.markdown("#### 🎯 X0 Sequential Descents (Model 24)")
    sequences = patterns.get('x0_sequential_descents', [])
    if sequences:
        st.success(f"Found {len(sequences)} sequential descent patterns")
        
        for seq in sequences:
            pattern_type = seq['pattern_type']
            color = "green" if "Flip" in pattern_type else "blue"
            
            with st.expander(f":{color}[{pattern_type}] - {seq['x0p_sequence'] if seq['x0p_sequence'] else seq['sequence'][:50]}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Anchor:** M#{seq['anchor_m']:+.0f} from {seq['anchor_origin']}")
                    st.write(f"**Feed:** {seq['feed']}")
                    st.write(f"**Sequence Length:** {seq['sequence_length']}")
                with col2:
                    st.write(f"**X0p Count:** {seq['x0p_count']}")
                    st.write(f"**Crosses Zero:** {'✅' if seq['crosses_zero'] else '❌'}")
                    st.write(f"**Output Spread:** {seq['output_spread']:.2f}")
                
                st.write(f"**Full Sequence:** {seq['sequence']}")
    else:
        st.info("No X0 sequential descents found")
    
    st.markdown("---")
    
    st.markdown("#### 🎯 X0 Alignments")
    x0_align = patterns.get('x0_alignments', [])
    if x0_align:
        st.success(f"Found {len(x0_align)} X0 alignments")
        st.write(f"Showing first 10...")
        
        for match in x0_align[:10]:
            same_type = "✅" if match['same_x0_type'] else ""
            st.write(f"M#{match['m1']:+.0f} ({match['tag1']}) → M#{match['m2']:+.0f} ({match['tag2']}) {same_type} | Spread: {match['spread']:.2f}")
    else:
        st.info("No X0 alignments found")
    
    st.markdown("---")
    
    st.markdown("#### 🎯 Same Origin Tag Descents")
    tag_descents = patterns.get('same_origin_tag_descents', [])
    if tag_descents:
        st.success(f"Found {len(tag_descents)} same-origin tag descents")
        
        for descent in tag_descents[:10]:
            st.write(f"{descent['origin']}: M#{descent['m_x0p']:+.0f} ({descent['tag_x0p']}) → M#{descent['m_x0d']:+.0f} ({descent['tag_x0d']}) | Spread: {descent['spread']:.2f}")
    else:
        st.info("No same-origin tag descents found")

def display_constellation_patterns(patterns):
    """Display constellation patterns"""
    
    st.markdown("#### 🌟 Constellations")
    constellations = patterns.get('constellations', [])
    
    if constellations:
        st.success(f"Found {len(constellations)} constellations")
        
        for const in constellations[:10]:
            has_wild = "✅ BOTH WILD" if const['has_both_wild'] else ""
            
            with st.expander(f"M#{const['anchor_m']:+.0f} ({const['anchor_family']}) from {const['anchor_origin']} {has_wild}"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Members", const['member_count'])
                with col2:
                    st.metric("Epic Count", const['epic_count'])
                with col3:
                    st.metric("Same Family", const['same_family_count'])
                
                st.write("**Members:**")
                for member in const['members'][:15]:
                    epic_flag = "⭐" if member['origin'] in ['Trinidad', 'Tobago'] else ""
                    st.write(f"  M#{member['m']:+.0f} from {member['origin']} at {member['output']:.2f} {epic_flag}")
                
                if len(const['members']) > 15:
                    st.write(f"  ... and {len(const['members']) - 15} more")
    else:
        st.info("No constellations found")
    
    st.markdown("---")
    
    st.markdown("#### 🌟 Indigo Wild Pairs (0 + 40)")
    wild_pairs = patterns.get('wild_pairs', [])
    
    if wild_pairs:
        st.success(f"Found {len(wild_pairs)} Wild pairs")
        
        for wp in wild_pairs[:10]:
            epic_flag = "✅ BOTH EPIC" if wp['both_epic'] else ""
            st.write(f"M# 0 from {wp['zero_origin']} + M#{wp['forty_m']:+.0f} from {wp['forty_origin']} | Spread: {wp['spread']:.2f} {epic_flag}")
    else:
        st.info("No Wild pairs found")

def display_model_matches(patterns):
    """Display matches from the 23 models"""
    
    st.markdown("#### 📊 Model Matches (23 Models)")
    
    model_matches = patterns.get('model_matches', {})
    
    # Count total
    total_matches = sum(len(matches) for matches in model_matches.values())
    
    if total_matches > 0:
        st.success(f"Found {total_matches} total model matches")
        
        # Sort by match count
        sorted_models = sorted(
            model_matches.items(),
            key=lambda x: len(x[1]),
            reverse=True
        )
        
        # Show top models
        for model_name, matches in sorted_models:
            if len(matches) == 0:
                continue
            
            model_num = MODELS[model_name]['number']
            
            with st.expander(f"Model {model_num}: {model_name} ({len(matches)} matches)"):
                st.write(f"**Description:** {MODELS[model_name]['description']}")
                
                if len(matches) <= 20:
                    # Show all
                    for match in matches:
                        st.write(f"M#{match['m1']:+.0f} ({match['origin1']}) → M#{match['m2']:+.0f} ({match['origin2']}) | Spread: {match['spread']:.2f} | {match['feed']}")
                else:
                    # Show sample
                    st.write(f"**Showing first 20 of {len(matches)} matches:**")
                    for match in matches[:20]:
                        st.write(f"M#{match['m1']:+.0f} ({match['origin1']}) → M#{match['m2']:+.0f} ({match['origin2']}) | Spread: {match['spread']:.2f}")
    else:
        st.info("No model matches found")

def display_downgrade_patterns(patterns):
    """Display downgrade patterns"""
    
    st.markdown("#### ⬇️ Top Downgrades")
    
    downgrades = patterns.get('downgrades', [])
    
    if downgrades:
        # Sort by differential
        sorted_dg = sorted(downgrades, key=lambda x: x['differential'], reverse=True)
        
        st.success(f"Found {len(downgrades)} downgrades (showing top 20)")
        
        for dg in sorted_dg[:20]:
            same_fam = "✅ Same Family" if dg['same_family'] else ""
            st.write(f"M#{dg['m_large']:+.0f} → M#{dg['m_small']:+.0f} (Δ{dg['differential']:.0f}) | {dg['origin1']} → {dg['origin2']} | Spread: {dg['spread']:.2f} {same_fam}")
    else:
        st.info("No downgrades found")

def display_all_patterns_summary(patterns):
    """Display summary of all pattern counts"""
    
    st.markdown("#### 📋 Pattern Summary")
    
    summary_data = {
        'Pattern Type': [],
        'Count': [],
        'Weight': []
    }
    
    from pattern_scanner_v03 import WEIGHTS
    
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
        
        summary_data['Pattern Type'].append(display_name)
        summary_data['Count'].append(count)
        
        # Get base weight
        weight_key = key.replace('_presence', '').replace('_pairs', '_pair').replace('_descents', '_descent')
        weight = WEIGHTS.get(weight_key, 0)
        summary_data['Weight'].append(weight)
    
    # Add model matches
    model_matches = patterns.get('model_matches', {})
    total_model = sum(len(matches) for matches in model_matches.values())
    summary_data['Pattern Type'].append('Model Matches (23 models)')
    summary_data['Count'].append(total_model)
    summary_data['Weight'].append(WEIGHTS.get('model_match', 40))
    
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('Count', ascending=False)
    
    st.dataframe(summary_df, use_container_width=True)

def export_results(zones_df):
    """Export results to Excel"""
    
    try:
        output_file = '/mnt/user-data/outputs/pattern_scanner_results.xlsx'
        
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            # Summary sheet
            summary_cols = ['rank', 'zone_type', 'zone_subtype', 'center_price', 'score', 'num_arrivals']
            if 'success' in zones_df.columns:
                summary_cols.extend(['touched_zone', 'success', 'max_move_toward', 'reversal_confirmed'])
            
            zones_df[summary_cols].to_excel(writer, sheet_name='Summary', index=False)
            
            # Pattern counts for each zone
            pattern_counts = []
            for idx, zone in zones_df.iterrows():
                patterns = zone['patterns']
                row = {
                    'Rank': zone['rank'],
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
                pattern_counts.append(row)
            
            pd.DataFrame(pattern_counts).to_excel(writer, sheet_name='Pattern Counts', index=False)
        
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

if __name__ == "__main__":
    main()
