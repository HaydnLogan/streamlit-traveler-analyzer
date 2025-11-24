"""
Custom Reciprocal Traveler Report Generator
============================================
Generates focused traveler reports from raw HLC feed data.
Only includes:
- Anchor and Epic origins
- Arrivals on Day [0] (today at report time)
- Previous trading day (for weekend handling)
- Recipe M# pairs from G.11
- Reciprocal matches within Max Spread
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# === ORIGIN TYPES ===
EPIC_ORIGINS = {'trinidad', 'tobago', 'wasp-12b', 'wasp-12b[1]', 'wasp-12b[2]', 
                'macedonia', 'macedonia[1]', 'macedonia[2]'}

ANCHOR_ORIGINS = {'spain', 'saturn', 'jupiter', 'kepler-62', 'kepler-44'}

# === RECIPE M# PAIRS from G.11 ===
RECIPE_PAIRS = {
    'GR': [(30, 50)],
    'X0': [(22, 60), (14, 68), (10, 77), (6, 87), (5, 96), (3, 103), (2, 107), (1, 111)],
    'X1': [(36, 43), (26, 55)],
    'X2': [(39, 41)]
}

# Flatten all recipe M# values
ALL_RECIPE_M_NUMBERS = set()
for recipe_list in RECIPE_PAIRS.values():
    for m1, m2 in recipe_list:
        ALL_RECIPE_M_NUMBERS.add(m1)
        ALL_RECIPE_M_NUMBERS.add(-m1)
        ALL_RECIPE_M_NUMBERS.add(m2)
        ALL_RECIPE_M_NUMBERS.add(-m2)

# === ORIGIN UPDATE TIMES ===
ORIGIN_UPDATE_TIMES = {
    'spain': [(3, 15), (9, 30), (15, 45)],
    'jupiter': [(6, 0), (12, 0), (18, 0)],
    'saturn': [(0, 0), (8, 0), (16, 0)],
    'trinidad': [(18, 0)],  # Daily at 18:00
    'tobago': [(18, 0)],    # Daily at 18:00
    'kepler-62': [(18, 0)],
    'kepler-44': [(18, 0)]
}


def calculate_pivot(h, l, c, m_value):
    """
    Calculate traveler output using pivot formula.
    Output = (H + L + C) / 3 + M × (H - L)
    """
    if pd.isna(h) or pd.isna(l) or pd.isna(c):
        return None
    pivot = (h + l + c) / 3
    spread = h - l
    output = pivot + (m_value * spread)
    return output


def extract_origins_from_hlc(df):
    """
    Extract origin names from HLC column format.
    Looks for columns ending with ' H', ' L', ' C'
    
    Returns: dict of {origin_name: [h_col, l_col, c_col]}
    """
    origins = {}
    
    for col in df.columns:
        if col.endswith(' H'):
            origin = col[:-2].strip()
            h_col = col
            l_col = f"{origin} L"
            c_col = f"{origin} C"
            
            if l_col in df.columns and c_col in df.columns:
                origins[origin] = [h_col, l_col, c_col]
    
    return origins


def is_anchor_or_epic(origin_name):
    """Check if origin is Anchor or Epic type"""
    origin_lower = origin_name.lower()
    return origin_lower in ANCHOR_ORIGINS or origin_lower in EPIC_ORIGINS


def get_origin_type(origin_name):
    """Get origin type: Anchor, Epic, or Other"""
    origin_lower = origin_name.lower()
    if origin_lower in ANCHOR_ORIGINS:
        return 'Anchor'
    elif origin_lower in EPIC_ORIGINS:
        return 'Epic'
    else:
        return 'Other'


def get_last_trading_day(report_time):
    """
    Get the last trading day before report_time.
    Handles weekends: if report is Sunday/Monday early, returns Friday.
    """
    # Start with previous day
    prev_day = report_time - timedelta(days=1)
    
    # If report is on Sunday (6) or Monday before market open
    if report_time.weekday() == 6:  # Sunday
        prev_day = report_time - timedelta(days=2)  # Friday
    elif report_time.weekday() == 0 and report_time.hour < 8:  # Monday early morning
        prev_day = report_time - timedelta(days=3)  # Friday
    
    # Keep going back if it's still a weekend
    while prev_day.weekday() >= 5:  # Saturday or Sunday
        prev_day -= timedelta(days=1)
    
    return prev_day


def generate_custom_recip_report(hlc_df, measurement_df, report_time, feed_label='Small'):
    """
    Generate a custom reciprocal traveler report from raw HLC data.
    
    Parameters:
    -----------
    hlc_df : DataFrame
        Raw HLC data with columns like 'time', 'Spain H', 'Spain L', 'Spain C', etc.
    measurement_df : DataFrame
        Measurement file with 'M #' and 'R #' columns
    report_time : datetime
        Time to generate report for (e.g., 18:00)
    feed_label : str
        Feed name ('Small' or 'Big')
    
    Returns:
    --------
    DataFrame with columns: M #, R #, Origin, Output, Arrival, Day, Feed, Origin_Type
    """
    
    # Ensure time column is datetime
    if 'time' not in hlc_df.columns:
        time_col = hlc_df.columns[0]
        hlc_df = hlc_df.rename(columns={time_col: 'time'})
    
    hlc_df['time'] = pd.to_datetime(hlc_df['time'], errors='coerce')
    
    # Extract origins from HLC columns
    origins = extract_origins_from_hlc(hlc_df)
    
    print(f"📊 Found {len(origins)} origins in HLC data: {list(origins.keys())}")
    
    # Filter to only Anchor and Epic origins
    filtered_origins = {name: cols for name, cols in origins.items() 
                       if is_anchor_or_epic(name)}
    
    print(f"✅ Filtered to {len(filtered_origins)} Anchor/Epic origins: {list(filtered_origins.keys())}")
    
    if not filtered_origins:
        print("❌ No Anchor or Epic origins found in data!")
        return pd.DataFrame()
    
    # Filter measurements to only Recipe M# values
    recipe_measurements = measurement_df[
        measurement_df['M #'].abs().isin(ALL_RECIPE_M_NUMBERS)
    ].copy()
    
    print(f"✅ Filtered to {len(recipe_measurements)} Recipe M# values")
    
    if recipe_measurements.empty:
        print("❌ No Recipe M# values found in measurement file!")
        return pd.DataFrame()
    
    # Get today's date at report time
    report_date = report_time.date()
    
    # Get last trading day
    last_trading_day = get_last_trading_day(report_time)
    last_trading_date = last_trading_day.date()
    
    print(f"📅 Report date: {report_date} (Day [0])")
    print(f"📅 Last trading date: {last_trading_date} (Day [-1] or earlier)")
    
    # Find rows for today and last trading day
    hlc_df['date'] = hlc_df['time'].dt.date
    
    today_rows = hlc_df[hlc_df['date'] == report_date]
    last_day_rows = hlc_df[hlc_df['date'] == last_trading_date]
    
    print(f"📊 Found {len(today_rows)} rows for today, {len(last_day_rows)} rows for last trading day")
    
    # Generate travelers
    travelers = []
    
    for origin_name, (h_col, l_col, c_col) in filtered_origins.items():
        origin_type = get_origin_type(origin_name)
        
        # Process today's arrivals (Day [0])
        for _, row in today_rows.iterrows():
            h = row[h_col]
            l = row[l_col]
            c = row[c_col]
            
            if pd.isna(h) or pd.isna(l) or pd.isna(c):
                continue
            
            # Generate travelers for each Recipe M#
            for _, m_row in recipe_measurements.iterrows():
                m_num = m_row['M #']
                r_num = m_row['R #']
                
                output = calculate_pivot(h, l, c, m_num)
                
                if output is not None:
                    travelers.append({
                        'M #': m_num,
                        'R #': r_num,
                        'Origin': origin_name,
                        'Output': output,
                        'Arrival': row['time'],
                        'Day': '[0]',
                        'Feed': feed_label,
                        'Origin_Type': origin_type
                    })
        
        # Process last trading day's arrivals
        for _, row in last_day_rows.iterrows():
            h = row[h_col]
            l = row[l_col]
            c = row[c_col]
            
            if pd.isna(h) or pd.isna(l) or pd.isna(c):
                continue
            
            # Calculate days back
            days_back = -(report_date - last_trading_date).days
            day_label = f"[{days_back}]"
            
            # Generate travelers for each Recipe M#
            for _, m_row in recipe_measurements.iterrows():
                m_num = m_row['M #']
                r_num = m_row['R #']
                
                output = calculate_pivot(h, l, c, m_num)
                
                if output is not None:
                    travelers.append({
                        'M #': m_num,
                        'R #': r_num,
                        'Origin': origin_name,
                        'Output': output,
                        'Arrival': row['time'],
                        'Day': day_label,
                        'Feed': feed_label,
                        'Origin_Type': origin_type
                    })
    
    # Convert to DataFrame
    traveler_df = pd.DataFrame(travelers)
    
    if traveler_df.empty:
        print("❌ No travelers generated!")
        return traveler_df
    
    print(f"✅ Generated {len(traveler_df)} travelers")
    print(f"   - Day [0]: {len(traveler_df[traveler_df['Day'] == '[0]'])} travelers")
    print(f"   - Last trading day: {len(traveler_df[traveler_df['Day'] != '[0]'])} travelers")
    print(f"   - Anchor origins: {len(traveler_df[traveler_df['Origin_Type'] == 'Anchor'])} travelers")
    print(f"   - Epic origins: {len(traveler_df[traveler_df['Origin_Type'] == 'Epic'])} travelers")
    
    return traveler_df


def find_reciprocal_matches(traveler_df, max_spread=3.0):
    """
    Find reciprocal matches within the traveler report.
    Reciprocal: M1's R# = M2's M# AND M2's R# = M1's M#
    
    Parameters:
    -----------
    traveler_df : DataFrame
        Traveler report with M #, R #, Output columns
    max_spread : float
        Maximum output spread for matches (default 3.0)
    
    Returns:
    --------
    List of reciprocal match dictionaries
    """
    matches = []
    
    # Group by Feed to only match within same feed
    for feed in traveler_df['Feed'].unique():
        feed_df = traveler_df[traveler_df['Feed'] == feed].reset_index(drop=True)
        
        # Compare each pair
        for i in range(len(feed_df)):
            for j in range(i + 1, len(feed_df)):
                row1 = feed_df.iloc[i]
                row2 = feed_df.iloc[j]
                
                m1 = row1['M #']
                r1 = row1['R #']
                m2 = row2['M #']
                r2 = row2['R #']
                
                # Check reciprocal relationship
                if r1 == m2 and r2 == m1:
                    output_spread = abs(row1['Output'] - row2['Output'])
                    
                    if output_spread <= max_spread:
                        matches.append({
                            'Feed': feed,
                            'Origin1': row1['Origin'],
                            'M1': m1,
                            'R1': r1,
                            'Output1': row1['Output'],
                            'Day1': row1['Day'],
                            'Origin2': row2['Origin'],
                            'M2': m2,
                            'R2': r2,
                            'Output2': row2['Output'],
                            'Day2': row2['Day'],
                            'Output_Spread': output_spread,
                            'Zone_Price': (row1['Output'] + row2['Output']) / 2
                        })
    
    print(f"🎯 Found {len(matches)} reciprocal matches within {max_spread} point spread")
    
    return matches


def format_recip_report_summary(matches):
    """Format reciprocal matches into a readable summary"""
    if not matches:
        return "No reciprocal matches found."
    
    summary = f"📊 RECIPROCAL TRAVELER REPORT\n"
    summary += f"{'='*80}\n"
    summary += f"Found {len(matches)} reciprocal matches:\n\n"
    
    for i, match in enumerate(matches, 1):
        summary += f"Match #{i}:\n"
        summary += f"  Feed: {match['Feed']}\n"
        summary += f"  Pair: {match['Origin1']} m#{match['M1']:>3} ↔ {match['Origin2']} m#{match['M2']:>3}\n"
        summary += f"  Outputs: {match['Output1']:.2f} / {match['Output2']:.2f}\n"
        summary += f"  Spread: {match['Output_Spread']:.2f} points\n"
        summary += f"  Zone Price: {match['Zone_Price']:.2f}\n"
        summary += f"  Days: {match['Day1']} / {match['Day2']}\n"
        summary += f"\n"
    
    return summary


# === MAIN FUNCTION ===
def generate_recip_traveler_reports(small_hlc_df, big_hlc_df, measurement_df, 
                                    report_time, max_spread=3.0):
    """
    Main function: Generate custom reciprocal traveler reports for both feeds.
    
    Parameters:
    -----------
    small_hlc_df : DataFrame
        Small feed HLC data
    big_hlc_df : DataFrame
        Big feed HLC data
    measurement_df : DataFrame
        Measurement file
    report_time : datetime
        Report time (e.g., 18:00)
    max_spread : float
        Maximum output spread for reciprocal matches
    
    Returns:
    --------
    dict with:
        - 'small_report': Small feed traveler DataFrame
        - 'big_report': Big feed traveler DataFrame
        - 'small_matches': List of Small feed reciprocal matches
        - 'big_matches': List of Big feed reciprocal matches
        - 'combined_travelers': Combined DataFrame for zone analysis
    """
    
    print(f"\n{'='*80}")
    print(f"GENERATING CUSTOM RECIPROCAL TRAVELER REPORTS")
    print(f"{'='*80}")
    print(f"Report Time: {report_time.strftime('%Y-%m-%d %H:%M')}")
    print(f"Max Spread: {max_spread} points")
    print(f"{'='*80}\n")
    
    # Generate Small feed report
    print("📊 SMALL FEED:")
    print("-" * 80)
    small_report = generate_custom_recip_report(
        small_hlc_df, measurement_df, report_time, feed_label='Small'
    )
    
    # Find Small feed reciprocal matches
    small_matches = []
    if not small_report.empty:
        small_matches = find_reciprocal_matches(small_report, max_spread)
    
    print("\n")
    
    # Generate Big feed report
    print("📊 BIG FEED:")
    print("-" * 80)
    big_report = generate_custom_recip_report(
        big_hlc_df, measurement_df, report_time, feed_label='Big'
    )
    
    # Find Big feed reciprocal matches
    big_matches = []
    if not big_report.empty:
        big_matches = find_reciprocal_matches(big_report, max_spread)
    
    print("\n")
    
    # Combine reports
    combined_travelers = pd.concat([small_report, big_report], ignore_index=True)
    
    print(f"{'='*80}")
    print(f"SUMMARY:")
    print(f"  Small Feed: {len(small_report)} travelers, {len(small_matches)} recip matches")
    print(f"  Big Feed: {len(big_report)} travelers, {len(big_matches)} recip matches")
    print(f"  Combined: {len(combined_travelers)} travelers")
    print(f"{'='*80}\n")
    
    return {
        'small_report': small_report,
        'big_report': big_report,
        'small_matches': small_matches,
        'big_matches': big_matches,
        'combined_travelers': combined_travelers
    }


if __name__ == "__main__":
    print("Custom Reciprocal Traveler Report Generator")
    print("=" * 80)
    print("This module generates focused traveler reports from raw HLC feed data.")
    print("\nFeatures:")
    print("  ✅ Only Anchor and Epic origins")
    print("  ✅ Only Day [0] and last trading day")
    print("  ✅ Only Recipe M# pairs from G.11")
    print("  ✅ Finds reciprocal matches within Max Spread")
    print("  ✅ Handles weekend/holiday trading day logic")
