# bypass_mode_matcher.py
# Matches travelers from pre-generated reports for 23 trading models

import pandas as pd
import numpy as np
from collections import defaultdict

def match_travelers_bypass_mode(
    traveler_df,
    pass1_m_numbers,
    pass2_m_numbers,
    max_spread=3.0,
    day_filter="[0]",  # Which day to look for Pass 1 travelers
    feed_name=None     # 'Small', 'Big', or None for all
):
    """
    Match travelers for bypass mode using pre-generated traveler report.
    
    Process:
    1. Filter for Pass 1 M#s on specified day (typically day [0])
    2. For each Pass 1 traveler:
       - Get its Output value
       - Calculate spread range (Output ± max_spread)
       - Find all Pass 2 M#s within that range
       - Create matched pairs
    
    Args:
        traveler_df: DataFrame with columns: Feed, M #, Origin, Output, Day, Arrival, etc.
        pass1_m_numbers: List of M# values for Pass 1 (e.g., [-6, -5, -3, -2, -1, 0, 1, 2, 3, 5, 6])
        pass2_m_numbers: List of M# values for Pass 2 (e.g., [40, 41, 43, 50, 77, 96, 107, 111])
        max_spread: Maximum output spread for matching (default 3.0)
        day_filter: Day to filter Pass 1 travelers (default "[0]" for today)
        feed_name: Feed to filter ('Small', 'Big', or None for all)
    
    Returns:
        DataFrame with matched pairs containing:
        - Input1, Input2: Output values
        - M1, M2: M# values
        - Origin1, Origin2: Origins
        - Arrival1, Arrival2: Arrival times
        - Day1, Day2: Day brackets
        - Prox: Proximity (absolute difference in Output)
        - Feed1, Feed2: Feed names
        - Arrival_Brackets: Combined day brackets
        - Arrival_Output: Output value (same for both)
    """
    
    # Filter by feed if specified
    if feed_name:
        df = traveler_df[traveler_df['Feed'] == feed_name].copy()
    else:
        df = traveler_df.copy()
    
    # Ensure Day column is string for comparison
    df['Day'] = df['Day'].astype(str).str.strip()
    
    # Step 1: Filter for Pass 1 travelers on specified day
    pass1_travelers = df[
        (df['M #'].isin(pass1_m_numbers)) & 
        (df['Day'] == day_filter)
    ].copy()
    
    # Step 2: Filter for Pass 2 travelers (all days)
    pass2_travelers = df[df['M #'].isin(pass2_m_numbers)].copy()
    
    # Step 3: Match each Pass 1 traveler with Pass 2 travelers
    matched_pairs = []
    
    for idx1, trav1 in pass1_travelers.iterrows():
        output1 = trav1['Output']
        
        # Calculate spread range
        range_low = output1 - max_spread
        range_high = output1 + max_spread
        
        # Find Pass 2 travelers within range
        matches = pass2_travelers[
            (pass2_travelers['Output'] >= range_low) &
            (pass2_travelers['Output'] <= range_high)
        ]
        
        # Create pairs
        for idx2, trav2 in matches.iterrows():
            prox = abs(trav1['Output'] - trav2['Output'])
            
            pair = {
                'Input1': trav1['Output'],
                'Input2': trav2['Output'],
                'M1': trav1['M #'],
                'M2': trav2['M #'],
                'Origin1': trav1['Origin'],
                'Origin2': trav2['Origin'],
                'Arrival1': trav1['Arrival'],
                'Arrival2': trav2['Arrival'],
                'Day1': trav1['Day'],
                'Day2': trav2['Day'],
                'Prox': prox,
                'Feed1': trav1['Feed'],
                'Feed2': trav2['Feed'],
                'Arrival_Brackets': f"{trav1['Day']}, {trav2['Day']}",
                'Arrival_Output': trav1['Output']  # Use Pass 1 output as reference
            }
            
            # Add any additional columns
            if 'R #' in trav1:
                pair['R1'] = trav1['R #']
            if 'R #' in trav2:
                pair['R2'] = trav2['R #']
            
            matched_pairs.append(pair)
    
    # Convert to DataFrame
    if matched_pairs:
        result_df = pd.DataFrame(matched_pairs)
        
        # Sort by Arrival_Output descending, then by Prox ascending
        result_df = result_df.sort_values(['Arrival_Output', 'Prox'], ascending=[False, True])
        
        return result_df.reset_index(drop=True)
    else:
        return pd.DataFrame()


def process_model_bypass_mode(
    traveler_df,
    model_config,
    max_spread=3.0
):
    """
    Process a specific model using bypass mode matching.
    
    Args:
        traveler_df: Traveler report DataFrame
        model_config: Model configuration dict with 'pass1_ms' and 'pass2_ms'
        max_spread: Maximum spread for matching
    
    Returns:
        DataFrame with matched pairs for this model
    """
    
    # Extract Pass 1 and Pass 2 M#s from model config
    pass1_ms = model_config.get('pass1_ms', [])
    pass2_ms = model_config.get('pass2_ms', [])
    
    # Get any special filtering requirements
    day_filter = model_config.get('day_filter', '[0]')
    
    # Check if model has separate feed requirements
    feed_selection = model_config.get('feed_selection', 'both')
    
    all_matches = []
    
    if feed_selection == 'both':
        # Process each feed separately
        for feed_name in ['Small', 'Big']:
            if feed_name in traveler_df['Feed'].unique():
                matches = match_travelers_bypass_mode(
                    traveler_df,
                    pass1_ms,
                    pass2_ms,
                    max_spread=max_spread,
                    day_filter=day_filter,
                    feed_name=feed_name
                )
                all_matches.append(matches)
    else:
        # Process all together
        matches = match_travelers_bypass_mode(
            traveler_df,
            pass1_ms,
            pass2_ms,
            max_spread=max_spread,
            day_filter=day_filter,
            feed_name=None
        )
        all_matches.append(matches)
    
    # Combine all matches
    if all_matches:
        combined = pd.concat(all_matches, ignore_index=True)
        return combined
    else:
        return pd.DataFrame()


def apply_model_filters(matched_df, model_config):
    """
    Apply model-specific filters to matched pairs.
    
    Args:
        matched_df: DataFrame of matched pairs
        model_config: Model configuration with filter criteria
    
    Returns:
        Filtered DataFrame
    """
    
    if matched_df.empty:
        return matched_df
    
    df = matched_df.copy()
    
    # Apply origin filters if specified
    if 'origin_filter' in model_config:
        origin_criteria = model_config['origin_filter']
        if origin_criteria:
            df = df[df['Origin1'].isin(origin_criteria) | df['Origin2'].isin(origin_criteria)]
    
    # Apply M# combination filters
    if 'required_m_pairs' in model_config:
        required_pairs = model_config['required_m_pairs']
        mask = pd.Series([False] * len(df))
        for m1, m2 in required_pairs:
            mask |= ((df['M1'] == m1) & (df['M2'] == m2)) | ((df['M1'] == m2) & (df['M2'] == m1))
        df = df[mask]
    
    # Apply feed combination filters
    if 'feed_requirement' in model_config:
        feed_req = model_config['feed_requirement']
        if feed_req == 'same_feed':
            df = df[df['Feed1'] == df['Feed2']]
        elif feed_req == 'cross_feed':
            df = df[df['Feed1'] != df['Feed2']]
    
    # Apply proximity filters
    if 'max_prox' in model_config:
        max_prox = model_config['max_prox']
        df = df[df['Prox'] <= max_prox]
    
    return df


# Example usage for FOGZ PD model
def process_fogz_pd_bypass(traveler_df, max_spread=3.0):
    """
    Process FOGZ PD model in bypass mode.
    
    FOGZ PD looks for:
    - Pass 1: M# 0 on day [0]
    - Pass 2: Premium M#s (40, 41, 43, 50, 54, 96, 107, 111)
    - Within max_spread proximity
    - Premium/Discount direction
    """
    
    model_config = {
        'pass1_ms': [0],
        'pass2_ms': [40, 41, 43, 50, 54, 96, 107, 111],
        'day_filter': '[0]',
        'feed_selection': 'both'
    }
    
    matched_df = process_model_bypass_mode(traveler_df, model_config, max_spread)
    
    # FOGZ-specific filtering: M1 must be 0, M2 must be premium
    if not matched_df.empty:
        matched_df = matched_df[
            (matched_df['M1'] == 0) & 
            (matched_df['M2'].isin([40, 41, 43, 50, 54, 96, 107, 111]))
        ]
    
    return matched_df
