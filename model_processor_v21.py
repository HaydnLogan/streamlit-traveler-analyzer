"""
Model Processor v21 - Batch Processing for All 23 Trading Models

This module handles the generation and matching of all trading models defined in
model_definitions_v21.py. It provides a clean interface for the main Swing Analysis Tool.

Key Functions:
- process_all_models(): Main entry point - processes all 23 models
- Returns organized results ready for display and Excel export
"""

import streamlit as st
import pandas as pd
import time as time_module
from model_definitions_v21 import MODELS, get_reciprocal_lookup, apply_special_matching
from custom_range_calculator_1125_21 import (
    process_cluster_tables_two_pass,
    match_cluster_table_entries
)


def process_all_models(
    measurement_df,
    small_df,
    big_df,
    report_time,
    lookback_days,
    max_spread,
    window_radius,
    allowed_origins,
    segment_size,
    combine_segments,
    feed_selection,
    match_type_selection
):
    """
    Process all 23 trading models and return organized results.
    
    Parameters:
    -----------
    measurement_df : DataFrame
        Measurement data with M# values
    small_df, big_df : DataFrame
        HLC data for small and big feeds
    report_time : datetime
        Report datetime
    lookback_days : int
        Number of days to look back
    max_spread : float
        Maximum output spread for matching
    window_radius : float
        Window radius around open
    allowed_origins : set or None
        Set of allowed origin names
    segment_size : int or None
        Segment size for processing
    combine_segments : bool
        Whether to combine segments
    feed_selection : str
        "Both feeds", "Small feed only", or "Big feed only"
    match_type_selection : str
        "Same feed only" or "Allow mixed feed"
    
    Returns:
    --------
    dict : {
        'results': {model_name: matched_df, ...},
        'prep_tables': {model_name: prep_df, ...},
        'summaries': {model_name: summary_dict, ...},
        'timings': {model_name: time_seconds, ...},
        'total_time': total_seconds,
        'report_time': report_time
    }
    """
    total_start = time_module.time()
    
    results = {}
    prep_tables = {}
    summaries = {}
    timings = {}
    
    # Get reciprocal lookup for models that need it
    recip_lookup = get_reciprocal_lookup()
    
    # Sort models by number for consistent order
    sorted_models = sorted(MODELS.items(), key=lambda x: x[1]['number'])
    
    st.info(f"🚀 Processing {len(sorted_models)} trading models...")
    
    # Create progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, (model_name, model_config) in enumerate(sorted_models):
        model_start = time_module.time()
        
        # Update progress
        progress = (idx) / len(sorted_models)
        progress_bar.progress(progress)
        status_text.text(f"Processing Model {model_config['number']}/{len(sorted_models)}: {model_name}...")
        
        try:
            # Generate prep table for this model
            prep_df, summary = process_cluster_tables_two_pass(
                measurement_df=measurement_df,
                small_df=small_df,
                big_df=big_df,
                report_time=report_time,
                scope_days=lookback_days,
                valid_list_pass1=model_config['pass1'],
                valid_list_pass2=model_config['pass2'],
                max_output_spread=max_spread,
                window_radius=window_radius,
                allowed_origins=allowed_origins,
                segment_size=segment_size,
                combine_segments=combine_segments,
                feed_selection=feed_selection
            )
            
            # ITEM 2 FIX: Apply Pass 1 Day restrictions for Premium-Premium models (19-23)
            # These models should only use [0], [-1], [-2], [-3] in Pass 1
            prem_prem_models = {
                'Prem x0s PP', 'Prem x1s PP', 'Prem xD0s PP', 'Prem xD1s PP', 'Prem xCs PP'
            }
            
            if model_name in prem_prem_models:
                # Restrict Pass 1 to recent days only
                allowed_pass1_days = {'[0]', '[-1]', '[-2]', '[-3]'}
                
                # Filter Pass 1 results to only include allowed days
                if not prep_df.empty and 'Day' in prep_df.columns:
                    # Keep only Pass 1 arrivals with allowed Day values
                    prep_df = prep_df[prep_df['Day'].isin(allowed_pass1_days)]
            
            # Store prep table and summary
            prep_tables[model_name] = prep_df
            summaries[model_name] = summary
            
            # Perform matching if prep table is not empty
            if not prep_df.empty:
                # Determine if reciprocal checking is needed
                check_recip = model_config.get('check_recip', False)
                
                # Perform matching
                matched_df = match_cluster_table_entries(
                    prep_df=prep_df,
                    valid_list_pass1=model_config['pass1'],
                    valid_list_pass2=model_config['pass2'],
                    max_output_spread=max_spread,
                    measurement_df=measurement_df if check_recip else None,
                    check_recip=check_recip,
                    allow_mixed_feed=(match_type_selection == "Allow mixed feed"),
                    table_name=model_config['display_name'],
                    feed_opens=summary.get('feed_opens', {})
                )
                
                # Apply special matching logic if specified
                if model_config.get('special_matching') and not matched_df.empty:
                    # Extract M#s from M_#s column
                    def extract_m_pair(row):
                        try:
                            m_str = row['M_#s']
                            m1, m2 = m_str.split(',')
                            return int(m1.strip()), int(m2.strip())
                        except:
                            return None, None
                    
                    # Filter rows based on special matching logic
                    valid_rows = []
                    for idx, row in matched_df.iterrows():
                        m1, m2 = extract_m_pair(row)
                        if m1 is not None and apply_special_matching(model_name, m1, m2):
                            valid_rows.append(idx)
                    
                    matched_df = matched_df.loc[valid_rows]
                
                results[model_name] = matched_df
            else:
                # Empty prep table means no matches
                results[model_name] = pd.DataFrame()
            
            model_time = time_module.time() - model_start
            timings[model_name] = model_time
            
        except Exception as e:
            st.error(f"❌ Error processing {model_name}: {str(e)}")
            results[model_name] = pd.DataFrame()
            prep_tables[model_name] = pd.DataFrame()
            summaries[model_name] = {}
            timings[model_name] = 0
    
    # Complete progress
    progress_bar.progress(1.0)
    status_text.text(f"✅ Completed all {len(sorted_models)} models!")
    
    total_time = time_module.time() - total_start
    
    return {
        'results': results,
        'prep_tables': prep_tables,
        'summaries': summaries,
        'timings': timings,
        'total_time': total_time,
        'report_time': report_time
    }


def get_model_display_info(model_name):
    """
    Get display information for a model.
    
    Returns dict with:
    - number: Model number (1-23)
    - display_name: Display name
    - description: Model description
    - pass1_desc: Description of Pass 1 M#s
    - pass2_desc: Description of Pass 2 M#s
    """
    if model_name not in MODELS:
        return None
    
    config = MODELS[model_name]
    
    # Generate descriptions
    pass1_list = sorted(list(config['pass1']))
    pass2_list = sorted(list(config['pass2']))
    
    # Simplify display - show first few and count
    def format_m_list(m_list, max_display=10):
        if len(m_list) <= max_display:
            return str(set(m_list))
        else:
            displayed = m_list[:max_display]
            return f"{{{', '.join(map(str, displayed))}, ... ({len(m_list)} total)}}"
    
    return {
        'number': config['number'],
        'display_name': config['display_name'],
        'description': config['description'],
        'pass1_desc': format_m_list(pass1_list),
        'pass2_desc': format_m_list(pass2_list),
        'check_recip': config['check_recip'],
        'special_matching': config['special_matching']
    }


def organize_results_by_category(all_results):
    """
    Organize model results into logical categories for display.
    
    Returns:
    --------
    dict : {
        'FOGZ Models': [model_names...],
        'Large Discount Models': [model_names...],
        'Reciprocal Models': [model_names...],
        'Premium/Discount Patterns': [model_names...],
        'Premium/Premium Patterns': [model_names...]
    }
    """
    categories = {
        'FOGZ Models (1-3)': [],
        'Large Discount Models (4-6)': [],
        'Reciprocal Models (7-8)': [],
        'Premium/Discount Patterns (9-18)': [],
        'Premium/Premium Patterns (19-23)': []
    }
    
    for model_name, config in MODELS.items():
        num = config['number']
        
        if 1 <= num <= 3:
            categories['FOGZ Models (1-3)'].append(model_name)
        elif 4 <= num <= 6:
            categories['Large Discount Models (4-6)'].append(model_name)
        elif 7 <= num <= 8:
            categories['Reciprocal Models (7-8)'].append(model_name)
        elif 9 <= num <= 18:
            categories['Premium/Discount Patterns (9-18)'].append(model_name)
        elif 19 <= num <= 23:
            categories['Premium/Premium Patterns (19-23)'].append(model_name)
    
    # Sort within each category by model number
    for category in categories:
        categories[category] = sorted(
            categories[category], 
            key=lambda x: MODELS[x]['number']
        )
    
    return categories


def create_summary_stats(all_results):
    """
    Create summary statistics across all models.
    
    Returns:
    --------
    dict : {
        'total_models': int,
        'models_with_matches': int,
        'total_matches': int,
        'matches_by_category': {...},
        'top_models': [(model_name, count), ...]
    }
    """
    results = all_results['results']
    
    total_models = len(results)
    models_with_matches = sum(1 for df in results.values() if len(df) > 0)
    total_matches = sum(len(df) for df in results.values())
    
    # Count by category
    categories = organize_results_by_category(results)
    matches_by_category = {}
    for category, model_list in categories.items():
        count = sum(len(results[m]) for m in model_list if m in results)
        matches_by_category[category] = count
    
    # Top models by match count
    model_counts = [(name, len(df)) for name, df in results.items()]
    top_models = sorted(model_counts, key=lambda x: x[1], reverse=True)[:10]
    
    return {
        'total_models': total_models,
        'models_with_matches': models_with_matches,
        'total_matches': total_matches,
        'matches_by_category': matches_by_category,
        'top_models': top_models
    }
