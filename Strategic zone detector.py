"""
Strategic Zone Detector - Market Intelligence Layer
====================================================
Identifies high-probability turning zones by analyzing:
1. High-rank Recip pairs (Epic + Anchor combinations)
2. HUGE HMA confluences (h1-h20)
3. Wildcard M# emergences (0, ±40, ±54)
4. MA role transitions (resistance → support)

Generates 3-4 high-confidence zone recommendations per report.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import re

# ============================================================================
# CONSTANTS
# ============================================================================

# High-Rank Recipe Pairs (from model_g_11.py)
RECIPE_GR = [(30, 50)]  # Great Recipe
RECIPE_X0 = [(22, 60), (14, 68), (10, 77), (6, 87), (5, 96), (3, 103), (2, 107), (1, 111)]
RECIPE_X1 = [(36, 43), (26, 55)]
RECIPE_X2 = [(39, 41)]

ALL_RECIPES = RECIPE_GR + RECIPE_X0 + RECIPE_X1 + RECIPE_X2

# Wildcard M# values (add confluence)
WILDCARDS = {0, 40, -40, 54, -54}

# Origin Classifications
EPIC_ORIGINS = {'trinidad', 'tobago', 'wasp-12b', 'wasp-12b[1]', 'wasp-12b[2]', 
                'macedonia', 'macedonia[1]', 'macedonia[2]'}
ANCHOR_ORIGINS = {'spain', 'saturn', 'jupiter', 'kepler-62', 'kepler-44'}

# Origin Update Times (approximate)
ORIGIN_UPDATE_TIMES = {
    'spain': ['03:15', '09:30', '15:45'],  # Updates ~every 6 hours
    'jupiter': ['06:00', '12:00', '18:00'],
    'saturn': ['00:00', '08:00', '16:00'],
    'trinidad': ['18:00'],  # Daily
    'tobago': ['18:00']  # Daily
}

# ============================================================================
# HUGE HMA DETECTION
# ============================================================================

def detect_huge_hmas(df):
    """
    Detect HUGE HMA columns (h1-h20 pattern).
    These are higher-rank MAs than regular EMAs, SMAs, or smaller HMAs.
    """
    huge_hmas = []
    
    for col in df.columns:
        # Match h followed by 1-2 digits, optionally followed by 'a' or 'b'
        match = re.match(r'^h(\d{1,2})([ab]?)$', col, re.IGNORECASE)
        if match:
            number = int(match.group(1))
            suffix = match.group(2)
            if 1 <= number <= 20:
                huge_hmas.append({
                    'column': col,
                    'number': number,
                    'suffix': suffix,
                    'rank': 1000 + number  # High rank score
                })
    
    # Sort by number
    huge_hmas.sort(key=lambda x: (x['number'], x['suffix']))
    return huge_hmas


def get_ma_rank(col_name):
    """
    Assign rank to MAs for prioritization.
    Higher rank = more important MA.
    
    Ranking:
    - HUGE HMAs (h1-h20): 1000+
    - Very high timeframe Hull (12Hr+, daily+, weekly, monthly): 800-900
    - High period Hull (500+, 800): 700-800
    - Standard MAs: < 700
    """
    # Check for HUGE HMA
    match = re.match(r'^h(\d{1,2})([ab]?)$', col_name, re.IGNORECASE)
    if match:
        number = int(match.group(1))
        return 1000 + number
    
    # Parse standard MA format
    if ' ' not in col_name:
        return 0
    
    parts = col_name.split()
    if len(parts) < 2:
        return 0
    
    timeframe = parts[0]
    period_str = parts[1]
    
    # Extract period number
    period_match = re.match(r'(\d+)([hse])', period_str)
    if not period_match:
        return 0
    
    period = int(period_match.group(1))
    ma_type = period_match.group(2)
    
    # Hull MAs get bonus
    type_bonus = 100 if ma_type == 'h' else 0
    
    # Timeframe scoring
    tf_scores = {
        'M': 900, 'W': 850, 'd2': 830, 'd1': 820,
        's8.2': 810, 's8.1': 800, 's7': 790, 's6': 780, 's3': 770,
        '12Hr': 760, '8Hr': 750, '4Hr': 740, '3Hr': 730, '2Hr': 720,
        '90m': 710, '1Hr': 700, '45m': 650, '30m': 600,
        '24m': 550, '20m': 520, '18m': 510, '15m': 500,
        '12m': 450, '10m': 440, '6m': 430, '5m': 420, '3m': 410, '1m': 400
    }
    
    tf_score = tf_scores.get(timeframe, 400)
    
    # Period bonus (800 period gets +50, 500 gets +30)
    period_bonus = 0
    if period >= 800:
        period_bonus = 50
    elif period >= 500:
        period_bonus = 30
    elif period >= 200:
        period_bonus = 15
    
    return tf_score + type_bonus + period_bonus


def get_high_rank_mas(df, min_rank=800):
    """Get all high-rank MAs (HUGE HMAs and very high timeframe MAs)."""
    high_rank_mas = []
    
    for col in df.columns:
        rank = get_ma_rank(col)
        if rank >= min_rank:
            high_rank_mas.append({
                'column': col,
                'rank': rank
            })
    
    # Sort by rank descending
    high_rank_mas.sort(key=lambda x: x['rank'], reverse=True)
    return high_rank_mas


# ============================================================================
# RECIP PAIR DETECTION
# ============================================================================

def normalize_origin(origin):
    """Normalize origin name for comparison."""
    if not origin:
        return ''
    origin = str(origin).lower().strip()
    # Remove brackets for base comparison
    if '[' in origin:
        origin = origin[:origin.find('[')]
    return origin


def is_recipe_pair(m1, m2):
    """Check if two M# values form a recipe pair."""
    abs_m1, abs_m2 = abs(m1), abs(m2)
    m_set = {abs_m1, abs_m2}
    
    for recipe in ALL_RECIPES:
        if m_set == set(recipe):
            return True, recipe
    return False, None


def score_recip_pair(origin1, origin2, output_spread, is_recipe=False):
    """
    Score a recip pair based on:
    - Origin types (Epic + Anchor = 100, Anchor + Anchor = 80)
    - Output spread (smaller = better)
    - Recipe status (bonus +50)
    """
    norm_o1 = normalize_origin(origin1)
    norm_o2 = normalize_origin(origin2)
    
    is_epic1 = norm_o1 in EPIC_ORIGINS
    is_epic2 = norm_o2 in EPIC_ORIGINS
    is_anchor1 = norm_o1 in ANCHOR_ORIGINS
    is_anchor2 = norm_o2 in ANCHOR_ORIGINS
    
    # Base score from origins
    base_score = 0
    
    if (is_epic1 and is_anchor2) or (is_anchor1 and is_epic2):
        base_score = 100  # Epic + Anchor
    elif is_anchor1 and is_anchor2:
        base_score = 80  # Anchor + Anchor
    elif is_epic1 and is_epic2:
        base_score = 60  # Epic + Epic
    else:
        base_score = 40  # Other combinations
    
    # Output spread penalty (smaller = better)
    spread_penalty = min(output_spread * 5, 50)  # Max penalty 50
    
    # Recipe bonus
    recipe_bonus = 50 if is_recipe else 0
    
    # Final score
    score = base_score + recipe_bonus - spread_penalty
    
    return max(score, 0)


def find_recip_pairs(travelers_df, max_spread=3.0, same_feed_only=True):
    """
    Find high-rank reciprocal pairs in travelers.
    
    Reciprocal = M1's R# matches M2's M#, and M2's R# matches M1's M#
    
    Returns list of recip pairs sorted by score.
    """
    if travelers_df.empty:
        return []
    
    recip_pairs = []
    
    # Group by feed if required
    if same_feed_only:
        feeds = travelers_df['Feed'].unique()
        feed_groups = {feed: travelers_df[travelers_df['Feed'] == feed] for feed in feeds}
    else:
        feed_groups = {'all': travelers_df}
    
    for feed_name, feed_df in feed_groups.items():
        for i, row1 in feed_df.iterrows():
            m1 = row1.get('M #')
            r1 = row1.get('R #')
            
            if pd.isna(m1) or pd.isna(r1):
                continue
            
            try:
                m1 = float(m1)
                r1 = float(r1)
            except:
                continue
            
            # Look for matching recip
            for j, row2 in feed_df.iterrows():
                if i >= j:
                    continue
                
                m2 = row2.get('M #')
                r2 = row2.get('R #')
                
                if pd.isna(m2) or pd.isna(r2):
                    continue
                
                try:
                    m2 = float(m2)
                    r2 = float(r2)
                except:
                    continue
                
                # Check reciprocal relationship
                if abs(r1 - m2) < 0.01 and abs(r2 - m1) < 0.01:
                    output1 = row1.get('Output', 0)
                    output2 = row2.get('Output', 0)
                    output_spread = abs(output1 - output2)
                    
                    if output_spread <= max_spread:
                        # Check if recipe
                        is_recipe, recipe = is_recipe_pair(m1, m2)
                        
                        # Score the pair
                        score = score_recip_pair(
                            row1.get('Origin'),
                            row2.get('Origin'),
                            output_spread,
                            is_recipe
                        )
                        
                        avg_output = (output1 + output2) / 2
                        
                        recip_pairs.append({
                            'feed': feed_name if same_feed_only else 'all',
                            'origin1': row1.get('Origin'),
                            'origin2': row2.get('Origin'),
                            'm1': m1,
                            'm2': m2,
                            'r1': r1,
                            'r2': r2,
                            'output1': output1,
                            'output2': output2,
                            'output_spread': output_spread,
                            'avg_output': avg_output,
                            'is_recipe': is_recipe,
                            'recipe_type': recipe if is_recipe else None,
                            'score': score,
                            'arrival1': row1.get('Arrival'),
                            'arrival2': row2.get('Arrival'),
                            'day1': row1.get('Day', '[0]'),
                            'day2': row2.get('Day', '[0]')
                        })
    
    # Sort by score descending
    recip_pairs.sort(key=lambda x: x['score'], reverse=True)
    
    return recip_pairs


def find_wildcards_near_zone(travelers_df, zone_price, tolerance=24):
    """
    Find wildcard M# values (0, ±40, ±54) near a zone.
    These add confluence to existing zones.
    """
    if travelers_df.empty:
        return []
    
    wildcards_found = []
    
    for _, row in travelers_df.iterrows():
        m_num = row.get('M #')
        output = row.get('Output')
        
        if pd.isna(m_num) or pd.isna(output):
            continue
        
        try:
            m_num = float(m_num)
            output = float(output)
        except:
            continue
        
        # Check if wildcard
        if m_num in WILDCARDS:
            distance = abs(output - zone_price)
            
            if distance <= tolerance:
                wildcards_found.append({
                    'm_num': m_num,
                    'origin': row.get('Origin'),
                    'output': output,
                    'distance': distance,
                    'arrival': row.get('Arrival'),
                    'day': row.get('Day', '[0]'),
                    'feed': row.get('Feed')
                })
    
    # Sort by distance
    wildcards_found.sort(key=lambda x: x['distance'])
    
    return wildcards_found


# ============================================================================
# MA ROLE TRACKING
# ============================================================================

def track_ma_role_at_price(df, ma_column, target_price, lookback_days=5):
    """
    Track whether an MA acted as support or resistance near a price level.
    
    Returns:
    - 'support': MA below price, price bounced up from it
    - 'resistance': MA above price, price rejected down from it
    - 'neutral': MA crossed or no clear role
    """
    if ma_column not in df.columns:
        return 'unknown'
    
    # Get recent data
    recent_df = df.tail(lookback_days * 240)  # Approx 5 days of 6m data
    
    roles = []
    
    for idx, row in recent_df.iterrows():
        ma_value = row[ma_column]
        if pd.isna(ma_value):
            continue
        
        high = row['high']
        low = row['low']
        close = row['close']
        
        # Check if price tested the MA
        if abs(ma_value - target_price) > 50:
            continue
        
        # Determine role
        if low <= ma_value <= high:
            # Price touched MA
            if close > ma_value:
                roles.append('support')
            elif close < ma_value:
                roles.append('resistance')
        elif ma_value < low and close > low:
            # MA below, price stayed above
            roles.append('support')
        elif ma_value > high and close < high:
            # MA above, price stayed below
            roles.append('resistance')
    
    if not roles:
        return 'neutral'
    
    # Most common role
    support_count = roles.count('support')
    resistance_count = roles.count('resistance')
    
    if support_count > resistance_count:
        return 'support'
    elif resistance_count > support_count:
        return 'resistance'
    else:
        return 'neutral'


def detect_ma_role_flip(df, ma_column, hod_price, lod_price):
    """
    Detect if an MA flipped roles between HOD and LOD.
    
    Example: h14 at HOD (resistance) → same h14 at LOD (support)
    
    Returns dict with role transition info.
    """
    if ma_column not in df.columns:
        return None
    
    hod_role = track_ma_role_at_price(df, ma_column, hod_price, lookback_days=2)
    lod_role = track_ma_role_at_price(df, ma_column, lod_price, lookback_days=2)
    
    if hod_role == 'resistance' and lod_role == 'support':
        return {
            'ma': ma_column,
            'transition': 'resistance → support',
            'significance': 'HIGH',
            'hod_price': hod_price,
            'lod_price': lod_price,
            'description': f'{ma_column} rejected at HOD, then supported at LOD'
        }
    elif hod_role == 'support' and lod_role == 'resistance':
        return {
            'ma': ma_column,
            'transition': 'support → resistance',
            'significance': 'HIGH',
            'hod_price': hod_price,
            'lod_price': lod_price,
            'description': f'{ma_column} supported at LOD, then rejected at HOD'
        }
    
    return None


# ============================================================================
# ZONE RECOMMENDATION ENGINE
# ============================================================================

def generate_zone_recommendations(
    ohlc_df,
    travelers_df,
    report_time,
    current_price,
    max_zones=4,
    zone_tolerance=24,
    recip_max_spread=3.0
):
    """
    Generate 3-4 high-confidence zone recommendations.
    
    Process:
    1. Find high-rank Recip pairs
    2. Check for HUGE HMA confluence at those zones
    3. Look for wildcard M# confluence
    4. Check MA role transitions
    5. Score and rank zones
    6. Return top 3-4 recommendations
    """
    recommendations = []
    
    # Find Recip pairs
    recip_pairs = find_recip_pairs(
        travelers_df,
        max_spread=recip_max_spread,
        same_feed_only=True
    )
    
    if not recip_pairs:
        return []
    
    # Get high-rank MAs
    high_rank_mas = get_high_rank_mas(ohlc_df, min_rank=800)
    
    # Analyze each Recip pair as potential zone
    for recip in recip_pairs[:10]:  # Top 10 recips
        zone_price = recip['avg_output']
        distance_from_current = zone_price - current_price
        
        # Initialize zone score with recip score
        zone_score = recip['score']
        confluence_factors = []
        
        # Add recip info
        recip_desc = f"Recip pair: {recip['origin1']} m#{int(recip['m1'])} ↔ {recip['origin2']} m#{int(recip['m2'])}"
        confluence_factors.append({
            'type': 'Recip Pair',
            'description': recip_desc,
            'score': recip['score'],
            'is_recipe': recip['is_recipe']
        })
        
        # Check for wildcards near this zone
        wildcards = find_wildcards_near_zone(travelers_df, zone_price, zone_tolerance)
        
        for wildcard in wildcards[:3]:  # Top 3 wildcards
            wildcard_score = 30 if wildcard['m_num'] == 0 else 20  # M#0 gets higher score
            zone_score += wildcard_score
            
            wc_desc = f"Wildcard m#{int(wildcard['m_num'])} @ {wildcard['output']:.2f} ({wildcard['origin']})"
            confluence_factors.append({
                'type': 'Wildcard',
                'description': wc_desc,
                'score': wildcard_score,
                'm_num': wildcard['m_num']
            })
        
        # Check for HUGE HMA confluence
        for ma_info in high_rank_mas[:10]:  # Top 10 high-rank MAs
            ma_col = ma_info['column']
            
            # Get MA value at most recent candle
            recent_ma_val = ohlc_df[ma_col].dropna().iloc[-1] if ma_col in ohlc_df.columns else None
            
            if recent_ma_val and abs(recent_ma_val - zone_price) <= zone_tolerance:
                ma_score = min(ma_info['rank'] // 20, 50)  # Scale rank to score
                zone_score += ma_score
                
                ma_desc = f"{ma_col} @ {recent_ma_val:.2f} (rank {ma_info['rank']})"
                confluence_factors.append({
                    'type': 'High-Rank MA',
                    'description': ma_desc,
                    'score': ma_score,
                    'rank': ma_info['rank']
                })
        
        # Build recommendation
        direction = 'BUY' if distance_from_current < 0 else 'SELL'
        distance_abs = abs(distance_from_current)
        
        recommendation = {
            'zone_price': zone_price,
            'direction': direction,
            'distance_from_current': distance_abs,
            'zone_score': zone_score,
            'recip_info': recip,
            'confluence_factors': confluence_factors,
            'confidence': 'HIGH' if zone_score >= 150 else 'MEDIUM' if zone_score >= 100 else 'LOW',
            'description': f"{direction} zone: ~{distance_abs:.0f} units from current price @ {zone_price:.2f}"
        }
        
        recommendations.append(recommendation)
    
    # Sort by score descending
    recommendations.sort(key=lambda x: x['zone_score'], reverse=True)
    
    # Return top zones
    return recommendations[:max_zones]


def format_recommendation_report(recommendations, report_time, current_price):
    """Format recommendations as a readable report."""
    report_lines = []
    
    report_lines.append("=" * 80)
    report_lines.append(f"STRATEGIC ZONE RECOMMENDATIONS - {report_time.strftime('%Y-%m-%d %H:%M')}")
    report_lines.append(f"Current Price: {current_price:.2f}")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    for i, rec in enumerate(recommendations, 1):
        report_lines.append(f"{'='*80}")
        report_lines.append(f"ZONE #{i} - {rec['confidence']} CONFIDENCE (Score: {rec['zone_score']:.0f})")
        report_lines.append(f"{'='*80}")
        report_lines.append(f"Direction: {rec['direction']}")
        report_lines.append(f"Target Zone: {rec['zone_price']:.2f}")
        report_lines.append(f"Distance: ~{rec['distance_from_current']:.0f} units from current price")
        report_lines.append("")
        
        # Recip info
        recip = rec['recip_info']
        report_lines.append(f"Primary Signal: {recip['origin1']} m#{int(recip['m1'])} ↔ {recip['origin2']} m#{int(recip['m2'])}")
        report_lines.append(f"  Output Spread: {recip['output_spread']:.2f} units")
        if recip['is_recipe']:
            report_lines.append(f"  ✨ RECIPE PAIR: {recip['recipe_type']}")
        report_lines.append(f"  Arrivals: {recip['day1']} & {recip['day2']}")
        report_lines.append("")
        
        # Confluence factors
        report_lines.append("Confluence Factors:")
        for factor in rec['confluence_factors']:
            report_lines.append(f"  • {factor['type']}: {factor['description']} (+{factor['score']} points)")
        
        report_lines.append("")
    
    return "\n".join(report_lines)


# ============================================================================
# UPDATE TIME TRACKING
# ============================================================================

def get_next_origin_updates(current_time):
    """
    Get upcoming origin update times.
    Useful for knowing when to check for new wildcards or confluence.
    """
    updates = []
    
    for origin, times in ORIGIN_UPDATE_TIMES.items():
        for time_str in times:
            hour, minute = map(int, time_str.split(':'))
            update_time = current_time.replace(hour=hour, minute=minute, second=0, microsecond=0)
            
            # If time already passed today, get next occurrence
            if update_time < current_time:
                update_time += timedelta(days=1)
            
            updates.append({
                'origin': origin,
                'time': update_time,
                'hours_until': (update_time - current_time).total_seconds() / 3600
            })
    
    # Sort by time
    updates.sort(key=lambda x: x['time'])
    
    return updates[:5]  # Next 5 updates
