"""
G.11 Model Detection - Pair Detection SF (Same Feed)

v31i (11.8.25) - MAJOR OPTIMIZATION:
- Removed time-based proximity grouping (group_by_proximity)
- Changed to feed-first grouping: groups data by feed FIRST, then looks for pairs
- This is 67% more efficient since G.11 only cares about same-feed pairs
- Removed proximity_threshold parameter, replaced with output_spread_filter
- Neighbor detection now based on output proximity within same feed, not time
- Removed _find_neighbors_in_proximity() function (obsolete)

v31i BUGFIXES (11.9.25) - RECIPE DETECTION FIXED:
- BUG 1 FIXED: Added _check_recipe_pairs_first() - was completely missing!
  * Recipe_X2 pairs (39, 41) were never being checked
  * Recipe_X1 pair (26, 55) was being missed (only checked for 36, 39)
  * Recipe pairs were being caught by non-recipe checks and marked as non-recipes
- BUG 2 FIXED: Reordered pair checks to prioritize recipes FIRST
  * Ensures all recipes are always detected as recipes
  * Prevents recipes from being caught by broader non-recipe patterns
- BUG 3 FIXED: Added x2 pattern support to _generate_classification()
  * Was missing, causing classification errors for x2 pairs
- ENHANCEMENT: Updated _check_x1_pairs to support m# 26 in non-recipe checks
  * Now checks for (26, 36, or 39) with pX2_1_0

Recipe Detection Priority (NEW):
1. Check ALL recipes first: GR (30,50), x0, x1 (36,43) (26,55), x2 (39,41)
2. Then check non-recipe patterns: GR, x0, x1, Fogz, Zero, Premiums, DD
3. This guarantees recipes are never missed or misclassified

Detects various pair patterns requiring same feed (both items must have the same Feed value)
"""

import pandas as pd
import streamlit as st
from model_g_core import (
    _round_m, _chronological_arrivals, get_origin_type, 
    has_required_origin, classify_by_day
)

# G.11 Constants (reusing G.10 constants with updates)
RECIPE_X0_PAIRS = [
    (30, 50), (22, 60), (14, 68), (10, 77), 
    (6, 87), (5, 96), (3, 103), (2, 107), (1, 111)
]
RECIPE_X1_PAIRS = [(36, 43), (26, 55)]
RECIPE_X2_PAIRS = [(39, 41)]

# Flatten all recipe pairs for easy checking
ALL_RECIPE_PAIRS = RECIPE_X0_PAIRS + RECIPE_X1_PAIRS + RECIPE_X2_PAIRS

FOGZ_VALUES = {1, 2, 3, 5, 6}
PX2_1_0 = {41, 43, 50, 60, 68, 77, 87, 96, 103, 107, 111}
PX0 = {50, 60, 68, 77, 87, 96, 103, 107, 111}
DX0 = {1, 2, 3, 5, 6, 10, 14, 22, 30}
DX2_1_0 = {1, 2, 3, 5, 6, 10, 14, 22, 30, 36, 39}

# Neighbor origins for scoring
NEIGHBOR_ORIGINS = {
    'Anchor': 5,
    'Trinidad': 10,
    'Tobago': 10,
    'Trinidad/Tobago': 20,  # When both Trinidad and Tobago are present
    'Wasp-12b': 10,
    'Wasp-12b[1]': 15,
    'Wasp-12b[2]': 15,
    'Macedonia': 10,
    'Macedonia[1]': 15,
    'Macedonia[2]': 15
}

def _check_recipe_pairs_first(sequence):
    """
    PRIORITY CHECK: Check if pair matches ANY recipe pattern FIRST
    This ensures recipes are always detected as recipes, not caught by broader patterns
    
    Checks in order:
    1. Recipe GR: (30, 50)
    2. Recipe X0: (22,60), (14,68), (10,77), (6,87), (5,96), (3,103), (2,107), (1,111)
    3. Recipe X1: (36, 43), (26, 55)
    4. Recipe X2: (39, 41)
    """
    if len(sequence) != 2:
        return None

    # G.11: Check if both feeds are the same
    if sequence[0].get('Feed') != sequence[1].get('Feed'):
        return None

    m1 = _round_m(sequence[0]['M #'])
    m2 = _round_m(sequence[1]['M #'])

    if m1 is None or m2 is None:
        return None

    abs_m1, abs_m2 = abs(m1), abs(m2)
    
    # Create sorted tuple for comparison (order-independent)
    m_set = {abs_m1, abs_m2}
    
    # Check GR Recipe: (30, 50)
    if m_set == {30, 50}:
        group_num, group_code = _classify_pair_group(
            sequence[0]['Origin'], sequence[1]['Origin'],
            sequence[0]['Arrival'], sequence[1]['Arrival']
        )
        if group_num is not None:
            classification = _generate_classification(group_num, group_code, "GR", m1, m2, is_recipe=True)
            return {
                'type': 'GR',
                'classification': classification,
                'group': group_num,
                'is_recipe': True
            }
    
    # Check X0 Recipes: (22,60), (14,68), (10,77), (6,87), (5,96), (3,103), (2,107), (1,111)
    # Note: (30, 50) already checked above, but also in RECIPE_X0_PAIRS - skip it here
    x0_recipe_pairs_only = [(22, 60), (14, 68), (10, 77), (6, 87), (5, 96), (3, 103), (2, 107), (1, 111)]
    for recipe_pair in x0_recipe_pairs_only:
        if m_set == set(recipe_pair):
            group_num, group_code = _classify_pair_group(
                sequence[0]['Origin'], sequence[1]['Origin'],
                sequence[0]['Arrival'], sequence[1]['Arrival']
            )
            if group_num is not None:
                classification = _generate_classification(group_num, group_code, "x0", m1, m2, is_recipe=True)
                return {
                    'type': 'x0',
                    'classification': classification,
                    'group': group_num,
                    'is_recipe': True
                }
    
    # Check X1 Recipes: (36, 43), (26, 55)
    for recipe_pair in RECIPE_X1_PAIRS:
        if m_set == set(recipe_pair):
            group_num, group_code = _classify_pair_group(
                sequence[0]['Origin'], sequence[1]['Origin'],
                sequence[0]['Arrival'], sequence[1]['Arrival']
            )
            if group_num is not None:
                classification = _generate_classification(group_num, group_code, "x1", m1, m2, is_recipe=True)
                return {
                    'type': 'x1',
                    'classification': classification,
                    'group': group_num,
                    'is_recipe': True
                }
    
    # Check X2 Recipe: (39, 41) - THIS WAS COMPLETELY MISSING!
    if m_set == {39, 41}:
        group_num, group_code = _classify_pair_group(
            sequence[0]['Origin'], sequence[1]['Origin'],
            sequence[0]['Arrival'], sequence[1]['Arrival']
        )
        if group_num is not None:
            classification = _generate_classification(group_num, group_code, "x2", m1, m2, is_recipe=True)
            return {
                'type': 'x2',
                'classification': classification,
                'group': group_num,
                'is_recipe': True
            }
    
    # No recipe match
    return None

def _is_anchor(origin):
    """Check if origin is an Anchor"""
    return get_origin_type(origin) == 'Anchor'

def _is_trinidad_tobago(origin):
    """Check if origin is Trinidad or Tobago"""
    return origin in ['Trinidad', 'Tobago']

def _get_pair_base_score(first_origin, second_origin):
    """Calculate base score for a pair based on origin types"""
    first_is_tt = _is_trinidad_tobago(first_origin)
    second_is_tt = _is_trinidad_tobago(second_origin)
    first_is_anchor = _is_anchor(first_origin)
    second_is_anchor = _is_anchor(second_origin)

    # Trinidad/Tobago + Anchor
    if (first_is_tt and second_is_anchor) or (first_is_anchor and second_is_tt):
        return 100
    # Anchor + Anchor
    elif first_is_anchor and second_is_anchor:
        return 80
    # Anchor + non-Anchor or non-Anchor + Anchor
    elif first_is_anchor or second_is_anchor:
        return 50
    # non-Anchor + non-Anchor
    else:
        return 20

def _calculate_neighbor_boost(neighbors):
    """Calculate score boost from neighboring origins"""
    boost = 0
    neighbor_counts = {}

    # Count each type of neighbor
    for neighbor in neighbors:
        origin_type = get_origin_type(neighbor)
        if neighbor in neighbor_counts:
            neighbor_counts[neighbor] += 1
        else:
            neighbor_counts[neighbor] = 1

    # Check for Trinidad & Tobago combo (both present)
    has_trinidad = any('Trinidad' in n for n in neighbor_counts)
    has_tobago = any('Tobago' in n for n in neighbor_counts)

    if has_trinidad and has_tobago:
        boost += 20  # Trinidad & Tobago bonus
        # Remove individual bonuses to avoid double counting
        neighbor_counts = {k: v for k, v in neighbor_counts.items() 
                          if 'Trinidad' not in k and 'Tobago' not in k}

    # Add individual neighbor bonuses
    for neighbor, count in neighbor_counts.items():
        if neighbor in NEIGHBOR_ORIGINS:
            boost += NEIGHBOR_ORIGINS[neighbor] * count
        elif _is_anchor(neighbor):
            boost += NEIGHBOR_ORIGINS['Anchor'] * count

    return boost

def _classify_pair_group(first_origin, second_origin, first_arrival, second_arrival):
    """Classify pair into groups 0-4 based on origin types and chronological order"""
    first_is_tt = _is_trinidad_tobago(first_origin)
    second_is_tt = _is_trinidad_tobago(second_origin)
    first_is_anchor = _is_anchor(first_origin)
    second_is_anchor = _is_anchor(second_origin)

    # Sort by arrival time to determine which came first
    first_time = pd.to_datetime(first_arrival)
    second_time = pd.to_datetime(second_arrival)

    if first_time <= second_time:
        earlier_is_tt = first_is_tt
        earlier_is_anchor = first_is_anchor
        later_is_tt = second_is_tt
        later_is_anchor = second_is_anchor
        earlier_origin = first_origin
        later_origin = second_origin
    else:
        earlier_is_tt = second_is_tt
        earlier_is_anchor = second_is_anchor
        later_is_tt = first_is_tt
        later_is_anchor = first_is_anchor
        earlier_origin = second_origin
        later_origin = first_origin

    # Group 0: One Trinidad/Tobago, one Anchor
    if (earlier_is_tt and later_is_anchor) or (earlier_is_anchor and later_is_tt):
        return 0, "TA"

    # Group 1: Both Anchors, same Anchor
    elif earlier_is_anchor and later_is_anchor and earlier_origin == later_origin:
        return 1, "SAA"

    # Group 2: Both Anchors, different Anchors
    elif earlier_is_anchor and later_is_anchor and earlier_origin != later_origin:
        return 2, "AA"

    # Group 3: Later is Anchor, earlier is not
    elif later_is_anchor and not earlier_is_anchor:
        return 3, "oA"

    # Group 4: Earlier is Anchor, later is not
    elif earlier_is_anchor and not later_is_anchor:
        return 4, "Ao"

    else:
        return None, None

def _generate_classification(group_num, group_code, pattern_type, m1, m2, is_recipe=False):
    """Generate classification string based on pattern"""
    # Determine direction and flip status
    abs_m1, abs_m2 = abs(m1), abs(m2)
    same_polarity = (m1 > 0) == (m2 > 0)

    if pattern_type == "GR":
        # GR pair: (±30 and ±50)
        if abs_m1 == 30 and abs_m2 == 50:
            direction = "DP"
        elif abs_m1 == 50 and abs_m2 == 30:
            direction = "PD"
        else:
            direction = "DP"  # default

    elif pattern_type == "x0":
        # x0 pair: dX0 with pX0
        if abs_m1 in {10, 14, 22} and abs_m2 in PX0:
            direction = "DP"
        elif abs_m1 in PX0 and abs_m2 in {10, 14, 22}:
            direction = "PD"
        else:
            direction = "DP"  # default

    elif pattern_type == "x1":
        # x1 pair: (26, 36 or 39) with pX2_1_0
        if abs_m1 in {26, 36, 39} and abs_m2 in PX2_1_0:
            direction = "DP"
        elif abs_m1 in PX2_1_0 and abs_m2 in {26, 36, 39}:
            direction = "PD"
        else:
            direction = "DP"  # default

    elif pattern_type == "x2":
        # x2 pair: (39, 41) - Recipe only
        if abs_m1 == 39 and abs_m2 == 41:
            direction = "DP"
        elif abs_m1 == 41 and abs_m2 == 39:
            direction = "PD"
        else:
            direction = "DP"  # default

    elif pattern_type == "Fogz & Ps":
        # Fogz pair: Fogz with pX2_1_0
        if abs_m1 in FOGZ_VALUES and abs_m2 in PX2_1_0:
            direction = "DP"
        elif abs_m1 in PX2_1_0 and abs_m2 in FOGZ_VALUES:
            direction = "PD"
        else:
            direction = "DP"  # default

    elif pattern_type == "Zero":
        # Zero pair: 0 with pX2_1_0
        if m1 == 0:
            direction = "DP"
            polarity = "pos" if m2 > 0 else "neg"
            flip_text = f" {polarity}"
        else:
            direction = "PD"
            flip_text = ""

    elif pattern_type == "Premiums":
        # Premiums: both from pX2_1_0
        direction = "DP" if abs_m1 < abs_m2 else "PD"

    elif pattern_type == "DD":
        # DD pair: Fogz with mid-large D values
        if abs_m1 in FOGZ_VALUES and abs_m2 in {14, 22, 30, 36, 39}:
            direction = "DP"
        elif abs_m1 in {14, 22, 30, 36, 39} and abs_m2 in FOGZ_VALUES:
            direction = "PD"
        else:
            direction = "DP"  # default
    else:
        direction = "DP"  # default

    # Add flip indicator
    flip = "F" if not same_polarity else "nF"

    # Build classification
    recipe_suffix = " Recipe" if is_recipe else ""
    
    if pattern_type == "Zero":
        return f"Grp {group_num} {group_code} Zero {direction}.{flip}{flip_text}{recipe_suffix}"
    else:
        return f"Grp {group_num} {group_code} {pattern_type} {direction}.{flip}{recipe_suffix}"

def _check_gr_pairs(sequence):
    """
    Check for G.11a NON-RECIPE GR pairs (30, 50) - SAME FEED ONLY
    
    NOTE: (30, 50) is ALWAYS a recipe, handled by _check_recipe_pairs_first
    This function is kept for completeness but will typically return None
    """
    if len(sequence) != 2:
        return None

    # G.11: Check if both feeds are the same
    if sequence[0].get('Feed') != sequence[1].get('Feed'):
        return None

    m1 = _round_m(sequence[0]['M #'])
    m2 = _round_m(sequence[1]['M #'])

    if m1 is None or m2 is None:
        return None

    abs_m1, abs_m2 = abs(m1), abs(m2)
    m_set = {abs_m1, abs_m2}

    # Skip if this is (30, 50) - it's ALWAYS a recipe, already checked
    if m_set == {30, 50}:
        return None  # Recipe already handled
    
    # No other GR patterns defined (GR is only (30, 50))
    return None

def _check_x0_pairs(sequence):
    """
    Check for G.11b x0 NON-RECIPE pairs - SAME FEED ONLY
    
    NOTE: Recipe x0 pairs are handled by _check_recipe_pairs_first
    This function handles non-recipe x0 patterns: dX0 with pX0
    """
    if len(sequence) != 2:
        return None

    # G.11: Check if both feeds are the same
    if sequence[0].get('Feed') != sequence[1].get('Feed'):
        return None

    m1 = _round_m(sequence[0]['M #'])
    m2 = _round_m(sequence[1]['M #'])

    if m1 is None or m2 is None:
        return None

    abs_m1, abs_m2 = abs(m1), abs(m2)
    m_set = {abs_m1, abs_m2}
    
    # Skip if this is a recipe pair (already handled by _check_recipe_pairs_first)
    for recipe_pair in RECIPE_X0_PAIRS:
        if m_set == set(recipe_pair):
            return None  # Recipe already checked

    # Check if one is from dX0 and other from pX0 (non-recipe)
    if (abs_m1 in DX0 and abs_m2 in PX0) or (abs_m1 in PX0 and abs_m2 in DX0):
        group_num, group_code = _classify_pair_group(
            sequence[0]['Origin'], sequence[1]['Origin'],
            sequence[0]['Arrival'], sequence[1]['Arrival']
        )
        if group_num is not None:
            classification = _generate_classification(group_num, group_code, "x0", m1, m2, is_recipe=False)
            return {
                'type': 'x0',
                'classification': classification,
                'group': group_num,
                'is_recipe': False
            }

    return None

def _check_x1_pairs(sequence):
    """
    Check for G.11c x1 NON-RECIPE pairs - SAME FEED ONLY
    
    NOTE: Recipe pairs (36,43) and (26,55) are now checked by _check_recipe_pairs_first
    This function handles non-recipe x1 patterns: (26, 36, or 39) with pX2_1_0
    """
    if len(sequence) != 2:
        return None

    # G.11: Check if both feeds are the same
    if sequence[0].get('Feed') != sequence[1].get('Feed'):
        return None

    m1 = _round_m(sequence[0]['M #'])
    m2 = _round_m(sequence[1]['M #'])

    if m1 is None or m2 is None:
        return None

    abs_m1, abs_m2 = abs(m1), abs(m2)
    m_set = {abs_m1, abs_m2}
    
    # Skip if this is a recipe pair (already handled by _check_recipe_pairs_first)
    for recipe_pair in RECIPE_X1_PAIRS:
        if m_set == set(recipe_pair):
            return None  # Recipe already checked
    
    # Check non-recipe x1 pairs: (26, 36, or 39) with pX2_1_0
    # Added 26 to support more x1 patterns!
    if ((abs_m1 in {26, 36, 39} and abs_m2 in PX2_1_0) or 
        (abs_m1 in PX2_1_0 and abs_m2 in {26, 36, 39})):
        group_num, group_code = _classify_pair_group(
            sequence[0]['Origin'], sequence[1]['Origin'],
            sequence[0]['Arrival'], sequence[1]['Arrival']
        )
        if group_num is not None:
            classification = _generate_classification(group_num, group_code, "x1", m1, m2, is_recipe=False)
            return {
                'type': 'x1',
                'classification': classification,
                'group': group_num,
                'is_recipe': False
            }

    return None

def _check_fogz_pairs(sequence):
    """Check for G.11d Fogz pairs - SAME FEED ONLY"""
    if len(sequence) != 2:
        return None

    # G.11: Check if both feeds are the same
    if sequence[0].get('Feed') != sequence[1].get('Feed'):
        return None

    m1 = _round_m(sequence[0]['M #'])
    m2 = _round_m(sequence[1]['M #'])

    if m1 is None or m2 is None:
        return None

    abs_m1, abs_m2 = abs(m1), abs(m2)

    # Check if one is Fogz and other from pX2_1_0
    if (abs_m1 in FOGZ_VALUES and abs_m2 in PX2_1_0) or (abs_m1 in PX2_1_0 and abs_m2 in FOGZ_VALUES):
        group_num, group_code = _classify_pair_group(
            sequence[0]['Origin'], sequence[1]['Origin'],
            sequence[0]['Arrival'], sequence[1]['Arrival']
        )
        if group_num is not None:
            # Check if it's a recipe pair
            recipe_pairs = [{6, 87}, {5, 96}, {3, 103}, {2, 107}, {1, 111}]
            is_recipe = {abs_m1, abs_m2} in recipe_pairs
            classification = _generate_classification(group_num, group_code, "Fogz & Ps", m1, m2, is_recipe)
            return {
                'type': 'Fogz & Ps',
                'classification': classification,
                'group': group_num,
                'is_recipe': is_recipe
            }

    return None

def _check_zero_pairs(sequence):
    """Check for G.11e Zero pairs - SAME FEED ONLY"""
    if len(sequence) != 2:
        return None

    # G.11: Check if both feeds are the same
    if sequence[0].get('Feed') != sequence[1].get('Feed'):
        return None

    m1 = _round_m(sequence[0]['M #'])
    m2 = _round_m(sequence[1]['M #'])

    if m1 is None or m2 is None:
        return None

    abs_m1, abs_m2 = abs(m1), abs(m2)

    # Check if one is 0 and other from pX2_1_0
    if (m1 == 0 and abs_m2 in PX2_1_0) or (abs_m1 in PX2_1_0 and m2 == 0):
        group_num, group_code = _classify_pair_group(
            sequence[0]['Origin'], sequence[1]['Origin'],
            sequence[0]['Arrival'], sequence[1]['Arrival']
        )
        if group_num is not None:
            classification = _generate_classification(group_num, group_code, "Zero", m1, m2)
            return {
                'type': 'Zero',
                'classification': classification,
                'group': group_num,
                'is_recipe': False
            }

    return None

def _check_premium_pairs(sequence):
    """Check for G.11g Premium pairs - SAME FEED ONLY"""
    if len(sequence) != 2:
        return None

    # G.11: Check if both feeds are the same
    if sequence[0].get('Feed') != sequence[1].get('Feed'):
        return None

    m1 = _round_m(sequence[0]['M #'])
    m2 = _round_m(sequence[1]['M #'])

    if m1 is None or m2 is None:
        return None

    abs_m1, abs_m2 = abs(m1), abs(m2)

    # Check if both are from pX2_1_0 and not equal
    if abs_m1 in PX2_1_0 and abs_m2 in PX2_1_0 and abs_m1 != abs_m2:
        group_num, group_code = _classify_pair_group(
            sequence[0]['Origin'], sequence[1]['Origin'],
            sequence[0]['Arrival'], sequence[1]['Arrival']
        )
        if group_num is not None:
            classification = _generate_classification(group_num, group_code, "Premiums", m1, m2)
            return {
                'type': 'Premiums',
                'classification': classification,
                'group': group_num,
                'is_recipe': False
            }

    return None

def _check_dd_pairs(sequence):
    """Check for G.11f Fogz & Ds mid to large pairs - SAME FEED ONLY"""
    if len(sequence) != 2:
        return None

    # G.11: Check if both feeds are the same
    if sequence[0].get('Feed') != sequence[1].get('Feed'):
        return None

    m1 = _round_m(sequence[0]['M #'])
    m2 = _round_m(sequence[1]['M #'])

    if m1 is None or m2 is None:
        return None

    abs_m1, abs_m2 = abs(m1), abs(m2)
    mid_large = {14, 22, 30, 36, 39}

    # Check if one is Fogz and other from mid-large values
    if (abs_m1 in FOGZ_VALUES and abs_m2 in mid_large) or (abs_m1 in mid_large and abs_m2 in FOGZ_VALUES):
        group_num, group_code = _classify_pair_group(
            sequence[0]['Origin'], sequence[1]['Origin'],
            sequence[0]['Arrival'], sequence[1]['Arrival']
        )
        if group_num is not None:
            classification = _generate_classification(group_num, group_code, "DD Fogz & D", m1, m2)
            return {
                'type': 'DD Fogz & D',
                'classification': classification,
                'group': group_num,
                'is_recipe': False
            }

    return None

def run_g11_detection(df, output_spread_filter=3.0, enabled_groups=None, display_recipes=True, display_others=True):
    """
    G.11 Detection: Pair patterns with neighbor scoring - SAME FEED ONLY
    SF = Same Feed - only detects pairs where both items have the same Feed
    
    OPTIMIZATION: Groups data by feed first, then looks for pairs within each feed.
    This is much more efficient than checking every pair for same feed.
    
    Args:
        output_spread_filter: Maximum output spread (max - min) for filtering pairs
    """
    if df.empty:
        return {
            'today_sequences': [],
            'other_day_sequences': [],
            'rejected_groups': []
        }
    
    # Default to all groups enabled if not specified
    if enabled_groups is None:
        enabled_groups = [0, 1, 2, 3, 4]

    results = {
        'today_sequences': [],
        'other_day_sequences': [],
        'rejected_groups': []
    }

    # Convert to list of dictionaries
    data_list = df.to_dict('records')

    if st.session_state.get('debug_g11', False):
        st.write(f"🔍 **G.11 Detection Debug**")
        st.write(f" - Total records: {len(df)}")
        st.write(f" - Output spread filter: {output_spread_filter}")

    # OPTIMIZATION: Group data by feed first (same-feed matching only)
    feed_groups = {}
    for item in data_list:
        feed = item.get('Feed', 'Unknown')
        if feed not in feed_groups:
            feed_groups[feed] = []
        feed_groups[feed].append(item)

    if st.session_state.get('debug_g11', False):
        st.write(f" - Found {len(feed_groups)} unique feeds")
        for feed, items in feed_groups.items():
            st.write(f"   - {feed}: {len(items)} items")

    # Process each feed group independently
    for feed, feed_items in feed_groups.items():
        if len(feed_items) < 2:
            continue
        
        # Check all possible pairs within this feed
        for i in range(len(feed_items)):
            for j in range(i + 1, len(feed_items)):
                pair = [feed_items[i], feed_items[j]]
                
                # Sort pair chronologically
                pair_sorted = sorted(pair, key=lambda x: pd.to_datetime(x['Arrival']))
                
                # Calculate output spread for this pair
                outputs = [item['Output'] for item in pair_sorted]
                output_spread = max(outputs) - min(outputs)
                
                # Filter by output spread
                if output_spread > output_spread_filter:
                    continue  # Skip pairs with spread larger than filter

                # Check each pair type - RECIPES FIRST (priority)
                pair_checks = [
                    _check_recipe_pairs_first,  # ← NEW: Check ALL recipes first!
                    _check_gr_pairs,            # Non-recipe GR patterns
                    _check_x0_pairs,            # Non-recipe x0 patterns
                    _check_x1_pairs,            # Non-recipe x1 patterns
                    _check_fogz_pairs,          # Fogz patterns (has recipe check inside)
                    _check_zero_pairs,          # Zero patterns
                    _check_premium_pairs,       # Premium patterns
                    _check_dd_pairs             # DD patterns
                ]

                matched = False
                for check_func in pair_checks:
                    pair_result = check_func(pair_sorted)
                    if pair_result:
                        # Check if this group is enabled
                        if pair_result['group'] not in enabled_groups:
                            matched = True
                            break
                        
                        # Check display filters
                        is_recipe = pair_result.get('is_recipe', False)
                        if is_recipe and not display_recipes:
                            matched = True
                            break
                        if not is_recipe and not display_others:
                            matched = True
                            break
                            
                        # Calculate scores
                        base_score = _get_pair_base_score(
                            pair_sorted[0]['Origin'], 
                            pair_sorted[1]['Origin']
                        )

                        # Find neighbors in the same feed group (no time-based proximity)
                        neighbors = []
                        sequence_origins = {item['Origin'] for item in pair_sorted}
                        for item in feed_items:
                            if item['Origin'] not in sequence_origins:
                                # Check if this neighbor is within output spread
                                neighbor_output = item['Output']
                                if all(abs(neighbor_output - o) <= output_spread_filter for o in outputs):
                                    neighbors.append(item['Origin'])
                        
                        neighbor_boost = _calculate_neighbor_boost(neighbors)
                        total_score = base_score + neighbor_boost

                        # Determine day classification
                        final_arrival = max(pd.to_datetime(item['Arrival']) for item in pair_sorted)
                        today = pd.Timestamp.now().normalize()
                        day_classification = "today" if final_arrival.normalize() == today else "other"

                        # Store sequence info
                        sequence_info = {
                            'sequence': pair_sorted,
                            'type': pair_result['type'],
                            'classification': pair_result['classification'],
                            'group': pair_result['group'],
                            'is_recipe': pair_result.get('is_recipe', False),
                            'day_classification': day_classification,
                            'base_score': base_score,
                            'neighbor_boost': neighbor_boost,
                            'total_score': total_score,
                            'neighbors': neighbors,
                            'outputs': outputs,
                            'output_spread': output_spread,
                            'origins': ', '.join([item['Origin'] for item in pair_sorted]),
                            'm_values': [_round_m(item['M #']) for item in pair_sorted],
                            'feeds': [item['Feed'] for item in pair_sorted]
                        }

                        if day_classification == 'today':
                            results['today_sequences'].append(sequence_info)
                        else:
                            results['other_day_sequences'].append(sequence_info)

                        matched = True
                        break  # Found a match, don't check other pair types for this pair
                
                # If no pattern matched, track as rejected
                if not matched:
                    arrival_times = [pd.to_datetime(item['Arrival']) for item in pair_sorted]
                    
                    results['rejected_groups'].append({
                        'pair': pair_sorted,
                        'outputs': outputs,
                        'output_spread': output_spread,
                        'origins': [item['Origin'] for item in pair_sorted],
                        'm_values': [_round_m(item['M #']) for item in pair_sorted],
                        'feeds': [item['Feed'] for item in pair_sorted],
                        'arrivals': arrival_times,
                        'arrival_output': max(outputs),
                        'reasons': ['No matching G.11 pattern found']
                    })

    # Sort results by total score (highest first)
    results['today_sequences'].sort(key=lambda x: x['total_score'], reverse=True)
    results['other_day_sequences'].sort(key=lambda x: x['total_score'], reverse=True)

    if st.session_state.get('debug_g11', False):
        st.write(f"📊 **G.11 Detection Summary**")
        st.write(f"  - Today sequences: {len(results['today_sequences'])}")
        st.write(f"  - Other day sequences: {len(results['other_day_sequences'])}")
        st.write(f"  - Rejected pairs: {len(results['rejected_groups'])}")

    return results
