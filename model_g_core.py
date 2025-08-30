"""
Core G Model Detection Framework
Shared utilities and base functions for all G model variants.
"""

import pandas as pd
import streamlit as st
from collections import defaultdict
import numpy as np
from datetime import datetime

# --- Shared Constants ---
try:
    # Prefer using helpers' canonical group
    from a_helpers import GROUP_1B_TRAVELERS as _G1B_SRC
    GROUP_1B_TRAVELERS = set(_G1B_SRC)
except Exception:
    # Fallback (copied from helpers)
    GROUP_1B_TRAVELERS = {
        111, 107, 103, 96, 87, 77, 68, 60, 55, 50, 43, 41, 40, 39, 36, 30, 22, 14, 10, 6, 5, 3, 2, 1, 0,
        -1, -2, -3, -5, -6, -10, -14, -22, -30, -36, -39, -40, -41, -43, -50, -55, -60, -68, -77, -87, -96, -103, -107, -111
    }

# x0Pd.w lists (descending order; both polarities)
X0PDW_POS = [111, 107, 103, 96, 87, 77, 68, 60, 50, 40, 39, 36, 30, 22, 14, 10, 6, 5, 3, 2, 1, 0]
X0PDW_NEG = [-111, -107, -103, -96, -87, -77, -68, -60, -50, -40, -39, -36, -30, -22, -14, -10, -6, -5, -3, -2, -1, 0]

# --- Shared Utilities ---

def _round_m(m):
    """Round M# value to integer"""
    try:
        return int(round(float(m)))
    except Exception:
        return None

def _chronological_arrivals(items):
    """Return arrivals as pandas Timestamps; handle bypass mode (time-only) and ISO formats"""
    out = []
    for it in items:
        a = it.get("Arrival")
        try:
            # Handle various timestamp formats including bypass mode
            if isinstance(a, str):
                if ':' in a and len(a.split(':')) >= 2:
                    # Bypass mode: time only (e.g., "12:30:00" or "12:30")
                    # Add today's date for comparison purposes
                    from datetime import datetime
                    today = datetime.now().strftime('%Y-%m-%d')
                    full_timestamp = f"{today}T{a}"
                    parsed = pd.to_datetime(full_timestamp)
                else:
                    # ISO format or other date formats
                    parsed = pd.to_datetime(a)
            elif hasattr(a, 'isoformat'):
                parsed = pd.to_datetime(a)
            else:
                parsed = pd.to_datetime(str(a))
            out.append(parsed)
        except Exception:
            # If all parsing fails, use NaT
            out.append(pd.NaT)
    return out

def get_origin_type(origin):
    """
    Classify origin as Anchor, EPC, or Neither - handles bracket variations like [1], [2]
    """
    anchor_origins = ['spain', 'saturn', 'jupiter', 'kepler-62', 'kepler-44']
    epc_origins = ['trinidad', 'tobago', 'wasp-12b', 'macedonia']
    
    if not origin:
        return 'Neither'
    
    # Normalize origin name: convert to lowercase and remove brackets
    normalized = str(origin).lower().strip()
    
    # Remove bracket variations like [1], [2], etc.
    if '[' in normalized and ']' in normalized:
        normalized = normalized[:normalized.find('[')]
    
    if normalized in anchor_origins:
        return 'Anchor'
    elif normalized in epc_origins:
        return 'EPC'
    else:
        return 'Neither'

def group_by_proximity(outputs, proximity_threshold):
    """Group outputs by proximity threshold"""
    if not outputs:
        return []
    
    sorted_outputs = sorted(outputs, key=lambda x: x['Output'])
    groups = []
    current_group = [sorted_outputs[0]]
    
    for i in range(1, len(sorted_outputs)):
        current_output = sorted_outputs[i]['Output']
        last_output = current_group[-1]['Output']
        
        if abs(current_output - last_output) <= proximity_threshold:
            current_group.append(sorted_outputs[i])
        else:
            groups.append(current_group)
            current_group = [sorted_outputs[i]]
    
    groups.append(current_group)
    return groups

def has_required_origin(group):
    """Check if group contains required Anchor or EPC origin"""
    for item in group:
        origin_type = get_origin_type(item['Origin'])
        if origin_type in ['Anchor', 'EPC']:
            return True
    return False

def classify_by_day(group):
    """Classify sequence by day (today vs other day)"""
    current_date = datetime.now().date()
    
    for item in group:
        try:
            arrival = pd.to_datetime(item['Arrival'])
            if arrival.date() == current_date:
                return 'today'
        except:
            continue
    
    return 'other_day'

def find_temporal_descending_sequences(group):
    """Find all valid temporal descending sequences within a group"""
    if len(group) < 3:
        return []
    
    def get_sort_key(item):
        arrival = item['Arrival']
        try:
            # Handle various timestamp formats including bypass mode
            if isinstance(arrival, str):
                if ':' in arrival and len(arrival.split(':')) >= 2:
                    # Bypass mode: time only format
                    from datetime import datetime
                    today = datetime.now().strftime('%Y-%m-%d')
                    full_timestamp = f"{today}T{arrival}"
                    return pd.to_datetime(full_timestamp)
                else:
                    # ISO format or other date formats
                    return pd.to_datetime(arrival)
            elif hasattr(arrival, 'isoformat'):
                return pd.to_datetime(arrival) if not isinstance(arrival, pd.Timestamp) else arrival
            else:
                return pd.to_datetime(str(arrival))
        except:
            return pd.NaT
    
    sorted_by_time = sorted(group, key=get_sort_key)
    valid_sequences = []
    
    # Check all possible subsequences of length 3 or more
    for i in range(len(sorted_by_time)):
        for j in range(i + 3, len(sorted_by_time) + 1):
            subseq = sorted_by_time[i:j]
            m_values = [abs(float(item['M #'])) for item in subseq]
            
            # Check if M# values are in descending order
            is_descending = all(m_values[k] >= m_values[k+1] for k in range(len(m_values)-1))
            
            if is_descending:
                valid_sequences.append(subseq)
    
    # Remove subsequences that are contained within larger sequences
    filtered_sequences = []
    for seq in valid_sequences:
        is_contained = False
        for other_seq in valid_sequences:
            if seq != other_seq and len(seq) < len(other_seq):
                if is_subsequence_contained(seq, other_seq):
                    is_contained = True
                    break
        if not is_contained:
            filtered_sequences.append(seq)
    
    return filtered_sequences

def is_subsequence_contained(small_seq, large_seq):
    """Check if small_seq is contained within large_seq"""
    if len(small_seq) > len(large_seq):
        return False
    
    # Convert to comparable format (using M# and normalized Arrival)
    def normalize_arrival(arrival):
        try:
            if isinstance(arrival, str):
                if ':' in arrival and len(arrival.split(':')) >= 2:
                    # Bypass mode: time only format
                    from datetime import datetime
                    today = datetime.now().strftime('%Y-%m-%d')
                    full_timestamp = f"{today}T{arrival}"
                    return pd.to_datetime(full_timestamp).isoformat()
                else:
                    # ISO format or other date formats
                    return pd.to_datetime(arrival).isoformat()
            elif hasattr(arrival, 'isoformat'):
                return pd.to_datetime(arrival).isoformat() if not isinstance(arrival, pd.Timestamp) else arrival.isoformat()
            else:
                return pd.to_datetime(str(arrival)).isoformat()
        except:
            return str(arrival)
    
    small_items = [(float(item['M #']), normalize_arrival(item['Arrival'])) for item in small_seq]
    large_items = [(float(item['M #']), normalize_arrival(item['Arrival'])) for item in large_seq]
    
    # Check if small_items is a subsequence of large_items
    i = 0
    for large_item in large_items:
        if i < len(small_items) and small_items[i] == large_item:
            i += 1
    
    return i == len(small_items)

def ends_with_m50_and_anchor(sequence):
    """Check if sequence ends with M# 50 and Anchor origin"""
    if not sequence:
        return False
    
    # Sort by time to get chronological order
    def get_sort_key(item):
        arrival = item['Arrival']
        try:
            if isinstance(arrival, str):
                if ':' in arrival and len(arrival.split(':')) >= 2:
                    # Bypass mode: time only format
                    from datetime import datetime
                    today = datetime.now().strftime('%Y-%m-%d')
                    full_timestamp = f"{today}T{arrival}"
                    return pd.to_datetime(full_timestamp)
                else:
                    # ISO format or other date formats
                    return pd.to_datetime(arrival)
            elif hasattr(arrival, 'isoformat'):
                return pd.to_datetime(arrival) if not isinstance(arrival, pd.Timestamp) else arrival
            else:
                return pd.to_datetime(str(arrival))
        except:
            return pd.NaT
    
    sorted_by_time = sorted(sequence, key=get_sort_key)
    last_item = sorted_by_time[-1]
    
    # Check if last M# is 50 and origin is Anchor
    last_m = abs(float(last_item['M #']))
    last_origin_type = get_origin_type(last_item['Origin'])
    
    return last_m == 50.0 and last_origin_type == 'Anchor'

def ends_with_opposite_m50_pair(sequence):
    """Check if sequence ends with opposite M# 50 pair (+50 and -50)"""
    if len(sequence) < 2:
        return False

    # Get the last two items in temporal order
    def get_sort_key(item):
        arrival = item['Arrival']
        try:
            if isinstance(arrival, str):
                if ':' in arrival and len(arrival.split(':')) >= 2:
                    # Bypass mode: time only format
                    from datetime import datetime
                    today = datetime.now().strftime('%Y-%m-%d')
                    full_timestamp = f"{today}T{arrival}"
                    return pd.to_datetime(full_timestamp)
                else:
                    # ISO format or other date formats
                    return pd.to_datetime(arrival)
            elif hasattr(arrival, 'isoformat'):
                return pd.to_datetime(arrival) if not isinstance(arrival, pd.Timestamp) else arrival
            else:
                return pd.to_datetime(str(arrival))
        except:
            return pd.NaT

    sorted_by_time = sorted(sequence, key=get_sort_key)
    last_two = sorted_by_time[-2:]

    # Check if the last two are 50 and -50 (in any order)
    last_m_values = [float(item['M #']) for item in last_two]
    return set(last_m_values) == {50.0, -50.0}
