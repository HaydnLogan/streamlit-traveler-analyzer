"""
Model Definitions v21 - All 23 Trading Models

This file contains definitions for all trading models used in the Swing Analysis Tool.
Each model defines Pass 1 and Pass 2 M# lists, and any special matching logic.

Models are organized into categories:
- FOGZ models (1-3): FOGZ matched with Premiums or Discounts
- Large Discount models (4-6): Large Discounts matched with Premiums or other Discounts
- Reciprocal models (7-8): Reciprocal pairs
- Premium/Discount pattern models (9-18): Specific x0, x1, xD0, xD1, xC patterns
- Premium/Premium models (19-23): Premium-to-Premium patterns

March 2026 update:
Added Bravo and Charlie X1 & X2. Recip Combos (64, 17), (53, 29), and (46, 32)
"""

# ============================================================================
# M# LIST DEFINITIONS
# ============================================================================

# Core Lists
FOGZ = {0, 1, -1, 2, -2, 3, -3, 5, -5, 6, -6}
LRG_D = {36, -36, 38, -38, 39, -39}
MED_D = {10, -10, 14, -14, 22, -22, 30, -30}

# Premium Lists
P_WX012 = {40, -40, 41, -41, 43, -43, 46, -46, 50, -50, 53, -53, 55, -55, 60, -60, 64, -64, 68, -68, 77, -77, 87, -87, 96, -96, 103, -103, 107, -107, 111, -111}
P_XD012 = {40, -40, 42, -42, 45, -45, 54, -54, 67, -67, 74, -74, 80, -80, 85, -85, 89, -89, 92, -92, 95, -95, 97.2, -97.2, 98.2, -98.2, 99.3, -99.3}
P_ALL = {40, -40, 41, -41, 42, -42, 43, -43, 45, -45, 46, -46, 50, -50, 53, -53, 54, -54, 55, -55, 60, -60, 64, -64, 67, -67, 68, -68, 74, -74, 77, -77, 80, -80, 85, -85, 87, -87, 89, -89, 92, -92, 95, -95, 96, -96, 97.2, -97.2, 98.2, -98.2, 99.3, -99.3, 103, -103, 107, -107, 111, -111}

# Pattern-specific Premium Lists
P_X0 = {50, -50, 60, -60, 68, -68, 77, -77, 87, -87, 96, -96, 103, -103, 107, -107, 111, -111}
P_X012 = {41, -41, 43, -43, 46, -46, 50, -50, 53, -53, 55, -55, 60, -60, 64, -64, 68, -68, 77, -77, 87, -87, 96, -96, 103, -103, 107, -107, 111, -111}
P_XD0 = {40, -40, 54, -54, 67, -67, 74, -74, 80, -80, 85, -85, 89, -89, 92, -92, 95, -95, 97.2, -97.2, 98.2, -98.2, 99.3, -99.3}
P_XC = {47, -47, 57, -57, 71, -71, 93.5, -93.5, 101, -101}

# Discount Lists (Reciprocal)
RECIPS_D = {1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6, 10, -10, 12, -12, 14, -14, 15, -15, 17, -17, 22, -22, 24, -24, 26, -26, 27, -27, 29, -29, 30, -30, 31, -31, 32, -32, 33, -33, 36, -36, 38, -38, 39, -39}
RECIPS_P = {41, -41, 42, -42, 43, -43, 45, -45, 46, -46, 47, -47, 50, -50, 53, -53, 54, -54, 55, -55, 57, -57, 60, -60, 64, -64, 67, -67, 68, -68, 71, -71, 77, -77, 87, -87, 96, -96, 101, -101, 103, -103, 107, -107, 111, -111}
D_ALL = {0, 1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6, 10, -10, 12, -12, 14, -14, 15, -15, 17, -17, 21, -21, 22, -22, 24, -24, 25, -25, 26, -26, 27, -27, 29, -29, 30, -30, 31, -31, 32, -32, 33, -33, 36, -36, 37, -37, 38, -38, 39, -39}

# Pattern-specific Discount Lists
D_X0 = {1, -1, 2, -2, 3, -3, 5, -5, 6, -6, 10, -10, 14, -14, 22, -22, 30, -30}
D_X12 = {17, -17, 26, -26, 29, -29, 32, -32, 36, -36, 39, -39}
D_XD0 = {15, -15, 27, -27}
D_XD12 = {33, -33, 38, -38}
D_XC = {4, -4, 12, -12, 24, -24, 31, -31}

# ============================================================================
# MODEL DEFINITIONS
# ============================================================================

MODELS = {
    # FOGZ Models (1-3)
    'Fogz PD': {
        'number': 1,
        'display_name': 'Fogz PD',
        'description': 'FOGZ arriving matched with Premium M#s',
        'pass1': FOGZ,
        'pass2': P_WX012,
        'check_recip': False,
        'special_matching': None
    },
    
    'Fogz Lrg DD': {
        'number': 2,
        'display_name': 'Fogz Lrg DD',
        'description': 'FOGZ arriving matched with Large Discount M#s',
        'pass1': FOGZ,
        'pass2': LRG_D,
        'check_recip': False,
        'special_matching': None
    },
    
    'Fogz Med DD': {
        'number': 3,
        'display_name': 'Fogz Med DD',
        'description': 'FOGZ arriving matched with Medium Discount M#s',
        'pass1': FOGZ,
        'pass2': MED_D,
        'check_recip': False,
        'special_matching': None
    },
    
    # Large Discount Models (4-6)
    'Lrg Disc PD': {
        'number': 4,
        'display_name': 'Lrg Disc PD',
        'description': 'Large Discount arriving matched with Premium M#s',
        'pass1': LRG_D,
        'pass2': P_ALL,
        'check_recip': False,
        'special_matching': 'lrg_disc_pd'  # Special: 36,39→P_WX012; 38→P_XD012
    },
    
    'Lrg Disc Med DD': {
        'number': 5,
        'display_name': 'Lrg Disc Med DD',
        'description': 'Large Discount arriving matched with Medium Discount M#s',
        'pass1': LRG_D,
        'pass2': MED_D,
        'check_recip': False,
        'special_matching': None
    },
    
    'Lrg Disc Fogz DD': {
        'number': 6,
        'display_name': 'Lrg Disc Fogz DD',
        'description': 'Large Discount arriving matched with FOGZ',
        'pass1': LRG_D,
        'pass2': FOGZ,
        'check_recip': False,
        'special_matching': None
    },
    
    # Reciprocal Models (7-8)
    'Recips PD': {
        'number': 7,
        'display_name': 'Recips PD',
        'description': 'Discount arriving matched with its reciprocal',
        'pass1': RECIPS_D,
        'pass2': RECIPS_P,
        'check_recip': True,
        'special_matching': None
    },
    
    'Recips DP': {
        'number': 8,
        'display_name': 'Recips DP',
        'description': 'Premium arriving matched with its reciprocal',
        'pass1': RECIPS_P,
        'pass2': RECIPS_D,
        'check_recip': True,
        'special_matching': None
    },
    
    # Premium x0s/x1s DP Models (9-10)
    'Prem x0s DP': {
        'number': 9,
        'display_name': 'Prem x0s DP',
        'description': 'Premium x0 arriving matched with Discount x0',
        'pass1': P_X0,
        'pass2': D_X0,
        'check_recip': False,
        'special_matching': None
    },
    
    'Prem x1s DP': {
        'number': 10,
        'display_name': 'Prem x1s DP',
        'description': 'Premium x0,x1,x2 arriving matched with Discount x0,x1,x2',
        'pass1': P_X012,
        'pass2': D_X12,
        'check_recip': False,
        'special_matching': None
    },
    
    # Premium xD0s/xD1s DP Models (11-12)
    'Prem xD0s DP': {
        'number': 11,
        'display_name': 'Prem xD0s DP',
        'description': 'Premium xD0 arriving matched with Discount xD0',
        'pass1': P_XD0,
        'pass2': D_XD0,
        'check_recip': False,
        'special_matching': None
    },
    
    'Prem xD1s DP': {
        'number': 12,
        'display_name': 'Prem xD1s DP',
        'description': 'Premium xD0,xD1,xD2 arriving matched with Discount xD0,xD1,xD2',
        'pass1': P_XD012,
        'pass2': D_XD12,
        'check_recip': False,
        'special_matching': None
    },
    
    # Premium xCs DP Model (13)
    'Prem xCs DP': {
        'number': 13,
        'display_name': 'Prem xCs DP',
        'description': 'Premium xCs arriving matched with Discount xCs',
        'pass1': P_XC,
        'pass2': D_XC,
        'check_recip': False,
        'special_matching': None
    },
    
    # Discount x0s/x1s PD Models (14-15)
    'Disc x0s PD': {
        'number': 14,
        'display_name': 'Disc x0s PD',
        'description': 'Discount x0 arriving matched with Premium x0',
        'pass1': D_X0,
        'pass2': P_X0,
        'check_recip': False,
        'special_matching': None
    },
    
    'Disc x1s PD': {
        'number': 15,
        'display_name': 'Disc x1s PD',
        'description': 'Discount x0,x1,x2 arriving matched with Premium x0,x1,x2',
        'pass1': D_X12,
        'pass2': P_X012,
        'check_recip': False,
        'special_matching': None
    },
    
    # Discount xD0s/xD1s PD Models (16-17)
    'Disc xD0s PD': {
        'number': 16,
        'display_name': 'Disc xD0s PD',
        'description': 'Discount xD0 arriving matched with Premium xD0',
        'pass1': D_XD0,
        'pass2': P_XD0,
        'check_recip': False,
        'special_matching': None
    },
    
    'Disc xD1s PD': {
        'number': 17,
        'display_name': 'Disc xD1s PD',
        'description': 'Discount xD0,xD1,xD2 arriving matched with Premium xD0,xD1,xD2',
        'pass1': D_XD12,
        'pass2': P_XD012,
        'check_recip': False,
        'special_matching': None
    },
    
    # Discount xCs PD Model (18)
    'Disc xCs PD': {
        'number': 18,
        'display_name': 'Disc xCs PD',
        'description': 'Discount xC arriving matched with Premium xC',
        'pass1': D_XC,
        'pass2': P_XC,
        'check_recip': False,
        'special_matching': None
    },
    
    # Premium-to-Premium Models (19-23)
    'Prem x0s PP': {
        'number': 19,
        'display_name': 'Prem x0s PP',
        'description': 'Premium x0 arriving matched with Premium x0',
        'pass1': P_X0,
        'pass2': P_X0,
        'check_recip': False,
        'special_matching': None
    },
    
    'Prem x1s PP': {
        'number': 20,
        'display_name': 'Prem x1s PP',
        'description': 'Premium x0,x1,x2 arriving matched with Premium x0,x1,x2',
        'pass1': P_X012,
        'pass2': P_X012,
        'check_recip': False,
        'special_matching': 'prem_x1s_pp'  # Special: 41,43,55→any; x0s→41,43,55 only
    },
    
    'Prem xD0s PP': {
        'number': 21,
        'display_name': 'Prem xD0s PP',
        'description': 'Premium xD0 arriving matched with Premium xD0',
        'pass1': P_XD0,
        'pass2': P_XD0,
        'check_recip': False,
        'special_matching': None
    },
    
    'Prem xD1s PP': {
        'number': 22,
        'display_name': 'Prem xD1s PP',
        'description': 'Premium xD0,xD1,xD2 arriving matched with Premium xD0,xD1,xD2',
        'pass1': P_XD012,
        'pass2': P_XD012,
        'check_recip': False,
        'special_matching': 'prem_xd1s_pp'  # Special: 42,45→any; xD0s→42,45 only
    },
    
    'Prem xCs PP': {
        'number': 23,
        'display_name': 'Prem xCs PP',
        'description': 'Premium xC arriving matched with Premium xC',
        'pass1': P_XC,
        'pass2': P_XC,
        'check_recip': False,
        'special_matching': None
    }
}

# ============================================================================
# SPECIAL MATCHING LOGIC
# ============================================================================

def apply_special_matching(model_name, m1, m2):
    """
    Apply special matching logic for specific models.
    
    Returns True if the pair should be matched, False otherwise.
    """
    special = MODELS[model_name].get('special_matching')
    
    if special is None:
        return True  # No special logic, accept all pairs
    
    m1_abs = abs(int(float(m1)))
    m2_abs = abs(int(float(m2)))
    
    if special == 'lrg_disc_pd':
        # Model #4: Lrg Disc PD
        # Pass 1 +/- (36 or 39) → any Pass 2 from P_WX012
        # Pass 1 +/- 38 → any Pass 2 from P_XD012
        if m1_abs in {36, 39}:
            return m2_abs in {abs(x) for x in P_WX012}
        elif m1_abs == 38:
            return m2_abs in {abs(x) for x in P_XD012}
        return False
    
    elif special == 'prem_x1s_pp':
        # Model #20: Prem x1s PP
        # Pass 1 +/- (41, 43, 46, 53, 55) → any Pass 2
        # Pass 1 +/- (50, 60, 68, 77, 87, 96, 103, 107, 111) → Pass 2 +/- (41, 43, 46, 53, 55) only
        if m1_abs in {41, 43, 46, 53, 55}:
            return True  # Can match with any Pass 2
        elif m1_abs in {50, 60, 68, 77, 87, 96, 103, 107, 111}:
            return m2_abs in {41, 43, 46, 53, 55}
        return False
    
    elif special == 'prem_xd1s_pp':
        # Model #22: Prem xD1s PP
        # Pass 1 +/- (42, 45) → any Pass 2
        # Pass 1 +/- (40, 54, 67, 74, 80, 85, 89, 92, 95, 97.2, 98.2, 99.3) → Pass 2 +/- (42, 45) only
        if m1_abs in {42, 45}:
            return True  # Can match with any Pass 2
        elif m1_abs in {40, 54, 67, 74, 80, 85, 89, 92, 95, 97.2, 98.2, 99.3}:
            return m2_abs in {42, 45}
        return False
    
    return True  # Default: accept the pair

# ============================================================================
# RECIPROCAL PAIRS (for Recips PD and Recips DP)
# ============================================================================

RECIP_PAIRS = {
    # X0 pairs - ALL 8 combinations (same signs + opposite signs)
    # (30, 50)
    (30, 50): 'X0', (50, 30): 'X0', (-30, -50): 'X0', (-50, -30): 'X0',
    (30, -50): 'X0', (50, -30): 'X0', (-30, 50): 'X0', (-50, 30): 'X0',
    # (22, 60)
    (22, 60): 'X0', (60, 22): 'X0', (-22, -60): 'X0', (-60, -22): 'X0',
    (22, -60): 'X0', (60, -22): 'X0', (-22, 60): 'X0', (-60, 22): 'X0',
    # (14, 68)
    (14, 68): 'X0', (68, 14): 'X0', (-14, -68): 'X0', (-68, -14): 'X0',
    (14, -68): 'X0', (68, -14): 'X0', (-14, 68): 'X0', (-68, 14): 'X0',
    # (10, 77)
    (10, 77): 'X0', (77, 10): 'X0', (-10, -77): 'X0', (-77, -10): 'X0',
    (10, -77): 'X0', (77, -10): 'X0', (-10, 77): 'X0', (-77, 10): 'X0',
    # (6, 87)
    (6, 87): 'X0', (87, 6): 'X0', (-6, -87): 'X0', (-87, -6): 'X0',
    (6, -87): 'X0', (87, -6): 'X0', (-6, 87): 'X0', (-87, 6): 'X0',
    # (5, 96)
    (5, 96): 'X0', (96, 5): 'X0', (-5, -96): 'X0', (-96, -5): 'X0',
    (5, -96): 'X0', (96, -5): 'X0', (-5, 96): 'X0', (-96, 5): 'X0',
    # (3, 103)
    (3, 103): 'X0', (103, 3): 'X0', (-3, -103): 'X0', (-103, -3): 'X0',
    (3, -103): 'X0', (103, -3): 'X0', (-3, 103): 'X0', (-103, 3): 'X0',
    # (2, 107)
    (2, 107): 'X0', (107, 2): 'X0', (-2, -107): 'X0', (-107, -2): 'X0',
    (2, -107): 'X0', (107, -2): 'X0', (-2, 107): 'X0', (-107, 2): 'X0',
    # (1, 111)
    (1, 111): 'X0', (111, 1): 'X0', (-1, -111): 'X0', (-111, -1): 'X0',
    (1, -111): 'X0', (111, -1): 'X0', (-1, 111): 'X0', (-111, 1): 'X0',
    
    # XD0 pairs - ALL 8 combinations
    # (27, 54)
    (27, 54): 'XD0', (54, 27): 'XD0', (-27, -54): 'XD0', (-54, -27): 'XD0',
    (27, -54): 'XD0', (54, -27): 'XD0', (-27, 54): 'XD0', (-54, 27): 'XD0',
    # (15, 67)
    (15, 67): 'XD0', (67, 15): 'XD0', (-15, -67): 'XD0', (-67, -15): 'XD0',
    (15, -67): 'XD0', (67, -15): 'XD0', (-15, 67): 'XD0', (-67, 15): 'XD0',
    
    # X1 pairs - ALL 8 combinations
    # (64, 17)
    (64, 17): 'X1', (17, 64): 'X1', (-64, -17): 'X1', (-17, -64): 'X1',
    (64, -17): 'X1', (17, -64): 'X1', (-64, 17): 'X1', (-17, 64): 'X1',   
    # (36, 43)
    (36, 43): 'X1', (43, 36): 'X1', (-36, -43): 'X1', (-43, -36): 'X1',
    (36, -43): 'X1', (43, -36): 'X1', (-36, 43): 'X1', (-43, 36): 'X1',
    # (26, 55)
    (26, 55): 'X1', (55, 26): 'X1', (-26, -55): 'X1', (-55, -26): 'X1',
    (26, -55): 'X1', (55, -26): 'X1', (-26, 55): 'X1', (-55, 26): 'X1',
    
    # XD1 pairs - ALL 8 combinations
    # (33, 45)
    (33, 45): 'XD1', (45, 33): 'XD1', (-33, -45): 'XD1', (-45, -33): 'XD1',
    (33, -45): 'XD1', (45, -33): 'XD1', (-33, 45): 'XD1', (-45, 33): 'XD1',
    
    # X2 pairs - ALL 8 combinations
    # (53, 29)
    (53, 29): 'X2', (29, 53): 'X2', (-53, -29): 'X2', (-29, -53): 'X2',
    (53, -29): 'X2', (29, -53): 'X2', (-53, 29): 'X2', (-29, 53): 'X2',
    # (46, 32)
    (46, 32): 'X2', (32, 46): 'X2', (-46, -32): 'X2', (-32, -46): 'X2',
    (46, -32): 'X2', (32, -46): 'X2', (-46, 32): 'X2', (-32, 46): 'X2',    
    # (39, 41)
    (39, 41): 'X2', (41, 39): 'X2', (-39, -41): 'X2', (-41, -39): 'X2',
    (39, -41): 'X2', (41, -39): 'X2', (-39, 41): 'X2', (-41, 39): 'X2',
    
    # XD2 pairs - ALL 8 combinations
    # (38, 42)
    (38, 42): 'XD2', (42, 38): 'XD2', (-38, -42): 'XD2', (-42, -38): 'XD2',
    (38, -42): 'XD2', (42, -38): 'XD2', (-38, 42): 'XD2', (-42, 38): 'XD2',
    
    # XC pairs - ALL 8 combinations
    # (4, 101)
    (4, 101): 'XC', (101, 4): 'XC', (-4, -101): 'XC', (-101, -4): 'XC',
    (4, -101): 'XC', (101, -4): 'XC', (-4, 101): 'XC', (-101, 4): 'XC',
    # (12, 71)
    (12, 71): 'XC', (71, 12): 'XC', (-12, -71): 'XC', (-71, -12): 'XC',
    (12, -71): 'XC', (71, -12): 'XC', (-12, 71): 'XC', (-71, 12): 'XC',
    # (24, 57)
    (24, 57): 'XC', (57, 24): 'XC', (-24, -57): 'XC', (-57, -24): 'XC',
    (24, -57): 'XC', (57, -24): 'XC', (-24, 57): 'XC', (-57, 24): 'XC',
    # (31, 47)
    (31, 47): 'XC', (47, 31): 'XC', (-31, -47): 'XC', (-47, -31): 'XC',
    (31, -47): 'XC', (47, -31): 'XC', (-31, 47): 'XC', (-47, 31): 'XC',
}

def get_reciprocal_lookup():
    """
    Build reciprocal lookup dictionary.
    Returns dict mapping each M# to its reciprocal.
    """
    lookup = {}
    for (m1, m2), pattern in RECIP_PAIRS.items():
        lookup[m1] = m2
        lookup[m2] = m1
    return lookup
