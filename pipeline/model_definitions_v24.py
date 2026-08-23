"""
Model Definitions v24 - All 24 Trading Models

This file contains definitions for all trading models used in the Swing Analysis Tool.
Each model defines Pass 1 and Pass 2 M# lists, and any special matching logic.

Models are organized into categories:
- FOGZ models (1-3): FOGZ matched with Premiums or Discounts
- Large Discount models (4-6): Large Discounts matched with Premiums or other Discounts
- Reciprocal models (7-8): Reciprocal pairs
- Premium/Discount pattern models (9-18): Specific x0, x1, xD0, xD1, xC patterns
- Premium/Premium models (19-23): Premium-to-Premium patterns
- Discount Opposite model (24): Discount matched with its sign-opposite Discount

August 2026 update (v24) — Model definition cleanup:
- FOGZ kept as-is. Added FOZ (= FOGZ | FOBZ | FOOZ), FOBZ = {0, 5.5, -5.5},
  FOOZ = {0, 4, -4} as new pass1 pools for Models 1-3.
- MED_D renamed to D_X0_MED (values unchanged). Added D_X12_MED, D_X_MED
  (= D_X0_MED | D_X12_MED), D_XD_MED, D_XC_MED, and D_MED_ALL (union of
  the four MED sub-lists).
- Added D_X012 (= D_X0 | D_X12) for Model 10's expanded pass2.
- P_ALL corrected/expanded to the true union of every other Premium list
  (P_WX012 | P_XD012 | P_X0 | P_X012 | P_XD0 | P_XC).
- Added reciprocal pair (5.5, 95) [all 8 sign/order combos] to RECIP_PAIRS
  under the XD0 pattern. Added 5.5/-5.5 to RECIPS_D and D_XD0; added
  95/-95 to RECIPS_P; added 5.5/-5.5 to D_ALL.
- Model 1 (Fogz PD): pass1 widened FOGZ -> FOZ, pass2 widened P_WX012 ->
  P_ALL, with new special matching 'fogz_pd' routing by FOGZ/FOBZ/FOOZ
  sub-group membership (FOGZ->P_WX012, FOBZ->P_XD012, FOOZ->P_XC).
- Model 2 (Fogz Lrg DD): pass1 widened FOGZ -> FOZ, new special matching
  'fogz_lrg_dd' (FOGZ->{36,39}, FOBZ->{38}; FOOZ has no target and is
  excluded — a FOOZ-origin pair produces no Model 2 match).
- Model 3 (Fogz Med DD): pass1 widened FOGZ -> FOZ, pass2 widened
  D_X0_MED -> D_MED_ALL, new special matching 'fogz_med_dd'
  (FOGZ->D_X_MED, FOBZ->D_XD_MED, FOOZ->D_XC_MED).
- Model 10 (Prem x1s DP): pass2 widened D_X12 -> D_X012, new special
  matching 'dedupe_model9' excludes any pair also produced by Model 9
  (Prem x0s DP) so it isn't double-reported.
- Model 22 (Prem xD1s PP) 'prem_xd1s_pp' branch fixed: it compared Pass 1
  against abs(int(float(m1))), which truncates 97.2/98.2/99.3-style
  values down to 97/98/99 — never matching either branch's set, so a
  decimal-magnitude Pass 1 M# could never match through this model at
  all. Rewritten with _in_set() (exact rounded-float comparison).
- NOTE: adding 5.5/-5.5 to D_XD0 means D_OPP (Model 24's Disc Opp pool)
  now contains a 5.5 magnitude alongside D_X0's existing 5 magnitude.
  The old truncated-abs-int comparison used elsewhere in this file would
  have collapsed those two together (int(float(5.5)) == 5), so
  is_invalid_self_match's sibling disc_opp check below was rewritten to
  compare rounded float magnitudes instead of truncated ints — see
  apply_special_matching()'s 'disc_opp' branch and the new _magnitude()
  helper.

March 2026 update:
Added Bravo and Charlie X1 & X2. Recip Combos (64, 17), (53, 29), and (46, 32)

June 2026 update:
Added Model 24 "Disc Opp" — Discount M# arriving matched with its Opposite
(same magnitude, flipped sign, e.g. 5 <-> -5). Pass 1/Pass 2 use a single
combined list of all five pattern-specific Discount sets (D_X0, D_X12, D_XD0,
D_XD12, D_XC), since each of those lists is internally symmetric (contains
both signs of every magnitude) and the magnitudes never collide across lists.
That means the 'disc_opp' special-matching rule (same |M#|, opposite sign)
naturally keeps each match inside its own sub-pattern — no per-list model
needed.

July 2026 update (v23) — REDUNDANT MATCH FIX MOVED TO THE SOURCE:
Pipeline v4.0.24+ was post-filtering two conditions out of every model's
raw pass1 x pass2 cross-product, downstream in the pipeline file itself:
  1. Invalid self-matches — the same M# matched to itself at the same
     arrival instant (only possible for symmetric models where pass1 and
     pass2 overlap, e.g. the PP series, Recips, Disc Opp).
  2. Overlay arrivals — two *distinct* M#s that arrive at the exact same
     instant, which a directional model label would otherwise misreport
     as a sequential Pass1-then-Pass2 relationship.
Both conditions are properties of a *pair of M#s*, exactly like
apply_special_matching() above — so the rule now lives here, next to the
other pairing rules, as is_invalid_self_match() / is_overlay_pair().
Every matcher/processor that consumes MODELS calls these two functions
instead of re-implementing its own copy of the filter. The pipeline's
post-processing step still exists (it drops/tags rows on the assembled
DataFrame) but now delegates to this module for the actual rule, so
there is exactly one definition of "invalid self-match" and "overlay"
in the codebase. See unified_traveler_pipeline_6_0_0.py.
"""

# ============================================================================
# M# LIST DEFINITIONS
# ============================================================================

# Core Lists
FOGZ = {0, 1, -1, 2, -2, 3, -3, 5, -5, 6, -6}
FOBZ = {0, 5.5, -5.5}
FOOZ = {0, 4, -4}
FOZ = FOGZ | FOBZ | FOOZ  # {0, 1,-1, 2,-2, 3,-3, 4,-4, 5,-5, 5.5,-5.5, 6,-6}
LRG_D = {36, -36, 38, -38, 39, -39}

# Medium Discount lists (MED_D renamed to D_X0_MED, values unchanged)
D_X0_MED = {10, -10, 14, -14, 22, -22, 30, -30}
D_X12_MED = {17, -17, 26, -26, 29, -29}
D_X_MED = D_X0_MED | D_X12_MED
D_XD_MED = {15, -15, 27, -27, 33, -33}
D_XC_MED = {12, -12, 24, -24, 31, -31}
D_MED_ALL = D_X0_MED | D_X12_MED | D_XD_MED | D_XC_MED

# Premium Lists
P_WX012 = {40, -40, 41, -41, 43, -43, 46, -46, 50, -50, 53, -53, 55, -55, 60, -60, 64, -64, 68, -68, 77, -77, 87, -87, 96, -96, 103, -103, 107, -107, 111, -111}
P_XD012 = {40, -40, 42, -42, 45, -45, 54, -54, 67, -67, 74, -74, 80, -80, 85, -85, 89, -89, 92, -92, 95, -95, 97.2, -97.2, 98.2, -98.2, 99.3, -99.3}

# Pattern-specific Premium Lists
P_X0 = {50, -50, 60, -60, 68, -68, 77, -77, 87, -87, 96, -96, 103, -103, 107, -107, 111, -111}
P_X012 = {41, -41, 43, -43, 46, -46, 50, -50, 53, -53, 55, -55, 60, -60, 64, -64, 68, -68, 77, -77, 87, -87, 96, -96, 103, -103, 107, -107, 111, -111}
P_XD0 = {40, -40, 54, -54, 67, -67, 74, -74, 80, -80, 85, -85, 89, -89, 92, -92, 95, -95, 97.2, -97.2, 98.2, -98.2, 99.3, -99.3}
P_XC = {47, -47, 57, -57, 71, -71, 93.5, -93.5, 101, -101}

# P_ALL = union of every other Premium list above (kept as the true union
# so it can never silently drift out of sync with the lists it's built from).
P_ALL = P_WX012 | P_XD012 | P_X0 | P_X012 | P_XD0 | P_XC

# Discount Lists (Reciprocal)
RECIPS_D = {1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 5.5, -5.5, 6, -6, 10, -10, 12, -12, 14, -14, 15, -15, 17, -17, 22, -22, 24, -24, 26, -26, 27, -27, 29, -29, 30, -30, 31, -31, 32, -32, 33, -33, 36, -36, 38, -38, 39, -39}
RECIPS_P = {41, -41, 42, -42, 43, -43, 45, -45, 46, -46, 47, -47, 50, -50, 53, -53, 54, -54, 55, -55, 57, -57, 60, -60, 64, -64, 67, -67, 68, -68, 71, -71, 77, -77, 87, -87, 95, -95, 96, -96, 101, -101, 103, -103, 107, -107, 111, -111}
D_ALL = {0, 1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 5.5, -5.5, 6, -6, 10, -10, 12, -12, 14, -14, 15, -15, 17, -17, 21, -21, 22, -22, 24, -24, 25, -25, 26, -26, 27, -27, 29, -29, 30, -30, 31, -31, 32, -32, 33, -33, 36, -36, 37, -37, 38, -38, 39, -39}

# Pattern-specific Discount Lists
D_X0 = {1, -1, 2, -2, 3, -3, 5, -5, 6, -6, 10, -10, 14, -14, 22, -22, 30, -30}
D_X12 = {17, -17, 26, -26, 29, -29, 32, -32, 36, -36, 39, -39}
D_X012 = D_X0 | D_X12
D_XD0 = {5.5, -5.5, 15, -15, 27, -27}
D_XD12 = {33, -33, 38, -38}
D_XC = {4, -4, 12, -12, 24, -24, 31, -31}

# Combined Discount list for Model 24 (Disc Opp).
# Union of all five pattern-specific Discount lists above. Each list
# contains both signs of every magnitude. Magnitudes used to never repeat
# across the five lists, which let the 'disc_opp' special-matching rule
# use a cheap truncated-int magnitude comparison. That invariant broke in
# the August 2026 update: D_XD0 now includes 5.5/-5.5, and int(float(5.5))
# truncates to 5 — the same truncated magnitude as D_X0's 5/-5. The
# 'disc_opp' branch in apply_special_matching() below was updated to
# compare rounded float magnitudes (round(abs(float(m)), 4)) instead of
# truncated ints, so 5 and 5.5 stay distinct and disc_opp still only
# pairs within its own sub-pattern.
D_OPP = D_X0 | D_X12 | D_XD0 | D_XD12 | D_XC

# ============================================================================
# MODEL DEFINITIONS
# ============================================================================

MODELS = {
    # FOGZ Models (1-3)
    'Fogz PD': {
        'number': 1,
        'display_name': 'Fogz PD',
        'description': 'FOGZ/FOBZ/FOOZ arriving matched with Premium M#s',
        'pass1': FOZ,
        'pass2': P_ALL,
        'check_recip': False,
        'special_matching': 'fogz_pd'  # Special: FOGZ->P_WX012; FOBZ->P_XD012; FOOZ->P_XC
    },
    
    'Fogz Lrg DD': {
        'number': 2,
        'display_name': 'Fogz Lrg DD',
        'description': 'FOGZ/FOBZ arriving matched with Large Discount M#s',
        'pass1': FOZ,
        'pass2': LRG_D,
        'check_recip': False,
        'special_matching': 'fogz_lrg_dd'  # Special: FOGZ->36,39; FOBZ->38; FOOZ excluded (no match)
    },
    
    'Fogz Med DD': {
        'number': 3,
        'display_name': 'Fogz Med DD',
        'description': 'FOGZ/FOBZ/FOOZ arriving matched with Medium Discount M#s',
        'pass1': FOZ,
        'pass2': D_MED_ALL,
        'check_recip': False,
        'special_matching': 'fogz_med_dd'  # Special: FOGZ->D_X_MED; FOBZ->D_XD_MED; FOOZ->D_XC_MED
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
        'pass2': D_X0_MED,  # was MED_D — renamed, values unchanged; not part of today's update
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
        'pass2': D_X012,
        'check_recip': False,
        'special_matching': 'dedupe_model9'  # Special: exclude pairs already reported by Model 9 (e.g. (50, 30))
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
    },

    # Discount Opposite Model (24)
    'Disc Opp': {
        'number': 24,
        'display_name': 'Disc Opp',
        'description': 'Discount arriving matched with its Opposite Discount M# (same magnitude, flipped sign)',
        'pass1': D_OPP,
        'pass2': D_OPP,
        'check_recip': False,
        'special_matching': 'disc_opp'
    }
}

# ============================================================================
# SPECIAL MATCHING LOGIC
# ============================================================================

def _magnitude(v):
    """
    Decimal-safe |M#|, rounded to avoid float-precision noise. Unlike
    abs(int(float(v))), this does NOT truncate — 5.5 stays 5.5 and is
    never confused with 5. Returns None if v can't be parsed as a number.
    """
    try:
        return round(abs(float(v)), 4)
    except (TypeError, ValueError):
        return None


def _in_set(v, s):
    """Membership test using _magnitude-equivalent float equality (signed, not abs)."""
    try:
        fv = round(float(v), 4)
    except (TypeError, ValueError):
        return False
    return any(fv == round(float(x), 4) for x in s)


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

    if special == 'fogz_pd':
        # Model #1: Fogz PD — pass1 is FOZ (FOGZ | FOBZ | FOOZ).
        # Route Pass 1's M# to a premium pool based on which FOGZ/FOBZ/FOOZ
        # sub-group(s) it belongs to. All three sub-groups include 0, so a
        # Pass 1 value of 0 is routed to the union of every applicable
        # pool it's a member of (in practice: all three, i.e. all of P_ALL).
        allowed = set()
        if _in_set(m1, FOGZ):
            allowed |= P_WX012
        if _in_set(m1, FOBZ):
            allowed |= P_XD012
        if _in_set(m1, FOOZ):
            allowed |= P_XC
        if not allowed:
            return False
        return _in_set(m2, allowed)

    elif special == 'fogz_lrg_dd':
        # Model #2: Fogz Lrg DD — pass1 is FOZ.
        # FOGZ -> {36, -36, 39, -39}; FOBZ -> {38, -38}.
        # FOOZ has no target: a FOOZ-origin pair (4/-4, or 0 acting only
        # as FOOZ) produces no match in this model.
        allowed = set()
        if _in_set(m1, FOGZ):
            allowed |= {36, -36, 39, -39}
        if _in_set(m1, FOBZ):
            allowed |= {38, -38}
        if not allowed:
            return False
        return _in_set(m2, allowed)

    elif special == 'fogz_med_dd':
        # Model #3: Fogz Med DD — pass1 is FOZ.
        # FOGZ -> D_X_MED; FOBZ -> D_XD_MED; FOOZ -> D_XC_MED.
        allowed = set()
        if _in_set(m1, FOGZ):
            allowed |= D_X_MED
        if _in_set(m1, FOBZ):
            allowed |= D_XD_MED
        if _in_set(m1, FOOZ):
            allowed |= D_XC_MED
        if not allowed:
            return False
        return _in_set(m2, allowed)

    elif special == 'dedupe_model9':
        # Model #10: exclude any pair already produced by Model #9
        # (Prem x0s DP: pass1=P_X0, pass2=D_X0), e.g. (50, 30), so it's
        # only reported once, under Model 9.
        if _in_set(m1, P_X0) and _in_set(m2, D_X0):
            return False
        return True

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
        # NOTE (Aug 2026 fix): this used to compare abs(int(float(m1))),
        # which truncates 97.2/98.2/99.3-style values down to 97/98/99 —
        # none of which appear in either branch's set, so any Pass 1 M#
        # with a decimal component could never match through this special
        # matching at all. Switched to _in_set(), which compares exact
        # (rounded) float values instead of truncated ints.
        if _in_set(m1, {42, -42, 45, -45}):
            return True  # Can match with any Pass 2
        elif _in_set(m1, {40, -40, 54, -54, 67, -67, 74, -74, 80, -80, 85, -85, 89, -89, 92, -92, 95, -95, 97.2, -97.2, 98.2, -98.2, 99.3, -99.3}):
            return _in_set(m2, {42, -42, 45, -45})
        return False

    elif special == 'disc_opp':
        # Model #24: Disc Opp
        # Pass 1 Discount M# must pair with a Pass 2 Discount M# of the SAME
        # magnitude but the OPPOSITE sign (e.g. 5 <-> -5, 17 <-> -17, and
        # now 5.5 <-> -5.5).
        # NOTE (Aug 2026): this used to compare abs(int(float(m))), relying
        # on magnitudes never repeating across D_X0/D_X12/D_XD0/D_XD12/D_XC.
        # Adding 5.5/-5.5 to D_XD0 broke that: int(float(5.5)) truncates to
        # 5, the same as D_X0's 5. Switched to rounded-float magnitude
        # comparison so 5 and 5.5 are never conflated.
        m1_mag = _magnitude(m1)
        m2_mag = _magnitude(m2)
        if m1_mag is None or m2_mag is None or m1_mag != m2_mag:
            return False
        try:
            m1_pos = float(m1) > 0
            m2_pos = float(m2) > 0
        except (TypeError, ValueError):
            return False
        return m1_pos != m2_pos  # opposite signs required

    return True  # Default: accept the pair

# ============================================================================
# PAIR-VALIDITY GUARDS  (self-match / overlay — single source of truth)
# ============================================================================
#
# These two functions replace the ad-hoc, duplicated filtering that used to
# live only inside unified_traveler_pipeline (_filter_invalid_self_matches /
# _flag_overlay_arrivals). Any matcher or processor that builds pairs from
# MODELS should call these on every candidate pair so the definitions never
# drift apart between the pipeline, the Streamlit batch processor, or any
# future consumer.
#
# Both are pure functions of (m1, m2, arrival1, arrival2) — no DataFrame
# dependency — so they can be called per-row during pair generation itself
# (preventing the invalid rows from ever being created) or applied after
# the fact as a filter/flag on an assembled DataFrame. The pipeline
# currently does the latter because the actual point-matching cross-product
# happens inside bypass_mode_matcher.py / custom_range_calculator*, which
# are not part of this file; wiring the guard directly into the generation
# loop there would be the ideal next step once those modules are in scope.

def _is_missing(v) -> bool:
    """True for None / NaN-like values, without requiring pandas."""
    if v is None:
        return True
    try:
        return v != v  # NaN != NaN
    except Exception:
        return False


def _arrivals_match(arrival1, arrival2) -> bool:
    """Robust equality for two arrival timestamps (str, Timestamp, datetime)."""
    if _is_missing(arrival1) or _is_missing(arrival2):
        return False
    try:
        import pandas as pd  # local import: keep this module pandas-optional
        return pd.to_datetime(arrival1) == pd.to_datetime(arrival2)
    except Exception:
        return str(arrival1).strip() == str(arrival2).strip()


def _m_values_match(m1, m2) -> bool:
    """Robust equality for two M# values (handles int/float/decimal M#s)."""
    if _is_missing(m1) or _is_missing(m2):
        return False
    try:
        return round(float(m1), 4) == round(float(m2), 4)
    except Exception:
        return str(m1).strip() == str(m2).strip()


def is_invalid_self_match(m1, m2, arrival1, arrival2) -> bool:
    """
    True when a candidate pair is the same M# matched to itself at the
    exact same arrival instant (Arrival1 == Arrival2 AND M1 == M2) — noise
    from the pass1 x pass2 cross-product on symmetric models, not a real
    journey between two distinct points.
    """
    return _arrivals_match(arrival1, arrival2) and _m_values_match(m1, m2)


def is_overlay_pair(m1, m2, arrival1, arrival2) -> bool:
    """
    True when a candidate pair is a genuine overlay: two *distinct* M#s
    (M1 != M2) that arrive at the exact same instant (Arrival1 == Arrival2).
    Reporting these under a directional model label (e.g. Model 10 'DP')
    would misleadingly imply one arrived before the other.
    """
    return _arrivals_match(arrival1, arrival2) and not _m_values_match(m1, m2)

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
    # (5.5, 95)
    (5.5, 95): 'XD0', (95, 5.5): 'XD0', (-5.5, -95): 'XD0', (-95, -5.5): 'XD0',
    (5.5, -95): 'XD0', (95, -5.5): 'XD0', (-5.5, 95): 'XD0', (-95, 5.5): 'XD0',
    
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
