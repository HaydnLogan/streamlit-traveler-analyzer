"""
Pattern Success Analysis Module
Tracks pattern performance and builds predictive scoring
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import sys
sys.path.append('/home/claude')

from pattern_scanner_v03 import HaydnPatternScanner, MODELS, WEIGHTS


class PatternSuccessAnalyzer:
    """
    Analyzes pattern success rates by tracking:
    - Zone hits (price reached zone)
    - Reversals (price bounced from zone)
    - Pattern-specific success rates
    - Predictive power of pattern combinations
    """
    
    def __init__(self):
        self.success_history = []
        self.pattern_stats = {}
    
    def analyze_zone_performance(self,
                                 zone_analysis: Dict,
                                 ohlc_df: pd.DataFrame,
                                 zone_time: datetime,
                                 zone_type: str,
                                 lookforward_hours: int = 4,
                                 success_threshold: float = 30.0) -> Dict:
        """
        Analyze how well a zone performed
        
        Args:
            zone_analysis: Zone analysis dict from scanner
            ohlc_df: OHLC dataframe
            zone_time: When zone was identified
            zone_type: 'High' or 'Low'
            lookforward_hours: Hours to look forward
            success_threshold: Minimum move to count as success
            
        Returns:
            Dict with performance metrics
        """
        
        # Parse times
        ohlc_df = ohlc_df.copy()
        ohlc_df['time'] = pd.to_datetime(ohlc_df['time'], utc=True).dt.tz_localize(None)
        zone_time = pd.to_datetime(zone_time).replace(tzinfo=None)
        
        zone_price = zone_analysis['center_price']
        
        # Get future bars
        end_time = zone_time + timedelta(hours=lookforward_hours)
        future_bars = ohlc_df[
            (ohlc_df['time'] > zone_time) &
            (ohlc_df['time'] <= end_time)
        ]
        
        if len(future_bars) == 0:
            return {
                'success': False,
                'reason': 'No future data',
                'touched': False,
                'reversal': False,
                'max_move_toward': 0,
                'max_move_away': 0,
                'time_to_touch': None,
                'bars_to_touch': None
            }
        
        # Analyze performance based on zone type
        if zone_type == 'Low':
            # For lows, success = price went down to zone and bounced
            touched = (future_bars['low'].min() <= zone_price)
            min_price = future_bars['low'].min()
            max_after_touch = 0
            time_to_touch = None
            bars_to_touch = None
            
            if touched:
                # Find when it touched
                touch_bars = future_bars[future_bars['low'] <= zone_price]
                if len(touch_bars) > 0:
                    first_touch = touch_bars.iloc[0]
                    time_to_touch = (first_touch['time'] - zone_time).total_seconds() / 3600
                    bars_to_touch = len(future_bars[future_bars['time'] <= first_touch['time']])
                    
                    # Check for bounce
                    after_touch = future_bars[future_bars['time'] >= first_touch['time']]
                    if len(after_touch) > 0:
                        max_after_touch = after_touch['high'].max() - zone_price
            
            max_move_toward = zone_price - min_price if touched else 0
            max_move_away = future_bars['high'].max() - zone_price
            
            reversal_confirmed = touched and (max_after_touch >= success_threshold)
            success = reversal_confirmed
            
        else:  # High
            # For highs, success = price went up to zone and bounced down
            touched = (future_bars['high'].max() >= zone_price)
            max_price = future_bars['high'].max()
            max_after_touch = 0
            time_to_touch = None
            bars_to_touch = None
            
            if touched:
                touch_bars = future_bars[future_bars['high'] >= zone_price]
                if len(touch_bars) > 0:
                    first_touch = touch_bars.iloc[0]
                    time_to_touch = (first_touch['time'] - zone_time).total_seconds() / 3600
                    bars_to_touch = len(future_bars[future_bars['time'] <= first_touch['time']])
                    
                    # Check for bounce
                    after_touch = future_bars[future_bars['time'] >= first_touch['time']]
                    if len(after_touch) > 0:
                        max_after_touch = zone_price - after_touch['low'].min()
            
            max_move_toward = max_price - zone_price if touched else 0
            max_move_away = zone_price - future_bars['low'].min()
            
            reversal_confirmed = touched and (max_after_touch >= success_threshold)
            success = reversal_confirmed
        
        return {
            'success': success,
            'reason': 'Reversal confirmed' if success else ('Touched but no reversal' if touched else 'Not touched'),
            'touched': touched,
            'reversal': reversal_confirmed,
            'max_move_toward': max_move_toward,
            'max_move_away': max_move_away,
            'time_to_touch': time_to_touch,
            'bars_to_touch': bars_to_touch
        }
    
    def build_pattern_statistics(self, results_history: List[Dict]) -> pd.DataFrame:
        """
        Build statistics for each pattern type
        
        Args:
            results_history: List of zone results with patterns and performance
            
        Returns:
            DataFrame with pattern statistics
        """
        
        stats = {}
        
        # Initialize stats for each pattern type
        pattern_types = [
            'epic_same_origin',
            'epic_epic_pairs',
            'x0_sequential_descents',
            'fogz_presence',
            'constellations',
            'wild_pairs',
            'same_origin_tag_descents',
            'x0_alignments',
            'downgrades',
            'large_m_presence'
        ]
        
        for ptype in pattern_types:
            stats[ptype] = {
                'total_zones': 0,
                'zones_with_pattern': 0,
                'successful_zones': 0,
                'touched_zones': 0,
                'reversal_zones': 0,
                'avg_score_contribution': 0,
                'success_rate': 0,
                'reversal_rate': 0,
                'avg_move_toward': 0
            }
        
        # Aggregate statistics
        for result in results_history:
            patterns = result['patterns']
            performance = result['performance']
            
            for ptype in pattern_types:
                pattern_list = patterns.get(ptype, [])
                if isinstance(pattern_list, dict):  # family_clusters
                    has_pattern = len(pattern_list) > 0
                    pattern_count = len(pattern_list)
                else:
                    has_pattern = len(pattern_list) > 0
                    pattern_count = len(pattern_list)
                
                stats[ptype]['total_zones'] += 1
                
                if has_pattern:
                    stats[ptype]['zones_with_pattern'] += 1
                    
                    if performance['success']:
                        stats[ptype]['successful_zones'] += 1
                    
                    if performance['touched']:
                        stats[ptype]['touched_zones'] += 1
                    
                    if performance['reversal']:
                        stats[ptype]['reversal_zones'] += 1
                    
                    stats[ptype]['avg_move_toward'] += performance['max_move_toward']
        
        # Calculate rates
        for ptype in pattern_types:
            if stats[ptype]['zones_with_pattern'] > 0:
                stats[ptype]['success_rate'] = (
                    stats[ptype]['successful_zones'] / stats[ptype]['zones_with_pattern'] * 100
                )
                stats[ptype]['reversal_rate'] = (
                    stats[ptype]['reversal_zones'] / stats[ptype]['zones_with_pattern'] * 100
                )
                stats[ptype]['avg_move_toward'] = (
                    stats[ptype]['avg_move_toward'] / stats[ptype]['zones_with_pattern']
                )
        
        # Convert to DataFrame
        stats_df = pd.DataFrame(stats).T
        stats_df = stats_df.sort_values('success_rate', ascending=False)
        
        return stats_df
    
    def analyze_pattern_combinations(self, results_history: List[Dict]) -> pd.DataFrame:
        """
        Find which pattern combinations are most predictive
        
        Returns:
            DataFrame with combination statistics
        """
        
        combinations = []
        
        for result in results_history:
            patterns = result['patterns']
            performance = result['performance']
            
            # Build pattern presence vector
            pattern_vector = {
                'has_epic_same': len(patterns.get('epic_same_origin', [])) > 0,
                'has_tt_pairs': len(patterns.get('epic_epic_pairs', [])) > 0,
                'has_x0_seq': len(patterns.get('x0_sequential_descents', [])) > 0,
                'has_fogz': len(patterns.get('fogz_presence', [])) > 0,
                'has_constellation': len(patterns.get('constellations', [])) > 0,
                'has_wild_pair': len(patterns.get('wild_pairs', [])) > 0,
                'has_tag_descent': len(patterns.get('same_origin_tag_descents', [])) > 0,
            }
            
            # Count how many pattern types are present
            pattern_count = sum(pattern_vector.values())
            
            combinations.append({
                **pattern_vector,
                'pattern_count': pattern_count,
                'score': result['score'],
                'success': performance['success'],
                'touched': performance['touched'],
                'reversal': performance['reversal'],
                'move_toward': performance['max_move_toward']
            })
        
        combo_df = pd.DataFrame(combinations)
        
        # Analyze by pattern count
        pattern_count_stats = combo_df.groupby('pattern_count').agg({
            'success': ['count', 'sum', 'mean'],
            'touched': 'mean',
            'reversal': 'mean',
            'move_toward': 'mean',
            'score': 'mean'
        }).round(3)
        
        return pattern_count_stats
    
    def rank_zones_by_predictive_power(self,
                                       results_history: List[Dict],
                                       pattern_stats: pd.DataFrame) -> List[Dict]:
        """
        Re-rank zones using learned predictive weights
        
        Args:
            results_history: Historical results
            pattern_stats: Pattern statistics from build_pattern_statistics
            
        Returns:
            List of zones ranked by predictive score
        """
        
        # Build predictive weights from success rates
        predictive_weights = {}
        for pattern_type, row in pattern_stats.iterrows():
            success_rate = row['success_rate'] / 100
            predictive_weights[pattern_type] = success_rate * WEIGHTS.get(pattern_type, 20)
        
        # Re-score zones
        rescored_zones = []
        
        for result in results_history:
            patterns = result['patterns']
            predictive_score = 0
            
            for ptype, weight in predictive_weights.items():
                pattern_list = patterns.get(ptype, [])
                if isinstance(pattern_list, dict):
                    count = len(pattern_list)
                else:
                    count = len(pattern_list)
                
                predictive_score += count * weight
            
            rescored_zones.append({
                'original_rank': result.get('rank', 0),
                'original_score': result['score'],
                'predictive_score': predictive_score,
                'zone_price': result['center_price'],
                'success': result['performance']['success'],
                'touched': result['performance']['touched']
            })
        
        # Sort by predictive score
        rescored_zones.sort(key=lambda x: x['predictive_score'], reverse=True)
        
        # Add predictive rank
        for i, zone in enumerate(rescored_zones, 1):
            zone['predictive_rank'] = i
        
        return rescored_zones
    
    def generate_success_report(self, results_history: List[Dict]) -> str:
        """
        Generate comprehensive success analysis report
        
        Returns:
            Formatted report string
        """
        
        pattern_stats = self.build_pattern_statistics(results_history)
        combo_stats = self.analyze_pattern_combinations(results_history)
        
        report = []
        report.append("=" * 80)
        report.append("PATTERN SUCCESS ANALYSIS REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Overall statistics
        total_zones = len(results_history)
        successful = sum(1 for r in results_history if r['performance']['success'])
        touched = sum(1 for r in results_history if r['performance']['touched'])
        
        report.append(f"Total Zones Analyzed: {total_zones}")
        report.append(f"Zones Touched: {touched} ({touched/total_zones*100:.1f}%)")
        report.append(f"Successful Reversals: {successful} ({successful/total_zones*100:.1f}%)")
        report.append("")
        
        # Pattern-specific success rates
        report.append("=" * 80)
        report.append("PATTERN SUCCESS RATES")
        report.append("=" * 80)
        report.append("")
        
        for pattern_type, row in pattern_stats.iterrows():
            if row['zones_with_pattern'] > 0:
                report.append(f"{pattern_type}:")
                report.append(f"  Zones with pattern: {int(row['zones_with_pattern'])}")
                report.append(f"  Success rate: {row['success_rate']:.1f}%")
                report.append(f"  Reversal rate: {row['reversal_rate']:.1f}%")
                report.append(f"  Avg move toward: {row['avg_move_toward']:.2f} points")
                report.append("")
        
        # Pattern combinations
        report.append("=" * 80)
        report.append("PATTERN COMBINATION ANALYSIS")
        report.append("=" * 80)
        report.append("")
        
        for pattern_count, row in combo_stats.iterrows():
            report.append(f"{pattern_count} pattern types present:")
            report.append(f"  Zones: {int(row['success']['count'])}")
            report.append(f"  Success rate: {row['success']['mean']*100:.1f}%")
            report.append(f"  Touch rate: {row['touched']['mean']*100:.1f}%")
            report.append(f"  Avg score: {row['score']['mean']:.0f}")
            report.append("")
        
        # Top performing patterns
        report.append("=" * 80)
        report.append("TOP PERFORMING PATTERNS")
        report.append("=" * 80)
        report.append("")
        
        top_patterns = pattern_stats.nlargest(5, 'success_rate')
        for i, (pattern_type, row) in enumerate(top_patterns.iterrows(), 1):
            if row['zones_with_pattern'] >= 3:  # Only if sufficient sample size
                report.append(f"{i}. {pattern_type}: {row['success_rate']:.1f}% success rate")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)


def batch_success_analysis(scanner: HaydnPatternScanner,
                           zones_df: pd.DataFrame,
                           ohlc_df: pd.DataFrame,
                           lookforward_hours: int = 4) -> Tuple[List[Dict], str]:
    """
    Run success analysis on batch of zones
    
    Args:
        scanner: Initialized scanner
        zones_df: Zones from scan_all_zones
        ohlc_df: OHLC data
        lookforward_hours: Hours to look forward
        
    Returns:
        (results_history, report_string)
    """
    
    analyzer = PatternSuccessAnalyzer()
    results_history = []
    
    for idx, zone in zones_df.iterrows():
        # Analyze performance
        performance = analyzer.analyze_zone_performance(
            zone_analysis=zone.to_dict(),
            ohlc_df=ohlc_df,
            zone_time=zone['zone_time'],
            zone_type=zone['zone_subtype'],
            lookforward_hours=lookforward_hours,
            success_threshold=30.0
        )
        
        results_history.append({
            'rank': zone['rank'],
            'center_price': zone['center_price'],
            'score': zone['score'],
            'patterns': zone['patterns'],
            'performance': performance
        })
    
    # Generate report
    report = analyzer.generate_success_report(results_history)
    
    return results_history, report


if __name__ == "__main__":
    print("Pattern Success Analysis Module")
    print("Import this module to track pattern performance")
