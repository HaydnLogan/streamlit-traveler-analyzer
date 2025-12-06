"""
HOD/LOD Diagnostic Script
Run this to check if your feed data has the necessary structure for HOD/LOD detection
"""
import pandas as pd
import datetime as dt

def diagnose_feed_data(feed_df, feed_name="Feed"):
    """Diagnose a feed dataframe for HOD/LOD compatibility"""
    print(f"\n{'='*60}")
    print(f"Diagnosing {feed_name}")
    print(f"{'='*60}")
    
    if feed_df.empty:
        print("❌ DataFrame is empty!")
        return False
    
    print(f"✓ Rows: {len(feed_df)}")
    print(f"✓ Columns: {list(feed_df.columns)}")
    
    # Check for time column
    if 'time' not in feed_df.columns:
        print("❌ No 'time' column found!")
        return False
    
    print(f"✓ Time column exists")
    print(f"  - dtype: {feed_df['time'].dtype}")
    
    # Check if time is datetime
    if not pd.api.types.is_datetime64_any_dtype(feed_df['time']):
        print("❌ Time column is not datetime type!")
        print(f"  - Current type: {feed_df['time'].dtype}")
        print(f"  - Sample values: {feed_df['time'].head(3).tolist()}")
        return False
    
    print(f"✓ Time is datetime type")
    
    # Show time range
    min_time = feed_df['time'].min()
    max_time = feed_df['time'].max()
    print(f"  - Range: {min_time} to {max_time}")
    print(f"  - Span: {(max_time - min_time).days} days")
    
    # Check for H/L/C columns (origin data)
    origin_cols = [col for col in feed_df.columns if col.endswith((' H', ' L', ' C'))]
    print(f"✓ Found {len(origin_cols)} origin columns:")
    for col in origin_cols[:5]:  # Show first 5
        print(f"  - {col}")
    if len(origin_cols) > 5:
        print(f"  ... and {len(origin_cols) - 5} more")
    
    # Check for actual high/low values
    high_cols = [col for col in feed_df.columns if col.endswith(' H')]
    low_cols = [col for col in feed_df.columns if col.endswith(' L')]
    
    if high_cols:
        print(f"\nHigh columns ({len(high_cols)} found):")
        for col in high_cols[:3]:
            non_null = feed_df[col].notna().sum()
            if non_null > 0:
                print(f"  ✓ {col}: {non_null} non-null values")
                print(f"    Range: {feed_df[col].min():.2f} to {feed_df[col].max():.2f}")
            else:
                print(f"  ❌ {col}: All null!")
    
    if low_cols:
        print(f"\nLow columns ({len(low_cols)} found):")
        for col in low_cols[:3]:
            non_null = feed_df[col].notna().sum()
            if non_null > 0:
                print(f"  ✓ {col}: {non_null} non-null values")
                print(f"    Range: {feed_df[col].min():.2f} to {feed_df[col].max():.2f}")
            else:
                print(f"  ❌ {col}: All null!")
    
    return True

def test_day_filtering(feed_df, report_time, day_start_hour=18):
    """Test if day filtering would find data"""
    print(f"\n{'='*60}")
    print(f"Testing Day Filtering")
    print(f"{'='*60}")
    
    if feed_df.empty or 'time' not in feed_df.columns:
        print("❌ Cannot test: Feed is empty or has no time column")
        return
    
    # Calculate day boundaries for the most recent day
    report_date = report_time.date()
    day_start = dt.datetime.combine(report_date, dt.time(day_start_hour, 0))
    day_end = day_start + dt.timedelta(hours=24)
    
    print(f"Report time: {report_time}")
    print(f"Day start hour: {day_start_hour}")
    print(f"Testing day: {day_start} to {day_end}")
    
    # Filter data for this day
    day_data = feed_df[(feed_df['time'] >= day_start) & (feed_df['time'] < day_end)]
    
    print(f"\n{'='*60}")
    if len(day_data) == 0:
        print(f"❌ No data found for this day!")
        print(f"\nFeed time range: {feed_df['time'].min()} to {feed_df['time'].max()}")
        print(f"Looking for: {day_start} to {day_end}")
        
        # Check if data exists nearby
        before = feed_df[feed_df['time'] < day_start]
        after = feed_df[feed_df['time'] >= day_end]
        
        if len(before) > 0:
            print(f"✓ Found {len(before)} rows BEFORE day start")
            print(f"  Latest: {before['time'].max()}")
        
        if len(after) > 0:
            print(f"✓ Found {len(after)} rows AFTER day end")
            print(f"  Earliest: {after['time'].min()}")
    else:
        print(f"✓ Found {len(day_data)} rows for this day")
        print(f"  Time range: {day_data['time'].min()} to {day_data['time'].max()}")

# Example usage:
if __name__ == "__main__":
    print("\n" + "="*60)
    print("HOD/LOD DIAGNOSTIC TOOL")
    print("="*60)
    print("\nLoad your CSV files and run:")
    print("  small_df = pd.read_csv('small_15m.csv', parse_dates=['time'])")
    print("  big_df = pd.read_csv('big_15m.csv', parse_dates=['time'])")
    print("  diagnose_feed_data(small_df, 'Small Feed')")
    print("  diagnose_feed_data(big_df, 'Big Feed')")
    print("  test_day_filtering(small_df, dt.datetime.now())")
