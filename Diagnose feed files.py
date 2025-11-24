"""
Diagnostic Script - Check Feed File Format
==========================================
Run this to see what columns your feed files actually have.
"""

import pandas as pd
import sys

print("="*80)
print("FEED FILE COLUMN CHECKER")
print("="*80)

if len(sys.argv) < 2:
    print("\nUsage: python diagnose_feed_files.py <your_feed_file.csv>")
    print("\nThis will show you what columns are in your file.")
    sys.exit(1)

filename = sys.argv[1]

try:
    df = pd.read_csv(filename)
    
    print(f"\n✅ Loaded: {filename}")
    print(f"   Rows: {len(df)}")
    print(f"   Columns: {len(df.columns)}")
    
    print("\n📋 COLUMNS FOUND:")
    print("-" * 80)
    for i, col in enumerate(df.columns, 1):
        sample = df[col].dropna().iloc[0] if not df[col].dropna().empty else "N/A"
        print(f"  {i:2d}. {col:25s} - Sample: {sample}")
    
    print("\n" + "="*80)
    print("REQUIRED COLUMNS FOR STRATEGIC ZONES:")
    print("="*80)
    
    required_cols = ['M #', 'R #', 'Origin', 'Output', 'Arrival', 'Feed']
    
    print("\nChecking for required columns...")
    for req_col in required_cols:
        if req_col in df.columns:
            print(f"  ✅ {req_col}")
        else:
            print(f"  ❌ {req_col} - MISSING!")
            
            # Check for similar column names
            similar = [col for col in df.columns if req_col.lower().replace(' ', '').replace('#', '') in col.lower().replace(' ', '').replace('#', '')]
            if similar:
                print(f"     Possible matches: {similar}")
    
    # Check for Day column (optional but helpful)
    if 'Day' in df.columns:
        print(f"  ✅ Day (optional)")
    
    print("\n" + "="*80)
    print("DIAGNOSIS:")
    print("="*80)
    
    missing = [col for col in required_cols if col not in df.columns]
    
    if not missing:
        print("\n✅ All required columns present!")
        print("   This file should work with Strategic Zones.")
    else:
        print(f"\n❌ Missing columns: {missing}")
        print("\n💡 SOLUTIONS:")
        print("   1. Check if column names are slightly different")
        print("   2. Make sure column names match exactly (including spaces and #)")
        print("   3. Common issues:")
        print("      - 'M#' instead of 'M #' (missing space)")
        print("      - 'R#' instead of 'R #' (missing space)")
        print("      - 'feed' instead of 'Feed' (wrong case)")
        print("\n   If your file has traveler data but different column names,")
        print("   you may need to rename the columns in Excel or add a mapping.")
    
    print("\n" + "="*80)
    print("SAMPLE DATA (first 3 rows):")
    print("="*80)
    print(df.head(3).to_string())
    
except Exception as e:
    print(f"\n❌ Error reading file: {e}")
    import traceback
    print("\nFull error:")
    traceback.print_exc()
