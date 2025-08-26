"""
Full Range Processing Module
Handles full range calculations separately from custom ranges for cleaner code organization.
"""

import pandas as pd
import streamlit as st
import time
from custom_range_calculator import apply_custom_ranges_advanced, process_custom_ranges_advanced

def process_full_range(master_measurements_df, small_df, big_df, report_time, 
                      input_value_at_start, full_range_value, 
                      GROUP_1A_TRAVELERS, GROUP_1B_TRAVELERS, GROUP_2A_TRAVELERS, GROUP_2B_TRAVELERS):
    """
    Process full range using unified approach.
    
    Args:
        master_measurements_df: Measurement dataframe
        small_df: Small CSV dataframe
        big_df: Big CSV dataframe  
        report_time: Report datetime
        input_value_at_start: Center value for range
        full_range_value: Plus/minus range value
        GROUP_*_TRAVELERS: Traveler group lists
        
    Returns:
        Dictionary with traveler_reports and processing_time
    """
    performance_start = time.time()
    
    # Convert full range to custom range format (one big range)
    range_center = input_value_at_start
    high_range = range_center + full_range_value
    low_range = range_center - full_range_value
    
    st.info(f"Processing full range from \"{low_range:.1f}\" to \"{high_range:.1f}\"")
    
    # Use the same advanced processing as custom ranges to ensure origin consistency
    # Create a single large custom range that covers the full range
    # NOTE: Full range treats range_center as a High range (value-24 to value)
    custom_ranges = {
        'High Full Range': {'enabled': True, 'value': high_range}
    }
    
    st.warning(f"🔧 FULL RANGE DEBUG: Using High Full Range with value {high_range} (range: {low_range:.1f} to {high_range:.1f})")
    
    # Use process_custom_ranges_advanced to get proper origin processing
    # This ensures Wasp-12b[1], Wasp-12b[2], Macedonia[1], Macedonia[2] are processed
    from custom_range_calculator import process_custom_ranges_advanced
    full_range_results = process_custom_ranges_advanced(
        measurement_df=master_measurements_df,
        small_df=small_df,
        report_time=report_time,
        custom_ranges=custom_ranges,
        big_df=big_df,
        run_model_g=False
    )
    
    st.info(f"🔍 Full range processing completed with {len(full_range_results)} entries")
    
    # DEBUG: Show what origins are actually in the results
    if isinstance(full_range_results, list) and len(full_range_results) > 0:
        if isinstance(full_range_results[0], dict) and 'Origin' in full_range_results[0]:
            result_origins = [entry.get('Origin', 'Unknown') for entry in full_range_results]
            unique_origins = list(set(result_origins))
            wasp_found = [o for o in unique_origins if 'wasp' in str(o).lower()]
            macedonia_found = [o for o in unique_origins if 'macedonia' in str(o).lower()]
            st.warning(f"🔍 DEBUG: Origins in results - Wasp: {wasp_found}, Macedonia: {macedonia_found}")
            st.warning(f"🔍 DEBUG: All unique origins: {unique_origins[:10]}...")  # First 10
        else:
            st.warning(f"🔍 DEBUG: Result format: {type(full_range_results[0]) if full_range_results else 'empty'}")
    elif hasattr(full_range_results, 'columns'):
        # It's a DataFrame
        if 'Origin' in full_range_results.columns:
            unique_origins = full_range_results['Origin'].unique().tolist()
            wasp_found = [o for o in unique_origins if 'wasp' in str(o).lower()]
            macedonia_found = [o for o in unique_origins if 'macedonia' in str(o).lower()]
            st.warning(f"🔍 DEBUG: DataFrame Origins - Wasp: {wasp_found}, Macedonia: {macedonia_found}")
        else:
            st.warning(f"🔍 DEBUG: DataFrame columns: {list(full_range_results.columns)}")
    else:
        st.warning(f"🔍 DEBUG: Unexpected result type: {type(full_range_results)}")
    
    # Process results
    if (isinstance(full_range_results, list) and len(full_range_results) > 0) or \
       (hasattr(full_range_results, '__len__') and len(full_range_results) > 0) or \
       (hasattr(full_range_results, 'empty') and not full_range_results.empty):
        # Handle different result formats
        if isinstance(full_range_results, list):
            # Convert list of dictionaries to DataFrame
            import pandas as pd
            final_df_filtered = pd.DataFrame(full_range_results)
        else:
            # Already a DataFrame
            final_df_filtered = full_range_results.copy()
        
        # Add Range column for consistency if not present
        if 'Range' not in final_df_filtered.columns:
            final_df_filtered['Range'] = 'Full Range'
        
        # Sort by Output descending, then Arrival ascending
        if 'Output' in final_df_filtered.columns:
            final_df_filtered = final_df_filtered.sort_values(['Output', 'Arrival'], ascending=[False, True])
        
        st.markdown(f"**Full range: found {len(final_df_filtered)} valid entries**")
        
        # Show origin processing summary to verify bracket variations are included
        if 'Origin' in final_df_filtered.columns:
            origins_found = final_df_filtered['Origin'].unique()
            wasp_origins = [o for o in origins_found if 'wasp' in str(o).lower()]
            macedonia_origins = [o for o in origins_found if 'macedonia' in str(o).lower()]
            
            if wasp_origins or macedonia_origins:
                st.info(f"🎯 Epic Origins found: Wasp variants: {wasp_origins}, Macedonia variants: {macedonia_origins}")
        
        # Processing Summary
        processing_time = time.time() - performance_start
        st.markdown(f"Advanced full range calculation completed in {processing_time:.1f} seconds.")
        
        # Group results by travelers
        if 'M #' in final_df_filtered.columns:
            traveler_reports = {}
            
            # Group 1a
            group_1a_mask = final_df_filtered['M #'].isin(GROUP_1A_TRAVELERS)
            traveler_reports["Grp 1a"] = final_df_filtered[group_1a_mask].copy()
            
            # Group 1b
            group_1b_mask = final_df_filtered['M #'].isin(GROUP_1B_TRAVELERS)
            traveler_reports["Grp 1b"] = final_df_filtered[group_1b_mask].copy()
            
            # Group 2a
            group_2a_mask = final_df_filtered['M #'].isin(GROUP_2A_TRAVELERS)
            traveler_reports["Grp 2a"] = final_df_filtered[group_2a_mask].copy()
            
            # Group 2b
            group_2b_mask = final_df_filtered['M #'].isin(GROUP_2B_TRAVELERS)
            traveler_reports["Grp 2b"] = final_df_filtered[group_2b_mask].copy()
            
            # Sort each group by Output descending, Arrival ascending
            for group_name, group_df in traveler_reports.items():
                if not group_df.empty:
                    if 'Output' in group_df.columns and 'Arrival' in group_df.columns:
                        traveler_reports[group_name] = group_df.sort_values(
                            ['Output', 'Arrival'], ascending=[False, True]
                        )
                    elif 'Output' in group_df.columns:
                        traveler_reports[group_name] = group_df.sort_values(
                            ['Output'], ascending=[False]
                        )
            
            return {
                'traveler_reports': traveler_reports,
                'processing_time': processing_time,
                'low_range': low_range,
                'high_range': high_range
            }
        else:
            st.warning("No 'M #' column found in full range results - cannot group by travelers")
            return {
                'traveler_reports': {},
                'processing_time': processing_time,
                'low_range': low_range,
                'high_range': high_range
            }
    else:
        processing_time = time.time() - performance_start
        st.warning("No data generated from full range processing")
        return {
            'traveler_reports': {},
            'processing_time': processing_time,
            'low_range': low_range,
            'high_range': high_range
        }

def display_full_range_results(result_data):
    """
    Display full range results in unified report format.
    
    Args:
        result_data: Dictionary from process_full_range
    """
    traveler_reports = result_data['traveler_reports']
    processing_time = result_data['processing_time']
    low_range = result_data['low_range']
    high_range = result_data['high_range']
    
    if traveler_reports:
        # Display unified report
        st.markdown("---")
        st.markdown("### 📊 Unified Report by Groups")
        
        total_entries = sum(len(df) for df in traveler_reports.values())
        st.info(f"Found {len(traveler_reports)} groups with {total_entries} total entries")
        
        # Display each group
        for group_name, group_df in traveler_reports.items():
            st.markdown(f"#### 📋 {group_name}")
            st.info(f"{len(group_df)} entries")
            
            if not group_df.empty:
                st.dataframe(group_df, use_container_width=True)
            else:
                st.warning(f"{group_name} is empty")
                
            # Add separator between groups
            if group_name != "Grp 2b":
                st.markdown("---")
        
        # Processing summary at the end
        st.markdown("---")
        st.markdown("### ✅ Full Range Processing Complete")
        st.success(f"Advanced range calculation completed in {processing_time:.1f} seconds")
        st.info(f"Processed full range: {low_range:.1f} to {high_range:.1f}")
        st.success("🎉 Full Range Processing Successfully Completed!")
        
        return traveler_reports
    else:
        st.warning("No traveler groups generated from full range processing")
        return {}
