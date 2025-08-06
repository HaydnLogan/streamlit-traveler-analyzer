import streamlit as st
import pandas as pd
import time
from datetime import datetime
from a_helpers2 import (
    generate_master_traveler_list,
    filter_master_list_by_group,
    create_excel_with_highlighting,
    GROUP_1A_VALUES,
    GROUP_1B_VALUES, 
    GROUP_2A_VALUES,
    GROUP_2B_VALUES
)

st.title("🚀 Master Traveler List Performance Test")
st.write("Testing master list approach vs individual tab processing")

# File uploads - same as original app
st.subheader("📁 File Upload")
col1, col2, col3 = st.columns(3)

with col1:
    data_file = st.file_uploader("Upload Data CSV", type=['csv'], key="data")
with col2:
    meas_file = st.file_uploader("Upload Measurements Excel", type=['xlsx'], key="meas") 
with col3:
    small_file = st.file_uploader("Upload Small CSV", type=['csv'], key="small")

if data_file and meas_file and small_file:
    # Load all three files
    try:
        data = pd.read_csv(data_file)
        measurements = pd.read_excel(meas_file, sheet_name=0)  # Use first sheet for master list
        small_df = pd.read_csv(small_file)
        
        st.success("✅ All 3 files loaded successfully!")
    except Exception as e:
        st.error(f"Error loading files: {str(e)}")
        st.stop()
    
    st.write(f"📊 Data loaded: {len(data)} rows, {len(data.columns)} columns")
    
    # Settings
    col1, col2 = st.columns(2)
    with col1:
        start_hour = st.selectbox("Start Hour", [17, 18], index=0)
        report_time = st.text_input("Report Time", "2025-01-05 17:00:00")
    
    with col2:
        fast_mode = st.checkbox("Fast Mode", value=True)
        minimal_display = st.checkbox("Minimal Display", value=True)
    
    if st.button("🚀 Generate Master List + 4 Sub-Reports"):
        try:
            report_datetime = pd.to_datetime(report_time)
            
            # Timer start
            start_time = time.time()
            
            st.write("⏱️ **Step 1: Creating Master Traveler List**")
            step1_start = time.time()
            
            # Generate master traveler list with all M# values
            master_list = generate_master_traveler_list(
                data, 
                measurements,
                small_df,
                report_datetime, 
                start_hour,
                fast_mode=fast_mode
            )
            
            step1_time = time.time() - step1_start
            st.write(f"✅ Master list created: {len(master_list)} entries in {step1_time:.1f}s")
            
            # Timer for sub-reports
            st.write("⏱️ **Step 2: Filtering for 4 Sub-Reports**")
            step2_start = time.time()
            
            # Create 4 filtered sub-reports
            reports = {}
            group_values = {
                "Grp 1a": GROUP_1A_VALUES,
                "Grp 1b": GROUP_1B_VALUES, 
                "Grp 2a": GROUP_2A_VALUES,
                "Grp 2b": GROUP_2B_VALUES
            }
            
            for group_name, m_values in group_values.items():
                filtered_df = filter_master_list_by_group(master_list, m_values)
                reports[group_name] = filtered_df
                st.write(f"📋 {group_name}: {len(filtered_df)} entries")
            
            step2_time = time.time() - step2_start
            st.write(f"✅ All 4 sub-reports created in {step2_time:.1f}s")
            
            # Total time
            total_time = time.time() - start_time
            st.success(f"🎉 **Total Processing Time: {total_time:.1f} seconds**")
            
            # Performance comparison
            st.write("📈 **Performance Comparison:**")
            estimated_old_time = 500  # Current 4-tab processing time
            improvement = ((estimated_old_time - total_time) / estimated_old_time) * 100
            st.write(f"- Old method: ~{estimated_old_time}s")
            st.write(f"- New method: {total_time:.1f}s")
            st.write(f"- **Improvement: {improvement:.1f}% faster** 🚀")
            
            # Display summary
            if not minimal_display:
                st.write("📊 **Data Preview:**")
                for group_name, df in reports.items():
                    with st.expander(f"{group_name} - {len(df)} entries"):
                        st.dataframe(df.head(10))
            
            # Excel export
            st.write("💾 **Excel Export:**")
            excel_start = time.time()
            
            excel_buffer = create_excel_with_highlighting(
                reports,
                f"master_traveler_reports_{report_datetime.strftime('%Y%m%d_%H%M%S')}.xlsx"
            )
            
            excel_time = time.time() - excel_start
            st.write(f"✅ Excel file created in {excel_time:.1f}s")
            
            # Download button
            st.download_button(
                label="📥 Download Master Reports Excel",
                data=excel_buffer,
                file_name=f"master_traveler_reports_{report_datetime.strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

else:
    st.info("👆 Please upload all 3 files to begin testing:")
    st.write("1. **Data CSV**: Main time series data with high/low/close prices")
    st.write("2. **Measurements Excel**: File with M# values and measurement definitions") 
    st.write("3. **Small CSV**: Input data for calculations")
    
    # Show group information
    st.write("📋 **Traveler Group Definitions:**")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Group 1a** (18 values):")
        st.code(str(GROUP_1A_VALUES[:10]) + "...")
        
        st.write("**Group 1b** (48 values):")
        st.code(str(GROUP_1B_VALUES[:10]) + "...")
    
    with col2:
        st.write("**Group 2a** (52 values):")
        st.code(str(GROUP_2A_VALUES[:10]) + "...")
        
        st.write("**Group 2b** (70 values):")
        st.code(str(GROUP_2B_VALUES[:10]) + "...")

