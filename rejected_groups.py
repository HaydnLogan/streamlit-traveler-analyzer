# Example: Integrating Rejected Groups Display into Streamlit App

"""
This file shows complete code examples for displaying rejected groups
in your Streamlit application. Choose the approach that fits your app best.
"""

import streamlit as st
import pandas as pd

# ============================================================================
# EXAMPLE 1: Simple Checkbox with Expanders (RECOMMENDED)
# ============================================================================

def display_rejected_groups_simple(results):
    """
    Simple display of rejected groups with checkbox toggle.
    Add this function to your app and call it after running detection.
    """
    if not results.get('rejected_groups'):
        return
    
    show_rejected = st.checkbox(
        f"🚫 Show Rejected Groups ({len(results['rejected_groups'])})", 
        value=False,
        help="View pairs/groups that were detected but filtered out"
    )
    
    if show_rejected:
        st.write("---")
        st.write("### Rejected Groups Details")
        
        # Group rejections by reason
        by_reason = {}
        for rejected in results['rejected_groups']:
            reason = rejected['reasons'][0]  # First reason
            if reason not in by_reason:
                by_reason[reason] = []
            by_reason[reason].append(rejected)
        
        # Display grouped by reason
        for reason, items in by_reason.items():
            st.write(f"**{reason}** ({len(items)} pairs)")
            
            for idx, rejected in enumerate(items[:10], 1):  # Show first 10 of each type
                with st.expander(f"Pair #{idx}: M#{rejected['m_values']}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Item 1:**")
                        st.write(f"Output: {rejected['outputs'][0]:.4f}")
                        st.write(f"Origin: {rejected['origins'][0]}")
                        st.write(f"M#: {rejected['m_values'][0]}")
                        st.write(f"Feed: {rejected['feeds'][0]}")
                    
                    with col2:
                        st.write("**Item 2:**")
                        st.write(f"Output: {rejected['outputs'][1]:.4f}")
                        st.write(f"Origin: {rejected['origins'][1]}")
                        st.write(f"M#: {rejected['m_values'][1]}")
                        st.write(f"Feed: {rejected['feeds'][1]}")
                    
                    # Show pattern info if available (G.11)
                    if 'type' in rejected:
                        st.write(f"**Pattern Type:** {rejected['type']}")
                        st.write(f"**Classification:** {rejected['classification']}")
                        st.write(f"**Group:** {rejected['group']}")
            
            if len(items) > 10:
                st.info(f"Showing 10 of {len(items)} rejected pairs for this reason")


# ============================================================================
# EXAMPLE 2: Tabbed Interface
# ============================================================================

def display_results_with_rejected_tab(results):
    """
    Display results in tabs including a rejected groups tab.
    Use this to integrate rejected groups into your existing results display.
    """
    tab1, tab2, tab3 = st.tabs([
        f"Today ({len(results['today_sequences'])})",
        f"Other Days ({len(results['other_day_sequences'])})",
        f"🚫 Rejected ({len(results['rejected_groups'])})"
    ])
    
    with tab1:
        st.write("### Today's Sequences")
        # Your existing display code for today sequences
        if results['today_sequences']:
            for seq in results['today_sequences']:
                st.write(f"- {seq['classification']}: {seq['outputs']}")
        else:
            st.info("No sequences found for today")
    
    with tab2:
        st.write("### Other Days Sequences")
        # Your existing display code for other day sequences
        if results['other_day_sequences']:
            for seq in results['other_day_sequences']:
                st.write(f"- {seq['classification']}: {seq['outputs']}")
        else:
            st.info("No sequences found for other days")
    
    with tab3:
        st.write("### Rejected Groups")
        if results['rejected_groups']:
            display_rejected_groups_simple(results)
        else:
            st.success("No rejected groups - all potential matches were accepted!")


# ============================================================================
# EXAMPLE 3: Detailed Table View
# ============================================================================

def display_rejected_groups_table(results):
    """
    Display rejected groups in a sortable table format.
    Good for when you have many rejections and want to analyze them.
    """
    if not results.get('rejected_groups'):
        st.info("No rejected groups")
        return
    
    show_rejected = st.checkbox("Show Rejected Groups Table", value=False)
    
    if show_rejected:
        # Convert rejected groups to DataFrame
        rejected_data = []
        for rejected in results['rejected_groups']:
            rejected_data.append({
                'Output 1': rejected['outputs'][0],
                'Output 2': rejected['outputs'][1],
                'Origin 1': rejected['origins'][0],
                'Origin 2': rejected['origins'][1],
                'M# 1': rejected['m_values'][0],
                'M# 2': rejected['m_values'][1],
                'Feed 1': rejected['feeds'][0],
                'Feed 2': rejected['feeds'][1],
                'Reason': rejected['reasons'][0],
                'Pattern': rejected.get('type', 'N/A'),
                'Group': rejected.get('group', 'N/A')
            })
        
        df_rejected = pd.DataFrame(rejected_data)
        
        # Add filters
        col1, col2 = st.columns(2)
        with col1:
            reason_filter = st.multiselect(
                "Filter by Reason",
                options=df_rejected['Reason'].unique(),
                default=df_rejected['Reason'].unique()
            )
        with col2:
            pattern_filter = st.multiselect(
                "Filter by Pattern",
                options=df_rejected['Pattern'].unique(),
                default=df_rejected['Pattern'].unique()
            )
        
        # Apply filters
        filtered = df_rejected[
            (df_rejected['Reason'].isin(reason_filter)) &
            (df_rejected['Pattern'].isin(pattern_filter))
        ]
        
        st.write(f"Showing {len(filtered)} of {len(df_rejected)} rejected groups")
        st.dataframe(filtered, use_container_width=True)
        
        # Download button
        csv = filtered.to_csv(index=False)
        st.download_button(
            "Download Rejected Groups CSV",
            csv,
            "rejected_groups.csv",
            "text/csv",
            key='download-rejected'
        )


# ============================================================================
# EXAMPLE 4: Statistics Summary
# ============================================================================

def display_rejection_statistics(results):
    """
    Show high-level statistics about rejections.
    Good for the sidebar or top of your app.
    """
    if not results.get('rejected_groups'):
        return
    
    total_rejected = len(results['rejected_groups'])
    total_accepted = len(results['today_sequences']) + len(results['other_day_sequences'])
    total_detected = total_accepted + total_rejected
    
    if total_detected > 0:
        rejection_rate = (total_rejected / total_detected) * 100
    else:
        rejection_rate = 0
    
    # Count by reason
    reason_counts = {}
    for rejected in results['rejected_groups']:
        reason = rejected['reasons'][0]
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
    
    with st.expander("📊 Detection Statistics"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Detected", total_detected)
        with col2:
            st.metric("Accepted", total_accepted)
        with col3:
            st.metric("Rejected", total_rejected, 
                     delta=f"{rejection_rate:.1f}%",
                     delta_color="inverse")
        
        if reason_counts:
            st.write("**Rejection Breakdown:**")
            for reason, count in sorted(reason_counts.items(), key=lambda x: x[1], reverse=True):
                st.write(f"- {reason}: {count}")


# ============================================================================
# EXAMPLE 5: Complete Integration Example
# ============================================================================

def complete_example():
    """
    Complete example showing how to integrate everything into your Streamlit app.
    """
    st.title("Model G Detection Results")
    
    # Sidebar controls
    with st.sidebar:
        st.header("Detection Settings")
        
        # Your existing controls
        proximity_threshold = st.slider("Proximity Threshold", 0.5, 5.0, 3.0)
        
        # Group filters
        st.subheader("Group Filters")
        enabled_groups = []
        for i in range(5):
            if st.checkbox(f"Group {i}", value=True, key=f"group_{i}"):
                enabled_groups.append(i)
        
        # Display filters
        display_recipes = st.checkbox("Show Recipes", value=True)
        display_others = st.checkbox("Show Others", value=True)
        
        # Debug mode
        st.subheader("Debug Options")
        debug_mode = st.checkbox("Enable Debug Mode", key='debug_g11')
        if debug_mode:
            st.session_state['debug_g11'] = True
        else:
            st.session_state['debug_g11'] = False
    
    # Run detection
    # df = load_your_data()  # Your data loading code
    # results = run_g11_detection(
    #     df,
    #     proximity_threshold=proximity_threshold,
    #     enabled_groups=enabled_groups,
    #     display_recipes=display_recipes,
    #     display_others=display_others
    # )
    
    # Mock results for example
    results = {
        'today_sequences': [],
        'other_day_sequences': [],
        'rejected_groups': []
    }
    
    # Display rejection statistics at the top
    display_rejection_statistics(results)
    
    # Main results display
    st.write("---")
    
    # Option 1: Use tabs
    display_results_with_rejected_tab(results)
    
    # OR Option 2: Use simple checkbox
    # display_rejected_groups_simple(results)
    
    # OR Option 3: Use table view
    # display_rejected_groups_table(results)


# ============================================================================
# USAGE INSTRUCTIONS
# ============================================================================

"""
TO USE THESE EXAMPLES:

1. Choose the display style you prefer (Simple, Tabbed, Table, or Statistics)

2. Copy the relevant function(s) to your Streamlit app

3. After running detection, call the display function:
   
   results = run_g11_detection(df, ...)
   display_rejected_groups_simple(results)  # or whichever function you chose

4. For debug mode, add this to your sidebar:
   
   if st.sidebar.checkbox("Enable G.11 Debug", key='debug_g11'):
       st.session_state['debug_g11'] = True

5. The same approaches work for G.05/G.06 - just use 'debug_g06' instead

TIPS:
- Start with the Simple checkbox approach - it's easiest to integrate
- Use the Table view if you have many rejections and need to analyze patterns
- Use Statistics in the sidebar for at-a-glance monitoring
- Combine multiple approaches for the most comprehensive view
"""

if __name__ == "__main__":
    complete_example()
