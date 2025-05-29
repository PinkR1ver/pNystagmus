import plistlib
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime
import streamlit as st
import tempfile
import zipfile
import shutil
from io import BytesIO
import pandas as pd

# Import utility modules (we'll need to create unified utils)
import utils  # Nystagmus analysis utilities
from new_plist_parser import parse_new_plist  # New plist parser for unified format
import hit_utils  # HIT analysis utilities

# Set page configuration
st.set_page_config(layout="wide", page_title="pVestibular - Vestibular Analysis Platform")

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.analysis-section {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 10px;
    margin: 1rem 0;
}
.metric-container {
    background-color: white;
    padding: 1rem;
    border-radius: 5px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

# Main title
st.markdown('<h1 class="main-header">🧠 pVestibular Analysis Platform</h1>', unsafe_allow_html=True)
st.markdown("**Comprehensive Nystagmus and Head Impulse Test (HIT) Analysis**")

# Sidebar for navigation and settings
st.sidebar.title("🔧 Analysis Control Panel")

# Analysis type selection
analysis_type = st.sidebar.selectbox(
    "Select Analysis Type",
    ["Nystagmus Analysis", "HIT Analysis"],
    help="Choose between nystagmus pattern analysis or head impulse test analysis"
)

st.sidebar.markdown("---")

# File upload section
st.sidebar.subheader("📁 Data Upload")

if analysis_type == "Nystagmus Analysis":
    st.sidebar.markdown("**Upload Nystagmus Data**")
    uploaded_files = st.sidebar.file_uploader(
        "Upload .plist files or ZIP archive",
        type=["plist", "zip"],
        accept_multiple_files=True,
        help="Upload individual .plist files or a ZIP archive containing nystagmus data"
    )
    
elif analysis_type == "HIT Analysis":
    st.sidebar.markdown("**Upload HIT Data**")
    uploaded_files = st.sidebar.file_uploader(
        "Upload HIT .plist files",
        type=["plist"],
        accept_multiple_files=True,
        help="Upload HIT data files (old format: separate head/eye files, or new format: unified file)"
    )

st.sidebar.markdown("---")

# Analysis parameters based on type
if analysis_type == "Nystagmus Analysis":
    st.sidebar.subheader("👁️ Nystagmus Parameters")
    
    # Analysis axis selection
    analysis_axis_option = st.sidebar.radio(
        "Analysis Axis", 
        ("Horizontal (X-axis)", "Vertical (Y-axis)"),
        help="Select the eye movement axis to analyze"
    )
    
    # Set direction_axis and data key based on selection
    if analysis_axis_option == "Horizontal (X-axis)":
        direction_axis = "horizontal"
        axis_data_key = "LeftEyeXDegList"
    else:
        direction_axis = "vertical"
        axis_data_key = "LeftEyeYDegList"
    
    # Preprocessing parameters - match original design
    st.sidebar.subheader("Preprocessing Parameters")
    
    # High-pass filter parameters
    st.sidebar.markdown("**High-pass Filter Parameters**")
    highpass_cutoff = st.sidebar.slider("High-pass Cutoff (Hz)", 0.01, 1.0, 0.1, 0.01)
    highpass_fs = st.sidebar.slider("High-pass Sampling Freq (Hz)", 30, 120, 60, 5)
    highpass_order = st.sidebar.slider("High-pass Filter Order", 1, 10, 5, 1)
    
    # Low-pass filter parameters
    st.sidebar.markdown("**Low-pass Filter Parameters**")
    lowpass_cutoff = st.sidebar.slider("Low-pass Cutoff (Hz)", 1.0, 20.0, 6.0, 0.5)
    lowpass_fs = st.sidebar.slider("Low-pass Sampling Freq (Hz)", 30, 120, 60, 5)
    lowpass_order = st.sidebar.slider("Low-pass Filter Order", 1, 10, 5, 1)
    
    # Other preprocessing parameters
    interpolate_ratio = st.sidebar.slider("Interpolation Ratio", 1, 20, 10, 1)
    
    # Turning point detection parameters
    st.sidebar.subheader("Turning Point Detection Parameters")
    prominence = st.sidebar.slider("Peak Prominence", 0.01, 1.0, 0.1, 0.01)
    distance = st.sidebar.slider("Min Distance Between Points", 50, 500, 150, 10)
    
    # Nystagmus pattern recognition parameters
    st.sidebar.subheader("Nystagmus Pattern Recognition Parameters")
    min_time = st.sidebar.slider("Min Time Threshold (s)", 0.1, 1.0, 0.3, 0.05)
    max_time = st.sidebar.slider("Max Time Threshold (s)", 0.5, 2.0, 0.8, 0.05)
    min_ratio = st.sidebar.slider("Min Slope Ratio", 1.0, 3.0, 1.4, 0.1)
    max_ratio = st.sidebar.slider("Max Slope Ratio", 3.0, 15.0, 8.0, 0.5)

elif analysis_type == "HIT Analysis":
    st.sidebar.subheader("🎯 HIT Parameters")
    
    # Eye selection for HIT analysis
    eye_selection = st.sidebar.radio(
        "Eye Selection",
        ("Left Eye", "Right Eye", "Both Eyes"),
        help="Select which eye(s) to analyze"
    )
    
    # Preprocessing parameters for HIT
    with st.sidebar.expander("🔧 Preprocessing", expanded=True):
        hit_highpass_cutoff = st.slider("High-pass Cutoff (Hz)", 0.01, 1.0, 0.1, 0.01)
        hit_lowpass_cutoff = st.slider("Low-pass Cutoff (Hz)", 1.0, 20.0, 6.0, 0.5)
        hit_window_size = st.slider("Moving Average Window", 3, 51, 5, 2)
        hit_interpolate_ratio = st.slider("Interpolation Ratio", 1, 20, 10, 1)
    
    # VOR analysis parameters
    with st.sidebar.expander("📊 VOR Analysis"):
        peak_detection_distance = st.slider("Peak Detection Distance", 100, 500, 200, 10)
        analysis_window_size = st.slider("Analysis Window Size", 200, 600, 400, 20)

# Helper functions for data processing
def detect_file_format(file_content):
    """Detect the format of uploaded plist file"""
    try:
        plist_data = plistlib.loads(file_content)
        
        # Check for new unified format
        if all(key in plist_data for key in ['HeadXList', 'LeftEyeXList', 'TimeList']):
            return 'new_hit_format'
        
        # Check for nystagmus format
        if 'TimeList' in plist_data and any(key.endswith('DegList') for key in plist_data.keys()):
            return 'nystagmus_format'
        
        # Check for old HIT format (single array)
        if isinstance(plist_data, list):
            return 'old_hit_format'
        
        return 'unknown'
    except:
        return 'invalid'

def process_nystagmus_data(uploaded_files):
    """Process uploaded nystagmus data files"""
    if not uploaded_files:
        st.warning("Please upload nystagmus data files.")
        return None
    
    results = []
    
    for uploaded_file in uploaded_files:
        try:
            # Handle ZIP files
            if uploaded_file.name.endswith('.zip'):
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_zip_path = os.path.join(temp_dir, "uploaded.zip")
                    with open(temp_zip_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
                        zip_ref.extractall(temp_dir)
                    
                    # Process extracted files
                    for root, dirs, files in os.walk(temp_dir):
                        for file in files:
                            if file.endswith('.plist'):
                                file_path = os.path.join(root, file)
                                with open(file_path, 'rb') as f:
                                    result = analyze_single_nystagmus_file(f, file)
                                    if result:
                                        results.append(result)
            else:
                # Process single plist file
                result = analyze_single_nystagmus_file(uploaded_file, uploaded_file.name)
                if result:
                    results.append(result)
                    
        except Exception as e:
            st.error(f"Error processing file {uploaded_file.name}: {str(e)}")
    
    return results

def analyze_single_nystagmus_file(file_obj, filename):
    """Analyze a single nystagmus file - updated to match original design"""
    try:
        # Parse the file
        timestamps, eye_angles, available_keys = parse_new_plist(file_obj, axis_data_key=axis_data_key)
        
        if eye_angles.size == 0:
            st.error(f"Could not load data for '{axis_data_key}' from the file {filename}, or data is empty.")
            if available_keys:
                st.info(f"Available eye movement data list keys in file: {available_keys}. Please check if the selected axis is correct or check the file content.")
            else:
                st.info("No eye movement data keys ending with 'DegList' found in the file.")
            return None
        
        if timestamps.size == 0:
            st.error(f"Could not load timestamp data (TimeList) from the file {filename}, or data is empty.")
            return None

        # Ensure equal lengths
        min_length = min(len(timestamps), len(eye_angles))
        timestamps = timestamps[:min_length]
        eye_angles = eye_angles[:min_length]
        
        # Set up parameters - match original parameter structure
        highpass_parameter = {'cutoff': highpass_cutoff, 'fs': highpass_fs, 'order': highpass_order}
        lowpass_parameter = {'cutoff': lowpass_cutoff, 'fs': lowpass_fs, 'order': lowpass_order}
        
        # Signal preprocessing - using original data structure
        filtered_signal, time = utils.signal_preprocess(
            timestamps, eye_angles,
            highpass_parameter=highpass_parameter,
            lowpass_parameter=lowpass_parameter,
            window_size=0,
            interpolate_ratio=interpolate_ratio
        )
        
        if filtered_signal.size == 0 or time.size == 0:
            st.error(f"预处理后信号为空，请检查参数或输入数据。文件: {filename}")
            return None
        
        # Turning point detection
        turning_points = utils.find_turning_points(filtered_signal, prominence=prominence, distance=distance)
        
        # Calculate slopes
        slope_times, slopes = utils.calculate_slopes(time, filtered_signal, turning_points)
        
        # Pattern analysis
        patterns, filtered_patterns, direction, pattern_spv, cv = utils.identify_nystagmus_patterns(
            filtered_signal, time, 
            min_time=min_time, max_time=max_time, 
            min_ratio=min_ratio, max_ratio=max_ratio,
            direction_axis=direction_axis
        )
        
        return {
            'filename': filename,
            'timestamps': timestamps,
            'eye_angles': eye_angles,
            'filtered_signal': filtered_signal,
            'time': time,
            'turning_points': turning_points,
            'slope_times': slope_times,
            'slopes': slopes,
            'patterns': patterns,
            'filtered_patterns': filtered_patterns,
            'direction': direction,
            'pattern_spv': pattern_spv,
            'cv': cv,
            'analysis_axis': analysis_axis_option,
            'highpass_parameter': highpass_parameter,
            'lowpass_parameter': lowpass_parameter
        }
        
    except ValueError as e:
        st.error(f"File parsing error (timestamp format issue) for {filename}: {e}")
        return None
    except Exception as e:
        st.error(f"Unknown error during file analysis for {filename}: {e}")
        return None

def process_hit_data(uploaded_files):
    """Process uploaded HIT data files"""
    if not uploaded_files:
        st.warning("Please upload HIT data files.")
        return None
    
    results = []
    
    for uploaded_file in uploaded_files:
        try:
            file_content = uploaded_file.read()
            file_format = detect_file_format(file_content)
            
            if file_format == 'new_hit_format':
                result = analyze_new_hit_format(file_content, uploaded_file.name)
            elif file_format == 'old_hit_format':
                result = analyze_old_hit_format(file_content, uploaded_file.name)
            else:
                st.warning(f"Unsupported file format for {uploaded_file.name}")
                continue
            
            if result:
                results.append(result)
                
        except Exception as e:
            st.error(f"Error processing HIT file {uploaded_file.name}: {str(e)}")
    
    return results

def analyze_new_hit_format(file_content, filename):
    """Analyze new unified HIT format"""
    try:
        # Parse the HIT data using the new utilities
        hit_data = hit_utils.parse_new_hit_format(file_content)
        
        # Set up analysis parameters
        highpass_params = {'cutoff': hit_highpass_cutoff, 'fs': 60, 'order': 5}
        lowpass_params = {'cutoff': hit_lowpass_cutoff, 'fs': 60, 'order': 5}
        
        # Perform complete HIT analysis
        analysis_results = hit_utils.analyze_hit_data(
            hit_data, eye_selection=eye_selection,
            highpass_params=highpass_params,
            lowpass_params=lowpass_params,
            window_size=hit_window_size,
            interpolate_ratio=hit_interpolate_ratio,
            peak_distance=peak_detection_distance,
            analysis_window_size=analysis_window_size
        )
        
        result = {
            'filename': filename,
            'format': 'new_unified',
            'raw_data': hit_data,
            'analysis_results': analysis_results,
            'eye_selection': eye_selection
        }
        
        return result
        
    except Exception as e:
        st.error(f"Error analyzing new HIT format {filename}: {str(e)}")
        return None

def analyze_old_hit_format(file_content, filename):
    """Analyze old separate HIT format"""
    try:
        # This would handle the old format where head/eye data are in separate files
        # Implementation depends on the specific old format structure
        pass
    except Exception as e:
        st.error(f"Error analyzing old HIT format {filename}: {str(e)}")
        return None

# Main analysis section
if uploaded_files:
    if analysis_type == "Nystagmus Analysis":
        st.subheader("👁️ Nystagmus Analysis Results")
        
        # Process files without spinner - direct processing
        results = process_nystagmus_data(uploaded_files)
        
        if results:
            # Display results for each file
            for i, result in enumerate(results):
                st.markdown(f"### 📊 Analysis: {result['filename']}")
                
                # Create comprehensive visualization matching original design
                fig = plt.figure(figsize=(18, 16))
                
                # ---------- 1. Signal Preprocessing Steps ----------
                plt.subplot(4, 1, 1)
                
                # Plot original data
                plt.plot(result['timestamps'], result['eye_angles'], 
                        label=f'Original Data ({axis_data_key})', alpha=0.7)
                
                # Apply and plot high-pass filter
                signal_highpass = utils.butter_highpass_filter(result['eye_angles'], **result['highpass_parameter'])
                plt.plot(result['timestamps'], signal_highpass, 
                        label='After High-pass Filter', alpha=0.7)
                
                # Apply and plot low-pass filter
                signal_lowpass = utils.butter_lowpass_filter(signal_highpass, **result['lowpass_parameter'])
                plt.plot(result['timestamps'], signal_lowpass, 
                        label='After Low-pass Filter', alpha=0.9, linewidth=2)
                
                plt.title(f'1. Signal Preprocessing Steps ({result["analysis_axis"]})')
                plt.ylabel('Position (°)')
                plt.grid(True)
                plt.legend()
                
                # ---------- 2. Turning Point Detection ----------
                plt.subplot(4, 1, 2)
                plt.plot(result['time'], result['filtered_signal'], 
                        'gray', alpha=0.3, label='Original Signal')
                plt.plot(result['time'][result['turning_points']], 
                        result['filtered_signal'][result['turning_points']], 
                        'r-', label='Turning Points Connection', linewidth=2)
                plt.plot(result['time'][result['turning_points']], 
                        result['filtered_signal'][result['turning_points']], 
                        'ro', markersize=5, label='Turning Points')
                
                plt.title(f'2. Eye Movement Signal with Turning Points ({result["analysis_axis"]})')
                plt.ylabel('Position (°)')
                plt.legend()
                plt.grid(True)
                
                # ---------- 3. Slope Calculation ----------
                plt.subplot(4, 1, 3)
                plt.scatter(result['slope_times'], result['slopes'], c='blue', s=30, alpha=0.7)
                plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
                plt.title(f'3. Calculated Slopes ({result["analysis_axis"]})')
                plt.ylabel('Slope (°/s)')
                plt.grid(True)
                plt.ylim([-40, 40])
                
                # ---------- 4. Nystagmus Pattern Recognition ----------
                plt.subplot(4, 1, 4)
                plt.plot(result['time'], result['filtered_signal'], 
                        'gray', alpha=0.5, label='Signal')
                
                # Plot filtered patterns (light colors)
                for pattern_item in result['filtered_patterns']:
                    idx = pattern_item['index']
                    if idx > 0 and idx + 1 < len(result['turning_points']):
                        idx1 = result['turning_points'][idx-1]
                        idx2 = result['turning_points'][idx]
                        idx3 = result['turning_points'][idx+1]
                        
                        fast_segment = result['time'][idx1:idx2+1] if pattern_item['fast_phase_first'] else result['time'][idx2:idx3+1]
                        slow_segment = result['time'][idx2:idx3+1] if pattern_item['fast_phase_first'] else result['time'][idx1:idx2+1]
                        fast_signal = result['filtered_signal'][idx1:idx2+1] if pattern_item['fast_phase_first'] else result['filtered_signal'][idx2:idx3+1]
                        slow_signal = result['filtered_signal'][idx2:idx3+1] if pattern_item['fast_phase_first'] else result['filtered_signal'][idx1:idx2+1]
                        
                        plt.plot(fast_segment, fast_signal, 'lightcoral', linewidth=2, alpha=0.5)
                        plt.plot(slow_segment, slow_signal, 'lightblue', linewidth=2, alpha=0.5)
                
                # Plot final patterns (bright colors)
                for pattern_item in result['patterns']:
                    idx = pattern_item['index']
                    if idx > 0 and idx + 1 < len(result['turning_points']):
                        idx1 = result['turning_points'][idx-1]
                        idx2 = result['turning_points'][idx]
                        idx3 = result['turning_points'][idx+1]
                        
                        fast_segment = result['time'][idx1:idx2+1] if pattern_item['fast_phase_first'] else result['time'][idx2:idx3+1]
                        slow_segment = result['time'][idx2:idx3+1] if pattern_item['fast_phase_first'] else result['time'][idx1:idx2+1]
                        fast_signal = result['filtered_signal'][idx1:idx2+1] if pattern_item['fast_phase_first'] else result['filtered_signal'][idx2:idx3+1]
                        slow_signal = result['filtered_signal'][idx2:idx3+1] if pattern_item['fast_phase_first'] else result['filtered_signal'][idx1:idx2+1]
                        
                        plt.plot(fast_segment, fast_signal, 'red', linewidth=2, alpha=0.8)
                        plt.plot(slow_segment, slow_signal, 'blue', linewidth=2, alpha=0.8)
                
                # Determine direction label in English
                english_direction_label = str(result['direction']).capitalize()
                if direction_axis == "horizontal":
                    if result['direction'] == "left": 
                        english_direction_label = "Left"
                    elif result['direction'] == "right": 
                        english_direction_label = "Right"
                elif direction_axis == "vertical":
                    if result['direction'] == "left": 
                        english_direction_label = "Up"
                    elif result['direction'] == "right": 
                        english_direction_label = "Down"
                
                plt.title(f'4. Pattern-Based Nystagmus Analysis ({result["analysis_axis"]} - Direction: {english_direction_label}, SPV: {result["pattern_spv"]:.1f}°/s, CV: {result["cv"]:.1f}%)')
                plt.xlabel('Time (s)')
                plt.ylabel('Position (°)')
                
                # Create legend items
                legend_items = ['Signal']
                if any(result['filtered_patterns']): 
                    legend_items.extend(['Filtered Fast Phase', 'Filtered Slow Phase'])
                if any(result['patterns']): 
                    legend_items.extend(['Fast Phase (Red)', 'Slow Phase (Blue)'])
                plt.legend(legend_items)
                plt.grid(True)
                
                plt.subplots_adjust(hspace=0.5, left=0.05, right=0.95, top=0.95, bottom=0.05)
                st.pyplot(fig)
                
                # Display metrics matching original design
                st.subheader(f"Nystagmus Analysis Results ({result['analysis_axis']})")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Detected Nystagmus Patterns", f"{len(result['patterns'])}")
                    st.metric("Nystagmus Direction", f"{english_direction_label}")
                with col2:
                    st.metric("Slow Phase Velocity (SPV)", f"{result['pattern_spv']:.2f}°/s")
                    st.metric("Coefficient of Variation (CV)", f"{result['cv']:.2f}%")
                
                # Additional metrics if patterns exist
                if len(result['patterns']) > 0:
                    avg_ratio_values = [p['ratio'] for p in result['patterns'] if 'ratio' in p and p['ratio'] is not None]
                    avg_ratio = np.mean(avg_ratio_values) if avg_ratio_values else 0.0
                    fast_first_count = sum(1 for p_item in result['patterns'] if p_item.get('fast_phase_first', False))
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Average Fast/Slow Phase Ratio", f"{avg_ratio:.2f}")
                    with col2:
                        st.metric("Patterns with Fast Phase First", f"{fast_first_count} ({fast_first_count/len(result['patterns'])*100:.1f}%)")
    
    elif analysis_type == "HIT Analysis":
        st.subheader("🎯 Head Impulse Test Analysis")
        
        # Process files without spinner - direct processing
        results = process_hit_data(uploaded_files)
        
        if results:
            st.success(f"Successfully processed {len(results)} HIT files")
            
            for result in results:
                st.markdown(f"### 📈 HIT Analysis: {result['filename']}")
                
                # Display basic file information
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Eye Selection", result['eye_selection'])
                with col2:
                    total_peaks = 0
                    if 'analysis_results' in result:
                        for eye_result in result['analysis_results'].values():
                            if isinstance(eye_result, dict):
                                for direction_result in eye_result.values():
                                    if isinstance(direction_result, dict) and 'num_peaks' in direction_result:
                                        total_peaks += direction_result['num_peaks']
                    st.metric("Total Detected Peaks", total_peaks)
                
                # Display analysis results for each eye and direction
                if 'analysis_results' in result and result['analysis_results']:
                    for eye_key, eye_data in result['analysis_results'].items():
                        if 'error' in eye_data:
                            st.error(f"Error analyzing {eye_key}: {eye_data['error']}")
                            continue
                        
                        # Create one combined overview figure for this eye (both directions)
                        st.markdown(f"#### 👁️ {eye_key.replace('_', ' ').title()} Analysis Overview")
                        
                        # Collect data from both directions
                        leftward_data = eye_data.get('leftward', {})
                        rightward_data = eye_data.get('rightward', {})
                        
                        # Only create overview if we have data from at least one direction
                        if leftward_data or rightward_data:
                            fig_overview = plt.figure(figsize=(20, 12))
                            
                            # Plot 1: Combined position signals
                            plt.subplot(2, 3, 1)
                            if leftward_data:
                                plt.plot(leftward_data['time_data'], leftward_data['head_data'], 
                                        label='Head Position', color='blue', alpha=0.8)
                                plt.plot(leftward_data['time_data'], leftward_data['eye_data'], 
                                        label=f'{eye_key.replace("_", " ").title()} Position', color='orange', alpha=0.8)
                            plt.title(f'Position Signals - {eye_key.replace("_", " ").title()}')
                            plt.xlabel('Time (s)')
                            plt.ylabel('Position (°)')
                            plt.legend()
                            plt.grid(True, alpha=0.3)
                            
                            # Plot 2: Combined velocity signals with peaks
                            plt.subplot(2, 3, 2)
                            if leftward_data:
                                plt.plot(leftward_data['time_velocity'], leftward_data['head_velocity'], 
                                        label='Head Velocity', color='blue', alpha=0.8)
                                plt.plot(leftward_data['time_velocity'], leftward_data['eye_velocity'], 
                                        label=f'{eye_key.replace("_", " ").title()} Velocity', color='orange', alpha=0.8)
                                
                                # Mark leftward peaks in red
                                if 'peaks' in leftward_data and len(leftward_data['peaks']) > 0:
                                    peaks = leftward_data['peaks']
                                    peak_times = leftward_data['time_velocity'][peaks]
                                    peak_velocities = leftward_data['head_velocity'][peaks]
                                    plt.scatter(peak_times, peak_velocities, 
                                              color='red', s=50, zorder=5, label='Leftward Peaks')
                            
                            if rightward_data:
                                # Mark rightward peaks in dark red
                                if 'peaks' in rightward_data and len(rightward_data['peaks']) > 0:
                                    peaks = rightward_data['peaks']
                                    peak_times = rightward_data['time_velocity'][peaks]
                                    peak_velocities = rightward_data['head_velocity'][peaks]
                                    plt.scatter(peak_times, peak_velocities, 
                                              color='darkred', s=50, zorder=5, label='Rightward Peaks')
                            
                            plt.title(f'Velocity Signals with Detected Peaks - {eye_key.replace("_", " ").title()}')
                            plt.xlabel('Time (s)')
                            plt.ylabel('Velocity (°/s)')
                            plt.legend()
                            plt.grid(True, alpha=0.3)
                            
                            # Plot 3: Combined VOR Gain distribution
                            plt.subplot(2, 3, 3)
                            all_gains = []
                            direction_labels = []
                            
                            if leftward_data.get('vor_gains'):
                                left_gains = [g for g in leftward_data['vor_gains'] if not np.isnan(g)]
                                all_gains.extend(left_gains)
                                direction_labels.extend(['Leftward'] * len(left_gains))
                            
                            if rightward_data.get('vor_gains'):
                                right_gains = [g for g in rightward_data['vor_gains'] if not np.isnan(g)]
                                all_gains.extend(right_gains)
                                direction_labels.extend(['Rightward'] * len(right_gains))
                            
                            if all_gains:
                                # Create histogram with different colors for directions
                                left_gains = [g for g, l in zip(all_gains, direction_labels) if l == 'Leftward']
                                right_gains = [g for g, l in zip(all_gains, direction_labels) if l == 'Rightward']
                                
                                if left_gains:
                                    plt.hist(left_gains, bins=max(3, len(left_gains)//2), 
                                           alpha=0.7, color='lightcoral', edgecolor='red', label='Leftward')
                                if right_gains:
                                    plt.hist(right_gains, bins=max(3, len(right_gains)//2), 
                                           alpha=0.7, color='lightblue', edgecolor='darkred', label='Rightward')
                                
                                # Add median lines
                                if left_gains:
                                    plt.axvline(np.median(left_gains), color='red', 
                                              linestyle='--', linewidth=2, alpha=0.8)
                                if right_gains:
                                    plt.axvline(np.median(right_gains), color='darkred', 
                                              linestyle='--', linewidth=2, alpha=0.8)
                                
                                plt.title(f'VOR Gain Distribution - {eye_key.replace("_", " ").title()}')
                                plt.xlabel('VOR Gain')
                                plt.ylabel('Frequency')
                                plt.legend()
                                plt.grid(True, alpha=0.3)
                            
                            # Plot 4: Combined gain over time
                            plt.subplot(2, 3, 4)
                            if leftward_data.get('peak_analysis'):
                                left_times = []
                                left_gains = []
                                for peak_info in leftward_data['peak_analysis']:
                                    peak_idx = peak_info['peak_index']
                                    if peak_idx < len(leftward_data['time_velocity']):
                                        left_times.append(leftward_data['time_velocity'][peak_idx])
                                        left_gains.append(peak_info['gain'])
                                
                                if left_times:
                                    plt.scatter(left_times, left_gains, 
                                              color='red', s=60, alpha=0.8, label='Leftward')
                            
                            if rightward_data.get('peak_analysis'):
                                right_times = []
                                right_gains = []
                                for peak_info in rightward_data['peak_analysis']:
                                    peak_idx = peak_info['peak_index']
                                    if peak_idx < len(rightward_data['time_velocity']):
                                        right_times.append(rightward_data['time_velocity'][peak_idx])
                                        right_gains.append(peak_info['gain'])
                                
                                if right_times:
                                    plt.scatter(right_times, right_gains, 
                                              color='darkred', s=60, alpha=0.8, label='Rightward')
                            
                            # Add normal range reference
                            if leftward_data.get('time_velocity') is not None or rightward_data.get('time_velocity') is not None:
                                time_data = leftward_data.get('time_velocity', rightward_data.get('time_velocity', []))
                                if len(time_data) > 0:
                                    plt.axhline(0.8, color='green', linestyle=':', alpha=0.5, label='Normal Range')
                                    plt.axhline(1.2, color='green', linestyle=':', alpha=0.5)
                                    plt.fill_between([time_data[0], time_data[-1]], 0.8, 1.2, alpha=0.1, color='green')
                            
                            plt.title(f'VOR Gain Over Time - {eye_key.replace("_", " ").title()}')
                            plt.xlabel('Time (s)')
                            plt.ylabel('VOR Gain')
                            plt.legend()
                            plt.grid(True, alpha=0.3)
                            
                            # Plot 5: Summary metrics table
                            plt.subplot(2, 3, 5)
                            plt.axis('off')
                            
                            summary_text = f"Combined Analysis for {eye_key.replace('_', ' ').title()}:\n\n"
                            
                            if leftward_data:
                                summary_text += f"Leftward Direction:\n"
                                summary_text += f"• Peaks: {leftward_data.get('num_peaks', 0)}\n"
                                summary_text += f"• Valid Gains: {leftward_data.get('num_valid_gains', 0)}\n"
                                summary_text += f"• Median Gain: {leftward_data.get('median_gain', np.nan):.3f}\n"
                                summary_text += f"• Std Dev: {leftward_data.get('std_gain', np.nan):.3f}\n\n"
                            
                            if rightward_data:
                                summary_text += f"Rightward Direction:\n"
                                summary_text += f"• Peaks: {rightward_data.get('num_peaks', 0)}\n"
                                summary_text += f"• Valid Gains: {rightward_data.get('num_valid_gains', 0)}\n"
                                summary_text += f"• Median Gain: {rightward_data.get('median_gain', np.nan):.3f}\n"
                                summary_text += f"• Std Dev: {rightward_data.get('std_gain', np.nan):.3f}\n\n"
                            
                            summary_text += "Clinical Interpretation:\n"
                            summary_text += "• Normal VOR Gain: 0.8 - 1.2\n"
                            
                            # Determine overall status
                            all_medians = []
                            if leftward_data.get('median_gain') and not np.isnan(leftward_data['median_gain']):
                                all_medians.append(leftward_data['median_gain'])
                            if rightward_data.get('median_gain') and not np.isnan(rightward_data['median_gain']):
                                all_medians.append(rightward_data['median_gain'])
                            
                            if all_medians:
                                overall_normal = all(0.8 <= gain <= 1.2 for gain in all_medians)
                                summary_text += f"• Overall Status: {'Normal' if overall_normal else 'Abnormal'}"
                            
                            plt.text(0.05, 0.95, summary_text, fontsize=10, 
                                   verticalalignment='top', fontfamily='monospace',
                                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
                            
                            # Plot 6: Space for additional info or leave empty
                            plt.subplot(2, 3, 6)
                            plt.axis('off')
                            plt.text(0.5, 0.5, 'Combined HIT Analysis\nBoth Directions', 
                                   ha='center', va='center', fontsize=16, 
                                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
                            
                            plt.tight_layout()
                            st.pyplot(fig_overview)
                        
                        # Now display individual direction analyses
                        for direction_key, direction_result in eye_data.items():
                            if 'error' in direction_result:
                                st.error(f"Error analyzing {direction_result.get('eye_name', direction_key)}: {direction_result['error']}")
                                continue
                            
                            st.markdown(f"##### 👁️ {direction_result['eye_name']} Detailed Analysis")
                            
                            # Display metrics for this eye direction
                            metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
                            with metrics_col1:
                                st.metric("Detected Peaks", direction_result.get('num_peaks', 0))
                            with metrics_col2:
                                st.metric("Valid Gains", direction_result.get('num_valid_gains', 0))
                            with metrics_col3:
                                median_gain = direction_result.get('median_gain', np.nan)
                                st.metric("Median VOR Gain", f"{median_gain:.3f}" if not np.isnan(median_gain) else "N/A")
                            with metrics_col4:
                                std_gain = direction_result.get('std_gain', np.nan)
                                st.metric("Gain Std Dev", f"{std_gain:.3f}" if not np.isnan(std_gain) else "N/A")
                            
                            # Add interactive peak selector for this direction
                            if direction_result.get('peak_analysis'):
                                st.markdown("##### 🔍 Interactive Peak Analysis")
                                
                                # Create three-figure layout similar to main.py
                                fig_interactive = plt.figure(figsize=(20, 8))
                                
                                # Calculate overall statistics for all peaks
                                all_gains = [peak['gain'] for peak in direction_result['peak_analysis'] if not np.isnan(peak['gain'])]
                                median_gain = np.median(all_gains) if all_gains else np.nan
                                std_gain = np.std(all_gains) if all_gains else np.nan
                                
                                # Peak selector
                                peak_index = st.selectbox(
                                    f"Select Peak for {direction_result['eye_name']} Analysis",
                                    range(len(direction_result['peak_analysis'])),
                                    format_func=lambda x: f"Peak {x+1} (Gain: {direction_result['peak_analysis'][x]['gain']:.3f})",
                                    key=f"peak_selector_{eye_key}_{direction_key}"  # Unique key for each selector
                                )
                                
                                selected_peak = direction_result['peak_analysis'][peak_index]
                                
                                # Show only VOR Gain metric
                                st.metric("Peak VOR Gain", f"{selected_peak['gain']:.3f}")
                                
                                # Figure 1: Velocity signals for selected peak
                                plt.subplot(1, 3, 1)
                                start_idx = selected_peak['window_start']
                                end_idx = selected_peak['window_end']
                                
                                time_segment = direction_result['time_velocity'][start_idx:end_idx]
                                head_segment = direction_result['head_velocity'][start_idx:end_idx]
                                eye_segment = direction_result['eye_velocity'][start_idx:end_idx]
                                
                                plt.plot(time_segment, head_segment, 'b-', linewidth=2, label='Head Velocity')
                                plt.plot(time_segment, eye_segment, 'r-', linewidth=2, label='Eye Velocity')
                                
                                # Add VOR area visualization if available
                                analysis_info = selected_peak.get('analysis_info', {})
                                if 'head_indices' in analysis_info and 'eye_indices' in analysis_info:
                                    head_start, head_end = analysis_info['head_indices']
                                    eye_start, eye_end = analysis_info['eye_indices']
                                    
                                    # Adjust indices to the segment
                                    segment_head_start = max(0, head_start)
                                    segment_head_end = min(len(head_segment), head_end)
                                    segment_eye_start = max(0, eye_start)
                                    segment_eye_end = min(len(eye_segment), eye_end)
                                    
                                    plt.fill_between(
                                        time_segment[segment_head_start:segment_head_end],
                                        head_segment[segment_head_start:segment_head_end],
                                        alpha=0.3, color='blue', label='Head AUC'
                                    )
                                    plt.fill_between(
                                        time_segment[segment_eye_start:segment_eye_end],
                                        eye_segment[segment_eye_start:segment_eye_end],
                                        alpha=0.3, color='red', label='Eye AUC'
                                    )
                                
                                plt.title(f'Peak {peak_index + 1} Velocity Signals (VOR Gain: {selected_peak["gain"]:.3f})')
                                plt.xlabel('Time (s)')
                                plt.ylabel('Velocity (°/s)')
                                plt.legend()
                                plt.grid(True, alpha=0.3)
                                
                                # Figure 2: Position signals for selected peak
                                plt.subplot(1, 3, 2)
                                pos_start_idx = start_idx
                                pos_end_idx = end_idx
                                
                                pos_time_segment = direction_result['time_data'][pos_start_idx:pos_end_idx]
                                pos_head_segment = direction_result['head_data'][pos_start_idx:pos_end_idx]
                                pos_eye_segment = direction_result['eye_data'][pos_start_idx:pos_end_idx]
                                
                                plt.plot(pos_time_segment, pos_head_segment, 'b-', linewidth=2, label='Head Position')
                                plt.plot(pos_time_segment, pos_eye_segment, 'r-', linewidth=2, label='Eye Position')
                                
                                plt.title('Corresponding Position Signals')
                                plt.xlabel('Time (s)')
                                plt.ylabel('Position (°)')
                                plt.legend()
                                plt.grid(True, alpha=0.3)
                                
                                # Figure 3: Overall analysis - Aligned peaks average
                                plt.subplot(1, 3, 3)
                                
                                # Align all peaks for average calculation
                                window_size = 240  # Similar to main.py
                                half_window = window_size // 2
                                sampling_rate = 60  # Assume 60Hz
                                
                                aligned_head_velocities = []
                                aligned_eye_velocities = []
                                aligned_times = np.arange(-half_window, half_window) / sampling_rate
                                
                                # Collect data from all peaks
                                for peak_info in direction_result['peak_analysis']:
                                    peak_idx = peak_info['peak_index']
                                    
                                    # Calculate window around this peak
                                    start_align = max(0, peak_idx - half_window)
                                    end_align = min(len(direction_result['head_velocity']), peak_idx + half_window)
                                    
                                    # Skip if window is incomplete
                                    if end_align - start_align != window_size:
                                        continue
                                    
                                    aligned_head_velocities.append(direction_result['head_velocity'][start_align:end_align])
                                    aligned_eye_velocities.append(direction_result['eye_velocity'][start_align:end_align])
                                
                                if aligned_head_velocities and aligned_eye_velocities:
                                    # Convert to numpy arrays and calculate statistics
                                    aligned_head_velocities = np.array(aligned_head_velocities)
                                    aligned_eye_velocities = np.array(aligned_eye_velocities)
                                    
                                    mean_head = np.mean(aligned_head_velocities, axis=0)
                                    mean_eye = np.mean(aligned_eye_velocities, axis=0)
                                    std_head = np.std(aligned_head_velocities, axis=0)
                                    std_eye = np.std(aligned_eye_velocities, axis=0)
                                    
                                    # Plot mean with standard deviation bands
                                    plt.plot(aligned_times, mean_head, 
                                           label='Mean Head Velocity', 
                                           color='blue', linewidth=2)
                                    plt.fill_between(aligned_times, 
                                                   mean_head - std_head, 
                                                   mean_head + std_head, 
                                                   color='blue', alpha=0.2)
                                    
                                    plt.plot(aligned_times, mean_eye, 
                                           label='Mean Eye Velocity', 
                                           color='orange', linewidth=2)
                                    plt.fill_between(aligned_times, 
                                                   mean_eye - std_eye, 
                                                   mean_eye + std_eye, 
                                                   color='orange', alpha=0.2)
                                
                                plt.title(f'Aligned Peaks Average (±1 SD) - {direction_result["direction_name"]}')
                                plt.xlabel('Time (s)')
                                plt.ylabel('Velocity (°/s)')
                                plt.legend()
                                plt.grid(True, alpha=0.3)
                                
                                plt.tight_layout()
                                st.pyplot(fig_interactive)
                                
                                # Display overall metrics
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Current Peak Gain", f"{selected_peak['gain']:.3f}")
                                with col2:
                                    st.metric("Median Gain (All Peaks)", f"{median_gain:.3f}" if not np.isnan(median_gain) else "N/A")
                                with col3:
                                    st.metric("Gain Std Dev", f"{std_gain:.3f}" if not np.isnan(std_gain) else "N/A")
                
                st.markdown("---")

else:
    # Welcome screen
    st.markdown("## 🚀 Welcome to pVestibular Analysis Platform")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 👁️ Nystagmus Analysis")
        st.markdown("""
        - **Pattern Recognition**: Automated detection of nystagmus patterns
        - **Fast/Slow Phase Analysis**: Comprehensive phase characterization
        - **Multi-axis Support**: Horizontal and vertical eye movements
        - **Advanced Filtering**: Customizable signal preprocessing
        - **Metrics Calculation**: SPV, CV, and pattern statistics
        """)
    
    with col2:
        st.markdown("### 🎯 HIT Analysis")
        st.markdown("""
        - **VOR Gain Calculation**: Precise vestibulo-ocular reflex analysis
        - **Multi-format Support**: Both old and new data formats
        - **Peak Detection**: Automated head impulse identification
        - **Bilateral Analysis**: Left and right eye comparison
        - **Interactive Visualization**: Peak-by-peak analysis
        """)
    
    st.markdown("---")
    st.markdown("### 📋 Getting Started")
    st.markdown("""
    1. **Select Analysis Type**: Choose between Nystagmus or HIT analysis in the sidebar
    2. **Upload Data**: Upload your .plist files or ZIP archives
    3. **Configure Parameters**: Adjust analysis parameters as needed
    4. **Review Results**: Examine the comprehensive analysis results and visualizations
    """)

# Footer
st.markdown("---")
st.markdown(
    "**pVestibular Platform** | Advanced Vestibular Function Analysis | "
    "Built with Streamlit & Python"
)

# Add application info to sidebar - matching original design
st.sidebar.markdown("---")
st.sidebar.subheader("About this App")

if analysis_type == "Nystagmus Analysis":
    st.sidebar.info(
        "This Nystagmus Analyzer allows you to upload .plist files and analyze nystagmus data by adjusting parameters. "
        "You can select horizontal or vertical eye movement data for analysis. "
        "Results will update automatically. Supports batch processing and ZIP archives."
    )
elif analysis_type == "HIT Analysis":
    st.sidebar.info(
        "This HIT Analyzer processes Head Impulse Test data from .plist files. "
        "Upload your HIT data files (new unified format with HeadXList, LeftEyeXList, TimeList) "
        "to perform VOR gain analysis and peak detection. Supports bilateral eye analysis."
    ) 