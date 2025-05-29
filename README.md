# 🧠 pVestibular Analysis Platform

A comprehensive web-based platform for vestibular function analysis, combining **Nystagmus** and **Head Impulse Test (HIT)** analysis capabilities.

## ✨ Features

### 👁️ Nystagmus Analysis
- **Pattern Recognition**: Automated detection and classification of nystagmus patterns
- **Fast/Slow Phase Analysis**: Comprehensive characterization of nystagmus phases
- **Multi-axis Support**: Analysis of both horizontal and vertical eye movements
- **Advanced Signal Processing**: Customizable high-pass, low-pass, and interpolation filters
- **Clinical Metrics**: Slow Phase Velocity (SPV), Coefficient of Variation (CV), and pattern statistics

### 🎯 Head Impulse Test (HIT) Analysis
- **VOR Gain Calculation**: Precise vestibulo-ocular reflex gain analysis
- **Multi-format Support**: Compatible with both old and new data formats
- **Peak Detection**: Automated head impulse identification and analysis
- **Bilateral Analysis**: Simultaneous left and right eye analysis
- **Interactive Visualization**: Peak-by-peak analysis with detailed metrics
- **Clinical Interpretation**: Normal range references and status assessment

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- Web browser (Chrome, Firefox, Safari, Edge)

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd pVestibular
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   streamlit run pvestibular.py
   ```

4. **Open your browser** and navigate to `http://localhost:8501`

## 📊 Data Formats

### Nystagmus Data
- **Format**: .plist files
- **Required Keys**: `TimeList`, `LeftEyeXDegList` (horizontal) or `LeftEyeYDegList` (vertical)
- **Multiple Files**: Supports batch processing and ZIP archives

### HIT Data

#### New Unified Format (.plist)
```
HeadXList: [float] - Head X-axis movement
HeadYList: [float] - Head Y-axis movement (optional)
HeadZList: [float] - Head Z-axis movement (optional)
LeftEyeXList: [float] - Left eye X-axis movement
LeftEyeYList: [float] - Left eye Y-axis movement (optional)
RightEyeXList: [float] - Right eye X-axis movement
RightEyeYList: [float] - Right eye Y-axis movement (optional)
TimeList: [string] - Timestamps in 'YYYY-MM-DD HH:MM:SS.ffffff' format
```

#### Legacy Format
- Separate files for head and eye data
- Simple array format in .plist files

## 🛠️ Usage Guide

### 1. Select Analysis Type
Choose between **Nystagmus Analysis** or **HIT Analysis** from the sidebar.

### 2. Upload Data Files
- **Single Files**: Upload individual .plist files
- **Multiple Files**: Select multiple files for batch processing
- **ZIP Archives**: Upload ZIP files containing multiple .plist files (nystagmus only)

### 3. Configure Parameters
Adjust analysis parameters in the sidebar:

#### Nystagmus Parameters
- **Analysis Axis**: Horizontal (X) or Vertical (Y)
- **Preprocessing**: High-pass/low-pass filter settings, interpolation ratio
- **Detection**: Peak prominence and distance parameters
- **Pattern Recognition**: Time and ratio thresholds

#### HIT Parameters
- **Eye Selection**: Left Eye, Right Eye, or Both Eyes
- **Preprocessing**: Filter settings and moving average window
- **VOR Analysis**: Peak detection and analysis window parameters

### 4. Review Results
Examine comprehensive analysis results including:
- Interactive visualizations
- Statistical metrics
- Clinical interpretations
- Downloadable reports

## 📈 Analysis Outputs

### Nystagmus Analysis Results
- **Signal Preprocessing Visualization**: Original, high-pass, and low-pass filtered signals
- **Turning Point Detection**: Identified peaks and valleys in eye movement
- **Pattern Classification**: Fast and slow phase identification
- **Clinical Metrics**:
  - Detected pattern count
  - Nystagmus direction
  - Slow Phase Velocity (SPV)
  - Coefficient of Variation (CV)

### HIT Analysis Results
- **Comprehensive Visualizations**:
  - Position and velocity signals
  - Detected head impulse peaks
  - VOR gain distribution
  - Individual peak analysis
  - Gain over time progression
- **Statistical Analysis**:
  - Median and mean VOR gains
  - Standard deviation
  - Peak detection statistics
- **Clinical Assessment**:
  - Normal range comparison (0.8-1.2)
  - Status classification
  - Interactive peak-by-peak analysis

## 🔧 Advanced Features

### Interactive Analysis
- **Peak Selection**: Click-to-analyze individual peaks in HIT data
- **Parameter Adjustment**: Real-time parameter tuning
- **Zoom and Pan**: Detailed signal examination

### Batch Processing
- **Multiple File Analysis**: Process several files simultaneously
- **Comparative Results**: Side-by-side analysis comparison
- **Export Capabilities**: Download results and visualizations

### Signal Processing
- **Butterworth Filtering**: Zero-phase high-pass and low-pass filters
- **Moving Average**: Configurable window smoothing
- **Interpolation**: Signal upsampling for improved analysis
- **Velocity Calculation**: Precise differentiation with configurable sampling rates

## 🏥 Clinical Applications

### Nystagmus Analysis
- **Vestibular Disorder Assessment**: Identify pathological nystagmus patterns
- **Treatment Monitoring**: Track changes in nystagmus characteristics
- **Research Applications**: Quantitative nystagmus analysis for studies

### HIT Analysis
- **Vestibular Function Testing**: Assess semicircular canal function
- **VOR Gain Measurement**: Quantify vestibulo-ocular reflex performance
- **Unilateral Weakness Detection**: Identify asymmetric vestibular function
- **Rehabilitation Monitoring**: Track recovery progress

## 🔬 Technical Details

### Signal Processing Pipeline
1. **Data Loading**: Robust .plist file parsing
2. **Preprocessing**: Configurable filtering and smoothing
3. **Feature Detection**: Peak and turning point identification
4. **Pattern Analysis**: Automated classification algorithms
5. **Metric Calculation**: Clinical parameter computation
6. **Visualization**: Interactive plot generation

### Performance Optimization
- **Efficient Algorithms**: Optimized signal processing routines
- **Memory Management**: Streaming data processing for large files
- **Parallel Processing**: Multi-threaded analysis where applicable

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines for:
- Code style requirements
- Testing procedures
- Documentation standards
- Issue reporting

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Support

For technical support or questions:
- Create an issue on GitHub
- Contact the development team
- Check the documentation wiki

## 🔄 Updates

- **v1.0.0**: Initial release with basic nystagmus and HIT analysis
- **v1.1.0**: Added new unified HIT format support
- **v1.2.0**: Enhanced interactive visualizations and batch processing

---

**pVestibular Platform** - Advancing vestibular function analysis through modern technology 