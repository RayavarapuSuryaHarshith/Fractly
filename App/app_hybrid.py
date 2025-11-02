import streamlit as st  
import sys
import os
from PIL import Image
import numpy as np
import torch

# Add the src directory to Python path to import our hybrid model
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from final_hybrid_model import FinalHybridFractureDetector

# Hide deprecation warnings
import warnings
warnings.filterwarnings("ignore")

# Configure Streamlit page
st.set_page_config(
    page_title="Bone Fracture Detection - Hybrid AI System",
    page_icon="🩻",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize hybrid model once and cache it
@st.cache_resource
def load_hybrid_model():
    """Load the hybrid model once and cache it for efficiency"""
    try:
        # Get the project root directory (parent of App directory)
        app_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(app_dir)
        
        # Use absolute paths
        yolo_weights = os.path.join(project_root, 'weights', 'yolo11n_fracture_high_accuracy.pt')
        fracnet_weights = os.path.join(project_root, 'weights', 'custom_fracnet2d_best.pth')
        
        # Debug: Check if files exist
        if not os.path.exists(yolo_weights):
            st.error(f"YOLO weights not found at: {yolo_weights}")
            # Try alternative weights
            alt_yolo_weights = os.path.join(project_root, 'weights', 'yolo11n_fracture_trained.pt')
            if os.path.exists(alt_yolo_weights):
                yolo_weights = alt_yolo_weights
                st.warning("Using alternative YOLO weights: yolo11n_fracture_trained.pt")
            else:
                st.error("No suitable YOLO weights found")
                return None
        
        if not os.path.exists(fracnet_weights):
            st.error(f"FracNet weights not found at: {fracnet_weights}")
            return None
        
        st.info(f"Loading YOLO weights from: {os.path.basename(yolo_weights)}")
        st.info(f"Loading FracNet weights from: {os.path.basename(fracnet_weights)}")
        
        hybrid_model = FinalHybridFractureDetector(
            yolo_weights_path=yolo_weights,
            fracnet_weights_path=fracnet_weights,
            device="cpu",
            verbose=False
        )
        return hybrid_model
    except Exception as e:
        st.error(f"Error loading hybrid model: {str(e)}")
        st.error(f"App directory: {os.path.dirname(os.path.abspath(__file__))}")
        st.error(f"Project root: {os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}")
        return None

# Sidebar configuration
def setup_sidebar():
    """Setup sidebar with model information and settings"""
    with st.sidebar:
        st.image("https://via.placeholder.com/300x150/1f77b4/ffffff?text=Medical+AI", 
                caption="Bone Fracture Detection System")
        
        # Debug information
        with st.expander("🔧 System Debug Info"):
            app_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(app_dir)
            yolo_path = os.path.join(project_root, 'weights', 'yolo11n_fracture_high_accuracy.pt')
            fracnet_path = os.path.join(project_root, 'weights', 'custom_fracnet2d_best.pth')
            
            st.code(f"App Dir: {app_dir}")
            st.code(f"Project Root: {project_root}")
            st.code(f"YOLO Path: {yolo_path}")
            st.code(f"YOLO Exists: {os.path.exists(yolo_path)}")
            st.code(f"FracNet Path: {fracnet_path}")
            st.code(f"FracNet Exists: {os.path.exists(fracnet_path)}")
        
        st.markdown("## 🎯 Model Information")
        st.success("**Hybrid AI System Active**")
        st.markdown("- **Accuracy**: 93.2%")
        st.markdown("- **Sensitivity**: 90.91%")
        st.markdown("- **Specificity**: 95.45%")
        st.markdown("- **Models**: YOLO v11 + FracNet2D")
        
        st.markdown("---")
        st.markdown("## 📊 Performance Metrics")
        st.markdown("Our hybrid system combines:")
        st.markdown("- YOLO v11: 88.6% standalone")
        st.markdown("- FracNet2D: 56.6% standalone")
        st.markdown("- **Combined**: 93.2% accuracy")
        
        st.markdown("---")
        st.markdown("## ℹ️ Instructions")
        st.markdown("1. Upload an X-ray image")
        st.markdown("2. Click 'Analyze X-ray'")
        st.markdown("3. Review results and confidence")
        st.markdown("4. Check detailed analysis")

def main():
    """Main application interface"""
    
    # Setup sidebar
    setup_sidebar()
    
    # Main page header
    st.markdown("""
    <div style='text-align: center; padding: 20px;'>
        <h1 style='color: #1f77b4; font-family: Arial, sans-serif;'>
            🩻 BONE FRACTURE DETECTION SYSTEM
        </h1>
        <h3 style='color: #666; font-weight: normal;'>
            Professional AI-Powered Fracture Analysis - 93.2% Accuracy
        </h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-container {
        padding: 20px;
        border-radius: 10px;
        background-color: #f8f9fa;
        margin: 20px 0;
    }
    .result-success {
        padding: 15px;
        border-radius: 5px;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        margin: 10px 0;
    }
    .result-warning {
        padding: 15px;
        border-radius: 5px;
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        margin: 10px 0;
    }
    .confidence-high { color: #28a745; font-weight: bold; }
    .confidence-medium { color: #ffc107; font-weight: bold; }
    .confidence-low { color: #dc3545; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)
    
    # Main content tabs
    tab1, tab2 = st.tabs(["🔬 X-ray Analysis", "📋 System Overview"])
    
    with tab1:
        st.markdown("### Upload X-ray Image for Analysis")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an X-ray image file",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a clear X-ray image for fracture detection"
        )
        
        if uploaded_file is not None:
            # Display uploaded image and results side by side
            col1, col2 = st.columns([1, 1])
            
            with col1:
                # Display uploaded image
                image = Image.open(uploaded_file)
                st.image(image, caption=f"Uploaded: {uploaded_file.name}", width="stretch")
                
                # Image information
                st.markdown("**Image Details:**")
                st.markdown(f"- Size: {image.size}")
                st.markdown(f"- Format: {image.format}")
                st.markdown(f"- Mode: {image.mode}")
            
            with col2:
                # Analysis section
                st.markdown("### 🔍 AI Analysis")
                
                if st.button("🩻 Analyze X-ray", type="primary"):
                    with st.spinner("🤖 AI models analyzing X-ray image..."):
                        # Load hybrid model
                        hybrid_model = load_hybrid_model()
                        
                        if hybrid_model is not None:
                            # Save uploaded image temporarily with absolute path
                            app_dir = os.path.dirname(os.path.abspath(__file__))
                            temp_path = os.path.join(app_dir, "temp_xray_analysis.jpg")
                            image.save(temp_path)
                            
                            try:
                                # Get prediction from hybrid model
                                result = hybrid_model.predict(temp_path)
                                
                                # Display main result
                                if result["is_fracture"]:
                                    st.markdown("""
                                    <div class='result-warning'>
                                        <h3>⚠️ FRACTURE DETECTED</h3>
                                        <p>The AI system has detected signs of bone fracture in the X-ray image.</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                else:
                                    st.markdown("""
                                    <div class='result-success'>
                                        <h3>✅ NO FRACTURE DETECTED</h3>
                                        <p>The AI system found no clear signs of bone fracture in the X-ray image.</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                # Confidence display
                                confidence = result['confidence']
                                if confidence >= 0.8:
                                    conf_class = "confidence-high"
                                    conf_text = "High"
                                elif confidence >= 0.6:
                                    conf_class = "confidence-medium"
                                    conf_text = "Medium"
                                else:
                                    conf_class = "confidence-low"
                                    conf_text = "Low"
                                
                                st.markdown(f"""
                                <div style='text-align: center; margin: 20px 0;'>
                                    <h4>Confidence Level: <span class='{conf_class}'>{confidence:.1%}</span></h4>
                                    <p>Confidence Rating: <span class='{conf_class}'>{conf_text}</span></p>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Severity classification based on confidence level
                                if result["is_fracture"]:
                                    st.markdown("---")
                                    st.markdown("### 🔍 Severity Assessment")
                                    
                                    if confidence >= 0.90:
                                        severity_type = "Major Fracture"
                                        severity_icon = "🔴"
                                        severity_desc = "Very high confidence indicates a likely major fracture. Immediate medical attention is strongly recommended."
                                        severity_color = "#dc3545"
                                    elif confidence >= 0.70:
                                        severity_type = "Minor Fracture or Air Crack"
                                        severity_icon = "🟡"
                                        severity_desc = "Moderate confidence suggests a possible minor fracture or air crack. Medical evaluation is recommended."
                                        severity_color = "#ffc107"
                                    else:
                                        severity_type = "Small Cut, Tear, or Minor Anomaly"
                                        severity_icon = "🟠"
                                        severity_desc = "Lower confidence may indicate a small cut, tear, or minor surface anomaly. Consider professional evaluation for confirmation."
                                        severity_color = "#fd7e14"
                                    
                                    st.markdown(f"""
                                    <div style='padding: 15px; border-radius: 8px; background-color: {severity_color}20; border-left: 4px solid {severity_color}; margin: 10px 0;'>
                                        <h4 style='margin: 0; color: {severity_color};'>{severity_icon} {severity_type}</h4>
                                        <p style='margin: 10px 0 0 0; color: #333;'>{severity_desc}</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    
                                    # Additional guidance
                                    # st.info("""
                                    # **💡 Understanding the Assessment:**
                                    # - **Very High Confidence (≥90%)**: Likely a major fracture requiring immediate attention
                                    # - **Moderate Confidence (70-89%)**: Possible minor fracture or air crack
                                    # - **Lower Confidence (<70%)**: Could indicate small cut, tear, or minor surface anomaly
                                    
                                    # This classification is based purely on the AI model's confidence level and helps assess the severity and reliability of detection.
                                    # """)
                                
                                # Detailed analysis in expandable section
                                # with st.expander("📊 Detailed AI Analysis", expanded=True):
                                #     # Decision reasoning
                                #     decision_reason = result['decision_reason'].replace('_', ' ').title()
                                #     st.markdown(f"**Decision Logic:** {decision_reason}")
                                    
                                #     # Model agreement
                                #     agreement = "✅ Yes" if result['models_agree'] else "❌ No"
                                #     st.markdown(f"**Models Agreement:** {agreement}")
                                    
                                #     # Individual model results
                                #     col_yolo, col_fracnet = st.columns(2)
                                    
                                #     with col_yolo:
                                #         st.markdown("**🎯 YOLO v11 Analysis**")
                                #         yolo_pred = "Fracture" if result['yolo_result']['is_fracture'] else "No Fracture"
                                #         yolo_conf = result['yolo_result']['confidence']
                                #         st.markdown(f"- Prediction: {yolo_pred}")
                                #         st.markdown(f"- Confidence: {yolo_conf:.1%}")
                                    
                                #     with col_fracnet:
                                #         st.markdown("**🧠 FracNet2D Analysis**")
                                #         fracnet_pred = "Fracture" if result['fracnet_result']['is_fracture'] else "No Fracture"
                                #         fracnet_conf = result['fracnet_result']['confidence']
                                #         st.markdown(f"- Prediction: {fracnet_pred}")
                                #         st.markdown(f"- Confidence: {fracnet_conf:.1%}")
                                
                                # Medical disclaimer
                                st.markdown("---")
                                st.warning("""
                                **⚕️ Medical Disclaimer:** This AI analysis is for research and educational purposes only. 
                                Always consult with qualified medical professionals for proper diagnosis and treatment decisions.
                                """)
                                
                                # Clean up temporary file
                                if os.path.exists(temp_path):
                                    os.remove(temp_path)
                                    
                            except Exception as e:
                                st.error(f"❌ Analysis Error: {str(e)}")
                                st.info("Please try uploading a different image or contact support.")
                                if os.path.exists(temp_path):
                                    os.remove(temp_path)
                        else:
                            st.error("❌ Model Loading Failed")
                            st.info("Unable to load the AI models. Please check system configuration.")
        
        else:
            # Instructions when no image is uploaded
            st.markdown("""
            <div class='main-container'>
                <h3>🚀 Get Started</h3>
                <p>Upload an X-ray image above to begin AI-powered fracture analysis.</p>
                
                <h4>📋 Supported Features:</h4>
                <ul>
                    <li><strong>High Accuracy:</strong> 93.2% detection accuracy</li>
                    <li><strong>Dual AI Models:</strong> YOLO v11 + FracNet2D ensemble</li>
                    <li><strong>Instant Results:</strong> Real-time analysis with confidence scores</li>
                    <li><strong>Detailed Breakdown:</strong> Individual model predictions and reasoning</li>
                    <li><strong>Medical Grade:</strong> Professional-level diagnostic assistance</li>
                </ul>
                
                <h4>📁 Supported Formats:</h4>
                <p>JPG, JPEG, PNG - Maximum file size: 200MB</p>
            </div>
            """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown("### 🏥 System Overview")
        
        # Performance metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Overall Accuracy", "93.2%", "+4.6%")
            st.metric("Sensitivity", "90.91%", "+2.3%")
        
        with col2:
            st.metric("Specificity", "95.45%", "+1.8%")
            st.metric("Processing Time", "< 3 sec", "-2 sec")
        
        with col3:
            st.metric("Model Type", "Hybrid AI", "✅")
            st.metric("Dataset Size", "1000+ images", "+500")
        
        # System architecture
        st.markdown("### 🏗️ Architecture")
        st.markdown("""
        Our hybrid system combines two complementary AI models:
        
        **Primary Model - YOLO v11:**
        - Specialized object detection for medical imaging
        - 88.6% standalone accuracy
        - Fast inference and high precision
        - Optimized for fracture localization
        
        **Secondary Model - FracNet2D:**
        - Deep learning classification network
        - Provides validation and confidence boosting
        - Custom trained on fracture datasets
        - Enhances overall system reliability
        
        **Ensemble Decision Logic:**
        - YOLO-dominant decision making
        - Intelligent confidence weighting
        - Medical-grade safety protocols
        - Transparent reasoning process
        """)
        
        # Development team
        st.markdown("### 👥 Development Team")
        st.markdown("""
        - R.Surya Harshith - 22BCE9912
        - S.Narendar Reddy  - 22BCE7427  
        - R.Sasi Varshith - 22BCE20052
        """)
        
        # Technical specifications
        with st.expander("🔧 Technical Specifications"):
            st.markdown("""
            **Model Specifications:**
            - YOLO v11: Classification variant, optimized for medical imaging
            - FracNet2D: Custom 2D fracnet(from 3D) with regularization and dropout
            - Input Resolution: 224x224 pixels (standardized)
            - Output Classes: Binary (Fracture/No Fracture)
            
            **Training Details:**
            - Dataset: 1000+ labeled X-ray images
            - Training Split: 70% train, 15% validation, 15% test
            - Data Augmentation: Rotation, scaling, brightness adjustment
            - Hardware: CPU-optimized for accessibility
            
            **Performance Metrics:**
            - Accuracy: 93.18%
            - Precision: 94.12%
            - Recall: 90.91%
            - F1-Score: 92.47%
            """)

if __name__ == "__main__":
    main()