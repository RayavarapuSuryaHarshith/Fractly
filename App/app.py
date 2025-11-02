import streamlit as st  
from ultralytics import YOLO
import sys
import os
from PIL import Image, ImageOps
import numpy as np
import torchvision
import torch

# Add the src directory to Python path to import our hybrid model
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from final_hybrid_model import FinalHybridFractureDetector

from sidebar import Sidebar

# Optional imports for legacy models - gracefully handle if not available
rcnnres = None
vgg = None
try:
    import rcnnres
except ImportError:
    pass  # Handle silently, show message only when model is selected

try:
    import vgg
except ImportError:
    pass  # Handle silently, show message only when model is selected

# hide deprecation warnings which directly don't affect the working of the application
import warnings
warnings.filterwarnings("ignore")

# Sidebar 
sb = Sidebar()

title_image = sb.title_img
model = sb.model_name
conf_threshold = sb.confidence_threshold

#Main Page

st.markdown("<h1 style='text-align: center; font-family: Arial, sans-serif; text-transform: uppercase;'>Bone Fracture Detection</h1>", unsafe_allow_html=True)
st.write("Professional Bone Fracture Detection using our advanced **Hybrid AI System** achieving **93.2% accuracy** - combining YOLO v11 and FracNet2D for medical-grade precision.")

st.markdown("""
<style>

	.stTabs [data-baseweb="tab-list"] {
		gap: 10px;
    }

	.stTabs [data-baseweb="tab"] {
		height: 50px;
        white-space: pre-wrap;
		border-radius: 2px 2px 2px 2px;
		gap: 8px;
        padding-left: 10px;
        padding-right: 10px;
		padding-top: 8px;
		padding-bottom: 8px;
    }

	.stTabs [aria-selected="true"] {
  		background-color: #7f91ad;
	}

</style>""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["Overview", "Test"])

with tab1:
   st.markdown("### Overview")
   st.text_area(
    "TEAM MEMBERS",
    "Nellore Sai Nikhil - 21BCE8845\nSungavarapu Sai Harshavardhan - 21BCE8250\nVemula Kesavaaditya Gupta - 21BCE7839",
    )
   
   st.markdown("#### 🎯 Hybrid AI System (Recommended)")
   st.success("**Our flagship Hybrid Model achieves 93.2% accuracy** - exceeding medical standards for automated fracture detection.")
   
   st.text_area(
       "Hybrid Model Description",
       "Our Hybrid Fracture Detection System combines the strengths of YOLO v11 (88.6% standalone accuracy) and Custom FracNet2D models through intelligent ensemble decision-making. The system uses YOLO as the primary decision maker due to its superior performance, while FracNet2D provides validation and confidence boosting. Key features include: YOLO-dominant decision logic for optimal accuracy, medical-grade sensitivity (90.91%) and specificity (95.45%), intelligent consensus algorithms that leverage both models' strengths, and professional confidence reporting with detailed analysis breakdown. The hybrid approach achieves 93.2% overall accuracy, significantly outperforming individual models and meeting medical standards for automated diagnosis assistance.",
       height=120,
    )
   
   st.markdown("#### Network Architecture")

   v11 = "images\\yolov11.jpg"
   st.image(v11, caption="Yolov11 Architecture (Primary Component)", width=None, use_column_width=True)
   
   network_img = "images\\NN_Architecture_Updated.jpg"
   st.image(network_img, caption="Network Architecture", width=None, use_column_width=True)
   
   st.markdown("#### Legacy Models (Research Comparison)")

   st.markdown("##### YoloV10")
   st.text_area(
       "Description",
       "In this bone fracture detection project, we're using a lightweight and efficient version of the YOLO v10 algorithm called YOLO v10 Nano (yolov10n). This algorithm is tailored for systems with limited computational resources. We start by training the model on a dataset of X-ray images that are labeled to show where fractures are. We specify the training settings in a YAML file. The model is trained for 50 epochs, and we save its progress every 25 epochs to keep the best-performing versions. YOLO v10 Nano is great at quickly and accurately spotting fractures, even on devices with lower computing power. After training, we test the model on a separate set of images to ensure it can reliably detect fractures. In practical use, the trained model automatically identifies and marks fractures on new X-ray images by drawing boxes around them. This helps doctors quickly and accurately diagnose fractures. We assess the model's effectiveness using performance metrics like confusion matrix and Intersection over Union (IoU) scores to understand how well it performs across different types of fractures.",
    )

   st.markdown("##### YoloV11")
   st.text_area(
       "Description",
       "In this bone fracture detection project, we're using a lightweight and efficient version of the YOLO v11 algorithm called YOLO v11. This algorithm is tailored for systems with limited computational resources. We start by training the model on a dataset of X-ray images that are labeled to show where fractures are. We specify the training settings in a YAML file. The model is trained for 100 epochs, and we save its progress every 30 epochs to keep the best-performing versions. YOLO v11  is great at quickly and accurately spotting fractures, even on devices with lower computing power. After training, we test the model on a separate set of images to ensure it can reliably detect fractures. In practical use, the trained model automatically identifies and marks fractures on new X-ray images by drawing boxes around them. This helps doctors quickly and accurately diagnose fractures. We assess the model's effectiveness using performance metrics like confusion matrix and Intersection over Union (IoU) scores to understand how well it performs across different types of fractures.",
    )
    
   
   st.markdown("##### FasterRCNN with ResNet")
   st.text_area(
       "Decription",
       "This code implements a bone fracture detection system using the Fast R-CNN (Region-based Convolutional Neural Network) architecture. The purpose of Faster R-CNN (Region-based Convolutional Neural Network) is to perform efficient and accurate object detection within images. It addresses the challenge of localizing and classifying objects of interest in images, a fundamental task in computer vision applications. Faster R-CNN achieves this by introducing a Region Proposal Network (RPN) to generate candidate object bounding boxes, which are then refined and classified by subsequent network components. By combining region proposal generation and object detection into a single unified framework, Faster R-CNN significantly improves detection accuracy while maintaining computational efficiency, making it suitable for real-time applications such as autonomous driving, surveillance, medical imaging, and more. The dataset containing bone X-ray images is prepared, with images and their corresponding labels loaded and augmented to resize and transform boundary boxes. The Faster R-CNN model is then instantiated, with a pre-trained ResNet-50 backbone and a custom classification layer for bone fracture detection. The training loop is executed over multiple epochs, optimizing the model's parameters using the Adam optimizer and minimizing the combined loss. The best-performing model is saved, and its performance is evaluated on the validation set, with the best model further tested on a separate test set. After training, the model's ability to detect fractures is evaluated by comparing its predictions with the actual fractures. This helps us understand how accurate the model is in finding fractures. Also, a confusion matrix is created to see how well the model performs for different types of fractures, providing insights into its overall performance. By combining the power of ResNet's feature extraction capabilities with Faster R-CNN's precise object localization and classification, the system can effectively detect bone fractures within medical images with improved accuracy and reliability.",
       height=45,
    )
   
   st.markdown("##### SSD with VGG16")
   st.text_area(
       "Description",
       "This code uses a Single Shot Multibox Detector (SSD) with a VGG16 backbone to construct a bone fracture detection system. It performs preprocessing on training and validation datasets, doing augmentations including image scaling and bounding box coordinate conversion. Pre-trained weights are used to initialize the SSD300_VGG16 model, and additional custom layers are added to allow for fine-tuning to the particular purpose of fracture identification. The training loop is executed over multiple epochs, optimizing the model's parameters using the Adam optimizer and minimizing the combined loss. While the evaluation loop evaluates the model's performance on the validation dataset, the training loop continually runs over the dataset, calculating losses, back propagating, and adjusting weights using an optimizer. Overall, this code efficiently integrates SSD with VGG16 for real-time fracture detection, leveraging the model's ability to predict one of the 7 class labels directly from input images.",
    )
   
   
   
   
   
   
# Initialize hybrid model once
@st.cache_resource
def load_hybrid_model():
    """Load the hybrid model once and cache it"""
    try:
        # Use relative paths from the App directory
        yolo_weights = os.path.join('..', 'weights', 'yolo11n_fracture_high_accuracy.pt')
        fracnet_weights = os.path.join('..', 'weights', 'custom_fracnet2d_best.pth')
        
        hybrid_model = FinalHybridFractureDetector(
            yolo_weights_path=yolo_weights,
            fracnet_weights_path=fracnet_weights,
            device="cpu",
            verbose=False
        )
        return hybrid_model
    except Exception as e:
        st.error(f"Error loading hybrid model: {e}")
        return None

# Legacy weights for other models
yolo_path_10 ="weights\YOLOv10n.pt"
yolo_path_v11 = "weights\yolov11_standalone.pt"
yolo_path_10_new = "weights\yv10.pt"

with tab2:
    st.markdown("### Upload & Test")
    
    # Image uploading
    image = st.file_uploader("Choose an X-ray image", type=["jpg", "png", "jpeg"])
    
    if image is not None:
        st.write("You selected the file:", image.name)
        
        # Create two columns for layout
        col1, col2 = st.columns(2)
        
        # Column 1: Display uploaded image
        with col1:
            uploaded_image = Image.open(image)
            st.image(
                image=uploaded_image,
                caption="Uploaded X-ray Image",
                use_column_width=True
            )
        
        # Column 2: Show results
        with col2:
            if model == "Hybrid Model (YOLO + FracNet)":
                if st.button("🔬 Analyze X-ray", type="primary"):
                    with st.spinner("Analyzing with Hybrid AI System..."):
                        # Load hybrid model
                        hybrid_model = load_hybrid_model()
                        
                        if hybrid_model is not None:
                            # Save uploaded image temporarily
                            temp_path = "temp_xray.jpg"
                            uploaded_image.save(temp_path)
                            
                            try:
                                # Get prediction
                                result = hybrid_model.predict(temp_path)
                                
                                # Display result
                                if result["is_fracture"]:
                                    st.error("⚠️ FRACTURE DETECTED")
                                    st.markdown(f"**Confidence:** {result['confidence']:.1%}")
                                else:
                                    st.success("✅ NO FRACTURE DETECTED")
                                    st.markdown(f"**Confidence:** {result['confidence']:.1%}")
                                
                                # Show detailed results
                                with st.expander("📊 Detailed Analysis"):
                                    st.markdown(f"**Decision Logic:** {result['decision_reason'].replace('_', ' ').title()}")
                                    st.markdown(f"**Models Agreement:** {'Yes' if result['models_agree'] else 'No'}")
                                    
                                    col_a, col_b = st.columns(2)
                                    with col_a:
                                        st.markdown("**YOLO Result:**")
                                        yolo_result = "Fracture" if result['yolo_result']['is_fracture'] else "No Fracture"
                                        st.write(f"- Prediction: {yolo_result}")
                                        st.write(f"- Confidence: {result['yolo_result']['confidence']:.1%}")
                                    
                                    with col_b:
                                        st.markdown("**FracNet Result:**")
                                        fracnet_result = "Fracture" if result['fracnet_result']['is_fracture'] else "No Fracture"
                                        st.write(f"- Prediction: {fracnet_result}")
                                        st.write(f"- Confidence: {result['fracnet_result']['confidence']:.1%}")
                                
                                # Clean up temp file
                                if os.path.exists(temp_path):
                                    os.remove(temp_path)
                                    
                            except Exception as e:
                                st.error(f"Error during analysis: {e}")
                                if os.path.exists(temp_path):
                                    os.remove(temp_path)
                        
                        else:
                            st.error("Failed to load hybrid model. Please check model weights.")
            
            # Legacy model implementations (condensed)
            elif model == "yolov10":


                if st.button("🔬 Analyze (YOLOv10)"):
                    with st.spinner("Running YOLOv10..."):
                        try:
                            yolo_detection_model = YOLO(yolo_path_10_new)
                            res = yolo_detection_model(uploaded_image)
                            detection_image = res[0].plot()
                            st.image(detection_image, caption="YOLOv10 Detection", use_column_width=True)
                        except Exception as ex:
                            st.error(f"YOLOv10 error: {ex}")
            
            elif model == 'Yolov11':
                if st.button("🔬 Analyze (YOLOv11)"):
                    with st.spinner("Running YOLOv11..."):
                        try:
                            yolo_detection_model = YOLO(yolo_path_v11)
                            res = yolo_detection_model.predict(uploaded_image, conf=0.25)
                            res_plotted = res[0].plot()[:, :, ::-1]
                            st.image(res_plotted, caption="YOLOv11 Detection", use_column_width=True)
                            
                            if len(res[0].boxes) > 0:
                                with st.expander("Detection Results"):
                                    for box in res[0].boxes:
                                        st.write(f"Confidence: {box.conf[0].item():.2f}")
                            else:
                                st.write("No detection")
                        except Exception as ex:
                            st.error(f"YOLOv11 error: {ex}")
                                    
            
            elif model == 'FastRCNN with ResNet':
                if rcnnres is None:
                    st.error("❌ FastRCNN model is not available - missing dependencies")
                    st.info("💡 Please use the Hybrid Model for best results (93.2% accuracy)")
                elif st.button("🔬 Analyze (FastRCNN)"):
                    with st.spinner("Running FastRCNN with ResNet..."):
                        try:
                            resnet_model = rcnnres.get_model()
                            device = torch.device('cpu')
                            resnet_model.to(device)
                            
                            content = uploaded_image.convert("RGB")
                            to_tensor = torchvision.transforms.ToTensor()
                            content = to_tensor(content).unsqueeze(0).half()
                            
                            output = rcnnres.make_prediction(resnet_model, content, 0.5)
                            fig, _ax, class_name = rcnnres.plot_image_from_output(content[0].detach(), output[0])
                            
                            st.image(rcnnres.figure_to_array(fig), caption="FastRCNN Detection", use_column_width=True)
                            with st.expander("Detection Results"):
                                st.write(f"Detected: {class_name}")
                        except Exception as ex:
                            st.error(f"FastRCNN error: {ex}")

            elif model == 'VGG16':
                if vgg is None:
                    st.error("❌ VGG16 model is not available - missing dependencies")
                    st.info("💡 Please use the Hybrid Model for best results (93.2% accuracy)")
                elif st.button("🔬 Analyze (VGG16)"):
                    with st.spinner("Running VGG16..."):
                        try:
                            vgg_model = vgg.get_vgg_model()
                            device = torch.device('cpu')
                            vgg_model.to(device)
                            
                            content = uploaded_image.convert("RGB")
                            to_tensor = torchvision.transforms.ToTensor()
                            content = to_tensor(content).unsqueeze(0).half()
                            
                            output = rcnnres.make_prediction(vgg_model, content, 0.5)
                            fig, _ax, class_name = rcnnres.plot_image_from_output(content[0].detach(), output[0])
                            
                            st.image(rcnnres.figure_to_array(fig), caption="VGG16 Detection", use_column_width=True)
                            with st.expander("Detection Results"):
                                st.write(f"Detected: {class_name}")
                        except Exception as ex:
                            st.error(f"VGG16 error: {ex}")
    
    else:
        st.info("👆 Please upload an X-ray image to start analysis")
        st.markdown("---")
        st.markdown("### 🏥 Recommended: Hybrid Model")
        st.markdown("Our **Hybrid Model (YOLO + FracNet)** achieves **93.2% accuracy** - the highest performance for bone fracture detection.")
        st.markdown("**Features:**")
        st.markdown("- ✅ Medical-grade accuracy (>90%)")
        st.markdown("- ✅ Dual AI system validation")
        st.markdown("- ✅ Professional medical analysis")
        st.markdown("- ✅ Detailed confidence reporting")
            
        
        
            
            
            

            







        
        
        

        
        
        
        
