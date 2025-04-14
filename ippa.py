import streamlit as st
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import io

# Configure page
st.set_page_config(page_title="Image Processing", layout="wide", page_icon="🎨")
st.title("Imageg Editor By Dhairya Sharma")

# Custom CSS for modern UI with new color scheme
st.markdown("""
<style>
    [data-testid="stSidebar"] {
        background: linear-gradient(45deg, #1e3a8a, #3b82f6) !important;
        color: white !important;
    }
    .comparison-container {
        display: flex;
        gap: 10px;
        margin-top: 20px;
    }
    .metric-card {
        background-color: #e0f2fe;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .stButton button {
        background-color: #3b82f6;
        color: white;
        border-radius: 8px;
    }
    .stButton button:hover {
        background-color: #1e3a8a;
    }
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1, h2, h3 {
        color: #1e3a8a;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #e0f2fe;
        border-radius: 4px 4px 0 0;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3b82f6 !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# Image upload with drag & drop zone
with st.sidebar.expander("📤 UPLOAD IMAGE", expanded=True):
    uploaded_file = st.file_uploader("", type=["png", "jpg", "jpeg"], label_visibility="collapsed")

# Function to calculate and display histogram
def display_histogram(image, title="Histogram"):
    fig, ax = plt.subplots(figsize=(5, 3))
    
    if len(image.shape) == 3:  # Color image
        colors = ('r', 'g', 'b')
        for i, color in enumerate(colors):
            hist = cv2.calcHist([image], [i], None, [256], [0, 256])
            ax.plot(hist, color=color)
    else:  # Grayscale image
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        ax.plot(hist, color='gray')
    
    ax.set_xlim([0, 256])
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.patch.set_facecolor('#e0f2fe')
    return fig

# Function to display image statistics
def display_stats(image):
    if len(image.shape) == 3:
        mean_val = np.mean(image, axis=(0, 1))
        std_val = np.std(image, axis=(0, 1))
        min_val = np.min(image, axis=(0, 1))
        max_val = np.max(image, axis=(0, 1))
        
        stats = {
            "Mean": f"R: {mean_val[0]:.2f}, G: {mean_val[1]:.2f}, B: {mean_val[2]:.2f}",
            "Std Dev": f"R: {std_val[0]:.2f}, G: {std_val[1]:.2f}, B: {std_val[2]:.2f}",
            "Min": f"R: {min_val[0]}, G: {min_val[1]}, B: {min_val[2]}",
            "Max": f"R: {max_val[0]}, G: {max_val[1]}, B: {max_val[2]}"
        }
    else:
        stats = {
            "Mean": f"{np.mean(image):.2f}",
            "Std Dev": f"{np.std(image):.2f}",
            "Min": f"{np.min(image)}",
            "Max": f"{np.max(image)}"
        }
    
    return stats

# NEW FEATURE 1: Image Overlay & Blending
def blend_images(img1, img2, alpha):
    # Resize img2 to match img1's dimensions
    img2_resized = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    # Blend the images
    blended = cv2.addWeighted(img1, alpha, img2_resized, 1-alpha, 0)
    return blended

# NEW FEATURE 2: Text & Watermark Overlay
def add_text_watermark(image, text, position, font_scale, color, thickness):
    result = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Calculate position from percentage to pixels
    h, w = image.shape[:2]
    x_pos = int(w * position[0] / 100)
    y_pos = int(h * position[1] / 100)
    
    # Add the text
    cv2.putText(result, text, (x_pos, y_pos), font, font_scale, color, thickness)
    return result

if uploaded_file:
    # Convert to numpy array
    image = np.array(Image.open(uploaded_file))
    
    # Store original image for restoration
    original_image = image.copy()
    
    # Main columns for layout
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.markdown("### Original Image")
        st.image(image, use_column_width=True)
        
        with st.expander("Original Image Analysis", expanded=False):
            st.pyplot(display_histogram(image, "Original Histogram"))
            
            stats = display_stats(image)
            stat_cols = st.columns(4)
            for i, (label, value) in enumerate(stats.items()):
                with stat_cols[i]:
                    st.metric(label, value, delta=None, delta_color="normal")
    
    # Processing controls
    with st.sidebar:
        st.markdown("## 🎚 Processing Controls")
        processor = st.radio("Select Operation:", [
            "Smoothing Filters",
            "Sharpening Filters",
            "Edge Detection",
            "Color Adjustments",
            "Image Blending",  # NEW FEATURE 1
            "Text & Watermark"  # NEW FEATURE 2
        ])
        
        # Add reset button
        if st.button("🔄 Reset Image"):
            processed = original_image.copy()
        
        # Add download option for processed image
        st.markdown("## 💾 Save Result")
        show_comparison = st.checkbox("Show Side-by-Side Comparison", value=True)
    
    # Processing pipeline
    processed = image.copy()
    with col2:
        st.markdown("### Processed Image")
        
        if processor == "Smoothing Filters":
            smooth_type = st.radio("Filter Type:", ["Gaussian", "Median", "Bilateral"])
            kernel_size = st.slider("Kernel Size", 3, 25, 9, 2)
            
            if smooth_type == "Gaussian":
                sigma = st.slider("Sigma", 0.1, 5.0, 1.5)
                processed = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
            elif smooth_type == "Median":
                processed = cv2.medianBlur(image, kernel_size)
            else:
                d = st.slider("Diameter", 1, 15, 9)
                sigma_color = st.slider("Color Sigma", 1, 200, 75)
                sigma_space = st.slider("Spatial Sigma", 1, 200, 75)
                processed = cv2.bilateralFilter(image, d, sigma_color, sigma_space)
        
        elif processor == "Sharpening Filters":
            sharp_type = st.radio("Technique:", ["Laplacian", "Unsharp Mask"])
            
            if sharp_type == "Laplacian":
                strength = st.slider("Strength", 1, 10, 5) 
                kernel = np.array([[0, -1, 0], [-1, strength, -1], [0, -1, 0]])
                processed = cv2.filter2D(image, -1, kernel)
            else:
                strength = st.slider("Strength", 0.5, 3.0, 1.5)
                blur = cv2.GaussianBlur(image, (0,0), 3)
                processed = cv2.addWeighted(image, strength, blur, -0.5, 0)
        
        elif processor == "Edge Detection":
            edge_type = st.selectbox("Detection Method:", ["Canny", "Sobel", "Laplacian"])
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            if edge_type == "Canny":
                threshold1 = st.slider("Low Threshold", 0, 255, 50)
                threshold2 = st.slider("High Threshold", 0, 255, 150)
                processed = cv2.Canny(gray, threshold1, threshold2)
                # Convert back to 3 channels for consistent display
                processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)
            elif edge_type == "Sobel":
                dx = st.slider("X Derivative", 0, 2, 1)
                dy = st.slider("Y Derivative", 0, 2, 1)
                ksize = st.slider("Kernel Size", 1, 7, 5, 2)
                sobel = cv2.Sobel(gray, cv2.CV_64F, dx, dy, ksize=ksize)
                processed = np.uint8(np.absolute(sobel))
                processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)
            else:
                ksize = st.slider("Kernel Size", 1, 7, 3, 2)
                processed = cv2.Laplacian(gray, cv2.CV_64F, ksize=ksize)
                processed = np.uint8(np.absolute(processed))
                processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)
        
        elif processor == "Color Adjustments":
            adjust_type = st.radio("Adjustment Type:", ["Brightness/Contrast", "HSV Adjustment", "Color Balance"])
            
            if adjust_type == "Brightness/Contrast":
                alpha = st.slider("Contrast", 0.0, 3.0, 1.0, 0.1)  # Contrast control
                beta = st.slider("Brightness", -100, 100, 0)  # Brightness control
                processed = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
            
            elif adjust_type == "HSV Adjustment":
                hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
                h_shift = st.slider("Hue Shift", -180, 180, 0)
                s_scale = st.slider("Saturation Scale", 0.0, 2.0, 1.0, 0.1)
                v_scale = st.slider("Value (Brightness) Scale", 0.0, 2.0, 1.0, 0.1)
                
                # Apply adjustments
                hsv[:,:,0] = (hsv[:,:,0] + h_shift) % 180
                hsv[:,:,1] = np.clip(hsv[:,:,1] * s_scale, 0, 255).astype(np.uint8)
                hsv[:,:,2] = np.clip(hsv[:,:,2] * v_scale, 0, 255).astype(np.uint8)
                processed = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
            
            elif adjust_type == "Color Balance":
                r_scale = st.slider("Red Channel", 0.0, 2.0, 1.0, 0.1)
                g_scale = st.slider("Green Channel", 0.0, 2.0, 1.0, 0.1)
                b_scale = st.slider("Blue Channel", 0.0, 2.0, 1.0, 0.1)
                
                # Split channels and scale them
                b, g, r = cv2.split(image)
                b = np.clip(b * b_scale, 0, 255).astype(np.uint8)
                g = np.clip(g * g_scale, 0, 255).astype(np.uint8)
                r = np.clip(r * r_scale, 0, 255).astype(np.uint8)
                processed = cv2.merge([b, g, r])
                
        # NEW FEATURE 1: Image Blending implementation
        elif processor == "Image Blending":
            st.info("Upload a second image to blend with the original")
            second_image_file = st.file_uploader("Select overlay image", type=["png", "jpg", "jpeg"])
            
            if second_image_file:
                second_image = np.array(Image.open(second_image_file))
                st.image(second_image, caption="Overlay Image", width=200)
                
                blend_alpha = st.slider("Blend Ratio (Original:Overlay)", 0.0, 1.0, 0.5, 0.05)
                blend_mode = st.radio("Blend Mode", ["Normal Blend", "Screen", "Multiply"])
                
                # Resize second image to match first
                second_image_resized = cv2.resize(second_image, (image.shape[1], image.shape[0]))
                
                if blend_mode == "Normal Blend":
                    processed = cv2.addWeighted(image, blend_alpha, second_image_resized, 1-blend_alpha, 0)
                elif blend_mode == "Screen":
                    # Screen blend mode formula
                    processed = 255 - ((255 - image) * (255 - second_image_resized) / 255)
                    processed = processed.astype(np.uint8)
                else:  # Multiply
                    # Multiply blend mode
                    processed = (image * second_image_resized / 255).astype(np.uint8)
            else:
                st.warning("Please upload a second image to use this feature")
                
        # NEW FEATURE 2: Text & Watermark implementation
        elif processor == "Text & Watermark":
            text_content = st.text_input("Watermark Text", "© My Image")
            
            # Position control (percentage of image dimensions)
            col_pos1, col_pos2 = st.columns(2)
            with col_pos1:
                pos_x = st.slider("X Position (%)", 0, 100, 10)
            with col_pos2:
                pos_y = st.slider("Y Position (%)", 0, 100, 90)
            
            # Appearance controls
            text_size = st.slider("Text Size", 0.5, 5.0, 1.0, 0.1)
            text_thickness = st.slider("Text Thickness", 1, 5, 2)
            
            # Color picker
            text_color = st.color_picker("Text Color", "#FFFFFF")
            # Convert from hex to RGB
            r = int(text_color[1:3], 16)
            g = int(text_color[3:5], 16)
            b = int(text_color[5:7], 16)
            
            # Apply watermark
            processed = add_text_watermark(
                image, 
                text_content, 
                (pos_x, pos_y), 
                text_size, 
                (r, g, b), 
                text_thickness
            )
            
            # Option for semi-transparent watermark
            if st.checkbox("Add semi-transparent background"):
                opacity = st.slider("Background Opacity", 0.1, 1.0, 0.5, 0.1)
                h, w = image.shape[:2]
                x_pos = int(w * pos_x / 100)
                y_pos = int(h * pos_y / 100)
                
                # Calculate text size
                (text_width, text_height), _ = cv2.getTextSize(
                    text_content, cv2.FONT_HERSHEY_SIMPLEX, text_size, text_thickness
                )
                
                # Create overlay for text background
                overlay = processed.copy()
                bg_color = (0, 0, 0)  # Black background
                
                # Draw rectangle behind text
                cv2.rectangle(
                    overlay, 
                    (x_pos - 5, y_pos - text_height - 5), 
                    (x_pos + text_width + 5, y_pos + 5), 
                    bg_color, 
                    -1
                )
                
                # Apply overlay with transparency
                processed = cv2.addWeighted(overlay, opacity, processed, 1 - opacity, 0)
                
                # Re-add text on top of semi-transparent background
                processed = add_text_watermark(
                    processed, 
                    text_content, 
                    (pos_x, pos_y), 
                    text_size, 
                    (r, g, b), 
                    text_thickness
                )
        
        # Display processed image
        st.image(processed, use_column_width=True)
        
        # Histogram and statistics for processed image
        with st.expander("Processed Image Analysis", expanded=False):
            st.pyplot(display_histogram(processed, "Processed Histogram"))
            
            stats = display_stats(processed)
            stat_cols = st.columns(4)
            for i, (label, value) in enumerate(stats.items()):
                with stat_cols[i]:
                    st.metric(label, value)
    
    # Side by side comparison if enabled
    if show_comparison:
        st.markdown("### Before & After Comparison")
        
        # Create tabs for different comparison views
        comparison_tabs = st.tabs(["Side by Side", "Slider Comparison"])
        
        with comparison_tabs[0]:
            comp_col1, comp_col2 = st.columns(2)
            
            with comp_col1:
                st.image(image, caption="Original", use_column_width=True)
            
            with comp_col2:
                st.image(processed, caption="Processed", use_column_width=True)
        
        with comparison_tabs[1]:
            # Create a matplotlib figure for the interactive comparison
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.imshow(np.hstack([image, processed]))
            ax.axis('off')
            st.pyplot(fig)
            st.caption("Drag the slider to compare Original (left) vs Processed (right)")
    
    # Download button for processed image
    buf = cv2.imencode('.png', cv2.cvtColor(processed, cv2.COLOR_RGB2BGR))[1].tobytes()
    st.sidebar.download_button(
        label="Download Processed Image",
        data=buf,
        file_name="processed_image.png",
        mime="image/png"
    )

else:
    # Show placeholder when no image is uploaded
    st.markdown("""
    <div style="text-align: center; padding: 100px 20px; background-color: #e0f2fe; border-radius: 12px; margin: 2rem 0;">
        <h2 style="color: #1e3a8a">📁 Drag & Drop Image to Begin</h2>
        <p style="color: #3b82f6">Supports JPG, PNG, JPEG formats</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Show feature overview when no image is uploaded
    st.markdown("## 🎨 Feature Overview")
    
    feature_cols = st.columns(3)
    with feature_cols[0]:
        st.markdown("""
        ### 🔍 Image Filters
        - Smoothing (Gaussian, Median, Bilateral)
        - Sharpening
        - Edge Detection
        - Color Adjustments
        """)
    
    with feature_cols[1]:
        st.markdown("""
        ### 🎭 Creative Tools
        - Image Blending
        - Watermarking
        - Multiple blend modes
        """)
    
    with feature_cols[2]:
        st.markdown("""
        ### 📊 Analysis
        - RGB Histograms
        - Image Statistics
        - Before/After Comparison
        """)