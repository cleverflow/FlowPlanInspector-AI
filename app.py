# Deepseek api - sk-cdc65d554f0d4daead2aa49c5642fbd7
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
from openai import OpenAI
import pandas as pd
from datetime import datetime
import base64
import logging
from supabase import create_client
import uuid

# Initialize Supabase
supabase = create_client(
    "https://fpahtdebjddftowpkgkr.supabase.co",
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImZwYWh0ZGViamRkZnRvd3BrZ2tyIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MzQ0MzI1NzUsImV4cCI6MjA1MDAwODU3NX0.w-gUfaHv-s2pSXv9IkPS6rkEUZ4S-CETwuEj6gLAtA0"
)

logging.basicConfig(level=logging.INFO)

DEEPSEEK_API_KEY = "sk-cdc65d554f0d4daead2aa49c5642fbd7"
client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
MODEL = "deepseek-chat"

def upload_to_supabase(file_bytes, filename, folder_path, content_type="image/png"):
    """Upload file to Supabase storage"""
    try:
        unique_filename = f"{uuid.uuid4()}_{filename}"
        path = f"{folder_path}/{unique_filename}"
        
        file_options = {
            "content-type": content_type,
            "x-upsert": "true"
        }
        
        supabase.storage.from_('floorplans').upload(
            path=path,
            file=file_bytes,
            file_options=file_options
        )
        return supabase.storage.from_('floorplans').get_public_url(path)
    except Exception as e:
        logging.error(f"Supabase upload error: {str(e)}")
        st.error(f"Failed to upload to Supabase: {str(e)}")
        return None

def get_deepseek_analysis(image_bytes):
    try:
        image_b64 = base64.b64encode(image_bytes).decode()
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant specialized in analyzing floor maps and architectural changes."},
            {"role": "user", "content": "Please analyze this floor map and describe any notable elements, layouts, and features you observe."}
        ]
        
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            max_tokens=1024,
            temperature=0.7,
            stream=False
        )
        return response.choices[0].message.content
    except Exception as e:
        logging.error(f"Deepseek API error: {str(e)}")
        return "Image analysis unavailable"

def align_and_overlay_images(img1_bytes, img2_bytes):
    img1 = cv2.imdecode(np.frombuffer(img1_bytes, np.uint8), cv2.IMREAD_COLOR)
    img2 = cv2.imdecode(np.frombuffer(img2_bytes, np.uint8), cv2.IMREAD_COLOR)
    
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)
    
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)
    
    if len(good_matches) > 10:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        h, w = img2.shape[:2]
        aligned_img1 = cv2.warpPerspective(img1, M, (w, h))
        
        return aligned_img1, img2
    
    return None, None

def compare_maps(img1_bytes, img2_bytes, min_diff_area=200, major_diff_area=500):
    aligned_img1, img2 = align_and_overlay_images(img1_bytes, img2_bytes)
    
    if aligned_img1 is None:
        raise Exception("Could not align images properly")
    
    gray1 = cv2.cvtColor(aligned_img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    diff = cv2.absdiff(gray1, gray2)
    _, thresh = cv2.threshold(diff, 50, 255, cv2.THRESH_BINARY)
    
    kernel = np.ones((3,3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    diff_colored = img2.copy()
    overlay = diff_colored.copy()
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    changes = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_diff_area:
            x, y, w, h = cv2.boundingRect(contour)
            change_type = "major" if area > major_diff_area else "minor"
            changes.append({
                "type": change_type,
                "area": area,
                "location": (x, y, w, h)
            })
            
            color = (0, 0, 255) if change_type == "major" else (0, 255, 255)
            cv2.drawContours(overlay, [contour], -1, color, -1)
            cv2.rectangle(diff_colored, (x, y), (x+w, y+h), color, 2)
    
    alpha = 0.3
    cv2.addWeighted(overlay, alpha, diff_colored, 1 - alpha, 0, diff_colored)
    
    similarity = 100 - (np.count_nonzero(thresh) / thresh.size * 100)
    
    high_quality_img = cv2.imwrite("high_quality_diff.png", diff_colored, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    
    return diff_colored, similarity, changes, high_quality_img

def verify_output(diff_image, changes):
    try:
        change_description = f"Found {len(changes)} changes in the floor map. "
        change_description += f"Major changes: {sum(1 for c in changes if c['type'] == 'major')}, "
        change_description += f"Minor changes: {sum(1 for c in changes if c['type'] == 'minor')}"
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant specialized in verifying floor map changes."},
            {"role": "user", "content": f"Please verify the following changes detected: {change_description}"}
        ]
        
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            max_tokens=1024,
            temperature=0.7,
            stream=False
        )
        return response.choices[0].message.content
    except Exception as e:
        logging.error(f"Verification error: {str(e)}")
        return "Verification unavailable"

# Initialize session state
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False
if 'result_image' not in st.session_state:
    st.session_state.result_image = None
if 'similarity' not in st.session_state:
    st.session_state.similarity = None
if 'changes' not in st.session_state:
    st.session_state.changes = None
if 'verification' not in st.session_state:
    st.session_state.verification = None

# Streamlit UI
st.set_page_config(
    page_title="FlowPlanInspector AI by CleverFlow",
    page_icon=":robot_face:",
)

st.title("FlowPlanInspector AI by CleverFlow")
st.divider()

# Reset button
if st.button("Reset", key="reset_button"):
    st.session_state.processing_complete = False
    st.session_state.result_image = None
    st.session_state.similarity = None
    st.session_state.changes = None
    st.session_state.verification = None
    st.cache_data.clear()
    st.rerun()

st.markdown("FlowPlanInspector AI is an AI-powered tool developed by CleverFlow to analyze and compare floor plans and architectural maps.")

# File uploaders
col1, col2 = st.columns(2)
with col1:
    file1 = st.file_uploader("Original Map", type=["png"], key="file_uploader_1")
with col2:
    file2 = st.file_uploader("Modified Map", type=["png"], key="file_uploader_2")

if file1 and file2 and not st.session_state.processing_complete:
    try:
        with st.spinner("Processing..."):
            # Generate folder name
            folder_name = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Upload original and modified images
            upload_to_supabase(file1.getvalue(), file1.name, f"{folder_name}/original")
            upload_to_supabase(file2.getvalue(), file2.name, f"{folder_name}/modified")
            
            # Process images
            diff_image, similarity, changes, high_quality_img = compare_maps(file1.getvalue(), file2.getvalue())
            verification = verify_output(diff_image, changes)
            
            # Store results in session state
            _, buffer = cv2.imencode('.png', diff_image)
            st.session_state.result_image = buffer.tobytes()
            st.session_state.similarity = similarity
            st.session_state.changes = changes
            st.session_state.verification = verification
            st.session_state.processing_complete = True
            
            # Upload difference image
            upload_to_supabase(
                st.session_state.result_image,
                f"diff_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                f"{folder_name}/result"
            )
            
            # Calculate change counts
            major_changes = sum(1 for c in changes if c['type'] == 'major')
            minor_changes = sum(1 for c in changes if c['type'] == 'minor')
            total_changes = len(changes)
            
            # Store metadata
            try:
                metadata = {
                    "run_id": folder_name,
                    "timestamp": datetime.now().isoformat(),
                    "similarity_score": float(similarity),
                    "major_changes": int(major_changes),
                    "minor_changes": int(minor_changes),
                    "total_changes": int(total_changes)
                }
                
                result = supabase.table('analysis_runs').insert(metadata).execute()
                if not result.data:
                    st.warning("Failed to store analysis metadata")
                    
            except Exception as e:
                logging.error(f"Failed to store metadata: {str(e)}")
                
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.info("Please check input files")

# Display results if processing is complete
if st.session_state.processing_complete and st.session_state.result_image is not None:
    # Display the image
    st.image(st.session_state.result_image, caption="Differences Highlighted")
    
    # Display metrics
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Similarity", f"{st.session_state.similarity:.1f}%")
    with col2:
        st.metric("Changes", len(st.session_state.changes))

    # Display verification
    if st.session_state.verification != "Verification unavailable":
        st.subheader("AI Analysis")
        st.write(st.session_state.verification)

    # Color guide
    st.subheader("Color Guide")
    legend_col1, legend_col2 = st.columns(2)
    with legend_col1:
        st.markdown("""
        🔴 **Red Areas**: Major structural changes  
        🟡 **Yellow Areas**: Minor modifications
        """)
    with legend_col2:
        st.markdown("""
        🟩 **Green Boxes**: Change boundaries  
        ⬜ **White Areas**: No changes detected
        """)
        
    st.markdown("*Note: Higher opacity indicates greater confidence in detected changes*")

    # Download button for high quality image
    with open("high_quality_diff.png", "rb") as file:
        st.download_button(
            label="Download High Quality Image",
            data=file,
            file_name="floor_map_comparison_hq.png",
            mime="image/png",
            key="download_button_main"
        )
