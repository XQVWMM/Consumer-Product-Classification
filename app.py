import cv2
import numpy as np
import os
import streamlit as st
import time
import pickle 
from collections import Counter

# --- CONFIGURATION ---
FEATURES_FILE = 'product_features.pkl'
DATASET_PATH = './Foto_dataset/'

# --- 1. INITIALIZE ENGINE (ONLY ONCE) ---
if 'matcher' not in st.session_state:
    st.session_state['matcher'] = None
    st.session_state['db_filenames'] = []

def load_engine():
    # If already loaded, skip
    if st.session_state['matcher'] is not None:
        return

    if not os.path.exists(FEATURES_FILE):
        st.error(f"File {FEATURES_FILE} not found.")
        return

    # Measure loading time
    t0 = time.time()
    
    try:
        with open(FEATURES_FILE, 'rb') as f:
            raw_data = pickle.load(f)
    except Exception as e:
        st.error(f"Error loading pickle: {e}")
        return

    db_descriptors = []
    filenames = []

    # Unpack data
    for name, des in raw_data:
        if des is not None and len(des) > 0:
            filenames.append(name)
            db_descriptors.append(des)

    if not db_descriptors:
        return

    # VECTORIZED MATCHER (All data in one C++ object)
    # We use BFMatcher because for Binary Descriptors (AKAZE), 
    # it uses CPU 'POPCNT' instructions which are insanely fast.
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    matcher.add(db_descriptors)
    matcher.train()

    # Store in Session State to persist across re-runs
    st.session_state['matcher'] = matcher
    st.session_state['db_filenames'] = filenames
    
    print(f"Engine loaded in {time.time() - t0:.4f}s")

# Load the engine immediately
load_engine()

# --- 2. FAST QUERY PROCESSING ---
def get_query_features(uploaded_image):
    # Decode
    file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, 1)
    
    # RESIZE (Crucial for speed)
    # 320px is the "sweet spot" for speed vs accuracy
    height, width = img_bgr.shape[:2]
    max_dim = 320
    if width > max_dim or height > max_dim:
        scale = max_dim / max(width, height)
        img_bgr = cv2.resize(img_bgr, None, fx=scale, fy=scale)

    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # High Threshold = Fewer Points = Faster Match
    akaze = cv2.AKAZE_create(threshold=0.005) 
    kp, des = akaze.detectAndCompute(img_gray, None)
    
    return img_bgr, kp, des

# --- STREAMLIT UI ---
st.set_page_config(page_title="Instant Retrieval", layout="wide")
st.title("⚡ 0.05s Product Retrieval")

# Check status
if st.session_state['matcher'] is None:
    st.warning("Building Index... (This happens only once)")
    load_engine()
    st.rerun() # Refresh to clear warning

uploaded_file = st.sidebar.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 1. START TIMER
    start_time = time.time()
    
    # Process
    query_img_bgr, kp_query, des_query = get_query_features(uploaded_file)
    
    if des_query is None:
        st.warning("No features found.")
    else:
        # 2. MATCHING (Vectorized C++)
        matcher = st.session_state['matcher']
        db_filenames = st.session_state['db_filenames']
        
        # Match Query against ALL DB images simultaneously
        matches = matcher.knnMatch(des_query, k=2)
        
        # 3. VOTING (Python optimized)
        votes = Counter()
        
        for match_pair in matches:
            if len(match_pair) < 2: continue
            m, n = match_pair
            
            # Strict Ratio Test
            if m.distance < 0.75 * n.distance:
                # m.imgIdx refers to the index of the image in db_filenames
                votes[m.imgIdx] += 1
        
        # 4. STOP TIMER
        end_time = time.time()
        elapsed = end_time - start_time
        
        # --- DISPLAY ---
        col1, col2 = st.columns(2)
        with col1:
            st.image(cv2.cvtColor(query_img_bgr, cv2.COLOR_BGR2RGB), caption="Query (Resized)", width=300)
            st.metric("Total Search Time", f"{elapsed:.4f} sec")

        if votes:
            best_idx, max_votes = votes.most_common(1)[0]
            best_filename = db_filenames[best_idx]
            
            with col2:
                match_path = os.path.join(DATASET_PATH, best_filename)
                if os.path.exists(match_path):
                    st.image(match_path, caption=f"Best Match: {best_filename}", width=300)
                else:
                    st.error(f"File {best_filename} missing.")
        else:
            st.warning("No matches found.")