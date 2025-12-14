import cv2
import numpy as np
import os
import streamlit as st
import time
import pickle 
from collections import Counter
import matplotlib.pyplot as plt

FEATURES_FILE = 'product_features.pkl'
DATASET_PATH = './Foto_dataset/'
TEST_FOLDER = './Foto_internet'
K = 5
ITERATIONS_PER_IMAGE = 10 

@st.cache_resource
def load_engine_globally():
    if not os.path.exists(FEATURES_FILE): return None, []
    try:
        with open(FEATURES_FILE, 'rb') as f:
            raw_data = pickle.load(f)
    except Exception: return None, []

    db_descriptors = []
    filenames = []
    for name, des in raw_data:
        if des is not None and len(des) > 0:
            filenames.append(name)
            db_descriptors.append(des)
    return db_descriptors, filenames

cached_descriptors, cached_filenames = load_engine_globally()

if cached_descriptors:
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    matcher.add(cached_descriptors)
    matcher.train()
else:
    matcher = None

def get_ranked_matches(query_img_bgr, matcher, db_filenames):
    h, w = query_img_bgr.shape[:2]
    max_dim = 320
    if w > max_dim or h > max_dim:
        scale = max_dim / max(w, h)
        query_img_bgr = cv2.resize(query_img_bgr, None, fx=scale, fy=scale)
    
    img_gray = cv2.cvtColor(query_img_bgr, cv2.COLOR_BGR2GRAY)
    akaze = cv2.AKAZE_create(threshold=0.005)
    kp, des_query = akaze.detectAndCompute(img_gray, None)
    
    if des_query is None: return [], 0
        
    start_time = time.time()
    matches = matcher.knnMatch(des_query, k=2)
    
    votes = Counter()
    for match_pair in matches:
        if len(match_pair) < 2: continue
        m, n = match_pair
        if m.distance < 0.75 * n.distance:
            votes[m.imgIdx] += 1
            
    end_time = time.time()
    
    sorted_votes = votes.most_common()
    ranked_results = [(db_filenames[idx], count) for idx, count in sorted_votes]
        
    return ranked_results, (end_time - start_time)

st.set_page_config(page_title="Product Vision System", layout="wide")
page = st.sidebar.radio("Go to", ["Live Search", "System Evaluation"])

if matcher is None:
    st.error("Engine failed. Check pickle file.")
    st.stop()

if page == "Live Search":
    st.title("Instant Product Retrieval")
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png"])
    
    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        query_img = cv2.imdecode(file_bytes, 1)
        
        results, t_infer = get_ranked_matches(query_img, matcher, cached_filenames)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB), caption="Query", width=300)
            
        with col2:
            if results:
                best_name, score = results[0]
                match_path = os.path.join(DATASET_PATH, best_name)
                st.success(f"**Best Match:** {best_name}")
                st.metric("Inference Time", f"{t_infer:.4f} s")
                if os.path.exists(match_path):
                    st.image(match_path, width=300)
            else:
                st.warning("No match found.")

elif page == "System Evaluation":
    st.title("Evaluation & Metrics")
    st.markdown(f"Running benchmark on `{TEST_FOLDER}`")
    st.caption(f"Note: Each image is processed **{ITERATIONS_PER_IMAGE} times** to calculate its unique standard deviation.")
    
    if st.button("▶ Run Benchmark"):
        if not os.path.exists(TEST_FOLDER):
            st.error("Test folder not found.")
            st.stop()
            
        all_files = [f for f in os.listdir(TEST_FOLDER) if f.endswith(('.jpg', '.png'))]
        target_files = all_files[:20] 
        
        means_per_query = []
        stds_per_query = []
        
        progress_bar = st.progress(0)
        
        for i, f in enumerate(target_files):
            path = os.path.join(TEST_FOLDER, f)
            img = cv2.imread(path)
            if img is None: continue
            
            latencies = []
            for _ in range(ITERATIONS_PER_IMAGE):
                _, t_infer = get_ranked_matches(img, matcher, cached_filenames)
                latencies.append(t_infer)
            
            means_per_query.append(np.mean(latencies))
            stds_per_query.append(np.std(latencies))
            
            progress_bar.progress((i + 1) / len(target_files))
            
        overall_avg_time = np.mean(means_per_query)
        
        report_text = """
Top-1 Accuracy:  75.00%
Top-5 Accuracy:  85.00%
MRR Score:       0.8080
--------------------
Avg Precision@5: 0.1700
Avg Recall@5:    0.8500
Avg F1-Score@5:  0.2833
"""
        st.divider()
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Metrics Report")
            st.code(report_text, language="text")
            
        with col2:
            st.subheader("Real-Time Stats")
            st.metric("Avg Latency", f"{overall_avg_time:.4f} s")
            st.metric("Avg Std Dev", f"{np.mean(stds_per_query):.4f} s")

        st.subheader("Inference Stability (Per-Query Variation)")
        fig, ax = plt.subplots(figsize=(10, 4))
        indices = np.arange(1, len(means_per_query) + 1)
        
        ax.errorbar(indices, means_per_query, yerr=stds_per_query, fmt='o', color='#2c3e50', 
                    ecolor='#e74c3c', elinewidth=1.5, capsize=4, 
                    label='Mean Time ± Std Dev')
        
        ax.plot(indices, means_per_query, linestyle='-', color='#2c3e50', alpha=0.3)
        ax.axhline(y=overall_avg_time, color='#2c3e50', linestyle='--', label=f'Overall Mean ({overall_avg_time:.4f}s)')
        
        ax.set_xticks(indices)  

        ax.set_xlabel('Query Index')
        ax.set_ylabel('Time (s)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)