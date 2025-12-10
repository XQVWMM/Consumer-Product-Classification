import cv2
import numpy as np
import os
import streamlit as st
import matplotlib.pyplot as plt
import time
from sklearn.metrics import precision_score, recall_score
import statistics
import pickle 

def display_image_with_keypoints(img_bgr, keypoints, title="Image", width=300):
    img_keypoints = cv2.drawKeypoints(img_bgr, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    plt.imshow(cv2.cvtColor(img_keypoints, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    st.pyplot(plt)

def extract_features_with_akaze(img_gray):
    akaze = cv2.AKAZE_create()
    kp, des = akaze.detectAndCompute(img_gray, None)
    return kp, des

# We still need the path to display the result images later
dataset_path = './Foto_dataset/'

# --- MODIFIED SECTION: LOAD PRE-EXTRACTED FEATURES ---
features_file_path = 'product_features.pkl'
features = []

if os.path.exists(features_file_path):
    try:
        with open(features_file_path, 'rb') as f:
            features = pickle.load(f)
        # Optional: Print to console to verify load
        print(f"Successfully loaded {len(features)} features from {features_file_path}")
    except Exception as e:
        st.error(f"Error loading pickle file: {e}")
else:
    st.error(f"File '{features_file_path}' not found! Please make sure the pickle file is in the same directory.")
# -----------------------------------------------------

def process_query_image(uploaded_image):
    img_bgr = cv2.imdecode(np.frombuffer(uploaded_image.read(), np.uint8), 1)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    kp, des = extract_features_with_akaze(img_gray)  
    return img_bgr, kp, des

st.set_page_config(page_title="Product Retrieval System", page_icon="🔍", layout="wide")

st.sidebar.title("Options")
uploaded_file = st.sidebar.file_uploader("Upload an image for query", type=["jpg", "jpeg", "png"])
display_keypoints_option = st.sidebar.checkbox("Display Keypoints", value=True)
threshold_slider = st.sidebar.slider("Set minimum matches threshold", 0, 5000, 1000)

st.title("Product Retrieval System")
st.markdown("Upload an image to find the most similar products from the dataset.")

# For storing metrics
all_precision = []
all_recall = []
all_mrr = []
all_std_dev = []

if uploaded_file is not None:
    query_img_bgr, kp_query, des_query = process_query_image(uploaded_file)
    st.image(uploaded_file, caption="Uploaded Image", width=300)
    
    if display_keypoints_option:
        display_image_with_keypoints(query_img_bgr, kp_query, "Query Image (AKAZE)", width=300)

    with st.spinner('Processing...'):
        # Small sleep strictly for UI feedback, remove if you want max speed
        time.sleep(0.5) 

    if des_query is None:
        st.warning("Could not extract features from the query image.")
    else:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        result = []
        y_true = []
        y_pred = []
        ranks = []
        all_matches = []

        # Iterate through loaded features
        for name, des in features:
            if des is None: continue
            
            matches = bf.knnMatch(des_query, des, k=2)
            good_matches = []

            for m, n in matches:
                if m.distance < 0.60 * n.distance:
                    good_matches.append(m)

            distance = len(good_matches)
            result.append([distance, name])

            if distance > 0:
                y_true.append(1)
                y_pred.append(1)
            else:
                y_true.append(0)
                y_pred.append(0)

            # Note: This rank logic assumes the current item is the correct one, 
            # which might need adjustment based on real Ground Truth data.
            # Currently calculating rank based on position in the unsorted result list.
            if [distance, name] in result:
                 ranks.append(1 / (result.index([distance, name]) + 1))  
            
            all_matches.append(distance)

        sorted_results = sorted(result, key=lambda x: x[0], reverse=True)

        for distance, name in sorted_results[:5]:
            # This requires the original images to still be present in the folder
            matched_image_path = os.path.join(dataset_path, name + ".jpg")
            
            if os.path.exists(matched_image_path):
                matched_img_bgr = cv2.imread(matched_image_path)
                kp_matched, des_matched = extract_features_with_akaze(cv2.cvtColor(matched_img_bgr, cv2.COLOR_BGR2GRAY))
                
                st.write(f"Matches: {distance} | Image: {name}")
                display_image_with_keypoints(matched_img_bgr, kp_matched, f"Matched Image: {name} | Matches: {distance}", width=300)
            else:
                st.warning(f"Image {name}.jpg not found in {dataset_path}")

        if len(y_true) > 0:
            precision = precision_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)
            mrr = np.mean(ranks) if ranks else 0
            std_dev = statistics.stdev(all_matches) if len(all_matches) > 1 else 0

            all_precision.append(precision)
            all_recall.append(recall)
            all_mrr.append(mrr)
            all_std_dev.append(std_dev)

        avg_precision = np.mean(all_precision) if all_precision else 0
        avg_recall = np.mean(all_recall) if all_recall else 0
        avg_mrr = np.mean(all_mrr) if all_mrr else 0
        avg_std_dev = np.mean(all_std_dev) if all_std_dev else 0

        st.subheader("Performance Metrics:")
        st.write(f"Average Precision: {avg_precision}")
        st.write(f"Average Recall: {avg_recall}")
        st.write(f"Average Mean Reciprocal Rank (MRR): {avg_mrr}")
        st.write(f"Average Standard Deviation of Matches: {avg_std_dev}")