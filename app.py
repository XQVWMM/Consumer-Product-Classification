import cv2
import numpy as np
import os
import streamlit as st
import matplotlib.pyplot as plt
import time

def display_image_with_keypoints(img_bgr, keypoints, title="Image"):
    img_keypoints = cv2.drawKeypoints(img_bgr, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    plt.imshow(cv2.cvtColor(img_keypoints, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    st.pyplot(plt)

def extract_features_with_akaze(img_gray):
    akaze = cv2.AKAZE_create()
    kp, des = akaze.detectAndCompute(img_gray, None)
    return kp, des

dataset_path = './Foto_dataset/'
features = []

for filename in os.listdir(dataset_path): 
    img_name = filename.split(".")[0]
    img = cv2.imread(dataset_path + "/" + filename)
    
    if img is None:
        continue
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kp, des = extract_features_with_akaze(img_gray)  
    if des is None:
        continue
    features.append([img_name, des])

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


if uploaded_file is not None:
    query_img_bgr, kp_query, des_query = process_query_image(uploaded_file)
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    
    if display_keypoints_option:
        display_image_with_keypoints(query_img_bgr, kp_query, "Query Image (AKAZE)")

    with st.spinner('Processing...'):
        time.sleep(2)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    result = []
    
    for name, des in features:
        matches = bf.match(des_query, des)
        distance = len(matches)
        if distance >= threshold_slider: 
            result.append([distance, name])
    
    sorted_results = sorted(result, key=lambda x: x[0], reverse=True)

    for distance, name in sorted_results[:5]:
        matched_image_path = dataset_path + "/" + name + ".jpg"
        matched_img_bgr = cv2.imread(matched_image_path)
        kp_matched, des_matched = extract_features_with_akaze(cv2.cvtColor(matched_img_bgr, cv2.COLOR_BGR2GRAY))
        
        st.write(f"Matches: {distance} | Image: {name}")
        display_image_with_keypoints(matched_img_bgr, kp_matched, f"Matched Image: {name} | Matches: {distance}")

    match_counts = [len(bf.match(des_query, des)) for _, des in features]
    st.bar_chart(match_counts)
