import tensorflow as tf
import cv2
import numpy as np
import streamlit as st

# Định nghĩa kích thước ảnh đầu vào cho model
IMG_SIZE = 128

# Định nghĩa mapping giữa số lớp và tên lớp
SEGMENT_CLASSES = {
    0: 'NOT tumor',
    1: 'NECROTIC/CORE',
    2: 'EDEMA',
    3: 'ENHANCING'
}

# Hàm tải model đã huấn luyện từ file .h5
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(
        "E:/Student/N4/H2/Computer_Vision/CK/model/3D_MRI_Brain_tumor_segmentation.h5",
        custom_objects={
            'dice_coef': lambda y_true, y_pred: tf.constant(0), # Các hàm loss/metrics tùy chỉnh (có thể cần định nghĩa lại nếu cần)
            'precision': lambda y_true, y_pred: tf.constant(0),
            'sensitivity': lambda y_true, y_pred: tf.constant(0),
            'specificity': lambda y_true, y_pred: tf.constant(0),
            'dice_coef_necrotic': lambda y_true, y_pred: tf.constant(0),
            'dice_coef_edema': lambda y_true, y_pred: tf.constant(0),
            'dice_coef_enhancing': lambda y_true, y_pred: tf.constant(0),
        }
    )
    return model

# Hàm tiền xử lý một slice ảnh đầu vào cho model
def preprocess_slice(image_slice):
    # Resize slice ảnh về kích thước mà model mong đợi
    resized_slice = cv2.resize(image_slice, (IMG_SIZE, IMG_SIZE)).astype(np.float32)
    # Chuẩn hóa giá trị pixel về khoảng [0, 1]
    normalized_slice = resized_slice / np.max(resized_slice) if np.max(resized_slice) > 0 else resized_slice
    # Tạo input tensor cho model (thêm một chiều batch và kênh thứ hai là zero padding)
    input_tensor = np.expand_dims(np.stack([normalized_slice, np.zeros_like(normalized_slice)], axis=-1), axis=0)
    return input_tensor

# Hàm thực hiện dự đoán trên một slice ảnh đã tiền xử lý
def predict_slice(model, image_slice):
    # Tiền xử lý slice ảnh
    preprocessed_image = preprocess_slice(image_slice)
    # Thực hiện dự đoán bằng model
    prediction = model.predict(preprocessed_image)
    return prediction[0] # Trả về output dự đoán cho slice đó