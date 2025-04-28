import streamlit as st
import numpy as np
import nibabel as nib
import cv2
from PIL import Image
import io
import tempfile
import os
from model_utils import load_model, predict_slice, IMG_SIZE, SEGMENT_CLASSES

st.title("🧠 Brain tumor segmentation")

model = load_model()

# Tạo bộ tải file để người dùng tải lên file ảnh FLAIR định dạng .nii
uploaded_flair_file = st.file_uploader("Tải lên file ảnh FLAIR (.nii)", type=["nii"])

# Kiểm tra xem người dùng đã tải lên file hay chưa
if uploaded_flair_file is not None:
    try:
        # Đọc nội dung của file đã tải lên dưới dạng bytes
        nifti_bytes = uploaded_flair_file.read()

        # Tạo một file tạm trên hệ thống để lưu nội dung file đã tải lên
        # Điều này cần thiết vì nibabel thường làm việc tốt nhất với đường dẫn file
        with tempfile.NamedTemporaryFile(suffix=".nii", delete=False) as tmp_file:
            tmp_file.write(nifti_bytes)
            temp_file_path = tmp_file.name

        # Hiển thị đường dẫn của file tạm (cho mục đích debug nếu cần)
        st.info(f"Đường dẫn file tạm: {temp_file_path}")

        # Kiểm tra xem file tạm có thực sự tồn tại trên hệ thống hay không
        if not os.path.exists(temp_file_path):
            st.error(f"Lỗi: File tạm không tồn tại tại {temp_file_path}")
            raise FileNotFoundError(f"File tạm không tồn tại tại {temp_file_path}")

        # Sử dụng nibabel để load dữ liệu ảnh NIfTI từ file tạm
        try:
            flair_data = nib.load(temp_file_path).get_fdata()
        except Exception as e:
            st.error(f"Lỗi khi load file NIfTI bằng nibabel: {e}")
            os.unlink(temp_file_path) # Xóa file tạm nếu xảy ra lỗi khi load
            raise e

        # Xóa file tạm sau khi đã load dữ liệu thành công vào bộ nhớ
        os.unlink(temp_file_path)

        # Xác định số lượng slice tối đa trong volume và cho phép người dùng chọn slice để xem
        max_slice = flair_data.shape[2] - 1
        slice_index = st.slider("Chọn slice", 0, max_slice, flair_data.shape[2] // 2)

        # Sử dụng hàm predict_slice từ model_utils.py để tiền xử lý và dự đoán trên slice đã chọn
        prediction = predict_slice(model, flair_data[:, :, slice_index])
        predicted_mask = np.argmax(prediction, axis=-1) # Lấy lớp có xác suất cao nhất cho mỗi pixel

        # Lấy các kênh dự đoán riêng lẻ cho từng lớp khối u
        core_pred = prediction[:, :, 1]
        edema_pred = prediction[:, :, 2]
        enhancing_pred = prediction[:, :, 3]

        # Hàm tạo ảnh overlay (kết hợp ảnh nền và mask dự đoán với màu và độ trong suốt)
        def create_overlay(background, mask, color=(255, 0, 0), alpha=0.5): # Thay đổi màu thành Red và tăng độ trong suốt
            img = np.stack([background] * 3, axis=-1).astype(np.uint8) # Chuyển ảnh nền thành ảnh RGB
            overlay = np.zeros_like(img, dtype=np.uint8) # Tạo mask overlay có cùng kích thước
            overlay[mask > 0] = color # Gán màu cho các pixel thuộc vùng mask
            return Image.blend(Image.fromarray(img), Image.fromarray(overlay), alpha) # Trộn ảnh nền và overlay

        # Tiền xử lý slice FLAIR để hiển thị
        normalized_flair_slice = cv2.resize(flair_data[:, :, slice_index], (IMG_SIZE, IMG_SIZE)).astype(np.float32) / np.max(flair_data[:, :, slice_index]) if np.max(flair_data[:, :, slice_index]) > 0 else cv2.resize(flair_data[:, :, slice_index], (IMG_SIZE, IMG_SIZE)).astype(np.float32)
        flair_pil = Image.fromarray((normalized_flair_slice * 255).astype(np.uint8)).convert("RGB") # Chuyển thành định dạng PIL để hiển thị

        # Tạo mask màu cho tất cả các lớp dự đoán
        all_classes_mask = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        colors = [(255, 0, 0), (0, 0, 255), (255, 255, 0)] # Red, Blue, Yellow
        for i, color in enumerate(colors):
            all_classes_mask[predicted_mask == i + 1] = color
        all_classes_pil = Image.blend(flair_pil, Image.fromarray(all_classes_mask), 0.5) # Tăng độ trong suốt

        # Tạo overlay cho từng lớp khối u riêng lẻ
        core_overlay_pil = create_overlay((normalized_flair_slice * 255).astype(np.uint8), predicted_mask == 1, color=(255, 0, 0)) # Red
        edema_overlay_pil = create_overlay((normalized_flair_slice * 255).astype(np.uint8), predicted_mask == 2, color=(0, 0, 255)) # Blue
        enhancing_overlay_pil = create_overlay((normalized_flair_slice * 255).astype(np.uint8), predicted_mask == 3, color=(255, 255, 0)) # Yellow

        # Hiển thị các hình ảnh kết quả trên giao diện Streamlit
        st.subheader("Predicted results")
        col1, col2, col3 = st.columns(3)
        col1.image(flair_pil, caption="Original image flair", use_container_width=True)
        col2.image(all_classes_pil, caption="All classes", use_container_width=True)
        col3.image(core_overlay_pil, caption=f"{SEGMENT_CLASSES[1]} Predicted", use_container_width=True)
        col1, col2, col3 = st.columns(3)
        col1.empty()
        col2.image(edema_overlay_pil, caption=f"{SEGMENT_CLASSES[2]} Predicted", use_container_width=True)
        col3.image(enhancing_overlay_pil, caption=f"{SEGMENT_CLASSES[3]} Predicted", use_container_width=True)

    except Exception as e:
        # Hiển thị thông báo lỗi nếu có bất kỳ vấn đề nào xảy ra trong quá trình xử lý
        st.error(f"Đã xảy ra lỗi: {e}")


# Use "streamlit run app.py" in terminal to run :>>>>