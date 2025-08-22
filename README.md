# 🧠 Brain Tumor Segmentation (Glioma) using U-Net  

## 📌 Giới thiệu  
Dự án này tập trung vào **nhiệm vụ phân vùng (segmentation) khối u glioma** trên ảnh chụp cộng hưởng từ (MRI) tiền phẫu thuật.  
Mục tiêu chính: gán nhãn chính xác cho từng điểm ảnh (pixel) để phân biệt vùng u và mô não khỏe mạnh, từ đó hỗ trợ:  
- Đánh giá kích thước, vị trí khối u.  
- Lập kế hoạch điều trị (phẫu thuật, xạ trị).  
- Theo dõi sự phát triển của bệnh theo thời gian.  

## 🎯 Định nghĩa bài toán  
Mỗi pixel trong ảnh MRI được phân loại vào một trong các nhãn:  
- **0**: Vùng không phải khối u.  
- **1**: Hoại tử & khối u không tăng quang (NCR & NET).  
- **2**: Phù nề (ED).  
- **4**: Khối u tăng quang (ET).  

Từ các nhãn này, ba vùng chính được xác định để đánh giá:  
- **ET (Enhancing Tumor)** = Nhãn 4.  
- **TC (Tumor Core)** = Nhãn 1 + 4.  
- **WT (Whole Tumor)** = Nhãn 1 + 2 + 4.  

## 📊 Dữ liệu  
- Dữ liệu lấy từ **BraTS (Brain Tumor Segmentation Challenge)**.  
- Ảnh MRI nhiều chuỗi (T1, T1CE, T2, FLAIR) + nhãn mask.  
- Định dạng: `.nii.gz`.  

## 🏗️ Kiến trúc mô hình  
Sử dụng **U-Net** – một CNN encoder-decoder với skip-connections, phù hợp cho phân đoạn y tế.  
- **Encoder**: Trích xuất đặc trưng qua convolution + pooling.  
- **Decoder**: Upsampling + nối skip-connection để khôi phục chi tiết.  
- **Output**: Softmax cho phân đoạn đa lớp.  

## ⚙️ Cài đặt  
```bash
git clone https://github.com/<username>/brain_tumor_segmentation.git
cd brain_tumor_segmentation
pip install -r requirements.txt
```

## 🚀 Huấn luyện  
```bash
python src/train.py --epochs 50 --batch_size 8 --lr 1e-4 --data_dir ./data
```

## 📈 Đánh giá  
Chỉ số sử dụng:  
- Dice Coefficient  
- IoU (Intersection over Union)  
- Precision, Recall  

Ví dụ kết quả:  

| MRI Gốc | Mask Ground Truth | Dự đoán (U-Net) |
|---------|------------------|------------------|
| ![](results/mri.png) | ![](results/mask.png) | ![](results/pred.png) |

## 🔮 Hướng phát triển  
- Thử nghiệm **Attention U-Net, ResUNet, 3D U-Net**.  
- Áp dụng **data augmentation nâng cao**.  
- Xây dựng ứng dụng web demo inference.  

## 📜 Tham khảo  
- Ronneberger O., Fischer P., Brox T. *U-Net: Convolutional Networks for Biomedical Image Segmentation*. MICCAI 2015.  
- [BraTS Challenge Dataset](https://www.med.upenn.edu/cbica/brats2020/data.html)  
