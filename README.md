# IbPRIA2015 - Data-Efficient Strategies for Object Detection

Welcome to this hands-on tutorial presented at **IbPRIA 2025**.

In this session, we walk through the full process of building a **custom object detection model** using YOLO, while demonstrating the power of **Transfer Learning (TL)**, **Active Learning (AL)**, and the **importance of data quality**.

Our example task is to detect **whitefly pests in tomato greenhouses** — a real-world problem that demands accurate, efficient, and cost-effective detection systems.

---

## 🎯 Tutorial Goals

By the end of this tutorial, you will:

- ✅ Train a YOLO-based object detector on custom data
- 🧪 Use **Active Learning** to reduce annotation workload
- 📊 Understand how annotation **quality affects model performance**
- 🚀 Apply **Transfer Learning** to boost performance and reduce training time
- 🔁 Build a deployable model with efficient training strategies

---

## ⚙️ How to Use This Tutorial

### 🔗 Run in Google Colab (Recommended)

You can run this tutorial in the cloud via Google Colab — **no setup required**:

[![Open in Colab](https://colab.research.google.com/github/dinisdcosta/IbPRIA2025---Data-Efficient-Strategies-for-Object-Detection/blob/main/hands_on_notebook.ipynb)

---

### 💻 Run Locally

    
    git clone https://github.com/dinisdcosta/IbPRIA2025---Data-Efficient-Strategies-for-Object-Detection 
    cd IbPRIA2025---Data-Efficient-Strategies-for-Object-Detection

---

## 📁 Project Structure

```text
├── dataset/
│   ├── original/                 # Original images and YOLO-format labels
│   ├── improved/                 # Same images but with improved (corrected) annotations
│   ├── test/                     # Test set (fixed)
│   ├── data.yaml                 # Dataset config file for YOLOv8-YOLOv12
│   └── data_yolov5.yaml          # Dataset config file adapted for YOLOv5
│
├── examples/                     # Media files used for inference demonstrations
│   ├── example_video.mp4
│   └── example_image.jpg
│
├── utils/                        # Utility scripts for processing, training, and AL
│   ├── dataset_splits.py
│   └── object_detection.py
│
├── pre-trained/                  # Pre-trained weights for TL experiments
│   ├── yellow_trap.pt            # YOLOv5 model trained on yellow sticky trap images       
│
├── hands_on_notebook.ipynb       # 📘 Main tutorial notebook
├── requirements.txt              # Dependencies list 
└── README.md                     # This file

```

---

**Enjoy the tutorial! 🚀**