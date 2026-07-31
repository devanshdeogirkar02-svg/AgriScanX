# 🌱 AgriScanX

> **AI-Powered Edge Platform for RGB-to-Multispectral Reconstruction and NDVI-Based Crop Health Monitoring**

AgriScanX is a deep learning–based Edge AI platform that reconstructs multispectral images from standard RGB images and generates vegetation health indices such as NDVI for crop monitoring. The project is designed to make precision agriculture more accessible by reducing the need for expensive multispectral cameras while enabling deployment on resource-constrained edge devices like the Raspberry Pi.

---

## 📖 Overview

Traditional crop health monitoring relies on multispectral imaging systems that capture spectral bands beyond the visible spectrum, such as Near-Infrared (NIR) and Red Edge. These systems provide valuable information for vegetation analysis but are often expensive and difficult to deploy at scale.

AgriScanX addresses this challenge by using a Convolutional Neural Network (CNN) to predict multispectral information from a single RGB image. The predicted spectral bands are then used to compute vegetation indices such as NDVI, allowing farmers and researchers to assess crop health using affordable imaging hardware.

---

## ❓ Problem Statement

Multispectral cameras are widely used in precision agriculture because they capture spectral information that cannot be obtained from conventional RGB cameras. However, their high cost limits adoption, particularly in small-scale farming and educational environments.

The objective of AgriScanX is to develop an AI-powered system capable of reconstructing multispectral imagery from RGB inputs while maintaining sufficient quality for vegetation analysis and deployment on edge devices.

---

## ✨ Planned Features

- RGB to Multispectral Image Reconstruction
- NDVI Generation
- FastAPI-based REST API
- Image Upload and Prediction
- Edge AI Deployment (Raspberry Pi)
- Optimized Deep Learning Inference
- Modular Python Codebase
- Professional Documentation

---

## 🏗️ Project Architecture

```

RGB Image
│
▼
Image Preprocessing
│
▼
Deep Learning Model
│
▼
Multispectral Prediction
│
▼
NDVI Calculation
│
▼
Visualization
│
▼
REST API Response

```

---

## 📂 Project Structure

```text
AgriScanX/
│
├── app/
├── assets/
├── configs/
├── docs/
├── models/
├── outputs/
├── sample_images/
├── tests/
│
├── README.md
├── requirements.txt
├── LICENSE
└── .gitignore
```

---

## 🛠️ Tech Stack

| Category | Technologies |
|-----------|--------------|
| Language | Python |
| Deep Learning | TensorFlow |
| Computer Vision | OpenCV |
| Backend | FastAPI |
| Numerical Computing | NumPy |
| Visualization | Matplotlib |
| Deployment | Raspberry Pi |
| Version Control | Git & GitHub |

---

## 🚧 Current Status

The project is currently under active development.

### Completed

- Project planning
- Repository setup
- Modular project architecture
- Documentation structure

### In Progress

- Model reconstruction
- Inference pipeline
- FastAPI backend

### Planned

- NDVI visualization
- REST API
- Docker support
- Edge deployment
- Performance optimization

---

## 🎯 Future Improvements

- Support for multiple vegetation indices (EVI, SAVI, GNDVI)
- Batch image processing
- Web dashboard
- ONNX model optimization
- Mobile deployment
- Cloud inference support

---

## 👨‍💻 Author

**Devansh Deogirkar**

Electronics & Communication Engineering Student

Interested in:
- Artificial Intelligence
- Computer Vision
- Edge AI
- Backend Development
- System Design

---

## ⭐ Star this repository

If you find this project useful, consider giving it a ⭐.