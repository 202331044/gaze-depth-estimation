# A Study on Correlation of Depth Fixation with Distance Between Dual Purkinje Images and Pupil Size

📄 **Paper**: Electronics, 2025  
🔗 https://doi.org/10.3390/electronics14091799  

---

## Overview

This project investigates a **monocular gaze depth estimation method** based on:

- **Distance between Dual Purkinje Images (DPI)**
- **Pupil size**

The goal is to develop a **low-cost, non-contact, single-camera-based approach** that overcomes the limitations of conventional gaze depth estimation systems.

---

## Background

With the growing demand for **3D gaze tracking** in applications such as AR/VR, HCI, and robotics, **gaze depth estimation** has become an increasingly important research topic.

However, existing methods typically suffer from the following limitations:

- Binocular-based systems  
- Requirement of wearable sensors  
- Use of multiple cameras  

To address these challenges, this study proposes a **monocular gaze depth estimation framework** that satisfies:

- Single-eye (monocular) operation  
- Non-contact, sensor-free measurement  
- Single-camera setup  

---

## Contributions

1. Propose a **monocular gaze depth estimation method** using **DPI distance and pupil size**.
2. Conduct experiments over a **wide gaze depth range (15–60 cm)** at **5 cm intervals**.
3. Demonstrate the feasibility of a **low-cost and non-invasive approach**.

---

## Dataset

- **Total participants**: `11` 
- **Subjects included in analysis**: `8`  

### Gaze Depth Setup

- Range: `15 cm – 60 cm` 
- Interval: `5 cm`
- Total depth levels: `10`

### Data Acquisition

- ~7 seconds of **right-eye video per depth level**
- **5 frames extracted per depth**
  - 3 for training  
  - 2 for testing  

### Image Specifications

- **Original resolution**: `1280 × 720 px`  
- **ROI (Region of Interest)**: `400 × 260 px`  
- **PROI (Purkinje ROI)**:
  - Width: `100, 120, 160, 170 px` (subject-dependent)
  - Height: `150 px`

---

## Feature Extraction

### 1. Pupil Size Detection

**Processing pipeline:**

1. Apply blurring and binary thresholding to the ROI
2. Detect contours  
3. Compute area and perimeter
   - Apply convex hull when criteria are satisfied
   - Extract ellipses from the convex hull only if the convex hull contains enough points
4. Use the length of the fitted ellipse as the pupil size

---
<img src="assets/pupil_detection.jpg" width="800">

### 2. 1st Purkinje Image Detection

**Processing pipeline:**

1. Apply binary thresholding to the PROI
2. Detect contours  
3. Compute area and perimeter
   - Apply convex hull when criteria are satisfied
   - Extract ellipses from the convex hull only if the convex hull contains enough points
4. Use the center of the fitted ellipse as the 1st Purkinje image center

---

### 3. 4th Purkinje Image Detection

Due to its **small size and low brightness**, the **4th Purkinje image** is difficult to detect using contour-based methods.

→ **Template matching** is employed instead.

Use the location of the maximum value in the template matching map as the center.

---
<img src="assets/dpi_detection.jpg" width="800">

## Analysis Methods

1. **Visualization Analysis**
   - 2D scatter plots to visualize relationships among:
     - DPI distance  
     - Pupil size  
     - Gaze depth  

2. **Correlation Analysis**
   - Spearman’s rank correlation analysis

3. **Regression Analysis**
   - Linear regression  
   - Nonlinear regression  
   - Multiple linear regression  

---
<img src="assets/analysis.jpg" width="800">

## Results

### Key Findings

1. **Statistically significant positive correlations** were observed **across all subjects** between the following variable pairs:
   - DPI distance ↔ gaze depth
   - Pupil size ↔ gaze depth
   - DPI distance ↔ pupil size

2. However, regression analysis demonstrated that a **higher correlation does not necessarily lead to better prediction performance**.

3. Based on prediction accuracy (**RMSE and R²**):
   - Models using **DPI distance generally outperformed those based on pupil size**.
   - This is likely because pupil size is influenced by uncontrollable factors such as psychological states, making it less reliable for depth estimation.
   - The **multiple linear regression model achieved higher accuracy than the single linear model** and **showed more stable performance compared to the logistic model**.

4. By **combining DPI distance and pupil size**, the proposed model **achieved consistently high and stable prediction performance**, demonstrating the effectiveness of multi-feature integration.

---

### Best Performing Model

Using data from all subjects, we derived the following **multiple linear regression model**:

$$ \text{Depth Fixation} = 20.746 \cdot \text{DPI Distance} + 5.223 \cdot \text{Pupil Size} + 16.495 \cdot (\text{DPI Distance} \times \text{Pupil Size}) + 13.880 $$

- **R²:** `0.71`  
- **RMSE:** `7.69`  

This model **achieved the best performance among all proposed general models**.
