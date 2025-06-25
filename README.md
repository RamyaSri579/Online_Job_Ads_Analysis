# Online_Job_Ads_Analysis
# 🧠 Job Title Classification from Online Job Ads using Deep Learning

This project presents a **two-stage job title identification methodology** for analyzing online job advertisements, especially in small or regional datasets like those from the Moroccan job market. By combining **BERT-based sector classification** with **unsupervised learning and document embedding**, this approach significantly improves job title prediction accuracy.

---

## 📌 Project Motivation

Classifying job titles from job ads is a valuable task in labor market analytics but often requires massive labeled datasets and systems optimized for specific markets (e.g., US-centric datasets like O*NET). Our goal is to:
- Build a scalable, accurate classification pipeline for **small datasets**.
- Improve prediction performance using **deep learning** and **embedding strategies**.
- Extend prior work by incorporating **advanced neural networks like CNN2D** for better feature extraction.

---

## 🔍 Problem Statement

Traditional methods (SVM, Naive Bayes, Logistic Regression) do not always yield optimal results for small-scale, non-US datasets. Moreover, prior research didn't explore modern deep learning approaches like CNNs or BI-LSTMs.

---

## 🚀 Proposed Solution

### 🔸 Stage 1: Sector Classification
- Use **BERT** (Bidirectional Encoder Representations from Transformers) to classify job ads into sectors (e.g., IT, Agriculture, Healthcare).

### 🔸 Stage 2: Job Title Identification
- Apply **unsupervised learning** and **document similarity techniques** to identify the most relevant job title within the predicted sector.
- Incorporate **custom document embedding strategies** including:
  - Weighted embeddings
  - Noise removal
  - Semantic similarity measures

### 🔸 Extension: Deep Learning with CNN2D
- We experiment with **CNN2D**, which outperforms traditional ML algorithms.
- CNN2D filters and learns optimal features via convolutional layers across multiple iterations, leading to **higher classification accuracy**.

---

## 📈 Results

- Our **two-stage BERT + unsupervised approach** achieves up to **85% accuracy** in job title prediction.
- **Embedding enhancements** improve classification accuracy by **23.5%** compared to Bag-of-Words.
- **CNN2D** further boosts performance by extracting deeper feature representations from the data.

---

## 🌍 Application

Applied on **Moroccan job advertisement datasets**, our approach:
- Identifies **emerging occupations**.
- Tracks **in-demand job roles**.
- Supports local job market analytics and decision-making.

---

## 🛠️ Technologies Used

- BERT (Transformers)
- Python (NumPy, pandas, scikit-learn)
- CNN2D (TensorFlow / Keras)
- Unsupervised Learning (Clustering, Cosine Similarity)
- Document Embeddings (TF-IDF, Doc2Vec)

---


