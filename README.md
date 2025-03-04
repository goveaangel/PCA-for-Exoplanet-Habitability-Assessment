# Exoplanet Habitability Analysis using PCA

## 📌 Project Overview
This project applies **Principal Component Analysis (PCA)** to analyze exoplanets and identify the most habitable candidates. By leveraging multivariate statistical techniques, we reduce the dimensionality of exoplanet datasets while preserving critical information relevant to habitability. 

## 📊 Dataset
We utilized data from the **NASA Exoplanet Archive**, which initially contained **36,490 observations** and **91 variables** describing various planetary characteristics. After data preprocessing and cleaning, we refined the dataset to **1,182 observations** and **15 key variables**.

## 🔬 Methodology
### 1️⃣ Data Cleaning & Preprocessing
- Removed missing values and irrelevant variables.
- Standardized numerical features to ensure PCA effectiveness.

### 2️⃣ Exploratory Data Analysis (EDA)
- Constructed **correlation matrices** to understand relationships between planetary features.
- Identified variable dependencies to optimize feature selection.

### 3️⃣ Principal Component Analysis (PCA)
- Implemented PCA using `scikit-learn` to reduce data dimensions.
- Retained **9 principal components** that explain **96% of the total variance**.
- Validated PCA results using **correlation matrices and eigenvector analysis**.

### 4️⃣ Habitability Scoring System
- Developed a **custom scoring algorithm** to evaluate exoplanet habitability.
- Assigned weights to each **Principal Component (PC)** based on its contribution to habitability.
- Established **optimal value ranges** for each PC based on scientific literature.

### 5️⃣ Regression Analysis
- Implemented **multiple linear regression** to predict habitability scores.
- Achieved a model with an **R² of 0.89**, indicating strong predictive power.

## 🚀 Key Findings
- The **top 10 most habitable exoplanets** were identified, with **HATS-38 b** achieving the highest habitability score of **0.934**.
- PCA revealed that **temperature, planetary mass, and orbital period** are key determinants of habitability.
- Regression analysis confirmed the strong influence of these variables in habitability estimation.

## 📂 Imnportant Files in This Repository
- `Habitabilidad_FINAL.ipynb` - Jupyter Notebook containing the full implementation (data preprocessing, PCA, habitability analysis, and regression model).
- `PCA_Report.pdf` - A detailed report summarizing methodology, results, and insights.

## 🔧 Technologies Used
- **Python** (Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn)
- **Jupyter Notebook**
- **NASA Exoplanet Archive** (Data Source)

## 👨‍💻 How to Use
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/your-repository.git
   ```
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Jupyter Notebook:
   ```bash
   jupyter notebook Habitabilidad_FINAL.ipynb
   ```

## 🏆 Project Impact
This project showcases **data science, machine learning, and astrophysics applications** in real-world datasets. The habitability scoring system provides an **efficient way to prioritize exoplanets for further research**.

## 📩 Contact
For inquiries, feel free to reach out via GitHub or email.

---
*This project was conducted as part of an academic research initiative at Tecnológico de Monterrey.*
