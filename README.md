---

## 📱 Phone Price Prediction (PMP)

This project builds a machine learning model to predict the **price range of mobile phones** based on technical specifications using a **Random Forest Classifier**. It involves data preprocessing, exploratory analysis, model training, and evaluation with visualizations.

---

### 📁 Project Structure

```
PMP/
├── PMP.py              # Main script for training and evaluation
├── dataset.csv         # Dataset containing mobile phone features and price range
├── README.md           # Project documentation (this file)
```

---

### 📌 Objective

To predict the **price range** (e.g., low, medium, high, very high) of smartphones using various hardware and feature specifications like RAM, battery power, screen dimensions, etc.

---

### 📊 Dataset

The dataset includes various smartphone features such as:

* Battery power
* RAM
* Internal memory
* Primary and front camera megapixels
* Bluetooth, Wi-Fi, Dual SIM
* Screen resolution
* 3G/4G support

Target column: `price_range` (classification into 4 categories: 0 to 3)

---

### 🧪 Steps and Workflow

#### 1. **Import Libraries**

Used libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `sklearn`.

#### 2. **Load Dataset**

```python
df = pd.read_csv("dataset.csv")
```

#### 3. **EDA & Visualization**

* Display basic statistics and data types
* Visualize feature correlations with a heatmap for insights

#### 4. **Data Preparation**

* Split features (`X`) and label (`y`)
* Train-test split with 80-20 ratio
* Feature scaling using `StandardScaler`

#### 5. **Model Building**

* Model: `RandomForestClassifier` with 100 trees
* Trained on scaled training data

#### 6. **Model Evaluation**

* Evaluation metrics: **Accuracy**, **Classification Report**, **Confusion Matrix**
* Accuracy score displayed
* Feature importance plot for interpretability

---

### 📈 Example Output

* Heatmap of feature correlation
* Bar plot of top 10 important features
* Classification metrics like precision, recall, f1-score
* Confusion matrix

---

### 🧰 Requirements

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

---

### 🚀 How to Run

1. Clone the repository or download the code.
2. Ensure `dataset.csv` is in the root folder.
3. Run the script:

   ```bash
   python PMP.py
   ```

---

### 🔍 Future Enhancements

* Hyperparameter tuning (GridSearchCV)
* Model comparison (SVM, XGBoost, etc.)
* Deploy as a web app using Streamlit or Flask

---


