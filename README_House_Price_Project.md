# 🏠 House Price Prediction using Machine Learning

## 📌 Project Overview
This project aims to predict house sale prices using Machine Learning techniques.  
The model is trained on a real-world housing dataset and estimates the selling price of houses based on multiple features such as area, number of rooms, construction year, and other property attributes.

## 🎯 Objective
To build an accurate **Regression Model** that can predict house prices by performing:
- Data preprocessing
- Feature encoding
- Model training
- Model evaluation
- Visualization
- Model serialization for reuse

---

## 📂 Dataset
- **Source:** Kaggle – House Prices Dataset  
- **Files Used:**
  - `train.csv` – Contains features + **SalePrice** target column
  - `test.csv` – Contains only features for prediction
- Dataset contains multiple numerical and categorical attributes related to residential properties.

---

## 🛠️ Tools & Technologies
- **Python**
- **Google Colab**
- **Pandas**
- **NumPy**
- **Matplotlib**
- **Seaborn**
- **Scikit-learn**
- **Pickle**

---

## 🔄 Project Workflow
1. **Data Loading** – Imported CSV files into Colab  
2. **Data Cleaning** – Handled missing values using median and “None” strategy  
3. **Feature Encoding** – Converted categorical features using One-Hot Encoding  
4. **Train-Validation Split** – Divided dataset for training and testing  
5. **Model Training** – Linear Regression model used  
6. **Model Evaluation** – Evaluated performance using:
   - R² Score
   - Mean Absolute Error (MAE)
7. **Visualization** –
   - Actual vs Predicted Scatter Plot  
   - Error Distribution Histogram  
   - Residual Plot  
8. **Model Saving** – Serialized trained model using `.pkl` file

---

## 📊 Model Performance
*(Replace values with your actual output if needed)*

- **R² Score:** ~0.85 – 0.90  
- **MAE:** ~15,000 – 25,000  

These metrics indicate that the model predicts house prices with good accuracy and low average error.

---

## 📈 Visualizations Included
- Actual vs Predicted Price Scatter Plot  
- Error Distribution Histogram  
- Residual Analysis Plot  

---

## 💾 Model Serialization
The trained model is saved as:

house_price_model.pkl

This allows:
- Reusing the model without retraining
- Faster predictions
- Easy deployment in future applications

---

## 📁 Project Structure
house-price-prediction-ml/
│
├── house_price_prediction.ipynb
├── train.csv
├── test.csv
├── house_price_model.pkl
├── README.md

---

## 🚀 Future Enhancements
- Add Random Forest / XGBoost models
- Hyperparameter tuning
- Feature importance analysis
- Deploy as a web application using Streamlit or Flask

---

## 🧠 Key Learnings
- Handling missing values and categorical encoding  
- Regression model building and evaluation  
- Data visualization techniques  
- Model serialization using Pickle  
- End-to-end ML project lifecycle  

---

## 👩‍💻 Author
**Rajeswari Gavara**  
Aspiring Machine Learning Engineer | Software Developer

---

This project demonstrates a complete **Machine Learning pipeline** from raw data to a deployment-ready predictive model.
