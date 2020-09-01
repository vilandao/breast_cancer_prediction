import streamlit as st
import pandas as pd
import numpy as np
import pickle
import io

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

st.set_option('deprecation.showfileUploaderEncoding', False)

st.markdown("""
# Breast Cancer Prediction App
This app predicts whether a cancer cell is benign or malignant based on
a variety of cell attributes that were computed for each cell nucleus such
as:
* radius (mean of distances from center to points on the perimeter)
* texture (standard deviation of gray-scale values)
* perimeter
* area
* smoothness (local variation in radius lengths)
* compactness (perimeter^2 / area - 1.0)
* concavity (severity of concave portions of the contour)
* concave points (number of concave portions of the contour)
* symmetry
* fractal dimension ("coastline approximation" - 1)
""")

st.markdown("""
Data obtained from [Kaggle](https://www.kaggle.com/uciml/breast-cancer-wisconsin-data)
""")

st.header('User Input Features')

# Collects user input features into dataframe
uploaded_file = st.file_uploader("Upload your input CSV file", type=["csv"])

if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        radius_mean = st.slider('radius_mean',5,30,14)
        texture_mean = st.slider('texture_mean',8,42,20 )
        smoothness_mean = st.slider('smoothness_mean',0.05,0.2,0.1)
        compactness_mean = st.slider('compactness_mean',0.01,0.4,0.1)
        symmetry_mean = st.slider('symmetry_mean',0.1,0.4,0.2)
        fractal_dimension_mean = st.slider('fractal_dimension_mean',0.03,0.1,0.06)
        radius_se = st.slider('radius_se',0.05,2.5,0.5)
        texture_se = st.slider('texture_se',0.25,5.0,1.5)
        smoothness_se = st.slider('smoothness_se',0.001,0.04,0.01)
        compactness_se = st.slider('compactness_se',0.002,0.15,0.03)
        concavity_se = st.slider('concavity_se',0.0,0.4,0.2)
        concave_points_se = st.slider('concave points_see',0.0,0.6,0.3)
        symmetry_se = st.slider('symmetry_se',0.005,0.08,0.02)
        fractal_dimension_se = st.slider('fractal_dimension_se',0.0005,0.04,0.004)
        smoothness_worst = st.slider('smoothness_worst',0.05,0.3,0.13)
        symmetry_worst = st.slider('symmetry_worst',0.1,0.8,0.3)
        fractal_dimension_worst = st.slider('fractal_dimension_worstt',0.04,0.3,0.08)
        data = {'radius_mean': radius_mean,
                'texture_mean': texture_mean,
                'smoothness_mean': smoothness_mean,
                'compactness_mean':compactness_mean,
                'symmetry_mean': symmetry_mean,
                'fractal_dimension_mean': fractal_dimension_mean,
                'radius_se': radius_se,
                'texture_se': texture_se,
                'smoothness_se': smoothness_se,
                'compactness_se': compactness_se,
                'concavity_se': concavity_se,
                'concave points_se':concave_points_se,
                'symmetry_se': symmetry_se,
                'fractal_dimension_se': fractal_dimension_se,
                'smoothness_worst':smoothness_worst,
                'symmetry_worst': symmetry_worst,
                'fractal_dimension_worst':fractal_dimension_worst
                }
        features = pd.DataFrame(data, index=[0])
        return features

    input_df = user_input_features()

# Combines user input features with entire breast_cancer dataset
df_cleaned = pd.read_csv('df_cleaned.csv')
df_cleaned= df_cleaned.drop(columns=['diagnosis'])
df = pd.concat([input_df,df_cleaned],axis=0)

# Scale the dataset
scaler = StandardScaler()
scaled_df = scaler.fit_transform(df)
scaled_df = pd.DataFrame(scaled_df, columns=df.columns)

df = scaled_df[:1]

# Displays the user input features
st.subheader('User Input features')

if uploaded_file is not None:
    st.write(df)
else:
    st.write('Awaiting CSV file to be uploaded. Currently using example input parameters (shown below) after performing StandardScaler')
    st.write(df)

# Perform PCA on the dataset
pca = PCA(n_components=8, svd_solver='randomized',whiten=True)
pca_components = pca.fit(scaled_df)
pca_df = pca.transform(scaled_df)

pca_df_frst = pca_df[:1]

# Reads in saved classification model
load_clf = pickle.load(open('breast_cancer_lr.pkl', 'rb'))

# Apply model to make predictions
prediction = load_clf.predict(pca_df_frst)

st.subheader('Prediction (1 = malignant, 0 = benign)')
results = np.array([0,1])
st.write(results[prediction])
