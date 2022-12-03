#%%load package
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
import shap
import sklearn
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import joblib

#%%不提示warning信息
st.set_option('deprecation.showPyplotGlobalUse', False)
#%%set title
st.set_page_config(page_title='Prediction model for Ocular metastasis of lung cancer')
st.title('Machine Learning for Eye Metastasis of Primary Lung Cancer: development and Verification of Predictive Model')
st.sidebar.markdown('## Variables')
Histopathological_type = st.sidebar.selectbox('Histopathological_type',('Squamous cell carcinoma','Adenocarcinoma','Large cell carcinoma',
                                                                        'Small cell lung cancer','Other non-small cell lung cancer','Unkown'),index=1)
AFP = st.sidebar.slider("AFP(μg/L)", 0.00, 20.00, value=7.00, step=0.01)
CEA = st.sidebar.slider("CEA(μg/L)", 0.00, 1000.00, value=400.00, step=0.01)
CA_125 = st.sidebar.slider("CA_125(μg/L)", 0.00, 1000.00, value=800.00, step=0.01)
CA_199 = st.sidebar.slider("CA_199(μg/L)", 0.00, 1000.00, value=500.00, step=0.01)
CA_153 = st.sidebar.slider("CA_153(μg/L)", 0.00, 500.00, value=200.00, step=0.01)
CYFRA21_1 = st.sidebar.slider("CYFRA21_1(μg/L)", 0.00, 15.00, value=5.00, step=0.01)
TPSA = st.sidebar.slider("TPSA(μg/L)", 0, 1000, value=215, step=1)

#分割符号
st.sidebar.markdown('#  ')
st.sidebar.markdown('#  ')
st.sidebar.markdown('##### All rights reserved') 
st.sidebar.markdown('##### For communication and cooperation, please contact wshinana99@163.com, Wu Shi-Nan, Nanchang university')
#传入数据
map = {'Squamous cell carcinoma':1,'Adenocarcinoma':2,'Large cell carcinoma':3,'Small cell lung cancer':4,'Other non-small cell lung cancer':5,'Unkown':6}

Histopathological_type =map[Histopathological_type]

#%%load model
xgb_model = joblib.load('xgb_model_lung_cancer_eye.pkl')
XGB_model = xgb_model
#%%load data
hp_train = pd.read_csv('lung_cancer_githubdata.csv')
hp_train['M'] = hp_train['M'].apply(lambda x : +1 if x==1 else 0)
features =["Histopathological_type","AFP","CEA","CA_125","CA_199","CA_153",'CYFRA21_1','TPSA']
target = 'M'
y = np.array(hp_train[target])
sp = 0.5
#figure
is_t = (XGB_model.predict_proba(np.array([[Histopathological_type,AFP,CEA,CA_125,CA_199,CA_153,CYFRA21_1,TPSA]]))[0][1])> sp
prob = (XGB_model.predict_proba(np.array([[Histopathological_type,AFP,CEA,CA_125,CA_199,CA_153,CYFRA21_1,TPSA]]))[0][1])*1000//1/10


if is_t:
    result = 'High Risk Ocular metastasis'
else:
    result = 'Low Risk Ocular metastasis'
if st.button('Predict'):
    st.markdown('## Result:  '+str(result))
    if result == '  Low Risk Ocular metastasis':
        st.balloons()
    st.markdown('## Probability of High Risk Ocular metastasis group:  '+str(prob)+'%')
    #%%cbind users data
    col_names = features
    X_last = pd.DataFrame(np.array([[Histopathological_type,AFP,CEA,CA_125,CA_199,CA_153,CYFRA21_1,TPSA]]))
    X_last.columns = col_names
    X_raw = hp_train[features]
    X = pd.concat([X_raw,X_last],ignore_index=True)
    if is_t:
        y_last = 1
    else:
        y_last = 0
    
    y_raw = (np.array(hp_train[target]))
    y = np.append(y_raw,y_last)
    y = pd.DataFrame(y)
    model = xgb_model
    #%%calculate shap values
    sns.set()
    explainer = shap.Explainer(model, X)
    shap_values = explainer.shap_values(X)
    a = len(X)-1
    #%%SHAP Force logit plot
    st.subheader('SHAP Force logit plot of XGB model')
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    force_plot = shap.force_plot(explainer.expected_value,
                    shap_values[a, :], 
                    X.iloc[a, :], 
                    figsize=(25, 3),
                    # link = "logit",
                    matplotlib=True,
                    out_names = "Output value")
    st.pyplot(force_plot)
    #%%SHAP Water PLOT
    st.subheader('SHAP Water plot of XGB model')
    shap_values = explainer(X) # 传入特征矩阵X，计算SHAP值
    fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
    waterfall_plot = shap.plots.waterfall(shap_values[a,:])
    st.pyplot(waterfall_plot)
    #%%ConfusionMatrix 
    st.subheader('Confusion Matrix of XGB model')
    xgb_prob = xgb_model.predict(X)
    cm = confusion_matrix(y, xgb_prob)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['NOM', 'OM'])
    sns.set_style("white")
    disp.plot(cmap='RdPu')
    plt.title("Confusion Matrix of XGB")
    disp1 = plt.show()
    st.pyplot(disp1)



