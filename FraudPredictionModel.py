import pickle as pkl
import streamlit as st
import pandas as pd
import numpy as np


model=pkl.load(open('model.pkl','rb'))
Transformer=pkl.load(open('Transformer.pkl','rb'))
colText=pkl.load(open('colText.pkl','rb'))


st.title("Fraud Prediction Model")
tab1,tab2=st.tabs(["Fraud Prediction Form","Fraud Prediction Via Upload"])

with tab1:
    
    age=st.number_input("Enter Account Age")
    merchant=st.text_input("Enter Merchant name")
    amount=st.number_input("Enter Amount")
    buttonForm=st.button("Predict")
    if buttonForm:
        if age<=0 or age=="":
            st.error("Please enter valid Age")
        elif merchant.isnumeric() or merchant=="":
            st.error("Please Enter Valid Value | Merchant")
        elif amount<=0 or amount=="":
            st.error("Please Enter Transaction Amount")
        else:
            data=pd.DataFrame({"age":[age],"merchant":[merchant],"amount":[amount]})
            transformData=Transformer.transform(data)
            frameWithFeatures=pd.DataFrame(transformData,columns=Transformer.get_feature_names_out())
            predict=model.predict(frameWithFeatures)
            st.write(f"Prediction: {predict}")
with tab2:
    
    fileUpload=st.file_uploader("Upload File",type="csv")
    fileButtton=st.button("Click to Predict")
    if fileButtton:
        if fileUpload:
            file=pd.read_csv(fileUpload)
            TransformFile=Transformer.transform(file)
            withEncodedFeatures=pd.DataFrame(TransformFile,columns=Transformer.get_feature_names_out())
            predictFile=model.predict(withEncodedFeatures)
            file['Prediction']=predictFile
            st.write(file)
        else:
            st.error("Upload File")