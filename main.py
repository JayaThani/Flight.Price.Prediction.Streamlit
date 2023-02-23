import streamlit as st
import pandas as pd
import numpy as np
from files import create_file, delete_tmp_files, transformations , load_models

#loading the serialized models
rf_model,encoder=load_models()

# navigation bar
st.sidebar.image("images/compunnel.png",width=100)
st.sidebar.title("Flight Price Prediction")

#file uploader for csv input
data=st.sidebar.file_uploader("Upload Data Here",type=["csv"],accept_multiple_files=False)




# Predict button
predict_btn=st.sidebar.button("Predict")


if predict_btn:
    #bytes like object of uploaded csv
    data_bytes=data.getvalue()    
    #file name of uploaded csv
    file_name=data.name
    if file_name:
        # creating temorary file for uploaded csv
        create_file(file_name,data_bytes)
        
        #loading the csv into a dataframe
        df=pd.read_csv(f"tmp_files/{file_name}")
        final_dataset=transformations(df,encoder)
        predictions=rf_model.predict(final_dataset)
        
        #saving output to a csv file
        df_predictions=pd.DataFrame(predictions,columns=["Predicted Price"])
        df_final=pd.concat([final_dataset,df_predictions],axis=1)
        csv=df_final.to_csv(index=False).encode("utf-8")
        

        #displaying the output
        st.dataframe(df_final)
        #st.success("Predictions are saved in outputs folder")
        st.download_button(
         label="Download data as CSV",
         data=csv,
         file_name='predictions.csv',
         mime='text/csv',
         )
        #delete the temporary files
        delete_tmp_files("tmp_files")
             
    else:
        # warning message if no files are uploaded
        st.warning("Please upload input csv file")








