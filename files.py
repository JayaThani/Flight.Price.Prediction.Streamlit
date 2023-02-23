import os
import shutil
import pandas as pd
import joblib

# function to create temporary csv data file from it's binary object
def create_file(file_name,data_bytes):
    
    with open("tmp_files/"+file_name, 'wb') as f:
        f.write(data_bytes)
        
  
#delete temporary files
def delete_tmp_files(folder_name):
    
	folder = folder_name
	for filename in os.listdir(folder):
		
		file_path = os.path.join(folder, filename)
		try:
			os.unlink(file_path)
		except Exception as e:
			print('Failed to delete %s. Reason: %s' % (file_path, e))
   
#function to perform transformations on the data
def transformations(df,encoder):
    final_dataset=df[['Airline','Date_of_Journey','Source','Destination','Route','Dep_Time','Arrival_Time','Duration','Total_Stops','Additional_Info']]
    final_dataset['Date']=final_dataset['Date_of_Journey'].str.split('/').str[0]
    final_dataset['Month']=final_dataset['Date_of_Journey'].str.split('/').str[1]
    final_dataset['Year']=final_dataset['Date_of_Journey'].str.split('/').str[2]
    final_dataset['Date']=final_dataset['Date'].astype(int)
    final_dataset['Month']=final_dataset['Month'].astype(int)
    final_dataset['Year']=final_dataset['Year'].astype(int)
    final_dataset=final_dataset.drop(['Date_of_Journey'],axis=1)
    final_dataset['Arrival_Time']=final_dataset['Arrival_Time'].str.split(' ').str[0]
    final_dataset['Total_Stops']=final_dataset['Total_Stops'].fillna('1 stop')
    final_dataset['Total_Stops']=final_dataset['Total_Stops'].replace('non-stop','0 stop')
    final_dataset['Stop'] = final_dataset['Total_Stops'].str.split(' ').str[0]
    final_dataset['Stop']=final_dataset['Stop'].astype(int)
    final_dataset=final_dataset.drop(['Total_Stops'],axis=1)
    final_dataset['Arrival_Hour'] = final_dataset['Arrival_Time'] .str.split(':').str[0]
    final_dataset['Arrival_Minute'] = final_dataset['Arrival_Time'] .str.split(':').str[1]
    final_dataset['Arrival_Hour']=final_dataset['Arrival_Hour'].astype(int)
    final_dataset['Arrival_Minute']=final_dataset['Arrival_Minute'].astype(int)
    final_dataset=final_dataset.drop(['Arrival_Time'],axis=1)
    final_dataset['Departure_Hour'] = final_dataset['Dep_Time'] .str.split(':').str[0]
    final_dataset['Departure_Minute'] = final_dataset['Dep_Time'] .str.split(':').str[1]
    final_dataset['Departure_Hour']=final_dataset['Departure_Hour'].astype(int)
    final_dataset['Departure_Minute']=final_dataset['Departure_Minute'].astype(int)
    final_dataset=final_dataset.drop(['Dep_Time'],axis=1)
    final_dataset['Route_1']=final_dataset['Route'].str.split('→ ').str[0]
    final_dataset['Route_2']=final_dataset['Route'].str.split('→ ').str[1]
    final_dataset['Route_3']=final_dataset['Route'].str.split('→ ').str[2]
    final_dataset['Route_4']=final_dataset['Route'].str.split('→ ').str[3]
    final_dataset['Route_5']=final_dataset['Route'].str.split('→ ').str[4]
    final_dataset['Route_1'].fillna("None",inplace=True)
    final_dataset['Route_2'].fillna("None",inplace=True)
    final_dataset['Route_3'].fillna("None",inplace=True)
    final_dataset['Route_4'].fillna("None",inplace=True)
    final_dataset['Route_5'].fillna("None",inplace=True)
    final_dataset=final_dataset.drop(['Route'],axis=1)
    final_dataset=final_dataset.drop(['Duration'],axis=1)
    final_dataset["Airline"]=encoder.fit_transform(final_dataset['Airline'])
    final_dataset["Source"]=encoder.fit_transform(final_dataset['Source'])
    final_dataset["Destination"]=encoder.fit_transform(final_dataset['Destination'])
    final_dataset["Additional_Info"]=encoder.fit_transform(final_dataset['Additional_Info'])
    final_dataset["Route_1"]=encoder.fit_transform(final_dataset['Route_1'])
    final_dataset["Route_2"]=encoder.fit_transform(final_dataset['Route_2'])
    final_dataset["Route_3"]=encoder.fit_transform(final_dataset['Route_3'])
    final_dataset["Route_4"]=encoder.fit_transform(final_dataset['Route_4'])
    final_dataset["Route_5"]=encoder.fit_transform(final_dataset['Route_5'])
    final_dataset=final_dataset.drop(['Year'],axis=1)
    return final_dataset

    
    
    

#function to load the model
def load_models():
    
	rf_model=joblib.load("models/rf.joblib")
	encoder=joblib.load("models/encoder.joblib")
	
	return rf_model,encoder