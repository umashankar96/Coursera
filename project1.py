import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier



pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

def blight_model():
    
    df=pd.read_csv("train.csv",encoding='mac_roman',low_memory=False)
    df=df.drop(['mailing_address_str_number','violation_zip_code','mailing_address_str_name',
               'state','non_us_str_code','payment_amount','balance_due','payment_date','grafitti_status',
               'collection_status','compliance_detail','violation_street_name','country','payment_status'],axis=1)
    
    df.dropna(subset=['compliance','hearing_date','zip_code'],inplace=True) 
    df.reset_index(drop=True,inplace=True)

    
    df=df.drop(['violator_name','violation_street_number','city',
                'violation_code','violation_description','hearing_date','ticket_issued_date',
                'fine_amount','admin_fee','state_fee','late_fee','zip_code','discount_amount','clean_up_cost'],axis=1)
    
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(df['disposition'])
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    df=df.drop(['disposition'],axis=1)
    
    df2=pd.DataFrame(onehot_encoded, columns = ['DISPO_A','DISPO_B','DISPO_C','DISPO_D'])
    df=df.join(df2)

    df.replace({'Neighborhood City Halls':'Buildings, Safety Engineering & Env Department'},inplace=True)
    integer_encoded = label_encoder.fit_transform(df['agency_name'])
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    df=df.drop(['agency_name'],axis=1)
    
    df2=pd.DataFrame(onehot_encoded, columns = ['Agency_A','Agency_B','Agency_C','Agency_D',])
    df=df.join(df2) 
    df.set_index('ticket_id',inplace=True)

    y=df['compliance']
    df=df.drop(['compliance'],axis=1)
    x_train, x_test, y_train, y_test = train_test_split(df,y,random_state=1,test_size=0.01)

    rf=RandomForestClassifier(n_estimators=100,max_depth=15,class_weight={0:0.1,1:0.9}).fit(x_train,y_train)
    y_pred_rf=rf.predict(x_test)
    auc_rf=roc_auc_score(y_test,y_pred_rf)
    
    df=pd.read_csv("test.csv",encoding='mac_roman',low_memory=False)
    
   
    df=df.drop(['violation_zip_code','mailing_address_str_number','mailing_address_str_name',
               'state','non_us_str_code','grafitti_status','violation_street_name','country'],axis=1)
    
    df=df.drop(['violator_name','violation_street_number','city','zip_code','inspector_name',
                'violation_code','violation_description','hearing_date','ticket_issued_date',
                'fine_amount','admin_fee','state_fee','late_fee','discount_amount','clean_up_cost'],axis=1)
    
    df.replace({'Responsible (Fine Waived) by Admis':'Responsible (Fine Waived) by Deter',
                'Responsible by Dismissal':'Responsible by Default','Responsible - Compl/Adj by Determi':'Responsible by Default',
                'Responsible - Compl/Adj by Default':'Responsible by Default'},inplace=True)            
    
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(df['disposition'])
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    df=df.drop(['disposition'],axis=1)
    
    df2=pd.DataFrame(onehot_encoded, columns = ['DISPO_A','DISPO_B','DISPO_C','DISPO_D'])
    df=df.join(df2)   
    
    integer_encoded = label_encoder.fit_transform(df['agency_name'])
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    df=df.drop(['agency_name'],axis=1)
    
    df2=pd.DataFrame(onehot_encoded, columns = ['Agency_A','Agency_B','Agency_C'])
    df=df.join(df2)  
    df['Agency_D']=0
    df.set_index('ticket_id',inplace=True)
    
    return pd.Series(rf.predict(df),index=df.index)
blight_model()

