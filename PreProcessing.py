import pandas as pd
import numpy as np
from pandas.api.types import CategoricalDtype

class PreProcessing():
    
    # def __init__(self,name):
    #     self.name = name

    def column_rename(self,data,column_name,new_column_name):
        data.rename(columns={column_name:new_column_name},inplace=True)
    
    def readcsv(self,filename):
        return pd.read_csv(filename)

    def add_year(self,data,year_num):
        """
        Adding year value to the corresponding dataset,here year format is YYYY
        add_year(data,2020)
        """
        data["year"]=year_num
    
    def get_common_cols(self,data_list):
        """
        Input= Arrary of all the datasets => l=[Data1,Data2,Data3]
        Selects the common columns in all the datasets 
        """
        common_cols=[]
        for i in range(len(data_list)):
            if(i==0):
                common_cols=data_list[i].columns
            else:
                common_cols=common_cols.join(data_list[i].columns,how="inner")
        return common_cols
    
    def get_common_data(self,data_list,common_cols):
        common_data=pd.DataFrame()
        for i in data_list:
            common_data=common_data.append(i[common_cols],ignore_index=True)
        return common_data
    
    def get_cat_col(self,df,col_name):
        col=df.columns
        emp=[i for i in col if(i[:len(col_name)]==col_name)]
        t=df[emp].iloc[:,:]
        return t

    def check_null_col(self,df_mph_new,col_name):
        ch_data=df_mph_new.filter(like=col_name)
        m=ch_data.transpose()
        for i in range(m.shape[1]):
            l=[]
            if(m[i].count() !=0):
                for j in range(m.shape[0]):
                    l.append(m.iloc[j,i])
                    m.iloc[j,i]=np.NaN
                cl = [x for x in l if str(x) != 'nan']
                for k in range(len(cl)):
                    m.iloc[k,i]=cl[k]
        df_mph_new=df_mph_new.drop(columns=ch_data.columns,axis=1)
        df_mph_new=pd.concat([df_mph_new,m.transpose()],axis=1)
        return df_mph_new
    
    def target_label(self,common_data,admitted_list,incomplete_list,denied_list):
        for i in admitted_list:
            common_data.loc[common_data.decision_code_0.str.contains(i),"decision_code_0"]='Admitted'
        for i in incomplete_list:
            common_data.loc[common_data.decision_code_0.str.contains(i),"decision_code_0"]='Incomplete'
        for i in denied_list:
            common_data.loc[common_data.decision_code_0.str.contains(i),"decision_code_0"]='Denied'

    def get_sum_column(self,df_mph_new,col_name):
        df_cols_to_sum=df_mph_new.filter(like=col_name)
        df_mph_new[col_name]=df_cols_to_sum.sum(axis=1)
        df_mph_new=df_mph_new.drop(columns=df_cols_to_sum.columns,axis=1)
        return df_mph_new

    def get_average_column(self,df,col):
        a=df.filter(like=col)
        l=a.columns
        df[col+'_avg']=a.sum(axis=1)/a.count(axis=1)
        df=df.drop(l,axis=1)
        return df

    def averge_data(self,df_num,col_list):
        for i in col_list:
            df_num=self.get_average_column(df_num,i)
        return df_num

    #Function to get count for continous data
    def get_count_column(self,  df,col_name):
        df_cols_to_sum=df.filter(like=col_name)
        df[col_name+'_count']=df_cols_to_sum.transpose().notnull().sum()
        df=df.drop(columns=df_cols_to_sum.columns,axis=1)
        return df
    
    def count_data(self,df_cat,col_list):
        for i in col_list:
            df_cat=self.get_count_column(df_cat,i)
        return df_cat

    #Function to get hot encoding techniques for nominal data
    def get_hot_enc_freq(self,df_mph_new,col_name,uniq_values):
        k=df_mph_new.filter(like=col_name).fillna('None')
        p=pd.get_dummies(k)
        enc_val=uniq_values
        for i in range(len(enc_val)):
            m=[]
            temp=[]
            for j in p.columns:
                if(j[(len(j)-len(enc_val[i])):len(j)]==enc_val[i]):
                    m.append(j)
            temp=p[m]
            df_mph_new[col_name+'_'+enc_val[i]]=temp.sum(axis=1)
        df_mph_new=df_mph_new.drop(columns=k.columns,axis=1)
        return df_mph_new
    

    def designation_rename(self,df_cat):
        df_cat.loc[df_cat.designation_0.str.contains('Health Promotion & Behavior'),'designation_0']='Health Promotion and Behavior'
        return df_cat

    def number_months(self,df_cat,cols_list,start_col,end_col,new_col):
        df_cat[cols_list]=df_cat[cols_list].apply(pd.to_datetime)
        for i in range((len(cols_list)//2)):
            diff=(df_cat[end_col+str(i)]-df_cat[start_col+str(i)])
            df_cat[new_col+str(i)]=(diff/np.timedelta64(1,'M')).round()
        
        a=df_cat.filter(like=new_col)
        df_cat[new_col+'avg']=a.sum(axis=1)/a.count(axis=1)
        df_cat=df_cat.drop(a.columns,axis=1)
        df_cat=df_cat.drop(cols_list,axis=1)
        return df_cat

    def fill_na(self,df,col_list,value):
        return  df[col_list].fillna(value)

    def label_encoding(self,category,df_cat,col_list,ordered):
        cat_type = CategoricalDtype(categories=category,ordered=ordered)
        return df_cat[col_list].apply(lambda x:x.astype(cat_type).cat.codes)

    def get_month(self,df_cat,col_list):
        return df_cat[col_list].apply(lambda x:pd.to_datetime(x).dt.month)

    def one_hot_freq(self,df_cat,col_list,uniq_val):
        for i in col_list:
            df_cat=self.get_hot_enc_freq(df_cat,i,uniq_val)
        return df_cat

    def drop_columns(self,df,col_list):
        for i in col_list:
            df=df.drop(df.filter(like=i).columns,axis=1)
        return df

    #Get maximum Gre score for an applicant
    def max_gre(self,df_num):
        k=df_num.filter(like='gre')
        first=k['gre_quantitative_scaled_0']+k['gre_verbal_scaled_0']+k['gre_analytical_scaled_0']
        second=k['gre_quantitative_scaled_1']+k['gre_verbal_scaled_1']+k['gre_analytical_scaled_1']
        cols=[]
        for i in k.columns:
            cols.append(i[0:len(i)-2])
        cols=np.unique(cols)

        for i in range(k.shape[0]):
                for j in cols:
                    if(first[i]>=second[i]):
                        df_num[j]=k[j+'_0'] 
                    elif(second[i]>first[i]):
                        df_num[j]=k[j+'_1'] 

        df_num=df_num.drop(k.columns,axis=1)
        return df_num

    #Get maximum Gre score for an applicant
    def max_mcat(self,df_num):
        k=df_num.filter(like='mcat').drop(['mcat_verbal_reasoning','mcat_physical_sciences','mcat_biological_sciences',
                                        'mcat_2015_aamc_id_0','mcat_2015_aamc_id_1'], axis=1)
        first=k['mcat_2015_total_score_0']
        second=k['mcat_2015_total_score_1']

        cols=[]
        for i in k.columns:
            cols.append(i[0:len(i)-2])
        cols=np.unique(cols)

        for i in range(k.shape[0]):
                for j in cols:
                    if(first[i]>=second[i]):
                        df_num[j]=k[j+'_0'] 
                    elif(second[i]>first[i]):
                        df_num[j]=k[j+'_1'] 

        df_num=df_num.drop(k.columns,axis=1)
        return df_num

    def full_column_null(self,df_num,sum_val):
        a=(df_num.isnull().sum()==sum_val)
        cols=a[a==True].index
        df_num=df_num.drop(cols,axis=1)
        return df_num

    
