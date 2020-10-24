import pandas as pd
import numpy as np
from PreProcessing import PreProcessing

from sklearn.feature_selection import VarianceThreshold

class Dimensional_Reduction():

    pp=PreProcessing()
    

    def designation_le(self,df_cat):
        df_cat=self.pp.designation_rename(df_cat)
        category=['Biostatistics', 'Epidemiology', 'FOR ADMISSIONS OFFICE USE ONLY', 'Health Management and Policy',
                'Health Promotion and Behavior']

        col_list=['designation_0','designation_1']

        df_cat[col_list]=self.pp.label_encoding(category,df_cat,col_list,False)
        return df_cat

    def desig_term_le(self,df_cat):
        category=['Fall', 'Spring']

        col_list=['designation_program_start_term_0','designation_program_start_term_1']

        df_cat[col_list]=self.pp.label_encoding(category,df_cat,col_list,False)
        return df_cat

    def lang_prof_le(self,df_cat):
        category=[ 'Beginner','Intermediate','Advanced','Native']

        col_list=['language_proficiency_level_0','language_proficiency_level_1','language_proficiency_level_2',
            'language_proficiency_level_3']

        df_cat[col_list]=self.pp.label_encoding(category,df_cat,col_list,True)
        return df_cat

    def lang_count(self,df_cat):
        col_list=['language_0','language_1','language_2','language_3']

        df_cat['language_count']=self.pp.get_count_column(df_cat[col_list],'language')

        df_cat=self.pp.drop_columns(df_cat,col_list)
        return df_cat

    def num_months_avg(self,df_cat,col_name):
        a=df_cat.filter(regex=col_name+".*start.*date") & df_cat.filter(regex=col_name+".*end.*date")
        col_list=a.columns
        df_cat=self.pp.number_months(df_cat,col_list,col_name+'_start_date_',col_name+'_end_date_',col_name+'_months_')
        return df_cat
    ## Imputing Status Data

    def label_enc_yes_no(self,df_cat):
        category=['No','Yes']

        col_list=['gmat_official','gre_official_0','gre_official_1','mcat_2015_official_0','mcat_2015_official_1',
                'mcat_official','academic_performance_suspension','policy_violation']

        df_cat[col_list]=self.pp.label_encoding(category,df_cat,col_list,False)
        return df_cat

    def label_enc_plan_taken(self,df_cat):
        category=['Planned','Taken']

        col_list=['gmat_status','gre_status_0','gre_status_1','mcat_2015_status_0','mcat_2015_status_1',
                'mcat_status']

        df_cat[col_list]=self.pp.label_encoding(category,df_cat,col_list,False)
        return df_cat

    def label_enc_first_mem(self,df_cat):
        category=['No', 'Yes', 'I prefer not to respond']

        col_list=['first_member_undergraduate_degree','first_member_graduate_degree']

        df_cat[col_list]=self.pp.label_encoding(category,df_cat,col_list,True)
        return df_cat


    def one_hot_enc_frequency(self,df_cat):
        uniq_val=['Part Time', 'Full Time', 'Temporary', 'Per Diem']

        col_list=['employment_frequency','research_frequency','volunteer_frequency']

        df_cat=self.pp.one_hot_freq(df_cat,col_list,uniq_val)
        return df_cat

    def count_columns(self,df_cat):
        col_list=['gpas_by_school_college_name','employment_title','honors_name','honors_sponsoring_organization',
                'honors_description','research_title','research_employer','research_duties','research_country',
                'volunteer_duties','volunteer_title','volunteer_organization','reference_occupation_name',
                'reference_institution','reference_state','employment_title','employment_employer','employment_duties',
                'employment_supervisor_title','honors_date_received']

        df_cat=self.pp.count_data(df_cat,col_list)
        return df_cat

    def one_hot_enc_yes_no(self,df_cat):
        uniq_val=['Yes','No']

        col_list=['employment_current','employment_academic_credit','employment_salary_or_payment','employment_volunteer',
                'research_academic_credit','research_salary_or_payment','research_volunteer','research_current',
                'research_contact_supervisor_permission','volunteer_academic_credit','volunteer_salary_or_payment',
                'volunteer_contact_supervisor_permission','volunteer_volunteer','volunteer_current']

        df_cat=self.pp.one_hot_freq(df_cat,col_list,uniq_val)
        return df_cat

    def drop_cat_columns(self,df_cat):
        """
        List of columns which will be dropped:
        ['state','city','zip','country','policy_violation_explanation','mcat_writing_sample',
                'academic_performance_suspension_explanation','decision_code_1','gpas_by_school_college_id']
        """
        col_list=['state','city','zip','country','policy_violation_explanation','mcat_writing_sample',
                'academic_performance_suspension_explanation','decision_code_1','gpas_by_school_college_id']

        df_cat=self.pp.drop_columns(df_cat,col_list)
        return df_cat

    def cat_data_processing(self,common_data):
        df_cat=common_data.select_dtypes(exclude=np.number) 
        df_cat=self.designation_le(df_cat)
        df_cat=self.desig_term_le(df_cat)
        df_cat=self.lang_prof_le(df_cat)
        df_cat=self.lang_count(df_cat)
        df_cat=self.label_enc_yes_no(df_cat)
        df_cat=self.label_enc_plan_taken(df_cat)
        df_cat=self.label_enc_first_mem(df_cat)
        df_cat=self.one_hot_enc_frequency(df_cat)
        df_cat=self.count_columns(df_cat)
        df_cat=self.one_hot_enc_yes_no(df_cat)
        df_cat=self.drop_cat_columns(df_cat)
        return df_cat

    ## Fetching all the numeric data

    def design_year_le(self,df_num):
        category=[2017,2018,2019]

        col_list=['designation_program_start_year_0','designation_program_start_year_1']

        df_num[col_list]=self.pp.label_encoding(category,df_num,col_list,True)
        return df_num

    def avg_data(self,df_num):
        col_list=['gpas_by_school_credit_hours','gpas_by_school_quality_points','gpas_by_school_gpa','employment_total_hours',
        'employment_hours','employment_weeks','research_weeks','research_hours','research_total_hours',
        'volunteer_weeks','volunteer_hours','volunteer_total_hours']

        df_num=self.pp.averge_data(df_num,col_list)
        return df_num

    #Fetching all the numeric data
    def num_data_processing(self,common_data):
        df_num=common_data._get_numeric_data()  ### 194 columns
        df_num=self.pp.max_gre(df_num)
        df_num=self.pp.max_mcat(df_num)
        df_num=self.design_year_le(df_num)
        df_num=self.avg_data(df_num)
        return df_num


    ## Joining Qualitative Data

    ## References & PS

    def drop_admisn_cols(self,ref_data):
        l=['upper_division_jrsr_undergraduate_total_gpa','upper_division_jrsr_undergraduate_total_hours',
        'upper_division_jrsr_undergraduate_total_quality_points','volunteer_academic_credit', 
        'volunteer_community_enrichment_experience_hours_total','volunteer_contact_supervisor_permission','volunteer_current',
        'volunteer_date_diff','volunteer_frequency','volunteer_hours','volunteer_salary_or_payment','volunteer_term_diff',
        'volunteer_total_hours','volunteer_weeks','wes_gpa_graduate','wes_gpa_professional','wes_gpa_undergraduate',
        'social_behavioral_gpa','social_behavioral_graded_hours','social_behavioral_quality_points', 'policy_violation',
        'public_health_gpa','public_health_graded_hours','public_health_quality_points','research_academic_credit',
        'research_contact_supervisor_permission','research_current','research_date_diff','research_frequency',
        'research_hours','research_salary_or_payment','research_term_diff','research_total_hours','research_volunteer',
        'research_weeks', 'other_gpa','other_graded_hours','other_quality_points','overall_total_gpa', 
        'overall_total_hours','overall_total_quality_points', 'lower_division_frso_undergraduate_total_gpa',
        'lower_division_frso_undergraduate_total_hours','lower_division_frso_undergraduate_total_quality_points', 
        'math_statistics_gpa','math_statistics_graded_hours','math_statistics_quality_points','mcat_2015_status', 
        'mcat_status', 'biology_chemistry_physics_life_science_gpa','biology_chemistry_physics_life_science_graded_hours',
        'biology_chemistry_physics_life_science_quality_points','business_gpa','business_graded_hours', 
        'business_quality_points', 'cumulative_undergraduate_total_gpa','cumulative_undergraduate_total_hours',
        'cumulative_undergraduate_total_quality_points', 'employment_academic_credit','employment_current',
        'employment_date_diff','employment_experience_hours_total','employment_frequency','employment_hours',
        'employment_salary_or_payment','employment_supervisor_contacted','employment_term_diff','employment_total_hours',
        'employment_volunteer','employment_weeks','gmat_status','gpas_by_school_credit_hours','gpas_by_school_gpa', 
        'gpas_by_school_quality_points','gre_analytical_percentile','gre_analytical_scaled','gre_official',
        'gre_quantitative_percentile','gre_quantitative_scaled','gre_status','gre_verbal_percentile','gre_verbal_scaled',
        'health_science_gpa','health_science_graded_hours','health_science_quality_points', 'language', 
        'language_proficiency_level']

        ref_data=ref_data.drop(l,axis=1)
        return ref_data

    def ref_avg_data(self,ref_data):
        col_list=['pct_syllable_n','pct_uniq_syllable_n','pct.inv_syllable_n','pct.inv_uniq_syllable_n','pct_inv_syllable_n',
        'pct_letter_n', 'pct_inv_letter_n','pct.inv_letter_n',
        'inv_uniq_syllable_n','inv_syllable_n1','sum_letter_n','sum_uniq_syllable_n',
        'cum.inv_letter','cum.pct_letter','cum.inv_syllable_n', 'cum.sum_syllable_n','cum_inv_syllable_n',
        'cum_inv_uniq_syllable_n','cum_inv_letter_n','cum_sum_syllable_n',
        'n_letters_l','n_syllables_s','num_letter_n','num_syllable_n','num_uniq_syllable_n']

        ref_data=self.pp.averge_data(ref_data,col_list)
        return ref_data

    def ref_afinn_avg(self,ref_data):
        a=ref_data.filter(regex='afinn_neg_.*_n')
        l=a.columns
        ref_data['afinn_neg_n_avg']=a.sum(axis=1)/a.count(axis=1)
        ref_data=ref_data.drop(l,axis=1)

        a=ref_data.filter(regex='afinn_neg_.*_p')
        l=a.columns
        ref_data['afinn_neg_p_avg']=a.sum(axis=1)/a.count(axis=1)
        ref_data=ref_data.drop(l,axis=1)

        a=ref_data.filter(regex='afinn_pos_.*_n')
        l=a.columns
        ref_data['afinn_pos_n_avg']=a.sum(axis=1)/a.count(axis=1)
        ref_data=ref_data.drop(l,axis=1)

        a=ref_data.filter(regex='afinn_pos_.*_p')
        l=a.columns
        ref_data['afinn_pos_p_avg']=a.sum(axis=1)/a.count(axis=1)
        ref_data=ref_data.drop(l,axis=1)
        return ref_data

    def ref_data_processing(self,qual_data):
        ref_data=pd.pivot_table(qual_data,index=['sophasid','year','term','admission']
                            ,columns='indicator',values='average').reset_index()
        ref_data=self.drop_admisn_cols(ref_data)
        ref_data=self.ref_avg_data(ref_data)
        ref_data=self.ref_afinn_avg(ref_data)
        ref_data=ref_data.rename(columns={'sophasid':'cas_id'})
        return ref_data

    def merge_final_data(self,admisn_data,ref_data):
        final_data=pd.merge(admisn_data,ref_data,on=['cas_id','year'])
        final_data=final_data.drop(['cas_id','year','term','admission'],axis=1)
        print('final_data',final_data.shape)
        return final_data

    ## Correlation
    def mask_corr(self,X):
        mask =np.array(X.corr())
        triangle_indices = np.triu_indices_from(mask)
        mask[triangle_indices] = True
        return mask
    
    def X_y_data(self,df,target_feature):
        X=df.loc[:,df.columns !=target_feature]
        y=df[target_feature]
        return X,y

    def get_col_corr(self,df,th):
        abv_th=[]
        bel_th=[]
        for i in range(df.shape[0]):
            for j in range(i):
                if(abs(df.iloc[i,j]) >= th):
                    if(df.columns[j] not in abv_th):
                        abv_th.append(df.columns[j])
                else:
                    if(df.columns[j] not in bel_th):
                        bel_th.append(df.columns[j])
        return abv_th,bel_th
    
    def mask_data(self,X):
        mask =np.array(X)
        triangle_indices = np.triu_indices_from(mask)
        mask[triangle_indices] = True
        return mask 

    def remove_nulls(self,df,th):
        df_null_sum=df.isnull().sum()
        b_cal=df_null_sum/df.shape[0]
        cols=df.columns[b_cal<th]
        return df[cols]
    ## Dropping Highly Correlated Data

    def thresholds(self,final_data,target_label,null_th,var_th,corr_th):
        final_data=self.remove_nulls(final_data,null_th)
        
        ## Define X and y
        X,y=self.X_y_data(final_data,target_label)
        print("After removing nulls: ", final_data.shape)
        
        ## Variance threshold
        sel = VarianceThreshold(threshold=var_th)
        sel_var=sel.fit_transform(X)
        X=X[X.columns[sel.get_support(indices=True)]]
        print("After removing low variance features: ", X.shape)
        
        ## Correlation threshold
        data_corr=pd.DataFrame(self.mask_corr(X),columns=X.columns,index=X.columns)
        abv_th,bel_th=self.get_col_corr(data_corr,corr_th)
        
        ## Drop highly correlated
        X=X.drop(abv_th,axis=1)
        features=X.columns
        print("After dropping high correlated features: ", X.shape)
        return X,y,features
