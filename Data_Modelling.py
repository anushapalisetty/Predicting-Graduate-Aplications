import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler,LabelEncoder
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier,AdaBoostClassifier,GradientBoostingClassifier,RandomForestClassifier
from Eval_Metrics import EvalMetrics
from sklearn import metrics

class DataModelling():
    
    em=EvalMetrics()

    def normalization(self,X):
        scalar=MinMaxScaler()
        scalar.fit(np.array(X))
        X=scalar.transform(np.array(X))
        return X

    def split_data(self,X,y,test_size,random_state,stratify_val):
        return train_test_split(X,y,test_size=test_size,random_state=random_state,stratify=stratify_val)

    def reg_decisiontree(self,depth_range,max_features,min_samples,random_state,X_train,y_train,X_test,y_test):
        resultsGini=pd.DataFrame(columns=['TreeDepth' , 'Score for Training' , 'Score for Testing',
                                            'Accuracy','Error'])
        for n in range(1,depth_range,1):
            clf = DecisionTreeClassifier(max_depth=n,max_features=max_features,min_samples_split=min_samples,random_state=random_state)
            scoreTrain,scoreTest,accuracy,err=self.em.score_accuracy(clf,X_train,y_train,X_test,y_test)
            resultsGini.loc[n]=[n,scoreTrain,scoreTest,accuracy,err]
        
        ## default
        clf = DecisionTreeClassifier(max_features=max_features,min_samples_split=min_samples,random_state=random_state)
        scoreTrain,scoreTest,accuracy,err=self.em.score_accuracy(clf,X_train,y_train,X_test,y_test)
        default=pd.DataFrame([['Default',scoreTrain,scoreTest,accuracy,err]],columns=resultsGini.columns)
        resultsGini=resultsGini.append(default, ignore_index=True)
        return resultsGini

    def random_forest(self,start,end,multiply,max_depth,random_state,X_train,y_train,X_test,y_test,max_features=0.1,min_samples=0.3):
        resultsGini=pd.DataFrame(columns=['LevelLimit' , 'Score for Training' , 'Score for Testing',
                                        'Accuracy','Error'])
        for n in range(start,end,multiply):
            clf = RandomForestClassifier(n_estimators=n,max_depth=max_depth,max_features=max_features,
                                        min_samples_split=min_samples)
            clf.fit(X_train,y_train)
            y_pred=clf.predict(X_test)
            err=error_rate(clf,X_test,y_test)
            scoreTrain=clf.score(X_train,y_train)
            scoreTest=clf.score(X_test,y_test)
            accuracy=metrics.accuracy_score(y_test,y_pred)
            resultsGini.loc[n]=[n,scoreTrain,scoreTest,accuracy,err]
            #a=tree_img(clf.estimators_[(n-1)],feature_names,class_names)
            #a.render('bagging_'+str(n))
        return clf,resultsGini
    
    def gradient_lr(self,lr_rate,max_depth,random_state,X_train,y_train,X_test,y_test,n_estimators=100,max_features=0.1,min_samples_split=0.3):
        resultsGini=pd.DataFrame(columns=['LevelLimit' , 'Score for Training' , 'Score for Testing',
                                        'Accuracy','Error'])
        for n in lr_rate:
                clf = GradientBoostingClassifier(n_estimators=n_estimators,max_depth=max_depth,max_features=max_features,min_samples_split=min_samples_split,
                                                learning_rate=n,random_state=random_state)
                clf.fit(X_train,y_train)
                y_pred=clf.predict(X_test)
                err=self.em.error_rate(clf,X_test,y_test)
                scoreTrain=clf.score(X_train,y_train)
                scoreTest=clf.score(X_test,y_test)
                accuracy=metrics.accuracy_score(y_test,y_pred)
                resultsGini.loc[n]=[n,scoreTrain,scoreTest,accuracy,err]
        return clf,resultsGini

    def gradient_booster(self,start,end,multiply,lr_rate,max_depth,random_state,X_train,y_train,X_test,y_test,n_estimators=100,max_features=0.1,min_samples_split=0.3):
        resultsGini=pd.DataFrame(columns=['LevelLimit' , 'Score for Training' , 'Score for Testing',
                                        'Accuracy','Error'])
        for n in range(start,end,multiply):
                clf = GradientBoostingClassifier(n_estimators=n,max_depth=max_depth,max_features=max_features,
                                                min_samples_split=min_samples_split,learning_rate=lr_rate,random_state=random_state)
                clf.fit(X_train,y_train)
                y_pred=clf.predict(X_test)
                err=self.em.error_rate(clf,X_test,y_test)
                scoreTrain=clf.score(X_train,y_train)
                scoreTest=clf.score(X_test,y_test)
                accuracy=metrics.accuracy_score(y_test,y_pred)
                resultsGini.loc[n]=[n,scoreTrain,scoreTest,accuracy,err]
        return clf,resultsGini

    