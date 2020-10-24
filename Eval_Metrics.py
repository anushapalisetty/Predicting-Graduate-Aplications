import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
import graphviz
from sklearn import tree 

class EvalMetrics():
    
    def roc_auc(self,clf,X_test,y_test):
        y_score=clf.predict_proba(X_test)
        metrics.roc_auc_score(y_test, y_score[:,1])
        fpr, tpr, _ = metrics.roc_curve(y_test, y_score[:,1],pos_label=1)
        roc_auc = metrics.auc(fpr, tpr)
        return roc_auc,fpr,tpr


    def error_rate(self,model,X_test,y_test):
        y_pred=model.predict(X_test)
        correct = (y_test == y_pred).sum()
        incorrect = (y_test != y_pred).sum()
        error_rate = incorrect / (correct + incorrect)
        return error_rate

    def tree_img(self,model,feature_names,class_names):
        dot_data=tree.export_graphviz(model, out_file=None, 
                    feature_names = feature_names,
                    class_names = class_names,
                    rounded = True, proportion = False, precision = 2, filled = True)
        img=graphviz.Source(dot_data,format="png")
        return img

    def score_accuracy(self,clf,X_train,y_train,X_test,y_test):
        clf.fit(X_train,y_train)
        y_pred=clf.predict(X_test)
        err=error_rate(clf,X_test,y_test)
        scoreTrain=clf.score(X_train,y_train)
        scoreTest=clf.score(X_test,y_test)
        accuracy=metrics.accuracy_score(y_test,y_pred)
        return scoreTrain,scoreTest,accuracy,err

    def confusion_matrix(self,y_test,y_pred):
        return pd.crosstab(y_test, y_pred)

    def confusion_report(self,y_test,y_pred):
        return metrics.classification_report(y_test, y_pred)

    def model_ypred(self,clf,X_train,y_train,X_test):
        clf.fit(X_train,y_train)
        y_pred=clf.predict(X_test)
        return y_pred

    def TSS(self,y_test,y_pred):
        TN,FP,FN,TP=metrics.confusion_matrix(y_test, y_pred).ravel()
        tp_rate=TP/float(TP+FN) if TP>0 else 0
        fp_rate=FP/float(FP+TN) if FP > 0 else 0
        return tp_rate-fp_rate  

    def HSS(self,y_test,y_pred):
        TN,FP,FN,TP=metrics.confusion_matrix(y_test, y_pred).ravel()
        N=TN+FP
        P=TP+FN
        HSS=(2*(TP * TN - FN * FP))/float((P * (FN + TN)) + ((TP + FP)*N))
        return HSS

    def F1(self,y_test,y_pred):
        TN,FP,FN,TP=metrics.confusion_matrix(y_test, y_pred,labels=[0,1]).ravel()
        precision=TP/float(TP+FP) ##Probabilty of Detection
        recall=TP/float(TP+FN)
        F1=2*((precision * recall)/(precision + recall))
        return F1

    def False_alaram_ratio(self,y_test, y_pred):
        TN,FP,FN,TP=metrics.confusion_matrix(y_test, y_pred).ravel()
        FAR= FP/float(FP + TN)
        return FAR

    def accuracy(self,y_test, y_pred):
        return metrics.accuracy_score(y_test, y_pred)

    def conf_matrix_table(self):
        TN,FP,FN,TP=metrics.confusion_matrix(y_test, clf.predict(X_test)).ravel()
        tp_rate=TP/float(TP+FN) if TP>0 else 0
        fp_rate=FP/float(FP+TN) if FP > 0 else 0
        tn_rate=TN/float(TN+FP) if TN>0 else 0
        fn_rate=FN/float(FN+TP) if FN > 0 else 0
        print('True Positive Rate: ',tp_rate)
        print('False Positive Rate: ',fp_rate)
        print('True Negative Rate: ',tn_rate)
        print('False Negative Rate: ',fn_rate)

    # def roc_auc(self,clf,X_test,y_test):
    #     y_score=clf.predict_proba(X_test)
    #     metrics.roc_auc_score(y_test, y_score[:,1])
    #     fpr, tpr, _ = metrics.roc_curve(y_test, y_score[:,1],pos_label=1)
    #     roc_auc = metrics.auc(fpr, tpr)
    # #     plt.plot(fpr, tpr)
    # #     plt.xlabel('FPR')
    # #     plt.ylabel('TPR')
    # #     plt.title('ROC curve')
    # #     plt.show()
    #     return roc_auc

    def eval_metrics(self,clf,X_test,y_test, y_pred):
        li=[]
        tss=self.TSS(y_test, y_pred)
        hss=self.HSS(y_test, y_pred)
        f1=self.F1(y_test, y_pred)
        FAR=self.False_alaram_ratio(y_test, y_pred)
        acc=self.accuracy(y_test,y_pred)
        auc,fpr,tpr=self.roc_auc(clf,X_test,y_test)
        li=[tss,hss,f1,FAR,auc,acc]
        eval_metr=pd.DataFrame(li, index=['TSS','HSS','F1','FAR','AUC','ACCURACY'])
        return eval_metr
    #     print('True Skill Statistics: ',tss.round(2))
    #     print('Heidke Skill Statistics: ',hss.round(2))
    #     print('F1 Score: ',f1.round(2))
    #     print('False Alaram Ratio: ',FAR.round(2))
    #     print('Accuracy: ',acc.round(2))

        
