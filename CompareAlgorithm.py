# -*- coding: utf-8 -*-
"""
Created on Thu Apr 05 17:56:20 2018

@author: CQQ

E-mail:chenqq18@126.com
"""

from sklearn.model_selection import StratifiedKFold,cross_val_predict,KFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import neighbors,metrics,tree
from sklearn.neighbors import KDTree
import Function as PF
from sklearn.svm import NuSVC,SVC
from scipy import stats
import numpy as np
import random
# %%                                   TwoStageEnsemble
def TWE(datamatrix,datalabel):
     
     '''
     parameter:
         df:represent input data contain attribute matrix and label
     variable setting:
         nsm:represent number of instance in df
         nat:represent number of attribute in  df
         dfmatrix:represent attribute matrix
         dfindex:represent index of df,which is index in original index
     '''
     min_error = 1
     '''
     Stage one :determine optimal sampling rate
     '''
     clf = RandomForestClassifier(n_estimators=50,oob_score=True)
     for sample_rate in [0.2,0.4,0.8]:
        oob_error = 0.0
        ns = np.ceil(1/sample_rate)
        skf = StratifiedKFold(n_splits=int(ns))
        for train,test in skf.split(datamatrix,datalabel):
            if sample_rate < 0.5:
                EnTrainMatrix = datamatrix[test]
                EnTrainLabel = datalabel[test]  
            else:
                EnTrainMatrix = datamatrix[train]
                EnTrainLabel = datalabel[train]  
            clf.fit(EnTrainMatrix,EnTrainLabel)
            oob_error = oob_error + 1-clf.oob_score_
        final_oob_error = oob_error/ns
        if min_error > final_oob_error:
            min_error = final_oob_error
            OptimalSamplingRate = sample_rate
     """
     According optimal sampling rate and using stratified sampling
     """
     labelrecord = PF.RecordIndexOfClass(datalabel)
     SecordTrianIndex = set()
     for i in range(len(labelrecord)):
         leni = int(np.ceil(len(labelrecord[i])*OptimalSamplingRate))
         rsi = random.sample(list(labelrecord[i]),leni)
         SecordTrianIndex = SecordTrianIndex.union(set(rsi))  
     SecordTrianIndex = list(SecordTrianIndex)
     """
     Stage two :determine optimal disagreement rate
     """
     clf.fit(datamatrix[SecordTrianIndex],datalabel[SecordTrianIndex])    
     min_error, index= 1, np.array(range(len(datalabel)))
     for theta in [0.5,0.6,0.7,0.8,0.9,1.0]:
        ProbaMatrix = clf.predict_proba(datamatrix)
        MaxProba = np.amax(ProbaMatrix,axis=1)
        MaxIndex = ProbaMatrix.argmax(axis=1)+1
        #Record error classification index
        TempNoiseIndex = np.nonzero(MaxIndex - datalabel)[0]
        RemovedIndex = TempNoiseIndex[np.where(MaxProba[TempNoiseIndex]>theta)[0]]
        CleansedIndex = np.array(index)[list(set(index)-set(index[RemovedIndex]))]
        #Wrapper CV
        predicted = cross_val_predict(clf,datamatrix[CleansedIndex],datalabel[CleansedIndex],cv=3)
        error = 1- metrics.accuracy_score(datalabel[CleansedIndex],predicted)
        if error < min_error:
          final_theta = theta
          min_error = error
          finalcleanindex = CleansedIndex
     """
     Final Clean Stage
     """
     clf.fit(datamatrix[finalcleanindex],datalabel[finalcleanindex])  
     ProbaMatrix = clf.predict_proba(datamatrix)
     MaxProba = np.amax(ProbaMatrix,axis=1)
     MaxIndex = ProbaMatrix.argmax(axis=1)+1
     TempNoiseIndex = np.nonzero(MaxIndex - datalabel)[0]
     NoisedIndex = TempNoiseIndex[np.where(MaxProba[TempNoiseIndex]>=final_theta)[0]]
     return NoisedIndex
# %%                                  PSAM 
def ConputeConfMatrix(datamatrix,datalabel):
     """
      parameter:
         df:represent input data contain attribute matrix and label
     variable setting:
         nsm:represent number of instance in df
         nat:represent number of attribute in  df
         dfmatrix:represent attribute matrix
         dfindex:represent index of df,which is index in original index
     """
     nsm,nat = datamatrix.shape  
     nc = len(set(np.ravel(datalabel)))
     ConfMatrix=np.zeros((nsm,nc))
     kf = KFold(n_splits=5,shuffle=True)
     T=10
     for i in range(T):
          neigh = KNeighborsClassifier(n_neighbors=3)
          dt = tree.DecisionTreeClassifier()
          gnb = GaussianNB()
          for train, test in kf.split(datamatrix):
                 #3-NN
                 neigh.fit(datamatrix[train],datalabel[train])
                 NN_pred = list(np.array(map(int,neigh.predict(datamatrix[test])))-1)
                 ConfMatrix[test,NN_pred] = ConfMatrix[test,NN_pred] + 1
                 #Naive bayes
                 gnb.fit(datamatrix[train],datalabel[train])
                 GND_pred = list(np.array(map(int,gnb.predict(datamatrix[test])))-1)
                 ConfMatrix[test,GND_pred] = ConfMatrix[test,GND_pred] + 1
                 #Decision Tree
                 dt.fit(datamatrix[train],datalabel[train])
                 DT_pred = list(np.array(map(int,dt.predict(datamatrix[test])))-1)
                 ConfMatrix[test,DT_pred] = ConfMatrix[test,DT_pred] + 1
     ConfMatrix = ConfMatrix*1.0/(T*nc)
     return  ConfMatrix 

def PSAM(neigh,svm,tree,TrainMatrix,TrainLabel,TestMatrix,TestLabel,ConfMatrix):
     """
    Parameters
    ----------
    dftr: traing data 
    dfte；test data
    ConfMatrix:

    Returns
    -------
    acc
     """
     
     nsd,nc = len(TrainLabel)*0.85,len(set(np.ravel(TrainLabel)))
     PSAM_KNN,PSAM_SVM,PSAM_C45 = 0,0,0
     k=0
     for t in range(10):
          si = range(len(TrainLabel))
          random.shuffle(si)
          smi = []  #sampling data index
          sml = []  #sampling data label
          for i in si:
               counter=0
               rr = random.random()
               for j in range(nc):
                    if ConfMatrix[i][j]>rr:
                         smi.append(i)
                         sml.append(j)
                         counter = counter + 1
                         break
               if counter==nsd:
                    break
          if len(set(sml))==1:
              k = k + 1
              continue
          neigh.fit(TrainMatrix[smi],np.array(sml)+1)
          svm.fit(TrainMatrix[smi],np.array(sml)+1)
          tree.fit(TrainMatrix[smi],np.array(sml)+1)
          
          PSAM_KNN = PSAM_KNN + neigh.score(TestMatrix,TestLabel)
          PSAM_SVM = PSAM_SVM + svm.score(TestMatrix,TestLabel)
          PSAM_C45 = PSAM_C45 + tree.score(TestMatrix,TestLabel)
     return PSAM_KNN*1.0/(10-k),PSAM_SVM*1.0/(10-k),PSAM_C45*1.0/(10-k)
# %%                                 INNFC    
def EnsembleFilter(datamatrix,datalabel,rfd):
    """
    Parameters
    ----------
    df: input data with label
    rtd: filtered data which remain index

    Returns
    -------
    ndi : noise data index
    """
    ar = np.ones([len(datalabel),3])
    # LOG filtering
    LOG = RidgeClassifier(alpha = 1e-8)
    trainmatrix ,trainlabel = datamatrix[rfd],datalabel[rfd]
    LOG.fit(trainmatrix ,trainlabel)
    ar[:,0] = LOG.predict(datamatrix)
    # KNN filtering
    NN = neighbors.KNeighborsClassifier(n_neighbors=3)
    NN.fit(trainmatrix ,trainlabel)
    ar[:,1] = NN.predict(datamatrix)
    # C4.5 filtering
    TR = tree.DecisionTreeClassifier()
    TR.fit(trainmatrix ,trainlabel)
    ar[:,2] = TR.predict(datamatrix)
    #ensemble filtering
    temp = (stats.mode(ar,axis=1)[0]-datalabel.reshape(-1,1)).astype(int)
    ndi = list(temp.nonzero()[0])
    return ndi

def INNFC (datamatrix,datalabel,ind):
    """
    Parameters
    ----------
    df: input data with label
    rtd: filtered data which remain index
    ind:index k-nn of each data

    Returns
    -------
    ndi : noise data index
    """
    # Preliminary filtering==================================================================
    rfd=range(len(datalabel))
    pf = EnsembleFilter(datamatrix,datalabel,rfd) 
    # Noise-free filtering==================================================================    
    ri = list(set(rfd)-set(pf))
    nff = EnsembleFilter(datamatrix,datalabel,ri)
    # Final removal of noise ===============================================================
    k=5#number of knn
    ndi=[]
    for i in range(len(nff)):
         sum=0
         for j in range(k):
              if datalabel[i]==datalabel[ind[i][j]]:
                   sum = sum - 1
              else:
                   sum = sum + 1
         if sum > 0:
            ndi.append(nff[i])  
    return ndi
# %%                                 CF                     
def CF(datamatrix,datalabel):
     ndi=set()
     skf = StratifiedKFold(n_splits=3)
     for train,test in skf.split(datamatrix,datalabel):
          TR = tree.DecisionTreeClassifier()
          TR.fit(datamatrix[train],datalabel[train])
          t = TR.predict(datamatrix[test])-datalabel[test]       
          ndi = ndi.union(set(test[t.nonzero()]))
     ndi = list(ndi)
     return ndi

# %%                                 MVF_ALNR        
def MVF_ALNR(DataMatrix,NoisedLable):
     skf = StratifiedKFold(n_splits=3)
     IN_SU = set()
     for train,test in skf.split(DataMatrix,NoisedLable):
         clf = SVC()
         clf.fit(DataMatrix[train],NoisedLable[train]) 
         IN_SU=IN_SU.union(set(train[list(clf.support_)]))
     return list(IN_SU)    
    
    
    
# %%                                 MVF        
def MVF(Tr_data,Tr_label,Te_data,Te_label):
     skf = StratifiedKFold(n_splits=3)
     ar = np.ones([len(Te_label),3])
     k=0
     for train,test in skf.split(Tr_data,Tr_label):
          if k==0:
               #Decision Tree
               TR = tree.DecisionTreeClassifier()
               TR.fit(Tr_data[train],Tr_label[train])
               ar[:,k] = TR.predict(Te_data)
               k= k + 1
          elif k==1:
               #LDA(LinearDiscriminantAnalysis)
               LDA = LinearDiscriminantAnalysis()
               LDA.fit(Tr_data[train],Tr_label[train])
               ar[:,k] = LDA.predict(Te_data)
               k= k + 1
          else:#1-NN
               NN = neighbors.KNeighborsClassifier(n_neighbors=1)
               NN.fit(Tr_data[train],Tr_label[train])
               ar[:,k] = NN.predict(Te_data)
     temp = (stats.mode(ar,axis=1)[0]-Te_label.reshape(-1,1)).astype(int)
     ndi = list(temp.nonzero()[0])
     return ndi
 
# %%                                 EF        
def EF(Tr_data,Tr_label,Te_data,Te_label):
     ar = np.ones([len(Te_label),3])
     #Decision Tree
     TR = tree.DecisionTreeClassifier()
     TR.fit(Tr_data,Tr_label)
     ar[:,0] = TR.predict(Te_data)
     #LDA(LinearDiscriminantAnalysis)
     LDA = LinearDiscriminantAnalysis()
     LDA.fit(Tr_data,Tr_label)
     ar[:,1] = LDA.predict(Te_data)
     #1-NN
     NN = neighbors.KNeighborsClassifier(n_neighbors=1)
     NN.fit(Tr_data,Tr_label)
     ar[:,2] = NN.predict(Te_data)
     temp = (stats.mode(ar,axis=1)[0]-Te_label.reshape(-1,1)).astype(int)
     ndi = list(temp.nonzero()[0])
     return ndi

# %%                                  ENN   
def ENN(NBM_Tr,NBL_TN,NBT_Kdt):
     """
    Parameters
    ----------
    df: train data 
    kdt:KD-Tree

    Returns
    -------
    NoiseIndex : filter find noise index in original data
    """
     NNIndex = NBT_Kdt.query(NBM_Tr,k=2,return_distance=False)
     NoiseIndex=[]
     for i in range(len(NBL_TN)):
          indexi = NNIndex[i][0]
          indexnn = NNIndex[i][1]
          if NBL_TN[indexi]!=NBL_TN[indexnn]:
               NoiseIndex.append(i)
     return NoiseIndex
# %%                                  IPF   
#singal processing
def FIPF(datamatrix ,datalabel): 
     n,k =3,0
     ar = np.ones((len(datalabel),3))
     skf = StratifiedKFold(n_splits=n)
     for tri,tei in skf.split(datamatrix ,datalabel):
          TR = tree.DecisionTreeClassifier(min_samples_leaf=2)
          TR.fit(datamatrix[tri] ,datalabel[tri])
          ar[:,k] = TR.predict(datamatrix)
          k = k + 1
     ml = (stats.mode(ar,axis=1)[0]).astype(int)
     return ml  
def IPF(datamatrix, datalabel):
     loop, ndi, Index = 0, set(),range(len(datalabel))
     while(loop!=3):
          temp = FIPF(datamatrix ,datalabel)-datalabel.reshape(-1,1)
          '''
          tni:represent identify noise index in df.index
          ndi:represent all of noise index set
          rei:represent remove noise index's and remain index of df.index
          '''
          tni = np.array(temp).nonzero()[0]
          ndi = ndi.union(set(np.array(Index)[tni]))
          rei = list(set(range(len(datalabel)))-set(tni))
          datamatrix, datalabel= datamatrix[rei], datalabel[rei]
          Index = np.array(Index)[rei]
          if len(tni)<=np.floor(len(datalabel)*1.0/100):
               loop = loop + 1
          else:
               loop=0
     ndi = list(ndi)
     return ndi
# %%                                  ALNR 
def ALNR(datamatrix,datalabel):  
     ndi=set()     #final noise data
     Index = range(len(datalabel))
     IN_F = [1]
     while len(IN_F)>0 :
          Tdatamatrix,Tdatalabel=datamatrix[Index],datalabel[Index]
          # build based svm classifier
          clf = SVC()
          clf.fit(Tdatamatrix,Tdatalabel)
          sv_ = list(clf.support_)
          sv_matrix,sv_label = Tdatamatrix[sv_],Tdatalabel[sv_]
          nsv_ = list(set(range(len(Index)))-set(sv_))
          nsv_matrix,nsv_label = Tdatamatrix[nsv_],Tdatalabel[nsv_]
          if len(set(nsv_label))==1 or len(nsv_label)<2:
              break
          #using nonsupport vector build classifier
          clf = SVC()
          clf.fit(nsv_matrix,nsv_label)
          temp = (clf.predict(sv_matrix)-sv_label).astype(int)
          IN_F = np.array(sv_)[list(temp.nonzero()[0])]
          ndi = ndi.union(set(np.array(Index)[IN_F]))
          Index = np.array(Index)[list(set(range(len(Index)))-set(IN_F))]
     ndi = list(ndi)
     return ndi

      
