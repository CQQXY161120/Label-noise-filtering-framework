# -*- coding: utf-8 -*-
"""
Created on Mon Aug 06 10:04:23 2018

@author: CQQ

E-mail:chenqq18@126.com
"""
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import CompareAlgorithm as CA
import Function as PF
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KDTree,KNeighborsClassifier

# %% preset format of result
datasets=[['seeds_210_7_3.txt']*3,['glass_175_9_3.txt']*3,['pima_768_8_2.txt']*3,['monk_432_6_2.txt']*3,['sonar_208_60_2.txt']*3,
          ['yeast_1136_8_3.txt']*3,['wdbc_569_30_2.txt']*3,['vehicle_846_18_4.txt']*3,['ionosphere_351_33_2.txt']*3,
          ['twonorm_7400_20_2.txt']*3,['Titanic_2201_3_2.txt']*3,['satimage_6435_36_6.txt']*3,['Letter_2204_16_3.txt']*3,
          ['segment_2310_19_7.txt']*3,['banana_5300_2_2.txt']*3,['Abalone_4177_9_3.txt']*3]
columns_n=[['f1','pre','KNN','SVM','C4.5']*8]

Result= pd.DataFrame(np.zeros([len(datasets)*3,40]),index=np.array(datasets).ravel(),columns=columns_n)
k = 5
# %% experiment for every datasets
#range(len(datasets))
for s in [2]:
      '''
      Variable declaration:
      '''
      ENN_F1,CF_F1,MVF_F1,IPF_F1,TWE_F1,INNF_F1,ALN_F1=0,0,0,0,0,0,0
      ENN_CMF1,CF_CMF1,MVF_CMF1,IPF_CMF1,TWE_CMF1,INNF_CMF1,ALN_CMF1=0,0,0,0,0,0,0
      ENN_CNF1,CF_CNF1,MVF_CNF1,IPF_CNF1,TWE_CNF1,INNF_CNF1,ALN_CNF1=0,0,0,0,0,0,0
      
      ENN_P1,CF_P1,MVF_P1,IPF_P1,TWE_P1,INNF_P1,ALN_P1=0,0,0,0,0,0,0
      ENN_CMP1,CF_CMP1,MVF_CMP1,IPF_CMP1,TWE_CMP1,INNF_CMP1,ALN_CMP1=0,0,0,0,0,0,0
      ENN_CNP1,CF_CNP1,MVF_CNP1,IPF_CNP1,TWE_CNP1,INNF_CNP1,ALN_CNP1=0,0,0,0,0,0,0
      
      ENN_KNN,CF_KNN,MVF_KNN,IPF_KNN,TWE_KNN,INNF_KNN,ALN_KNN=0,0,0,0,0,0,0
      ENN_CMKNN,CF_CMKNN,MVF_CMKNN,IPF_CMKNN,TWE_CMKNN,INNF_CMKNN,ALN_CMKNN=0,0,0,0,0,0,0
      ENN_CNKNN,CF_CNKNN,MVF_CNKNN,IPF_CNKNN,TWE_CNKNN,INNF_CNKNN,ALN_CNKNN=0,0,0,0,0,0,0
        
      ENN_SVM,CF_SVM,MVF_SVM,IPF_SVM,TWE_SVM,INNF_SVM,ALN_SVM=0,0,0,0,0,0,0
      ENN_CMSVM,CF_CMSVM,MVF_CMSVM,IPF_CMSVM,TWE_CMSVM,INNF_CMSVM,ALN_CMSVM=0,0,0,0,0,0,0
      ENN_CNSVM,CF_CNSVM,MVF_CNSVM,IPF_CNSVM,TWE_CNSVM,INNF_CNSVM,ALN_CNSVM=0,0,0,0,0,0,0
        
      ENN_C45,CF_C45,MVF_C45,IPF_C45,TWE_C45,INNF_C45,ALN_C45=0,0,0,0,0,0,0
      ENN_CMC45,CF_CMC45,MVF_CMC45,IPF_CMC45,TWE_CMC45,INNF_CMC45,ALN_CMC45=0,0,0,0,0,0,0
      ENN_CNC45,CF_CNC45,MVF_CNC45,IPF_CNC45,TWE_CNC45,INNF_CNC45,ALN_CNC45=0,0,0,0,0,0,0
      
      PSAM_KNN,PSAM_SVM,PSAM_C45 = 0,0,0
      
      print (s)
      # %% Load datasets and preprocess the datasets
      DataMatrix,DataLabel = PF.Load_Data(datasets[s][0])
      '''
      Normatrix,MapMatrix respectively represent Max-Min preprocess and map data matrix in a hypersphere which every data have same normal
      T:represent times for add artificial label noise
      NoiseRatio:represent add artificial label noise ratio  
      '''
      Normatrix = PF.PreprocessAtribute(DataMatrix)[0] 
      MapMatrix = PF.PreprocessAtribute(Normatrix)[1]
      kf = StratifiedKFold(n_splits=k) 
      T, NoiseRatio= 10, 0.1
      for train, test in kf.split(Normatrix,DataLabel):
          # %% separate dataset into train and test, and use traindata continue experiment
          '''
          NBM_T:represent based on NonBorderMatrix_N divided train and test
          Index_NBL:represent index of different kinds of label in NonBorderMatrix
          NBT_Kdt:represnt based on NBtrain_Matrix bulid KD-Tree
          SDNB:represent sored distance of NBtrain_Matrix
          '''
          DM_Tr,DL_Tr= Normatrix[train],DataLabel[train]
          DM_Te,DL_Te= Normatrix[test],DataLabel[test]
          
          MDM_Tr,MDL_Tr= MapMatrix[train],DataLabel[train]
          MDM_Te,MDL_Te= MapMatrix[test],DataLabel[test]
          
          Index_CL = PF.RecordIndexOfClass(DL_Tr)
          SDNB = np.argsort(PF.calculate(DM_Tr))
          neigh,tree,svm = KNeighborsClassifier(n_neighbors=3),DecisionTreeClassifier(min_samples_leaf=2),SVC()
          for Tt in range(T):
             # %% Add artificial label noise
             '''
             IN_NBL:represent Index of add noise in the NBL_Tr
             NBL_TN:represent noised lable of NBL_Tr
             '''
             print (Tt)
             NBT_Kdt = KDTree(DM_Tr,leaf_size=2)
             IN_NL,NL_TN = PF.AddNoise(DM_Tr,DL_Tr,NoiseRatio,NBT_Kdt,Index_CL)
             # %% experiment with no CCA preprocessing to deal with label noise
             #=========================  ENN  =================================
             IN_ENN = CA.ENN(DM_Tr,NL_TN,NBT_Kdt)
             pre,f1,ACC_Nei,ACC_SVM,ACC_Tree = PF.Indicator(IN_NL,IN_ENN,NL_TN,DM_Tr,neigh,tree,svm,DM_Te,DL_Te)

             ENN_F1 = ENN_F1 + f1
             ENN_P1 = ENN_P1 + pre
             ENN_KNN = ENN_KNN + ACC_Nei
             ENN_SVM = ENN_SVM + ACC_SVM
             ENN_C45 = ENN_C45 + ACC_Tree
             #=========================  CF  ==================================
             IN_CF = CA.CF(DM_Tr,NL_TN)
             pre,f1,ACC_Nei,ACC_SVM,ACC_Tree = PF.Indicator(IN_NL,IN_CF,NL_TN,DM_Tr,neigh,tree,svm,DM_Te,DL_Te)
             
             CF_F1 = CF_F1 + f1
             CF_P1 = CF_P1 + pre
             CF_KNN = CF_KNN + ACC_Nei
             CF_SVM = CF_SVM + ACC_SVM
             CF_C45 = CF_C45 + ACC_Tree
             
             #=========================  MVF  =================================
             IN_MVF = CA.MVF(DM_Tr,NL_TN,DM_Tr,NL_TN)
             pre,f1,ACC_Nei,ACC_SVM,ACC_Tree = PF.Indicator(IN_NL,IN_MVF,NL_TN,DM_Tr,neigh,tree,svm,DM_Te,DL_Te)
             
             MVF_F1 = MVF_F1 + f1
             MVF_P1 = MVF_P1 + pre
             MVF_KNN = MVF_KNN + ACC_Nei
             MVF_SVM = MVF_SVM + ACC_SVM
             MVF_NC45 = MVF_C45 + ACC_Tree

             #=========================  IPF  =================================
             IN_IPF = CA.IPF(DM_Tr,NL_TN)
             pre,f1,ACC_Nei,ACC_SVM,ACC_Tree = PF.Indicator(IN_NL,IN_IPF,NL_TN,DM_Tr,neigh,tree,svm,DM_Te,DL_Te)
             
             IPF_F1 = IPF_F1 + f1
             IPF_P1 = IPF_P1 + pre
             IPF_KNN = IPF_KNN + ACC_Nei
             IPF_SVM = IPF_SVM + ACC_SVM
             IPF_C45 = IPF_C45 + ACC_Tree
             #=========================  TWE  =================================
             IN_TWE = CA.TWE(DM_Tr,NL_TN)
             pre,f1,ACC_Nei,ACC_SVM,ACC_Tree = PF.Indicator(IN_NL,IN_TWE,NL_TN,DM_Tr,neigh,tree,svm,DM_Te,DL_Te)
             
             TWE_F1 =  TWE_F1 + f1 
             TWE_P1 = TWE_P1 + pre
             TWE_KNN = TWE_KNN + ACC_Nei
             TWE_SVM = TWE_SVM + ACC_SVM
             TWE_C45 = TWE_C45 + ACC_Tree
             #=========================  INFFC  ===============================
             dist,ind = NBT_Kdt.query(DM_Tr,k=5)
             IN_INNF = CA.INNFC(DM_Tr,NL_TN,ind)
             pre,f1,ACC_Nei,ACC_SVM,ACC_Tree = PF.Indicator(IN_NL,IN_INNF,NL_TN,DM_Tr,neigh,tree,svm,DM_Te,DL_Te)

             INNF_F1 = INNF_F1 + f1
             INNF_P1 = INNF_P1 + pre
             INNF_KNN = INNF_KNN + ACC_Nei
             INNF_SVM = INNF_SVM + ACC_SVM
             INNF_C45 = INNF_C45 + ACC_Tree
             #=========================  PSAM  ================================
             confmatrix = CA.ConputeConfMatrix(DM_Tr,NL_TN)
             knn_acc,svm_acc,c45_acc = CA.PSAM(neigh,svm,tree,DM_Tr,DL_Tr,DM_Te,DL_Te,confmatrix)
             PSAM_KNN =  PSAM_KNN + knn_acc
             PSAM_SVM =  PSAM_SVM + svm_acc
             PSAM_C45 =  PSAM_C45 + c45_acc
             #=========================  ALNR  ================================
             IN_ALN = CA.ALNR(DM_Tr,NL_TN)
             pre,f1,ACC_Nei,ACC_SVM,ACC_Tree = PF.Indicator(IN_NL,IN_ALN,NL_TN,DM_Tr,neigh,tree,svm,DM_Te,DL_Te)
             
             ALN_F1 = ALN_F1 + f1
             ALN_P1 = ALN_P1 + pre
             ALN_KNN = ALN_KNN + ACC_Nei
             ALN_SVM = ALN_SVM + ACC_SVM
             ALN_C45 = ALN_C45 + ACC_Tree
             # %% experiment with map data matrix CCA preprocessing to deal with label noise
             TSD = np.argsort(PF.calculate(MDM_Tr)) 
             CompareMatrix = PF.CompareNoiseLabel(TSD,NL_TN)
             TI_CCA = PF.RCCA(CompareMatrix) 
             
             TDM,TDL = MDM_Tr[TI_CCA],NL_TN[TI_CCA]
             DM_Kdt = KDTree(TDM,leaf_size=2)
             #=========================  ENN  =================================
             IN_ENN = PF.SCCA(np.array(TI_CCA)[CA.ENN(TDM,TDL,DM_Kdt)],CompareMatrix)
             pre,f1,ACC_Nei,ACC_SVM,ACC_Tree = PF.Indicator(IN_NL,IN_ENN,MDL_Tr,MDM_Tr,neigh,tree,svm,MDM_Te,MDL_Te)

             ENN_CMF1 = ENN_CMF1 + f1
             ENN_CMP1 = ENN_CMP1 + pre
             ENN_CMKNN = ENN_CMKNN + ACC_Nei
             ENN_CMSVM = ENN_CMSVM + ACC_SVM
             ENN_CMC45 = ENN_CMC45 + ACC_Tree
             #=========================  CF  ==================================
             IN_CF = PF.SCCA(np.array(TI_CCA)[CA.CF(TDM,TDL)],CompareMatrix)
             pre,f1,ACC_Nei,ACC_SVM,ACC_Tree = PF.Indicator(IN_NL,IN_CF,MDL_Tr,MDM_Tr,neigh,tree,svm,MDM_Te,MDL_Te)

             CF_CMF1 = CF_CMF1 + f1
             CF_CMP1 = CF_CMP1 + pre
             CF_CMKNN = CF_CMKNN + ACC_Nei
             CF_CMSVM = CF_CMSVM + ACC_SVM
             CF_CMC45 = CF_CMC45 + ACC_Tree
             
             #=========================  MVF  =================================
             IN_MVF = PF.SCCA(np.array(TI_CCA)[CA.MVF(TDM,TDL,TDM,TDL)],CompareMatrix)
             pre,f1,ACC_Nei,ACC_SVM,ACC_Tree = PF.Indicator(IN_NL,IN_MVF,MDL_Tr,MDM_Tr,neigh,tree,svm,MDM_Te,MDL_Te)
             
             MVF_CMF1 = MVF_CMF1 + f1
             MVF_CMP1 = MVF_CMP1 + pre
             MVF_CMKNN = MVF_CMKNN + ACC_Nei
             MVF_CMSVM = MVF_CMSVM + ACC_SVM
             MVF_CMC45 = MVF_CMC45 + ACC_Tree

             #=========================  IPF  =================================
             IN_IPF = PF.SCCA(np.array(TI_CCA)[CA.IPF(TDM,TDL)],CompareMatrix)
             pre,f1,ACC_Nei,ACC_SVM,ACC_Tree = PF.Indicator(IN_NL,IN_IPF,MDL_Tr,MDM_Tr,neigh,tree,svm,MDM_Te,MDL_Te)

             IPF_CMF1 = IPF_CMF1 + f1
             IPF_CMP1 = IPF_CMP1 + pre
             IPF_CMKNN = IPF_CMKNN + ACC_Nei
             IPF_CMSVM = IPF_CMSVM + ACC_SVM
             IPF_CMC45 = IPF_CMC45 + ACC_Tree
             #=========================  TWE  =================================
             IN_TWE = PF.SCCA(np.array(TI_CCA)[CA.TWE(TDM,TDL)],CompareMatrix)
             pre,f1,ACC_Nei,ACC_SVM,ACC_Tree = PF.Indicator(IN_NL,IN_TWE,MDL_Tr,MDM_Tr,neigh,tree,svm,MDM_Te,MDL_Te)

             TWE_CMF1 = TWE_CMF1 + f1
             TWE_CMP1 = TWE_CMP1 + pre
             TWE_CMKNN = TWE_CMKNN + ACC_Nei
             TWE_CMSVM = TWE_CMSVM + ACC_SVM
             TWE_CMC45 = TWE_CMC45 + ACC_Tree
             #=========================  INFFC  ===============================
             dist,ind = DM_Kdt.query(TDM,k=5)
             IN_INNF = PF.SCCA(np.array(TI_CCA)[CA.INNFC(TDM,TDL,ind)],CompareMatrix)
             pre,f1,ACC_Nei,ACC_SVM,ACC_Tree = PF.Indicator(IN_NL,IN_INNF,MDL_Tr,MDM_Tr,neigh,tree,svm,MDM_Te,MDL_Te)

             INNF_CMF1 = INNF_CMF1 + f1
             INNF_CMP1 = INNF_CMP1 + pre
             INNF_CMKNN = INNF_CMKNN + ACC_Nei
             INNF_CMSVM = INNF_CMSVM + ACC_SVM
             INNF_CMC45 = INNF_CMC45 + ACC_Tree
             #=========================  ALNR  ================================
             IN_ALNR = PF.SCCA(np.array(TI_CCA)[CA.ALNR(TDM,TDL)],CompareMatrix)
             pre,f1,ACC_Nei,ACC_SVM,ACC_Tree = PF.Indicator(IN_NL,IN_ALNR,MDL_Tr,MDM_Tr,neigh,tree,svm,MDM_Te,MDL_Te)

             ALN_CMF1 = ALN_CMF1 + f1
             ALN_CMP1 = ALN_CMP1 + pre
             ALN_CMKNN = ALN_CMKNN + ACC_Nei
             ALN_CMSVM = ALN_CMSVM + ACC_SVM
             ALN_CMC45 = ALN_CMC45 + ACC_Tree
             # %% experiment with Normal data matrix CCA preprocessing to deal with label noise
             TSD = np.argsort(PF.calculate(DM_Tr)) 
             CompareMatrix = PF.CompareNoiseLabel(TSD,NL_TN)
             TI_CCA = PF.RCCA(CompareMatrix) 
             
             TDM,TDL = DM_Tr[TI_CCA],NL_TN[TI_CCA]
             DM_Kdt = KDTree(TDM,leaf_size=2)
             #=========================  ENN  =================================
             IN_ENN = PF.SCCA(np.array(TI_CCA)[CA.ENN(TDM,TDL,DM_Kdt)],CompareMatrix)
             pre,f1,ACC_Nei,ACC_SVM,ACC_Tree = PF.Indicator(IN_NL,IN_ENN,DL_Tr,DM_Tr,neigh,tree,svm,DM_Te,DL_Te)

             ENN_CNF1 = ENN_CNF1 + f1
             ENN_CNP1 = ENN_CNP1 + pre
             ENN_CNKNN = ENN_CNKNN + ACC_Nei
             ENN_CNSVM = ENN_CNSVM + ACC_SVM
             ENN_CNC45 = ENN_CNC45 + ACC_Tree
             #=========================  CF  ==================================
             IN_CF = PF.SCCA(np.array(TI_CCA)[CA.CF(TDM,TDL)],CompareMatrix)
             pre,f1,ACC_Nei,ACC_SVM,ACC_Tree = PF.Indicator(IN_NL,IN_CF,DL_Tr,DM_Tr,neigh,tree,svm,DM_Te,DL_Te)

             CF_CNF1 = CF_CNF1 + f1
             CF_CNP1 = CF_CNP1 + pre
             CF_CNKNN = CF_CNKNN + ACC_Nei
             CF_CNSVM = CF_CNSVM + ACC_SVM
             CF_CNC45 = CF_CNC45 + ACC_Tree
             
             #=========================  MVF  =================================
             IN_MVF = PF.SCCA(np.array(TI_CCA)[CA.MVF(TDM,TDL,TDM,TDL)],CompareMatrix)
             pre,f1,ACC_Nei,ACC_SVM,ACC_Tree = PF.Indicator(IN_NL,IN_MVF,DL_Tr,DM_Tr,neigh,tree,svm,DM_Te,DL_Te)

             MVF_CNF1 = MVF_CNF1 + f1
             MVF_CNP1 = MVF_CNP1 + pre
             MVF_CNKNN = MVF_CNKNN + ACC_Nei
             MVF_CNSVM = MVF_CNSVM + ACC_SVM
             MVF_CNC45 = MVF_CNC45 + ACC_Tree
            
             #=========================  IPF  =================================
             IN_IPF = PF.SCCA(np.array(TI_CCA)[CA.IPF(TDM,TDL)],CompareMatrix)
             pre,f1,ACC_Nei,ACC_SVM,ACC_Tree = PF.Indicator(IN_NL,IN_IPF,DL_Tr,DM_Tr,neigh,tree,svm,DM_Te,DL_Te)
             
             IPF_CNF1 = IPF_CNF1 + f1
             IPF_CNP1 = IPF_CNP1 + pre
             IPF_CNKNN = IPF_CNKNN + ACC_Nei
             IPF_CNSVM = IPF_CNSVM + ACC_SVM
             IPF_CNC45 = IPF_CNC45 + ACC_Tree
             #=========================  TWE  =================================
             IN_TWE = PF.SCCA(np.array(TI_CCA)[CA.TWE(TDM,TDL)],CompareMatrix)
             pre,f1,ACC_Nei,ACC_SVM,ACC_Tree = PF.Indicator(IN_NL,IN_TWE,DL_Tr,DM_Tr,neigh,tree,svm,DM_Te,DL_Te)

             TWE_CNF1 = TWE_CNF1 + f1
             TWE_CNP1 = TWE_CNP1 + pre
             TWE_CNKNN = TWE_CNKNN + ACC_Nei
             TWE_CNSVM = TWE_CNSVM + ACC_SVM
             TWE_CNC45 = TWE_CNC45 + ACC_Tree
             #=========================  INFFC  ===============================
             dist,ind = DM_Kdt.query(TDM,k=5)
             IN_INNF = PF.SCCA(np.array(TI_CCA)[CA.INNFC(TDM,TDL,ind)],CompareMatrix)
             pre,f1,ACC_Nei,ACC_SVM,ACC_Tree = PF.Indicator(IN_NL,IN_INNF,DL_Tr,DM_Tr,neigh,tree,svm,DM_Te,DL_Te)

             INNF_CNF1 = INNF_CNF1 + f1
             INNF_CNP1 = INNF_CNP1 + pre
             INNF_CNKNN = INNF_CNKNN + ACC_Nei
             INNF_CNSVM = INNF_CNSVM + ACC_SVM
             INNF_CNC45 = INNF_CNC45 + ACC_Tree
             #=========================  ALNR  ================================
             IN_ALNR = PF.SCCA(np.array(TI_CCA)[CA.ALNR(TDM,TDL)],CompareMatrix)
             pre,f1,ACC_Nei,ACC_SVM,ACC_Tree = PF.Indicator(IN_NL,IN_MVF,DL_Tr,DM_Tr,neigh,tree,svm,DM_Te,DL_Te)

             ALN_CNF1 = ALN_CNF1 + f1
             ALN_CNP1 = ALN_CNP1 + pre
             ALN_CNKNN = ALN_CNKNN + ACC_Nei
             ALN_CNSVM = ALN_CNSVM + ACC_SVM
             ALN_CNC45 = ALN_CNC45 + ACC_Tree
             
    
      # %% result ensemble    
      Result.iloc[3*s][0],Result.iloc[3*s][1],Result.iloc[3*s][2],Result.iloc[3*s][3],Result.iloc[3*s][4] = ENN_F1*1.0/(T*k),ENN_P1*1.0/(T*k),ENN_KNN*1.0/(T*k),ENN_SVM*1.0/(T*k),ENN_C45*1.0/(T*k)     
      Result.iloc[3*s][5],Result.iloc[3*s][6],Result.iloc[3*s][7],Result.iloc[3*s][8],Result.iloc[3*s][9] = CF_F1*1.0/(T*k),CF_P1*1.0/(T*k),CF_KNN*1.0/(T*k),CF_SVM*1.0/(T*k),CF_C45*1.0/(T*k)
      Result.iloc[3*s][10],Result.iloc[3*s][11],Result.iloc[3*s][12],Result.iloc[3*s][13],Result.iloc[3*s][14] = MVF_F1*1.0/(T*k),MVF_P1*1.0/(T*k),MVF_KNN*1.0/(T*k),MVF_SVM*1.0/(T*k),MVF_C45*1.0/(T*k)  
      Result.iloc[3*s][15],Result.iloc[3*s][16],Result.iloc[3*s][17],Result.iloc[3*s][18],Result.iloc[3*s][19] = IPF_F1*1.0/(T*k),IPF_P1*1.0/(T*k),IPF_KNN*1.0/(T*k),IPF_SVM*1.0/(T*k),IPF_C45*1.0/(T*k)
      Result.iloc[3*s][20],Result.iloc[3*s][21],Result.iloc[3*s][22],Result.iloc[3*s][23],Result.iloc[3*s][24] = TWE_F1*1.0/(T*k),TWE_P1*1.0/(T*k),TWE_KNN*1.0/(T*k),TWE_SVM*1.0/(T*k),TWE_C45*1.0/(T*k)
      Result.iloc[3*s][25],Result.iloc[3*s][26],Result.iloc[3*s][27],Result.iloc[3*s][28],Result.iloc[3*s][29] = INNF_F1*1.0/(T*k),INNF_P1*1.0/(T*k),INNF_KNN*1.0/(T*k),INNF_SVM*1.0/(T*k),INNF_C45*1.0/(T*k)
      Result.iloc[3*s][30],Result.iloc[3*s][31],Result.iloc[3*s][32],Result.iloc[3*s][33],Result.iloc[3*s][34] = ALN_F1*1.0/(T*k),ALN_P1*1.0/(T*k),ALN_KNN*1.0/(T*k),ALN_SVM*1.0/(T*k),ALN_C45*1.0/(T*k)
      Result.iloc[3*s][37],Result.iloc[3*s][38],Result.iloc[3*s][39] = PSAM_KNN*1.0/(T*k),PSAM_SVM*1.0/(T*k),PSAM_C45*1.0/(T*k)
      
      Result.iloc[3*s+1][0],Result.iloc[3*s+1][1],Result.iloc[3*s+1][2],Result.iloc[3*s+1][3],Result.iloc[3*s+1][4] = ENN_CMF1*1.0/(T*k),ENN_CMP1*1.0/(T*k),ENN_CMKNN*1.0/(T*k),ENN_CMSVM*1.0/(T*k),ENN_CMC45*1.0/(T*k) 
      Result.iloc[3*s+1][5],Result.iloc[3*s+1][6],Result.iloc[3*s+1][7],Result.iloc[3*s+1][8],Result.iloc[3*s+1][9] = CF_CMF1*1.0/(T*k),CF_CMP1*1.0/(T*k),CF_CMKNN*1.0/(T*k),CF_CMSVM*1.0/(T*k),CF_CMC45*1.0/(T*k)
      Result.iloc[3*s+1][10],Result.iloc[3*s+1][11],Result.iloc[3*s+1][12],Result.iloc[3*s+1][13],Result.iloc[3*s+1][14] = MVF_CMF1*1.0/(T*k),MVF_CMP1*1.0/(T*k),MVF_CMKNN*1.0/(T*k),MVF_CMSVM*1.0/(T*k),MVF_CMC45*1.0/(T*k)
      Result.iloc[3*s+1][15],Result.iloc[3*s+1][16],Result.iloc[3*s+1][17],Result.iloc[3*s+1][18],Result.iloc[3*s+1][19] = IPF_CMF1*1.0/(T*k),IPF_CMP1*1.0/(T*k),IPF_CMKNN*1.0/(T*k),IPF_CMSVM*1.0/(T*k),IPF_CMC45*1.0/(T*k)
      Result.iloc[3*s+1][20],Result.iloc[3*s+1][21],Result.iloc[3*s+1][22],Result.iloc[3*s+1][23],Result.iloc[3*s+1][24] = TWE_CMF1*1.0/(T*k),TWE_CMP1*1.0/(T*k),TWE_CMKNN*1.0/(T*k),TWE_CMSVM*1.0/(T*k),TWE_CMC45*1.0/(T*k)
      Result.iloc[3*s+1][25],Result.iloc[3*s+1][26],Result.iloc[3*s+1][27],Result.iloc[3*s+1][28],Result.iloc[3*s+1][29] = INNF_CMF1*1.0/(T*k),INNF_CMP1*1.0/(T*k),INNF_CMKNN*1.0/(T*k),INNF_CMSVM*1.0/(T*k),INNF_CMC45*1.0/(T*k)
      Result.iloc[3*s+1][30],Result.iloc[3*s+1][31],Result.iloc[3*s+1][32],Result.iloc[3*s+1][33],Result.iloc[3*s+1][34] = ALN_CMF1*1.0/(T*k),ALN_CMP1*1.0/(T*k),ALN_CMKNN*1.0/(T*k),ALN_CMSVM*1.0/(T*k),ALN_CMC45*1.0/(T*k)
      
      Result.iloc[3*s+2][0],Result.iloc[3*s+2][1],Result.iloc[3*s+2][2],Result.iloc[3*s+2][3],Result.iloc[3*s+2][4] = ENN_CNF1*1.0/(T*k),ENN_CNP1*1.0/(T*k),ENN_CNKNN*1.0/(T*k),ENN_CNSVM*1.0/(T*k),ENN_CNC45*1.0/(T*k) 
      Result.iloc[3*s+2][5],Result.iloc[3*s+2][6],Result.iloc[3*s+2][7],Result.iloc[3*s+2][8],Result.iloc[3*s+2][9] = CF_CNF1*1.0/(T*k),CF_CNP1*1.0/(T*k),CF_CNKNN*1.0/(T*k),CF_CNSVM*1.0/(T*k),CF_CNC45*1.0/(T*k)
      Result.iloc[3*s+2][10],Result.iloc[3*s+2][11],Result.iloc[3*s+2][12],Result.iloc[3*s+2][13],Result.iloc[3*s+2][14] = MVF_CNF1*1.0/(T*k),MVF_CNP1*1.0/(T*k),MVF_CNKNN*1.0/(T*k),MVF_CNSVM*1.0/(T*k),MVF_CNC45*1.0/(T*k)
      Result.iloc[3*s+2][15],Result.iloc[3*s+2][16],Result.iloc[3*s+2][17],Result.iloc[3*s+2][18],Result.iloc[3*s+2][19] = IPF_CNF1*1.0/(T*k),IPF_CNP1*1.0/(T*k),IPF_CNKNN*1.0/(T*k),IPF_CNSVM*1.0/(T*k),IPF_CNC45*1.0/(T*k)
      Result.iloc[3*s+2][20],Result.iloc[3*s+2][21],Result.iloc[3*s+2][22],Result.iloc[3*s+2][23],Result.iloc[3*s+2][24] = TWE_CNF1*1.0/(T*k),TWE_CNP1*1.0/(T*k),TWE_CNKNN*1.0/(T*k),TWE_CNSVM*1.0/(T*k),TWE_CNC45*1.0/(T*k)
      Result.iloc[3*s+2][25],Result.iloc[3*s+2][26],Result.iloc[3*s+2][27],Result.iloc[3*s+2][28],Result.iloc[3*s+2][29] = INNF_CNF1*1.0/(T*k),INNF_CNP1*1.0/(T*k),INNF_CNKNN*1.0/(T*k),INNF_CNSVM*1.0/(T*k),INNF_CNC45*1.0/(T*k)
      Result.iloc[3*s+2][30],Result.iloc[3*s+2][31],Result.iloc[3*s+2][32],Result.iloc[3*s+2][33],Result.iloc[3*s+2][34] = ALN_CNF1*1.0/(T*k),ALN_CNP1*1.0/(T*k),ALN_CNKNN*1.0/(T*k),ALN_CNSVM*1.0/(T*k),ALN_CNC45*1.0/(T*k)
     
      Result.to_csv ("KN.csv" , encoding = "utf-8")     
