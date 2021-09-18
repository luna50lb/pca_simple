from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
#dir_fonts=FontProperties(fname='/System/Library/Fonts/Menlo.ttc', size=12, style='oblique', variant='normal'); 
#plt.rcParams['font.family'] = dir_fonts.get_name()

dir_fonts=FontProperties(fname='/System/Library/Fonts/ヒラギノ丸ゴ ProN W4.ttc', size=12, style='oblique', variant='normal'); 
plt.rcParams['font.family'] = dir_fonts.get_name()
import datetime

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow import keras
import tensorflow as tf 







from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import numpy as np
import seaborn as sns 
import scipy
from sklearn import decomposition
import pandas as pd



class PrincipalComponent():
    def __init__(self):
        print('class PrincipalComponent used')
        self.figsize=(10,6)
        self.scatter_marker_size=250;
    
    def draw(self):
        fig0=plt.figure(figsize=self.figsize )
        gspec0=fig0.add_gridspec(1,1);
        ax0=fig0.add_subplot(gspec0[0,0])
        df_data=pd.DataFrame(data=self.principal_comp_mat)
        df_data=df_data.rename(columns={0:'0th_principal', 1:'1st_principal'})
        sns.scatterplot(ax=ax0,data=df_data, x='0th_principal', y='1st_principal', s=self.scatter_marker_size)
        ax0.grid()
        plt.show()
        plt.close('all')
    
    def analyze(self, matA):
        matCov=np.cov(matA, rowvar=False)
        la0, u0=np.linalg.eig(matCov)
        df_eig=pd.concat( [pd.DataFrame(la0.reshape(1,-1)), pd.DataFrame(u0)], axis=0).reset_index(drop=True)
        df_eig=df_eig.transpose().sort_values(by=[0], ascending=[False]).transpose()
        la0_sorted=df_eig.iloc[0,:].to_numpy()
        u0_sorted=df_eig[1:].to_numpy()
        print('la sorted=', la0_sorted)
        print('u0 sorted=\n', u0_sorted)
        principal_mat=np.matmul( ( matA - matA.mean(axis=0)), u0_sorted)
        #print('principal comp mat=\n', principal_mat)
        # principal_mat
        principal_loading_mat=np.zeros(shape=(4,4)) * np.nan
        for jth_principal_comp in np.arange(0,principal_mat.shape[1]):
            for kth_original_comp in np.arange(0,matA.shape[1]):
                principal_loading_mat[jth_principal_comp,kth_original_comp]=np.corrcoef(matA[:,kth_original_comp], principal_mat[:,jth_principal_comp])[0,1]
        #
        self.principal_comp_mat=principal_mat;
        self.principal_loading_mat=principal_loading_mat;
        print('loading mat=\n', principal_loading_mat)


matA=np.array([ [2, 2, 3, 1], [9,8, 10,9], [8,3, 2,7], [7,1, 3,8], [2,9, 8,2],[5,4,5,5] ])
print('matA shape=', matA.shape)
matA


print(matA.shape)
# matA

if __name__=='__main__':
    pca0=PrincipalComponent();
    pca0.analyze(matA=matA);
    pca0.draw();
    
    print('principal mat=\n', pca0.principal_comp_mat)
    print(' done at ', datetime.datetime.now().strftime('%b %dth, %Y'))