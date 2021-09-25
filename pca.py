from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
#dir_fonts=FontProperties(fname='/System/Library/Fonts/Menlo.ttc', size=12, style='oblique', variant='normal'); 
#plt.rcParams['font.family'] = dir_fonts.get_name()

dir_fonts=FontProperties(fname='/System/Library/Fonts/ヒラギノ丸ゴ ProN W4.ttc', size=12, style='oblique', variant='normal'); 
plt.rcParams['font.family'] = dir_fonts.get_name()
import datetime

#from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
#from tensorflow.keras.models import load_model
#from tensorflow.keras.preprocessing.sequence import pad_sequences
#from tensorflow import keras
#import tensorflow as tf 







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
		dict_initialization={'figsize':(11,7), 'scatter_marker_size':250}
		for jv in list(dict_initialization):
			if hasattr(self, jv)==False:
				setattr(self, jv, dict_initialization.get(jv, 'Error!') )
		#self.figsize=(10,6)
		#self.scatter_marker_size=250;
	
	def __call__(self):
		print(' call method was used')	
	
	def draw(self):
		fig0=plt.figure(figsize=self.figsize )
		gspec0=fig0.add_gridspec(1,2);
		ax0=fig0.add_subplot(gspec0[0,0])
		axA=fig0.add_subplot(gspec0[0,1])
		df_data=pd.DataFrame(data=self.principal_comp_mat)
		df_data['student']=np.arange(1, df_data.shape[0]+1 )
		df_data['student']=df_data['student'].astype(str);
		df_data=df_data.rename(columns={0:'0th_principal', 1:'1st_principal'})
		sns.scatterplot(ax=ax0,data=df_data, x='0th_principal', y='1st_principal', s=self.scatter_marker_size, hue='student', palette='hsv' )
		ax0.grid()
		ax0.set_xlabel(ax0.get_xlabel() + ' {0:.2f}'.format( self.lambda_sorted[0] / self.sum_eigv) )
		ax0.set_ylabel(ax0.get_ylabel() + ' {0:.2f}'.format( self.lambda_sorted[1] / self.sum_eigv) )
		ax0.set_title("$ X U $")
		del(df_data)
		#
		df_data=pd.DataFrame(data=self.principal_comp_mat_cent)
		df_data['student']=np.arange(1, df_data.shape[0]+1 )
		df_data['student']=df_data['student'].astype(str);
		df_data=df_data.rename(columns={0:'0th_principal', 1:'1st_principal'})
		sns.scatterplot(ax=axA,data=df_data, x='0th_principal', y='1st_principal', s=self.scatter_marker_size, hue='student', palette='hsv'  )
		axA.grid()
		axA.set_xlabel(axA.get_xlabel() + ' {0:.2f}'.format( self.lambda_sorted[0] / self.sum_eigv) )
		axA.set_ylabel(axA.get_ylabel() + ' {0:.2f}'.format( self.lambda_sorted[1] / self.sum_eigv) )
		axA.set_title("$ X_{C} U $")
		plt.show()
		plt.close('all')
	
	def analyze(self, matA):
		matCov=np.cov(matA, rowvar=False)
		la0, u0=np.linalg.eig(matCov)
		print('la=', la0)
		self.sum_eigv=la0.sum()
		print('sum eigv=', self.sum_eigv)
		df_eig=pd.concat( [pd.DataFrame(la0.reshape(1,-1)), pd.DataFrame(u0)], axis=0).reset_index(drop=True)
		df_eig=df_eig.transpose().sort_values(by=[0], ascending=[False]).transpose()
		self.lambda_sorted=df_eig.iloc[0,:].to_numpy()
		eigvec_sorted=df_eig[1:].to_numpy()
		print('la sorted=', self.lambda_sorted)
		print('eigv sorted=\n', eigvec_sorted)
		principal_mat_cent=np.matmul( ( matA - matA.mean(axis=0)), eigvec_sorted)
		principal_mat=np.matmul( matA, eigvec_sorted)
		#print('principal comp mat=\n', principal_mat)
		# principal_mat
		principal_loading_mat=np.zeros(shape=(4,4)) * np.nan
		for jth_principal_comp in np.arange(0,principal_mat.shape[1]):
			for kth_original_comp in np.arange(0,matA.shape[1]):
				principal_loading_mat[jth_principal_comp,kth_original_comp]=np.corrcoef(matA[:,kth_original_comp], principal_mat_cent[:,jth_principal_comp])[0,1]
		#
		self.principal_comp_mat_cent=principal_mat_cent;
		self.principal_comp_mat=principal_mat;
		self.principal_loading_mat=principal_loading_mat;
		print('loading mat=\n', principal_loading_mat)


#matA=np.array([ [2, 2, 3, 1], [9,8, 10,9], [8,3, 2,7], [7,1, 3,8], [2,9, 8,2],[5,4,5,5] ])
#print('matA shape=', matA.shape)
#matA


matX=np.array([
[85, 50, 50, 90], 
[80, 60, 70, 80], 
[60, 90, 90, 50], 
[40, 40, 50, 60], 
[75, 50, 50, 40], 
[30, 60, 60, 45], 
[50, 80, 75, 60], 
[80, 90, 90, 95]  
])

#print(matA.shape)
# matA

if __name__=='__main__':
	print('matX.shape=', matX.shape)
	pca0=PrincipalComponent();
	pca0.analyze(matA=matX);
	pca0.draw();
	
	#print('principal mat=\n', pca0.principal_comp_mat)
	df_matX=pd.DataFrame(matX, columns=['japanese', 'math', 'basic sciences', 'sociology'])
	df_matX['student']=np.arange(1, df_matX.shape[0]+1)
	df_matX=df_matX.set_index('student')
	
	df_principal_loading_mat=pd.DataFrame(pca0.principal_loading_mat, columns=['japanese', 'math', 'basic sciences', 'sociology'])
	df_principal_loading_mat['principal']=np.arange(1,  df_principal_loading_mat.shape[0]+1 );
	df_principal_loading_mat['principal']=df_principal_loading_mat['principal'].astype(str) + 'th_comp'
	df_principal_loading_mat=df_principal_loading_mat.set_index('principal')
	print('loading mat=\n', df_principal_loading_mat)
	print(' done at ', datetime.datetime.now().strftime('%b %dth, %Y'))
	with pd.ExcelWriter('pca_.xlsx') as excel_writer:
		df_matX.to_excel(excel_writer, sheet_name='matX', index=True, header=True);
		df_principal_loading_mat.to_excel(excel_writer, sheet_name='loading', header=True, index=True)




