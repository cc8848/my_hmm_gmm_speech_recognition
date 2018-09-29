# -----------------------------------------------------------------------------------------------------
'''
&usage:		HMM-GMM的孤立词识别模型
@author:	hongwen sun
'''
# -----------------------------------------------------------------------------------------------------
# 导入依赖库，特别需要注意hmmlearn
from python_speech_features import mfcc
from scipy.io import wavfile
from hmmlearn import hmm
from sklearn.externals import joblib
import numpy as np
import os


# -----------------------------------------------------------------------------------------------------
'''
&usage:		准备所需数据
'''
# -----------------------------------------------------------------------------------------------------
# 生成wavdict，key=wavid，value=wavfile
def gen_wavlist(wavpath):
	wavdict = {}
	labeldict = {}
	for (dirpath, dirnames, filenames) in os.walk(wavpath):
		for filename in filenames:
			if filename.endswith('.wav'):
				filepath = os.sep.join([dirpath, filename])
				fileid = filename.strip('.wav')
				wavdict[fileid] = filepath
				label = fileid.split('_')[1]
				labeldict[fileid] = label
	return wavdict, labeldict




# 特征提取，feat = compute_mfcc(wadict[wavid])
def compute_mfcc(file):
	fs, audio = wavfile.read(file)
	# 这里我故意fs/2,有些类似减小step，不建议这样做，投机取巧做法
	mfcc_feat = mfcc(audio, samplerate=(fs/2), numcep=26)
	return mfcc_feat


# -----------------------------------------------------------------------------------------------------
'''
&usage:		搭建HMM-GMM的孤立词识别模型
参数意义：
	CATEGORY:	所有标签的列表
	n_comp:		每个孤立词中的状态数
	n_mix:		每个状态包含的混合高斯数量
	cov_type:	协方差矩阵的类型
	n_iter:		训练迭代次数
'''
# -----------------------------------------------------------------------------------------------------

class Model():
	"""docstring for Model"""
	def __init__(self, CATEGORY=None, n_comp=3, n_mix = 3, cov_type='diag', n_iter=1000):
		super(Model, self).__init__()
		self.CATEGORY = CATEGORY
		self.category = len(CATEGORY)
		self.n_comp = n_comp
		self.n_mix = n_mix
		self.cov_type = cov_type
		self.n_iter = n_iter
		# 关键步骤，初始化models，返回特定参数的模型的列表
		self.models = []
		for k in range(self.category):
			model = hmm.GMMHMM(n_components=self.n_comp, n_mix = self.n_mix, 
								covariance_type=self.cov_type, n_iter=self.n_iter)
			self.models.append(model)

	# 模型训练
	def train(self, wavdict=None, labeldict=None):
		for k in range(10):
			subdata = []
			model = self.models[k]
			for x in wavdict:
				if labeldict[x] == self.CATEGORY[k]:
					mfcc_feat = compute_mfcc(wavdict[x])
					model.fit(mfcc_feat)

	# 使用特定的测试集合进行测试
	def test(self, wavdict=None, labeldict=None):
		result = []
		for k in range(self.category):
			subre = []
			label = []
			model = self.models[k]
			for x in wavdict:
				mfcc_feat = compute_mfcc(wavdict[x])
				# 生成每个数据在当前模型下的得分情况
				re = model.score(mfcc_feat)
				subre.append(re)
				label.append(labeldict[x])
			# 汇总得分情况
			result.append(subre)
		# 选取得分最高的种类
		result = np.vstack(result).argmax(axis=0)
		# 返回种类的类别标签
		result = [self.CATEGORY[label] for label in result]
		print('识别得到结果：\n',result)
		print('原始标签类别：\n',label)
		# 检查识别率，为：正确识别的个数/总数
		totalnum = len(label)
		correctnum = 0
		for i in range(totalnum):
		 	if result[i] == label[i]:
		 	 	correctnum += 1 
		print('识别率:', correctnum/totalnum)


	def save(self, path="models.pkl"):
		# 利用external joblib保存生成的hmm模型
		joblib.dump(self.models, path)


	def load(self, path="models.pkl"):
		# 导入hmm模型
		self.models = joblib.load(path)


# -----------------------------------------------------------------------------------------------------
'''
&usage:		使用模型进行训练和识别
'''
# -----------------------------------------------------------------------------------------------------
# 准备训练所需数据
CATEGORY = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
wavdict, labeldict = gen_wavlist('training_data')
testdict, testlabel = gen_wavlist('test_data')	
# 进行训练
models = Model(CATEGORY=CATEGORY)
models.train(wavdict=wavdict, labeldict=labeldict)
models.save()
models.load()
models.test(wavdict=wavdict, labeldict=labeldict)
models.test(wavdict=testdict, labeldict=testlabel)
