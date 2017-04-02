import pandas as pd
from sklearn.preprocessing import scale
from numpy import *


def get_channel_ranks(file_name):
	matrix = pd.read_csv(file_name,header=None)
	rank = [[0 for i in range(132)] for j in range(132)]

	for i in range(len(matrix)):
		for j in range(len(matrix.loc[i])): 
			rank[matrix.loc[i][j]-1][j] += 1

	# print rank,'\n\n\n\n\n\n\n\n'
	#print sorted(range(len(rank)),key=lambda x:rank[x])

	df = pd.DataFrame(rank)
	df.to_csv('ranks-'+file_name)

# get_channel_ranks('ranks.csv')


def use_csp_channels(subject="",channels=""):
	left = pd.read_csv('./signals/'+subject+'-left-signals.csv',header=None)
	right = pd.read_csv('./signals/'+subject+'-right-signals.csv',header=None)
	print("data loaded")
	data = left.as_matrix()[:-(1375*10)]
	data = vstack((data,right.as_matrix()[:-(10*1375)][:]))
	test_data = left.as_matrix()[-(1375*10):]
	test_data = vstack((test_data,right.as_matrix()[-(10*1375):][:]))
	channels = pd.read_csv(channels,header=None)
	channels = list(channels)
	data= data[:,(channels)]
	data = scale(data)
	test_data = test_data[:,(channels)]
	test_data = scale(test_data)
	data = pd.DataFrame(data)
	test_data = pd.DataFrame(test_data)
	data.to_csv('scaled-training-data-'+subject+'.csv')
	test_data.to_csv('scaled-testing-data-'+subject+'.csv')

use_csp_channels('A02T','selected_channels.csv')