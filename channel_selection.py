import pandas as pd

def get_channel_ranks(file_name):
	matrix = pd.read_csv(file_name,header=None)
	rank = [[0 for i in range(22)] for j in range(22)]

	for i in range(len(matrix)):
		for j in range(len(matrix.loc[i])): 
			rank[matrix.loc[i][j]-1][j] += 1

	# print rank,'\n\n\n\n\n\n\n\n'
	#print sorted(range(len(rank)),key=lambda x:rank[x])

	df = pd.DataFrame(rank)
	df.to_csv('ranks-'+file_name)

get_channel_ranks('left_channel.csv')
get_channel_ranks('right_channel.csv')