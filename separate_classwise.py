import pandas as pd
# import pdb
# pdb.set_trace()


def separate_classwise(filename = ""):

	df = pd.read_csv(filename)
	mat1 = df[(df['132'] == 769)]
	mat2 = df[(df['132'] == 770)]

	del mat1['132']
	del mat1['133']
	del mat2['132']
	del mat2['133']

	mat1.index = range(len(mat1))
	mat2.index = range(len(mat2))


	# mat1 = mat1.transpose()
	# mat2 = mat2.transpose()

	mat1.to_csv('left-signals.csv')
	mat2.to_csv('right-signals.csv')

# this file has the first row as header, remove it manually before importing in matlab
separate_classwise('filtered-list.csv')