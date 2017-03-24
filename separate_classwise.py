import pandas as pd
# import pdb
# pdb.set_trace()


def separate_classwise(dataset = ""):

	df = pd.read_csv('./data/'+dataset+'-filtered-list.csv')
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
	print "done working " + dataset
	mat1.to_csv('./data/'+dataset+'-left-signals.csv')
	mat2.to_csv('./data/'+dataset+'-right-signals.csv')
	print "done saving " + dataset

# this file has the first row as header, remove it manually before importing in matlab
separate_classwise('A03T')
separate_classwise('A04T')
separate_classwise('A05T')
separate_classwise('A06T')
separate_classwise('A07T')
separate_classwise('A08T')
separate_classwise('A09T')
