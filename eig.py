import pandas as pd

df = pd.read_csv('list.csv')

mat1 = df[(df['25'] == 769)]

mat2 = df[(df['25'] == 770)]

del mat1['25']
del mat1['26']
del mat2['25']
del mat2['26']

mat1.index = range(len(mat1))
mat2.index = range(len(mat2))


# mat1 = mat1.transpose()
# mat2 = mat2.transpose()

mat1.to_csv('mat1.csv')
mat2.to_csv('mat2.csv')