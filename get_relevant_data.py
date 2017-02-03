import pandas
import pdb
#pdb.set_trace()




def get_relevant_data(filename = ""):
	
	H_data=pandas.read_csv('file-h.csv')
	S_data=pandas.read_csv(filename,header=None)
	A_data=pandas.read_csv('file-artifact.csv',header=None)

	count=0
	list_of_lists=[[] for x in range (0,134)]
	for i in range(0, len(H_data['TYP'])):
		if H_data['TYP'][i]==768:
			position=H_data['POS'][i+1]
			label=H_data['TYP'][i+1]
			count+=1
			if(H_data['TYP'][i+1] == 769 or H_data['TYP'][i+1] == 770 or H_data['TYP'][i+1] == 771) and A_data[0][count]==0:
				for x in range (0,132):
					list_of_lists[x] = list_of_lists[x]+list(S_data[x][position-1:position-1+H_data['DUR'][i]-500])
				list_class=[label for x in range (0,H_data['DUR'][i]-500)]
				list_of_lists[132] =list_of_lists[132]+ list_class
				list_count=[count for x in range (0,H_data['DUR'][i]-500)]
				list_of_lists[133] =list_of_lists[133] + list_count
	df = pandas.DataFrame(list_of_lists)
	df=df.transpose()
	df.to_csv('filtered-list.csv', index=False)


get_relevant_data('filtered.csv')