import pandas
import pdb
#pdb.set_trace()




def get_relevant_data(dataset = ""):
	
	H_data=pandas.read_csv('./data/' + dataset +'-h.csv')
	S_data=pandas.read_csv('./data/' + dataset +'-filtered.csv',header=None)
	A_data=pandas.read_csv('./data/' + dataset +'-artifacts.csv',header=None)

	count=0
	list_of_lists=[[] for x in range (0,134)]
	for i in range(0, len(H_data['TYP'])):
		
		if H_data['TYP'][i]==768:
			position=H_data['POS'][i+1]
			label=H_data['TYP'][i+1]
			count+=1
			if(H_data['TYP'][i+1] == 769 or H_data['TYP'][i+1] == 770 or H_data['TYP'][i+1] == 771) and A_data[0][count-1]==0:#
				for x in range (0,132):
					list_of_lists[x] = list_of_lists[x]+list(S_data[x][position-1:position-1+H_data['DUR'][i]-500])
				list_class=[label for x in range (0,H_data['DUR'][i]-500)]
				list_of_lists[132] =list_of_lists[132]+ list_class
				list_count=[count for x in range (0,H_data['DUR'][i]-500)]
				list_of_lists[133] =list_of_lists[133] + list_count
	df = pandas.DataFrame(list_of_lists)
	df=df.transpose()
	df.to_csv('./data/'+dataset+'-filtered-list.csv', index=False)
	print "done " + dataset


# get_relevant_data('A03T')
# get_relevant_data('A04T')
# get_relevant_data('A05T')
get_relevant_data('A06T')
get_relevant_data('A07T')
get_relevant_data('A08T')
get_relevant_data('A09T')

