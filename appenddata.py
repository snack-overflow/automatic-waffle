import pandas as pd
ldf=pd.read_csv('left-signals.csv',header=None)
rdf=pd.read_csv('right-signals.csv',header=None)
print "in left"
for i in range(2,10):
    print i
    ldf=ldf.append(pd.read_csv('./signals/A0'+str(i)+'T-left-signals.csv',header=None))

print "in right"
for i in range(2,10):
    print i
    rdf=rdf.append(pd.read_csv('./signals/A0'+str(i)+'T-right-signals.csv',header=None))

ldf.to_csv('left-signals-all.csv',index=False)
rdf.to_csv('right-signals-all.csv',index=False)
