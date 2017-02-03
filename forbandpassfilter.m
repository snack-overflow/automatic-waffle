data=[];
for i=1:1:22;
     i
     temp=s(:,i);
     [output,n ]= ourfilter(temp,2,3,40,250);
     data=[data output]; 
%      if i==1
%          data= output;
%      else 
%          data=[data output]; 
%      end
end
csvwrite('filtered.csv',data);