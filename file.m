% a = []
% for i= 1:1:25
%     for j= i+1:1:25
%         a = [a [i;j;corr(s(:,i),s(:,j))]]
%     end
% end


M1 = csvread('mat1.csv',1,1);
M2 = csvread('mat2.csv',1,1);
M1 = M1.';
M2 = M2.';

% cov_M1 = cov(M1);
% cov_M2 = cov(M2);
% [A,D] = eig(cov_M1,cov_M1+cov_M2);
% [right_coeff,right_index] = sort(A(1:22,25),'descend');
% [left_coeff,left_index] = sort(A(1:22,1),'descend');

right_channels = []
left_channels = []
for i = 1:1:100
    temp_M1 = []
    t = randperm(66, 44);
    for j = 1:1:44
        if(t(j)==1)
            temp_M1=[temp_M1;M1(1:(t(j)*1375),:)];
        else
            temp_M1=[temp_M1;M1((t(j)-1)*1375+1:(t(j)-1)*1375+1375,:)];
        end
    end
    temp_M2 = []
    t = randperm(63, 44);
    for j = 1:1:44
        if(t(j)==1)
            temp_M2=[temp_M2;M2(1:(t(j)*1375),:)];
        else
            temp_M2=[temp_M2;M2((t(j)-1)*1375+1:(t(j)-1)*1375+1375,:)];
        end
    %temp_M1 = []M1(randperm(msize, 1106892))
    end
    cov_M1 = cov(temp_M1);
    cov_M2 = cov(temp_M2);
    [A,D] = eig(cov_M1,cov_M1+cov_M2);
    [right_coeff,right_index] = sort(A(1:22,25),'descend');
    [left_coeff,left_index] = sort(A(1:22,1),'descend');
    right_channel = [right_channel; right_index.']
    left_channel = [left_channel; left_index.']
    
end
csvwrite('right_channel.csv',right_channel)
csvwrite('left_channel.csv',left_channel)