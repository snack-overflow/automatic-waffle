left_signals = csvread('left-signals.csv');
right_signals = csvread('right-signals.csv');
left_signals = permute(reshape(left_signals, 1375,66, 132), [1 3 2]);
right_signals = permute(reshape(right_signals, 1375,63, 132), [1 3 2]);
left_signals = permute(left_signals, [2 1 3]);
right_signals = permute(right_signals, [2 1 3]);
right_features=zeros(63,132);
left_features=zeros(66,132);
for i=1:1:63;
Z=W*right_signals(:,:,i);
for j=1:1:132;
right_features(i,j)=var((Z(j,:)));

end
right_features(i,:)=rdivide(right_features(i,:),sum(right_features(i,:)));
end

for i=1:1:66;
Z=W*left_signals(:,:,i);
for j=1:1:132;
left_features(i,j)=var((Z(j,:)));

end
left_features(i,:)=rdivide(left_features(i,:),sum(left_features(i,:)));
end

csvwrite('left_features.csv',left_features);
csvwrite('right_features.csv',right_features);