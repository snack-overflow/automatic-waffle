function [V, W]  = csp(left, right, row, column, left_trial, right_trial)
left_cov=zeros(132,132);
right_cov= zeros(132,132);
%the input left and right should be of the format 132*1375*66 or 132*1375*63
for i= 1:1:left_trial;
    trial=left(:,:,i);
    cov=(1/column)*trial*(eye(column)-ones([column 1])*ones([1 column]));
    left_cov = left_cov + (cov * cov');
end

left_cov=left_cov/left_trial;

    
for i= 1:1:right_trial;
    trial=right(:,:,i);
    cov=(1/column)*trial*(eye(column)-ones([column 1])*ones([1 column]));
    right_cov = right_cov + (cov * cov');
end

right_cov=right_cov/right_trial;

[V, W]=eig(right_cov,left_cov+right_cov);



