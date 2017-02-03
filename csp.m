function [V, W]  = csp(left, right, row, column, left_trial, right_trial)
left_cov=zeros(132,132);
right_cov= zeros(132,132);

for i= 1:1:left_trial;
    trial=left(:,:,i);
%     trial=left;
    
    cov=(1/row)*trial*(eye(row)-ones([row 1])*ones([1 row]));
    
    left_cov = left_cov + (cov * cov');
end
left_cov=left_cov/left_trial;
    
for i= 1:1:right_trial;
    trial=right(:,:,i);
%     trial=right;
    cov=(1/row)*trial*(eye(row)-ones([row 1])*ones([1 row]));
    right_cov = right_cov + (cov * cov');
end
right_cov=right_cov/right_trial;

[V W]=eig(right_cov,left_cov+right_cov);



