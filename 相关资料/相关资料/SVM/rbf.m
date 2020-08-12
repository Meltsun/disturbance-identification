function [H] = rbf(A,B,s)
%¾¶Ïò»ùºËº¯Êý
m = size(A,1);
n = size(B,1);
H = zeros(m,n);
for i = 1:m
    for j = 1:n
        H(i,j) = exp(-s*norm(A(i,:)'-B(:,j))^2);
    end
end
% r2 = repmat(sum(A.^2,2),1,size(B,1))+repmat(sum(B.^2,2),1,size(A,1))-2*A*B';
% H = exp(-s*r2);
end