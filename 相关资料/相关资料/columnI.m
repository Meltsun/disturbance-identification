function [ output ] = columnI( input )
%UNTITLED5 ��������������
%   �˴���ʾ��ϸ˵��
[m, ~] = size(input);
output = zeros(m, 1);
for i = 1:m
    y = input(i, :);
    output(i, 1) = (max(y)-min(y))/mean(abs(y));
end
end