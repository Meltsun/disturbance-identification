function [ output ] = columnpk( input )
%UNTITLED5 ���з�-��ֵ
%   �˴���ʾ��ϸ˵��
[m, ~] = size(input);
output = zeros(m, 1);
for i = 1:m
    output(i, 1) = max(input(i, :))-min(input(i, :));
end
end

