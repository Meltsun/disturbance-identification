function [ output ] = columnmin( input )
%UNTITLED5 ��������Сֵ
%   �˴���ʾ��ϸ˵��
[m, ~] = size(input);
output = zeros(m, 1);
for i = 1:m
    y = input(i, :);
    output(i, 1) = min(y);
end
end

