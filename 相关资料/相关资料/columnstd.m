function [ output ] = columnstd( input )
%UNTITLED5 ���б�׼��
%   �˴���ʾ��ϸ˵��
[m, ~] = size(input);
output = zeros(m, 1);
for i = 1:m
    output(i, 1) = std(input(i, :));
end
end
