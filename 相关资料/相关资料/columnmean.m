function [ output ] = columnmean( input )
%UNTITLED �˴���ʾ�йش˺�����ժҪ
%   �������ֵ
%
[m, ~] = size(input);
output = zeros(m, 1);
for i = 1:m
    output(i, 1) = mean(input(i, :));
end
end

