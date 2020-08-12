function [ output ] = columnstd( input )
%UNTITLED5 按行标准差
%   此处显示详细说明
[m, ~] = size(input);
output = zeros(m, 1);
for i = 1:m
    output(i, 1) = std(input(i, :));
end
end
