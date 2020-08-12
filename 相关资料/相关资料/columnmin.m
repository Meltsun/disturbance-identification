function [ output ] = columnmin( input )
%UNTITLED5 按行求最小值
%   此处显示详细说明
[m, ~] = size(input);
output = zeros(m, 1);
for i = 1:m
    y = input(i, :);
    output(i, 1) = min(y);
end
end

