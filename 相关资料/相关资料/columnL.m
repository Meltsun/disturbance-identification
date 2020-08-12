function [ output ] = columnL( input )
%UNTITLED5 按行求裕度因子
%   此处显示详细说明
[m, ~] = size(input);
output = zeros(m, 1);
for i = 1:m
    y = input(i, :);
    output(i, 1) = (max(y)-min(y))/(mean(sqrt(abs(y)))^2);
end
end