function [ output ] = columnpk( input )
%UNTITLED5 按行峰-峰值
%   此处显示详细说明
[m, ~] = size(input);
output = zeros(m, 1);
for i = 1:m
    output(i, 1) = max(input(i, :))-min(input(i, :));
end
end

