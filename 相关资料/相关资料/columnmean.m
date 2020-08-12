function [ output ] = columnmean( input )
%UNTITLED 此处显示有关此函数的摘要
%   按行求均值
%
[m, ~] = size(input);
output = zeros(m, 1);
for i = 1:m
    output(i, 1) = mean(input(i, :));
end
end

