function [ output ] = columnC( input )
%UNTITLED5 波峰因子
%   此处显示详细说明
[m, ~] = size(input);
output = zeros(m, 1);
for i = 1:m
    y = input(i, :);
    output(i, 1) = (max(y)-min(y))/(2*rms(y));
end
end
