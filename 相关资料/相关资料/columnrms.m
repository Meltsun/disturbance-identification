function [ output ] = columnrms( input )
%UNTITLED5 按行求均方根
%   此处显示详细说明
[m, ~] = size(input);
output = zeros(m, 1);
for i = 1:m
    output(i, 1) = rms(input(i, :));
end
end

