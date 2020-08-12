function [ output ] = columnKr( input )
%UNTITLED5 按行求峭度因子
%   此处显示详细说明
[m, ~] = size(input);
output = zeros(m, 1);
for i = 1:m
    y = input(i, :);
    v = sum((y-mean(y)).^2)/length(y);
    output(i, 1) = sum((y-mean(y)).^4)/(v.^2)/length(y);
    %output(i, 1) = sum(y.^4)/sqrt(sum(y.^2));
end
end