function [ output ] = columnSK( input )
%UNTITLED2 按行求偏度因子
%   此处显示详细说明
[m, ~] = size(input);
output = zeros(m, 1);
for i = 1:m
    y = input(i, :);
    v = sum((y-mean(y)).^2)/length(y);
    output(i, 1) = (sum((y - mean(y)).^3)/length(y))/(v.^(3/2));
end

end

