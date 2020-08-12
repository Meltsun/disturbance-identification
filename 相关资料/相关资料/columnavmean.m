function [output] = columnavmean(input)
[m, ~] = size(input);
output = zeros(m, 1);
for i = 1:m
    output(i, 1) = mean(abs(input(i, :)));
end
end