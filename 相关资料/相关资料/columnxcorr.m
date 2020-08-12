function [output] = columnxcorr(input)

output = mean(xcorr(input));
   