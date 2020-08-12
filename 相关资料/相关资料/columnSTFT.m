function [output] = columnSTFT(input)
[S,~,~] = specgram(input);
abs_S = abs(S);
abs_S = abs_S/sum(abs_S);
% output = -sum(abs_S.*log(abs_S));
output = abs_S(end);
end
    