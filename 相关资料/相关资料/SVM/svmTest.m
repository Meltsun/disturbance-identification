%---------------���Եĺ���-------------
function result = svmTest(svm, Xt, Yt, kertype)
temp = (svm.a'.*svm.Ysv)*kernel(svm.Xsv,svm.Xsv,kertype);
%total_b = svm.Ysv-temp;
b = mean(svm.Ysv-temp);  %bȡ��ֵ  �α���ʽ
w = (svm.a'.*svm.Ysv)*kernel(svm.Xsv,Xt,kertype);%����Ҫͨ�����е����w��ֻ��֧�������㼴�ɡ� �α���Ĺ�ʽ
result.score = w + b;
Y = sign(w+b);  %f(x)
result.Y = Y;
result.accuracy = size(find(Y==Yt))/size(Yt);
end

