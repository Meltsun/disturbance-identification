%---------------测试的函数-------------
function result = svmTest(svm, Xt, Yt, kertype)
temp = (svm.a'.*svm.Ysv)*kernel(svm.Xsv,svm.Xsv,kertype);
%total_b = svm.Ysv-temp;
b = mean(svm.Ysv-temp);  %b取均值  课本公式
w = (svm.a'.*svm.Ysv)*kernel(svm.Xsv,Xt,kertype);%不需要通过所有点计算w，只需支持向量点即可。 课本里的公式
result.score = w + b;
Y = sign(w+b);  %f(x)
result.Y = Y;
result.accuracy = size(find(Y==Yt))/size(Yt);
end

