function Predict_Y = TWSVM(TestX,DataTrain,FunPara)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% TWSVM: Twin Support Vector Machine 
%
%       Predict_Y =TWSVM(TestX,DataTrain,FunPara)
% 
%       Input:
%               TestX       - Test Data matrix. 
%                             Each row vector of fea is a data point.
%
%               DataTrain   - Struct value in Matlab------Training data.
%                   DataTrain.A: Positive input of Data matrix.
%                   DataTrain.B: Negative input of Data matrix.
%
%               FunPara - Struct value in Matlab. The fields in options
%                         that can be set:
%                   c1: [0,inf] Paramter to tune the weight. 
%                   c2: [0,inf] Paramter to tune the weight. 
%                   c3: [0,inf] Paramter to tune the weight. 
%                   c4: [0,inf] Paramter to tune the weight. 
%                   kerfPara:Kernel parameters. See kernelfun.m.
%
%       Output:
%               Predict_Y - Predict value of the TestX.
%
%
%       Examples:
%
%           DataTrain.A = rand(50,10);
%           DataTrain.B = rand(60,10);
%           TestX=rand(20,10);
%           FunPara.c1=0.1;
%           FunPara.c2=0.1;
%           FunPara.c3=0.1;
%           FunPara.c4=0.1;
%           FunPara.kerfPara.type = 'lin';
%           Predict_Y =TWSVM(TestX,DataTrain,FunPara);
% 
%Reference:
%   Y.-H. Shao, C.-H. Chun, X.-B. Wang, N.-Y. Deng.Improvements on Twin 
%    Support Vector Machines.IEEE Transactions on Neural Networks, 2011, 22
%   (6):962-968.
%
%   version 1.0 --Apr/2013 
%
%   Written by Yuan-Hai Shao (shaoyuanhai21@163.com)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initailization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%tic;
    Xpos = DataTrain.A;
    Xneg = DataTrain.B;
    cpos = FunPara.c1;
    cneg = FunPara.c2;
    eps1 = FunPara.c3;
    eps2 = FunPara.c4;
    kerfPara = FunPara.kerfPara;
    m1=size(Xpos,1);
    m2=size(Xneg,1);
    e1=-ones(m1,1);
    e2=-ones(m2,1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compute Kernel 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if strcmp(kerfPara.type,'lin')
        H=[Xpos,-e1];
        G=[Xneg,-e2];
    else
        X=[DataTrain.A;DataTrain.B];
        H=[kernelfun(Xpos,kerfPara,X),-e1];
        G=[kernelfun(Xneg,kerfPara,X),-e2];
    end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Compute (w1,b1) and (w2,b2)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%DTWSVM1
    HH=H'*H;
    HH = HH + eps1*eye(size(HH));%regularization
    HHG = HH\G';
    kerH1=G*HHG;
    kerH1=(kerH1+kerH1')/2;
    alpha=qpSOR(kerH1,0.5,cpos,0.05); %SOR
    vpos=-HHG*alpha;
%%%%DTWSVM2 equ(29)
    QQ=G'*G;
    QQ=QQ + eps2*eye(size(QQ));%regularization
    QQP=QQ\H';
    kerH1=H*QQP;
    kerH1=(kerH1+kerH1')/2;
    gamma=qpSOR(kerH1,0.5,cneg,0.05);
    vneg=QQP*gamma;
    clear kerH1 H G HH HHG QQ QQP;

    w1=vpos(1:(length(vpos)-1));
    b1=vpos(length(vpos));
    w2=vneg(1:(length(vneg)-1));
    b2=vneg(length(vneg));
%toc;    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Predict and output
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    m=size(TestX,1);
    if strcmp(kerfPara.type,'lin')
        H=TestX;
        w11=sqrt(w1'*w1);
        w22=sqrt(w2'*w2);
        y1=H*w1+b1*ones(m,1);
        y2=H*w2+b2*ones(m,1);    
    else
        C=[DataTrain.A;DataTrain.B];
        H=kernelfun(TestX,kerfPara,C);
        w11=sqrt(w1'*kernelfun(X,kerfPara,C)*w1);
        w22=sqrt(w2'*kernelfun(X,kerfPara,C)*w2);
        y1=H*w1+b1*ones(m,1);
        y2=H*w2+b2*ones(m,1);
    end
    wp=sqrt(2+2*w1'*w2/(w11*w22));
    wm=sqrt(2-2*w1'*w2/(w11*w22));
    clear H; clear C;
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
    m1=y1/w11;
    m2=y2/w22;
    MP=(m1+m2)/wp;
    MN=(m1-m2)/wm;
    mind=min(abs(MP),abs(MN));
    maxd=max(abs(MP),abs(MN));
    Predict_Y = sign(abs(m2)-abs(m1));
end