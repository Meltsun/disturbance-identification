function bestalpha=qpSOR(Q,t,C,smallvalue)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% NSVM: Nonparallel support vector machine
%
%       bestalpha=qpSOR(Q,t,C,smallvalue)
% 
%       Input:
%               Q     - Hessian matrix(Require positive definite). 
%
%               t     - (0,2) Paramter to control training.
%
%               C     - Upper bound
%
%               smallvalue - Termination condition
%
%       Output:
%               bestalpha - Solutions of QPPs.
% 
%Reference:
%   Yuan-Hai Shao, Wei-Jie Chen and Nai-Yang Deng, "Nonparallel support 
%   vector machine" Submitted 2013
%
%   version 1.0 --Apr/2013 
%
%   Written by Wei-Jie Chen (wjcper2008@126.com)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initailization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[m,n]=size(Q);
alpha0=zeros(m,1);
L=tril(Q);
E=diag(Q);
twinalpha=alpha0;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compute alpha
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for j=1:n
%     i=i+1;
    twinalpha(j,1)=alpha0(j,1)-(t/E(j,1))*(Q(j,:)*twinalpha(:,1)-1+L(j,:)*(twinalpha(:,1)-alpha0));
    if twinalpha(j,1)<0
        twinalpha(j,1)=0;
    elseif twinalpha(j,1)>C
        twinalpha(j,1)=C;
    else
        ;
    end
end

alpha=[alpha0,twinalpha];
while norm(alpha(:,2)-alpha(:,1))>smallvalue 
    for j=1:n
        twinalpha(j,1)=alpha(j,2)-(t/E(j,1))*(Q(j,:)*twinalpha(:,1)-1+L(j,:)*(twinalpha(:,1)-alpha(:,2)));
        if twinalpha(j,1)<0
            twinalpha(j,1)=0;
        elseif twinalpha(j,1)>C
            twinalpha(j,1)=C;
        else
            ;
        end
    end
    alpha(:,1)=[];
    alpha=[alpha,twinalpha];
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Output
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
bestalpha=alpha(:,2);


