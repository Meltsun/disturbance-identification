function Accuracy = PSO_twsvm_fitness(trDA,trDB,trDC,trDD,trDE,c1,c2)
g = 1;
m = 5;                     %类别数
n = 6;                     %特征数
t = 10;                    %基于模糊隶属度压缩的个数
T = 7;                     %交叉验证的比例。例如6：4
n1_ = 50;                  %第一类的样本数
n2_ = 50;                  %第一类的样本数
n1 = n1_*T/10-t;           %第一类的训练样本数
n2 = n2_*T/10-t;           %第二类的训练样本数
s_ = n1_*(10-T)/10;        %某一类的测试样本数
k = 0;                     %防止矩阵求逆奇异值的出现
    c = 0;                                 %计数
    trD = yasuo(trDA,trDB,trDC,trDD,trDE,t);
    for i = 1:m-1
        for j = i+1:m
            trD1 = trD((i-1)*n1+1:i*n1,:); %第一类 
            trD2 = trD((j-1)*n1+1:j*n1,:); %第二类

            %基于TLDM方法
    %         X  = [trD1;trD2];
    %         e1 = ones(n1,1);
    %         e2 = ones(n2,1);
    %         e  = ones(n1+n2,1);
    %         Y = [ones(n1,1);-ones(n2,1)];
    %         M  = [X,e];
    %         D  = M'*(ones(n1+n2,n1+n2)-Y*Y');
    %         H  = [trD1,e1];
    %         G  = [trD2,e2];
    %         Q  = lambda2/(n1+n2)*M'*Y;
    %         P  = lambda1/(n1+n2).^2*D*M+H'*H; 
    %         S  = lambda1/(n1+n2).^2*D*M+G'*G; 
    %         [w1,b1,w2,b2] = TLDMtrain(H,Q,P,G,S,e1,e2,n1,n2,c1,c2);

            %基于twsvm线性核函数方法
            e1 = ones(n1,1);
            e2 = ones(n2,1);
            H = [trD1,e1];
            G = [trD2,e2];
            P = [trD1,e1];
            Q = [trD2,e2];
            [w1,b1,w2,b2] = twsvmtrain(H,G,P,Q,e1,e2,c1,c2,n1,n2,k,T);

            %基于twsvm高斯核函数方法
    %         A = trD1;
    %         B = trD2;
    %         C = [A',B']';
    %         S = [rbf(A,C',s),e1];
    %         R = [rbf(B,C',s),e2];
    %         L = [rbf(A,C',s),e1];
    %         N = [rbf(B,C',s),e2];
    %         [w1,b1,w2,b2] = twsvmtrainRBF(S,R,L,N,e1,e2,c1,c2,50,50,k,T);

            %基于LSTWSVM
    %         e1 = ones(n1,1);
    %         e2 = ones(n2,1);
    %         H = [trD1,e1];
    %         G = [trD2,e2];
    %         I = ones(size(H'*H,1),size(H'*H,1));
    %         u1 = -c1*inv(H'*H+c1*(G'*G)+c3*I)*G'*e2;
    %         u2 = -c2*inv(G'*G+c2*(H'*H)+c4*I)*H'*e1;
    %         w1 = u1(1:end-1);
    %         b1 = u1(end);
    %         w2 = u2(1:end-1);
    %         b2 = u2(end);

            %分类器的个数
            c = c+1;
            %权值汇总
            W1(c,:) = w1';
            W2(c,:) = w2';
            B1(c)   = b1;
            B2(c)   = b2;
        end
    end
 %% 分类器汇总
    flq = [1,2;1,3;1,4;1,5;2,3;2,4;2,5;3,4;3,5;4,5];
    yxwht = [1,5;2,5;1,4;3,5;2,4;1,3;4,5;3,4;2,3;1,2];
    for i = 1:c
        for j = 1:c
           if(yxwht(i,:) == flq(j,:))
               lib(i) = j;
           end
        end
    end
    teD_z = trD;
    [row,col] = size(teD_z);
        biaoqian = [ones(n1,1)*1;
                ones(n1,1)*2;
                ones(n1,1)*3;
                ones(n1,1)*4;
                ones(n1,1)*5];
     for i = 1:row
        result1 = (W1(lib(1),:)*teD_z(i,:)'+B1(lib(1)))/(W1(lib(1),:)*W1(lib(1),:)');
        result2 = (W2(lib(1),:)*teD_z(i,:)'+B2(lib(1)))/(W2(lib(1),:)*W2(lib(1),:)');
        if(abs(result1) > abs(result2))
%                te_(i,k) = flq(lib(1),2);
            result1 = (W1(lib(2),:)*teD_z(i,:)'+B1(lib(2)))/(W1(lib(2),:)*W1(lib(2),:)');
            result2 = (W2(lib(2),:)*teD_z(i,:)'+B2(lib(2)))/(W2(lib(2),:)*W2(lib(2),:)'); 
            if(abs(result1) > abs(result2))   
               result1 = (W1(lib(4),:)*teD_z(i,:)'+B1(lib(4)))/(W1(lib(4),:)*W1(lib(4),:)');
               result2 = (W2(lib(4),:)*teD_z(i,:)'+B2(lib(4)))/(W2(lib(4),:)*W2(lib(4),:)');        
               if(abs(result1) > abs(result2))
                  result1 = (W1(lib(7),:)*teD_z(i,:)'+B1(lib(7)))/(W1(lib(7),:)*W1(lib(7),:)');
                  result2 = (W2(lib(7),:)*teD_z(i,:)'+B2(lib(7)))/(W2(lib(7),:)*W2(lib(7),:)');       
                  if(abs(result1) > abs(result2))
                      jg(g,i) = flq(lib(7),2);
                  else
                      jg(g,i) = flq(lib(7),1);
                  end
               else 
                  result1 = (W1(lib(8),:)*teD_z(i,:)'+B1(lib(8)))/(W1(lib(8),:)*W1(lib(8),:)');
                  result2 = (W2(lib(8),:)*teD_z(i,:)'+B2(lib(8)))/(W2(lib(8),:)*W2(lib(8),:)');      
                  if(abs(result1) > abs(result2))
                      jg(g,i) = flq(lib(8),2);
                  else
                      jg(g,i) = flq(lib(8),1);
                  end
               end  
            else 
               result1 = (W1(lib(5),:)*teD_z(i,:)'+B1(lib(5)))/(W1(lib(5),:)*W1(lib(5),:)');
               result2 = (W2(lib(5),:)*teD_z(i,:)'+B2(lib(5)))/(W2(lib(5),:)*W2(lib(5),:)'); 
               if(abs(result1) > abs(result2))
                  result1 = (W1(lib(8),:)*teD_z(i,:)'+B1(lib(8)))/(W1(lib(8),:)*W1(lib(8),:)');
                  result2 = (W2(lib(8),:)*teD_z(i,:)'+B2(lib(8)))/(W2(lib(8),:)*W2(lib(8),:)');   
                  if(abs(result1) > abs(result2))
                       jg(g,i) = flq(lib(8),2);
                  else
                       jg(g,i) = flq(lib(8),1);
                  end
               else
                  result1 = (W1(lib(9),:)*teD_z(i,:)'+B1(lib(9)))/(W1(lib(9),:)*W1(lib(9),:)');
                  result2 = (W2(lib(9),:)*teD_z(i,:)'+B2(lib(9)))/(W2(lib(9),:)*W2(lib(9),:)');  
                  if(abs(result1) > abs(result2))
                       jg(g,i) = flq(lib(9),2);
                  else
                       jg(g,i) = flq(lib(9),1);
                  end
               end
            end
        else
            result1 = (W1(lib(3),:)*teD_z(i,:)'+B1(lib(3)))/(W1(lib(3),:)*W1(lib(3),:)');
            result2 = (W2(lib(3),:)*teD_z(i,:)'+B2(lib(3)))/(W2(lib(3),:)*W2(lib(3),:)'); 
            if(abs(result1) > abs(result2))
               result1 = (W1(lib(5),:)*teD_z(i,:)'+B1(lib(5)))/(W1(lib(5),:)*W1(lib(5),:)');
               result2 = (W2(lib(5),:)*teD_z(i,:)'+B2(lib(5)))/(W2(lib(5),:)*W2(lib(5),:)');  
               if(abs(result1) > abs(result2))
                  result1 = (W1(lib(8),:)*teD_z(i,:)'+B1(lib(8)))/(W1(lib(8),:)*W1(lib(8),:)');
                  result2 = (W2(lib(8),:)*teD_z(i,:)'+B2(lib(8)))/(W2(lib(8),:)*W2(lib(8),:)');   
                  if(abs(result1) > abs(result2))
                       jg(g,i) = flq(lib(8),2);
                  else
                       jg(g,i) = flq(lib(8),1);
                  end 
               else
                  result1 = (W1(lib(9),:)*teD_z(i,:)'+B1(lib(9)))/(W1(lib(9),:)*W1(lib(9),:)');
                  result2 = (W2(lib(9),:)*teD_z(i,:)'+B2(lib(9)))/(W2(lib(9),:)*W2(lib(9),:)');   
                  if(abs(result1) > abs(result2))
                       jg(g,i) = flq(lib(9),2);
                  else
                       jg(g,i) = flq(lib(9),1);
                  end  
               end
            else
               result1 = (W1(lib(6),:)*teD_z(i,:)'+B1(lib(6)))/(W1(lib(6),:)*W1(lib(6),:)');
               result2 = (W2(lib(6),:)*teD_z(i,:)'+B2(lib(6)))/(W2(lib(6),:)*W2(lib(6),:)');   
               if(abs(result1) > abs(result2))
                  result1 = (W1(lib(9),:)*teD_z(i,:)'+B1(lib(9)))/(W1(lib(9),:)*W1(lib(9),:)');
                  result2 = (W2(lib(9),:)*teD_z(i,:)'+B2(lib(9)))/(W2(lib(9),:)*W2(lib(9),:)');   
                  if(abs(result1) > abs(result2))
                       jg(g,i) = flq(lib(9),2);
                  else
                       jg(g,i) = flq(lib(9),1);
                  end  
               else
                  result1 = (W1(lib(10),:)*teD_z(i,:)'+B1(lib(10)))/(W1(lib(10),:)*W1(lib(10),:)');
                  result2 = (W2(lib(10),:)*teD_z(i,:)'+B2(lib(10)))/(W2(lib(10),:)*W2(lib(10),:)');     
                  if(abs(result1) > abs(result2))
                       jg(g,i) = flq(lib(10),2);
                  else
                       jg(g,i) = flq(lib(10),1);
                  end     
               end
            end
        end
    end
jg = jg';
     Accuracy = 1-length(find((jg-biaoqian)~=0))/row;
end 
% toc;
% %% 画图
% figure(1)
% plot(vote_y,'og')
% hold on
% plot(biaoqian,'r*');
% legend('预测标签','实际标签')
% title('twsvm预测分类与实际类别比对','fontsize',12)
% ylabel('类别标签','fontsize',12)
% xlabel('样本数目','fontsize',12)
%% 分类正确率
% A_j = length(find(vote_y(1:s_,:) == 1))/(DD*s_);             %浇的正确率
% A_q = length(find(vote_y(s_+1:2*s_,:) == 2))/(DD*s_);        %敲的正确率
% A_y = length(find(vote_y(2*s_+1:3*s_,:) == 3))/(DD*s_);      %压的正确率
% A_p = length(find(vote_y(3*s_+1:4*s_,:) == 4))/(DD*s_);      %爬的正确率
% A_n = length(find(vote_y(4*s_+1:5*s_,:) == 5))/(DD*s_);      %无的正确率
                                     %平均正确率

