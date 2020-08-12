          DataTrain.A = rand(50,10);
          DataTrain.B = rand(60,10);
          TestX=rand(20,10);
          FunPara.c1=0.1;
          FunPara.c2=0.1;
          FunPara.c3=0.1;
          FunPara.c4=0.1;
          FunPara.kerfPara.type = 'lin';
          Predict_Y =TWSVM(TestX,DataTrain,FunPara);

