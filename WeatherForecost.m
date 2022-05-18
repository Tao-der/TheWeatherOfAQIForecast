clc
clear
%%  基于某市2021年的空气污染数据搭建的BP神经网络预测模型，
%并手动输入天气等影响AQI值的因素来对空气质量进行模糊推理  
          
%原始数据输入
    X_Oringin = [xlsread('C:\Users\HP\Desktop\yiyang2021aqi.xlsx','Sheet1')]';
%目标（输出）数据矩阵
    Y_Oringin = [xlsread('C:\Users\HP\Desktop\yiyang2021aqi.xlsx','Sheet2')]';
%     X_Oringin_test = [xlsread('C:\Users\HP\Desktop\AQIW.csv')]';
%     Y_Oringin_test = [xlsread('C:\Users\HP\Desktop\AQI.csv')]';

%% 归一化
 [X_train,PS] = mapminmax(X_Oringin,-1,1);%-1 1之间正则化
 [Y_train,PS] = mapminmax(Y_Oringin,-1,1);
    %% 神经网络参数定义
    m=0;  
    In=length(X_train(1,:));                                         %输入层神经元数
    H1=In-5; 
    H2=H1-4;
    Out=1;                                          %输出层神经元数                       
    w1=unifrnd(-1,1,In,H1);      %初始化
    w2=unifrnd(-1,1,H1,H2);         %随机初始化
    w3=unifrnd(-1,1,H2,Out);  
    %}  
    Epoch=100;   %数据集的重复次数代训练
    batch=1;%一次训练的数据个数%
    Iteration=10;%一个batch训练迭代次数
    Lr=0.001/Iteration+0.0001;    
    
    %% 训练
    while (m<=Epoch)                                   %进行5000轮迭代
    
        m=m+1;
        
       for  i=1:length(Y_train)
         for j=1:Iteration
           %% 前向传播
           In=X_train(1,:) ;   
           H_1i=In*w1;
           H_1o=tanh(H_1i);
           
           H_2i=H_1o*w2;
           H_2o=tanh(H_2i);         
           out=H_2o*w3;
         
           %% 反向传播
         %  e= 0.5*(Y_train(i)-out).^2;                     %计算误差=期望-输出
           
          Error=(Y_train(i)-out);

           D3=H_2o.*Error;
           D2=(((1-H_2o.^2))'.*(w3.*Error))*H_1o;
           D1=(((1-H_1o.^2))'.*w2*(( (1-H_2o.^2))'.*w3.*Error))*In;
           
           w1=w1+Lr*D1';%-a.*Lr.*w1;                                 %输出层权值更新
           w2=w2+Lr*D2';%-a.*Lr.*w2;   
           w3=w3+Lr*D3';%-a.*Lr.*w3;%隐含层权值更新
        
%          if j==1
%       
%            Err_squ = [Err_squ Error*Error];
% 
%          end 
%      end
      
       end%进行一轮的迭代
       end
    end
    


