clear all;clc;close all;
load('SampleInput/sample.mat')

%% initilize parameters
         ops.constraint{1} = 'nonnegative';
         ops.constraint{2} = 'nonnegative';
         ops.constraint{3} = 'fln';
         ops.la=0.0001;
         ops.lb=0.0001;
%% optional features: un-comment if you need to learn via code provided.

 %[lla,llb]=LARC_Lambda(X,K,ops);ops.la=lla;ops.lb=llb;
% Kmax=10;[K_est] = LARC_CCD(X,Kmax,ops); K=K_est;

 %% run code
 [Fac,timeSpent] = LARC(X,K, ops);
 fprintf('time taken : %5.5f seconds.\n',timeSpent);
 
 %% evaluation and get communities in  file , one line per community
 candidate.C=Fac{1};
 ground.C=communities;
 threshold=0.21; % change it and use dynamic threshold for evaluation like threshold=logspace(-3,0,30);
 [order, best]  = evalMatch(candidate, ground);
 C=Fac{1}(:,order);
 C(C>threshold)=1;C(C<=threshold)=0;
 myfilepath=sprintf('OutputFiles/Predicted_communities.dat');
 getCommunityFile(myfilepath,C,K);
 myfilepath=sprintf('OutputFiles/Actual_communities.dat');
 getCommunityFile(myfilepath,communities,K); % where communities is actual ground truth

 
 

 