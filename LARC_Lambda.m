%Alexander Gorovits, Ekta Gujral,Evangelos E. Papalexakis and Petko Bogdanov
%Department of Computer Science, University at Albany—SUNY,
%Department of Computer Science and Engineering, University of California Riverside
%LARC: Learning Activity-Regularized overlapping Communities across Time
%Code for learning lambda values using Minimum Description Length (MDL) model.

function [lla,llb]=LARC_Lambda(X,K,ops)
	% Input : X - 3 mode tensor 
	%		: K- known rank of tensor X
	%       : ops- supported parameters for LARC-TF
	% output: lla- learned reg param on sparseness
	%		: llb - learned reg param on piecewise constancy 
			la=[10e-5,10e-4,10e-3,10e-2,10e-1,10,100];
			lb=[10e-5,10e-4,10e-3,10e-2,10e-1,10,100];
			MDLSCORE=zeros(8,8);it=0;
	for a=1:size(la,2)
		for b=1:size(lb,2)
			it=it+1;
			ops.la = la(a); %reg param on sparseness 
			ops.lb = lb(b);  %reg param on piecewise constancy 
			 [P, timeSpent]  = LARC(X,K, ops);
			  mdlScore=getscore(X,P);
			  MDLSCORE(a+1,1)=la(a);
			  MDLSCORE(1,b+1)=lb(b);
			  MDLSCORE(a+1,b+1)= mdlScore.MDL;
			  fprintf('Iteration %d for LA: %6.8f and LB: %6.8f with MDL Score:%7.4f\n',it,ops.la,ops.lb,mdlScore.MDL);
	  
		end
	end

	[val,idxi]=min(MDLSCORE(2:end,2:end));
	[valj,idxj]=min(val);
	llb=lb(idxj);
	lla=la(idxi(idxj));
end
%% supported function to calculate mdlscore.
function Score=getscore(X,P)
        sz=size(X);
        A=P{1}; B=P{2};  C=P{3};
        lambda=ones(size(A,2),1);
        XC1=ktensor(lambda,A,B,C);
        normXC=norm(XC1)^2;
        normX=norm(X)^2;
        T=prod(size(X));
        %%part1 reconstruction Error    
        normDifflogAvgbefore=log(abs(normXC-normX)/T);
        T1=prod(size(diff(C)));
       sDA=log(sum(sum(abs(diff(C))))/T1);
       Score.MDL=-sDA-normDifflogAvgbefore;
end
