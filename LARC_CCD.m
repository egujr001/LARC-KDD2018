%Alexander Gorovits, Ekta Gujral,Evangelos E. Papalexakis and Petko Bogdanov
%Department of Computer Science, University at Albany—SUNY,
%Department of Computer Science and Engineering, University of California Riverside 
%LARC: Learning Activity-Regularized overlapping Communities across Time
%Code for learning rank K with Time-Warped CCD.

function [K_est] = LARC_CCD(X,Kmax,ops)
	% Input : X - 3 mode tensor 
	%		: Kmax- max rank K expected.
	%       : ops- supported parameters for LARC-TF
	% output: K_est- learned K with Time-Warped CCD
	allF = 2:Kmax;
	all_Fac_fro = {};
	thresh = 20;

	all_c_fro = zeros(length(allF),1);

	for f = allF
	   [Fac, his ] = LARC( X, f, ops );
	   [c_fro, idx] = time_warped_corcondia(X,Fac);
		  
	   all_c_fro(find(f == allF)) = c_fro;
	   all_Fac_fro{find(f == allF)} = Fac;
	end

	all_c_fro(all_c_fro<thresh) = 0;
	[K_est, c] = multi_objective_optim(allF,all_c_fro);
	fprintf('Estimated rank K is : %d \n',K_est);
end


function [x_best, y_best] = multi_objective_optim(x,y)
    %clustering heuristic method
    try
        clusters = kmeans(y,2,'Distance','cityblock');
    catch %if an error is thrown, then an empty cluster is formed, so then we just assign all elements to cluster 1
        disp('Empty cluster!!')
        clusters = ones(size(y));
        clusters(y>0)=2;
    end
    cent1 = mean(y(clusters == 1));
    cent2 = mean(y(clusters == 2));
    [maxval, cent_idx] = max([cent1 cent2]);
    x_to_choose = x(clusters == cent_idx);
    [x_best,x_idx] = max(x_to_choose);
    y_best = y(x == x_best);
end
function [max_c, idx] = time_warped_corcondia(X,P)
	A=P{1};B=P{2};C=P{3};
	[U, S, V] = svd(C,'econ');
	all_c2 = zeros(size(A,2),1);
	for r=1:size(A,2)
		Xnew = sptensor(ttm(X,U(:,1:r)*U(:,1:r)',3));
		Fac.U{1} = A; Fac.U{2} = B; Fac.U{3} = sparse(U(:,1:r)*U(:,1:r)'*C);
		Fac.lambda=ones(size(A,2),1);
		all_c2(r) = efficient_corcondia(Xnew,Fac,1);
	end
	[max_c,idx] = max(all_c2);
end

function [c,time] = efficient_corcondia(X,Fac,sparse_flag)
	%Vagelis Papalexakis - Carnegie Mellon University, School of Computer
	%Science (2014)
	%This is an efficient algorithm for computing the CORCONDIA diagnostic for
	%the PARAFAC decomposition (Bro and Kiers, "A new
	%efficient method for determining the number of components in PARAFAC
	%models", Journal of Chemometrics, 2003)
	%This algorithm is an implementation of the algorithm introduced in
	%    E.E. Papalexakis and C. Faloutsos, 
	%   Fast efficient and scalable core consistency diagnostic for the parafac decomposition for big sparse tensors,? 
	%    in IEEE ICASSP 2015

	if nargin == 2
		sparse_flag = 2;
	end

	C = Fac.U{3};
	B = Fac.U{2};
	A = Fac.U{1};
	A = A*diag(Fac.lambda);
	if sparse_flag
		A = sparse(A);
	end
	F = size(A,2);

	tic
	try
		if(sparse_flag)
			[Ua Sa Va] = svds(A,F);
			[Ub Sb Vb] = svds(B,F);
			[Uc Sc Vc] = svds(C,F);
		else
			[Ua Sa Va] = svd(A,'econ');
			[Ub Sb Vb] = svd(B,'econ');
			[Uc Sc Vc] = svd(C,'econ');   
		end
	catch exception
		warning('Factors are really bad! Returning zero');
		c = 0;
		return;
	end

	part1 = kron_mat_vec({Ua' Ub' Uc'},X);
	part2 = kron_mat_vec({pinv(Sa) pinv(Sb) pinv(Sc)},part1);
	G = kron_mat_vec({Va Vb Vc},part2);

	T = sptensor([F F F]);
	for i = 1:F; T(i,i,i) =1; end

	c = 100* (1 - sum(sum(sum(double(G-T).^2)))/F);
	time = toc;
	end

	function C = kron_mat_vec(Alist,X)
	K = length(Alist);
	for k = K:-1:1
		A = Alist{k};
		Y = ttm(X,A,k);
		X = Y;
		X = permute(X,[3 2 1]);
	end
	C = Y;
end