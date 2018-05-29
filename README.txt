This toolbox provides a overlapping community detection algorithm developed in the paper:
Alexander Gorovits, Ekta Gujral,Evangelos E. Papalexakis and Petko Bogdanov,
"LARC: Learning Activity-Regularized overlapping Communities across Time" (Submitted in KDD 2018)


To execute, run the DEMO.m file:
     
	 main function: 
	 [Fac,timeSpent] = LARC_TF(X,K, ops);
	 
	 % Input : X - 3 mode tensor , K- known rank of tensor X, ops- supported parameters for LARC-TF
     % Output: Fac: Factor matrices[C,C'and A]  and  timeSpent - Total CPU time

	 
EVALUATION: The format of output files from  getCommunityFile.m is like this: write the labels of the nodes belonging to the same cluster in the same line.
For example:

1 2 3
2 4 5 
Each row represent nodes in single community. It means there are two clusters: in one there are nodes 1, 2, 3 and in the other there are nodes 2, 4 and 5.

For NMI, refer to code at https://sites.google.com/site/andrealancichinetti/mutual
ref:
    A. Lancichinetti, S. Fortunato, J. Kertész
    Detecting the overlapping and hierarchical community structure of complex networks        
    New Journal of Physics 11, 033015 (2009) 
 
 
Optional Features
1) Learn Lambdas (C and D)
  
  Usage : 
  [lla,llb]=LARC_Lambda(X,K,ops);
 
 % Input : X - 3 mode tensor , K- known rank of tensor X, ops- supported parameters for LARC-TF
 % Output: lla- learned reg param on sparseness and  llb - learned reg param on piecewise constancy 
	

2) Learn Rank (K)
    [K_est] = LARC_CCD(X,Kmax,ops);
	
	% Input : X - 3 mode tensor , Kmax- max rank K expected and ops- supported parameters for LARC-TF
	% output: K_est- learned K with Time-Warped CCD
	
	NOTE: To handle large dataset, we need change in ttm.m file in tensor_toolbox under sptensor folder.
	for your reference file is included with LARC-TF code.
	
	Following changes are made: 
	Line 122: "Z = double(Xnt) * V';"  replaced with following code for making efficient and large computations.
	
	vals = Xnt.vals;
	subs = Xnt.subs;
	Xnts = sparse(subs(:,1),subs(:,2),vals,size(Xnt,1),size(Xnt,2));
	Xnt = Xnts;
	V=sparse(V);
    Z =Xnt * V';
	
	
	Line 130 and 131:  replaced with following code.
	
	Ynt = sptenmat(Z, rdims, cdims, siz);
    Y = sptensor(Ynt);

3) Synthetic generator
	generator(N, K, T, p, params)
	%Input : node count N, community count K, duration T, activity switching
 			probability p; params contains avg community size 'ksize', enforced
 			periodicity 'periodic', average spread 'spread', and other parameters (see
            first lines of code)
	%Output : Graph file "in.txt", (weighted) community membership file 
            "communities(_w).txt" - N by K, activations file "activations.txt"
            - T by K, graph.mat containing the graph as a sptensor and the
            weighted community matrix
     
If you are using our code for your research please cite:
"LARC: Learning Activity-Regularized overlapping Communities across Time", 
Gorovits et al 2018
