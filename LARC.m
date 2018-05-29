%Alexander Gorovits, Ekta Gujral,Evangelos E. Papalexakis and Petko Bogdanov
%Department of Computer Science, University at Albany—SUNY,
%Department of Computer Science and Engineering, University of California Riverside 
%"LARC: Learning Activity-Regularized overlapping Communities across Time", Submitted in KDD 2018

function [Fac,timeSpent] = LARC(X,K, ops)
 % Input : X - 3 mode tensor , K- known rank of tensor X, ops- supported parameters for LARC-TF
 % Output: Fac: Factor matrices[C,C'and A]  and  timeSpent - Total CPU time

    if ~isfield(ops, 'la'), ops.la = .001; end %reg param on sparseness
    if ~isfield(ops, 'lb'), ops.lb = .01; end   %reg param on piecewise constancy
    if ~isfield(ops, 'dense'), ops.dense = false; end
    if ~isfield(ops, 'minStep'), ops.minStep = 10^-10; end
    if ~isfield(ops, 'startStep'), ops.startStep = 0.01; end
    if ~isfield(ops, 'stepMult'), ops.stepMult = 0.1; end
    sz=size(X);
    N = sz(1);
    T = sz(3);
   
    if ops.dense, X = tensor(X); end
        
    if ~isfield(ops,'Hinit') 
        Hinit{1} = rand( N, K );
        Hinit{2} = rand( N, K );
        Hinit{3} = rand( T, K );
        for d = 1:3
            Hinit{d} = Hinit{d} / diag( sqrt( sum( Hinit{d}.^2 ) ) );
        end
        ops.Hinit = Hinit;
    end
    ops.constraint{1} = 'nonnegative';
    if isfield(ops,'noReg') && ops.noReg, ops.constraint{3} = 'nonnegative'; 
    else, ops.constraint{3} = 'fln'; end
    ops.constraint{2} = 'nonnegative';
    %% LARC-TF
    doFit = tic;
    [ Fac, ~ ] = AOadmm( X, K, ops );
    timeSpent = toc(doFit);
    %% normalizing the factor matrices.
    C1 = Fac{1};  mc1 = max(C1); C1 = C1./mc1; 
    C2 = Fac{2};  mc2 = max(C2); C2 = C2./mc2;  
    scale = mc1.*mc2;
    Fac{3}= Fac{3}.*scale;
    Fac{1} = (C1+C2)/2;
	Fac{2}=(C1+C2)/2;
   
end