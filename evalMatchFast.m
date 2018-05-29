%Alexander Gorovits, Ekta Gujral,Evangelos E. Papalexakis and Petko Bogdanov
%Department of Computer Science, University at Albany—SUNY,
%Department of Computer Science and Engineering, University of California Riverside 
%"LARC: Learning Activity-Regularized overlapping Communities across Time", Submitted in KDD 2018

function [order,best] = evalMatchFast(candidate, ground,noRep)
    %Pass noRep=true for true greedy matching w/o replacement, else order
    %may contain repeat communities
    if size(ground.C,1) > size(candidate.C,1)
        candidate.C = [candidate.C; zeros(size(ground.C,1)-size(candidate.C,1), size(ground.C, 2))];
    end
    [order, best] = greedyMatch(candidate.C, ground.C,noRep);
  
end

function [order, avgDist] = greedyMatch(A, B, noRep)
    K = size(A, 2);
    candidates = 1:K;
    order = repmat(-1,K,1);
    avgDist = 0;
    for i = 1:K
        best = -1;
        minD = 1;
        for j = candidates
            ATemp = A(:,j);        
            dist = JSDiv(ATemp,B(:, i));

            if dist < minD
                best = j;
                minD = dist;
            end
        end
        order(i) = best;
        avgDist = avgDist + minD;
        if noRep, candidates(candidates==best)=[]; end
    end
    
    avgDist = avgDist/K
end

function l = safeLogProd(x, y)
    if x > 0
        l = x*log2(x/y);
    else
        l = 0;
    end
end

function div = JSDiv(P, Q)
    if norm(P) == 0 || norm(Q) == 0
        div = 1;
        return;
    end
    P_norm = P/sum(P);
	Q_norm = Q/sum(Q);
	M = 0.5*(P_norm+Q_norm);
    KLDP = sum(arrayfun(@safeLogProd, P_norm, M));
    KLDQ = sum(arrayfun(@safeLogProd, Q_norm, M));
    
    div = 0.5*(KLDP + KLDQ);
end

