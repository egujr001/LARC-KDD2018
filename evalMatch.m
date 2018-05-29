
%Alexander Gorovits, Ekta Gujral,Evangelos E. Papalexakis and Petko Bogdanov
%Department of Computer Science, University at Albany—SUNY,
%Department of Computer Science and Engineering, University of California Riverside 
%"LARC: Learning Activity-Regularized overlapping Communities across Time", Submitted in KDD 2018
function [order, best]  = evalMatch(candidate, ground)
    if size(ground.C,1) > size(candidate.C,1)
        candidate.C = [candidate.C; zeros(size(ground.C,1)-size(candidate.C,1), size(ground.C, 2))];
    end
    [order, best] = bestMatch(candidate.C, ground.C);
 end

function [order, minD] = bestMatch(A, B)
    K = size(A, 2);
    p = perms(1:K);
    best = -1;
    minD = 1;
    for i = 1:size(p, 1)
        ATemp = A(:,p(i,:));
        dist = 0;
        for j = 1:K
            dist = dist + JSDiv(ATemp(:, j),B(:, j));
        end
        dist = dist/K;
        
        if dist < minD
            best = i;
            minD = dist;
        end
        
    end
    
    order = p(best,:);
end

function l = safeLogProd(x, y)
    if x > 0
        l = x*log2(x/y);
    else
        l = 0;
    end
end

function div = JSDiv(P, Q)
    P_norm = P/sum(P);
	Q_norm = Q/sum(Q);
	M = 0.5*(P_norm+Q_norm);
    KLDP = sum(arrayfun(@safeLogProd, P_norm, M));
    KLDQ = sum(arrayfun(@safeLogProd, Q_norm, M));
    
    div = 0.5*(KLDP + KLDQ);
end

