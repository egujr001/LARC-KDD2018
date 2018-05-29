
%Alexander Gorovits, Ekta Gujral,Evangelos E. Papalexakis and Petko Bogdanov
%Department of Computer Science, University at Albany—SUNY,
%Department of Computer Science and Engineering, University of California Riverside 
function [ H, U, GG, itr ] = lsADMMdense( Y, W, H, U, d, GG, ops)
% ADMM iterates to solve
%       minimize (1/2)*|| Y - W*H' ||^2 + r(H)
% for dense tensor factorization
%   Y is the approapriate matrix unfolding
%   W is the khatri-rao product of the rest of the factors

[ ~, k ] = size(H);
G = ones(k,k); prod = [ 1:d-1, d+1:length(GG) ];
for dd = prod
    G = G .* GG{dd}; 
end
rho = trace(G)/k;
L = chol( G + (rho+ops.mu)*eye(k), 'lower' );

F = W'*Y; Hp = H;
tol = 1e-2;
for itr = 1:5
    H0 = H;
    
    Ht = L'\ ( L\ ( F + rho*(H+U)' + ops.mu*Hp' ) );
    H  = proxr( Ht'-U, ops, d, rho, H);
    U  = U + H - Ht';
    
    r = H - Ht';
    s = H - H0;
    if norm(r(:)) < tol*norm(H(:)) && norm(s(:)) < tol*norm(U(:))
        break
    end
end
GG{d} = H'*H;
end


function H = proxr( Hb, ops, d, rho, Hd )
    switch ops.constraint{d}
        case 'nonnegative'
            H = max( 0, Hb );
        case 'simplex_col'
            H = ProjectOntoSimplex(Hb, 1);
        case 'simplex_row'
            H = ProjectOntoSimplex(Hb', 1);
            H = H';
        case 'l1'
            H = sign( Hb ) .* max( 0, abs(Hb) - (ops.l1{d}/rho) );
        case 'l1n'
            H = max( 0, Hb - ops.l1{d}/rho );
        case 'l2'
            H = ( rho/(ops.l2{d}+rho) ) * Hb;
        case 'l2n'
            H = ( rho/(ops.l2{d}+rho) ) * max(0,Hb);
        case 'l2-bound'
           nn = sqrt( sum( Hb.^2 ) );
            H = Hb * diag( 1./ max(1,nn) );
        case 'l2-boundn'
            H = max( 0, Hb );
           nn = sqrt( sum( H.^2 ) );
            H = H * diag( 1./ max(1,nn) );
        case 'l0'
            T = sort(Hb,2,'descend');
            t = T(:,4); T = repmat(t,1,size(T,2));
            H = Hb .* ( Hb >= T );
        case 'fln'
            H = fusedLASSO(Hb, ops, rho, Hd);
    end
end

function H = fusedLASSO(Hb, ops, rho, A)
    fixedStep = isfield(ops,'fixedStep'); %Do fixed step size instead of line search (speed)
    H = A;
    tol = 1e-2;
    [T, ~] = size(H);
    STEP = ops.startStep;
    BETA = ops.stepMult;
    for itr = 1:200
        A0 = H;
        for t = 1:T
            dA = (sign(H(t,:)))*ops.la + rho*(H(t,:)-Hb(t,:));
            if t > 1
                dA = dA + ops.lb*sign(H(t,:)-H(t-1,:));
            end
            if t < T
                dA = dA - ops.lb*sign(H(t+1,:) - H(t,:));
            end
            s = STEP;
            Atemp = H;
            i = 0;
            if fixedStep, Atemp(t,:) = max(0, H(t,:) - dA*ops.fixedStep);
            else
                baseline = sum(sum(abs(H(t,:))))*ops.la + sum(sum(abs(diff(H(max(t-1,1):min(t+1,T),:)))))*ops.lb + rho/2*norm(H(t,:)-Hb(t,:),'fro')^2;
                while s > ops.minStep
                    i = i + 1;
                    Atemp(t,:) = max(0,H(t,:) - dA*s);
                    if sum(sum(abs(Atemp(t,:))))*ops.la + sum(sum(abs(diff(Atemp(max(t-1,1):min(t+1,T),:)))))*ops.lb + rho/2*norm(Atemp(t,:)-Hb(t,:),'fro')^2 < baseline
                        break
                    end
                    Atemp = H;
                    s = s*BETA;
                end
            end
            H = Atemp;
        end
        r = A0 - H;
        if norm(r(:)) < tol*norm(H(:))
            break
        end
    end
end