%Alexander Gorovits, Ekta Gujral,Evangelos E. Papalexakis and Petko Bogdanov
%Department of Computer Science, University at Albany—SUNY,
%Department of Computer Science and Engineering, University of California Riverside 
%"LARC: Learning Activity-Regularized overlapping Communities across Time", Submitted in KDD 2018

function T_actual = generator(N, K, T, p, params)
    %Input : node count N, community count K, duration T, activity switching
    %probability p;
    %allowable params: 
    %   ksize : avg community size
    %   dir : directory to write files to
    %   file : output file suffix
    %   periodic : Enforce runs in un-stretched time
    %   eps : link rate when no communities are shared
    %   preMadeT/preMadeC : filename for provided activation/community
    %       matrices (K by T/N by K)
    %   spread : average stretch per time point
    %   overlap : community node overlap - see lines 56+
    %   weights : (mean, sdev) per community for membership weights
    %       (default (0.5, 0) for all)
    
	%Output : Graph file "in.txt", (weighted) community membership file 
    %   "communities(_w).txt" - N by K, activations file "activations.txt"
    %   - T by K, graph.mat containining sparse tensor graph and
    %   communities_w.
    
    if ~isfield(params,'ksize'), ksize = N/10;              %Average community size
    else, ksize = params.ksize; end
    if ~isfield(params, 'dir'), params.dir = './'; end
    if ~isfield(params, 'file'), params.file = ''; end      %output file suffix
    if ~isfield(params,'periodic'), params.periodic=0; end  %Enforce runs in un-stretched time
    if ~isfield(params, 'eps'), params.eps = 0; end         %Link rate if no shared communities
    if isfield(params, 'preMadeT')                          %Read activations from file
        Tbase = dlmread(params.preMadeT);
        Tbase = arrayfun(@(k) Tbase(k,:), 1:K,'UniformOutput',false);
    else
        Tbase = cellfun(@(k) getTimes(T, p, params.periodic), num2cell(1:K), 'UniformOutput',false);
    end
    if isfield(params,'spread') && params.spread>0
        spreads = max(poissrnd(params.spread,T,1),1);
		Tpart = 1;
		Ts = cell(1, K);
        for t=1:T
			Tpart = cat(1, Tpart, Tpart(end)+spreads(t));
            for k=1:K
				Ts{k} = cat(1, Ts{k}, ones(spreads(t),1)*Tbase{k}(t));
            end
        end
    else
		Tpart = 1:T+1;
		Ts = Tbase;
    end

	A = cellfun(@(t) diag(arrayfun(@(k) Tbase{k}(t), 1:K)), num2cell(1:T),'UniformOutput',false);

	sizes = min(max(2, poissrnd(ksize, K, 1)),N);
    
    if isfield(params,'preMadeC') %Pre-set community matrix
    elseif isfield(params, 'overlap') && sum(sum(params.overlap)) > 0 
        %Let overlap be a single value, or uncomment 45-48 and comment
        %43/44, allowing overlap as a K by K matrix
        communities = cell(K, 1);
        communities{1} = randperm(N, sizes(1));
        for k = 2:K
            kset = randsample(communities{k-1}, floor(min(sizes(k)*params.overlap,sizes(k-1))));
            communities{k} = cat(2,kset, randsample(setdiff(1:N, kset),sizes(k) - length(kset)));
%             kset = arrayfun(@(j) randsample(communities{j}, floor(min(sizes(k)*params.overlap(j,k),sizes(k-1)))),1:k-1, 'UniformOutput',false);
%             kset = cat(2, kset{:});
%             kset = unique(kset);
%             communities{k} = cat(2,kset, randsample(setdiff(1:N, kset),sizes(k) - length(kset)));

        end
    else
        communities = cellfun(@(k) randperm(N,sizes(k)),num2cell(1:K),'UniformOutput',false);
        
    end
    
    if isfield(params,'preMadeC')
        C = dlmread(params.preMadeC);
    else
        C = zeros(N,K);
        for n=1:N
            for k=1:K
                if ismember(n,communities{k}) 
                    if ~isfield(params, 'weights')
                        C(n, k) = 0.5;
                    else
                        C(n, k) = max(0,normrnd(params.weights{k}(1), params.weights{k}(2)));
                    end
                end
            end
        end
    end
    f = fopen(strcat(params.dir,sprintf('communities%s.txt',params.file)),'w');
    fprintf(f,[repmat('%d,',1,K-1) '%d\n'],(C > 0)');
    fclose(f);
	f = fopen(strcat(params.dir,sprintf('communities%s_w.txt',params.file)),'w');    
    fprintf(f,[repmat('%f,',1,K-1) '%f\n'],C');
    fclose(f);
    f = fopen(strcat(params.dir,sprintf('activations%s.txt',params.file)),'w');
    fprintf(f,[repmat('%f,',1,K-1) '%f\n'], cell2mat(Ts)');
    fclose(f);
    
    X = [];
    X1 = zeros(N,N);
    i = 1:N; j = 1:N;
    for t = 1:T
        X1(i,j) = triu(-C(i,:)*A{t}*C(j,:)',1);
        X1 = 1-exp(X1);
        X2 = triu((X1 == 0)*params.eps,1);
        p = rand(N, N);
        X3 = (X1 > p) + (X2 > p);
        ind = find(X3);
        [i, j] = ind2sub([N N], ind);
        Xf1 = horzcat(i, j, randi([Tpart(t) Tpart(t+1)-1],length(i),1));
        Xf2 = horzcat(j, i, Xf1(:,3));
        X = cat(1,X,Xf1,Xf2);
    end


    X = sortrows(X, 3);
    f = sprintf('in%s.txt',params.file);
    fID = fopen(strcat(params.dir,f),'w');
    fprintf(fID,'%d %d %d\n',X');
    fclose(fID);
    
    T_actual = Tpart(end)-1;
    
    X_ = sptensor(X, 1, [N N T_actual]);
    save(strcat(params.dir,'graph.mat'),'X_','C')

end
               
                    
function k = getTimes(T, p, periodic)
	k = zeros(T,1);
	k(1) = rand<0.5;
    while true
		run = 1;
        for t=2:T
            if run >= periodic && rand < p
				k(t) = 1 - k(t-1);
				run = 1;
            else
				k(t) = k(t-1);
                run = run + 1;
            end
        end
        if sum(k) >= 1
            break %We don't want communities that are NEVER active
        end
    end
end




