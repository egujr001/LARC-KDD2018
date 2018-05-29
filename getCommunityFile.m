function getCommunityFile(myfilepath,candidate,K)
fid = fopen(myfilepath, 'wt');
 fprintf(fid, '% this is the beginning of my file');    
   for l=1:K     
    candidate(candidate(:,l)==1,l)=l;
    idx=find(candidate(:,l)==l);     
    fprintf(fid, '%d ',idx);
    fprintf(fid,'\n');% if var1 is double
   end
   fclose(fid);  
end