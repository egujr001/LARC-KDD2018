
%Alexander Gorovits, Ekta Gujral,Evangelos E. Papalexakis and Petko Bogdanov
%Department of Computer Science, University at Albany—SUNY,
%Department of Computer Science and Engineering, University of California Riverside 
%"LARC: Learning Activity-Regularized overlapping Communities across Time", Submitted in KDD 2018

function Ym = matricize( Y )

Y = tensor(Y);
Ym = cell( ndims(Y), 1 );
for d = 1:ndims(Y)
    temp  = tenmat(Y,d);
    Ym{d} = temp.data';
end