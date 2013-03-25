function xmtx = streaming2mtx(x,L,N,offset)
% this function converts vector to matrix 

xmtx = zeros(L,N+offset);

for i = offset:N+offset,
    xtdl = x(i:-1:i-L+1);
    xmtx(:,i) = xtdl;
end