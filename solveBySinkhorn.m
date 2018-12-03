function [emd, T, flow] = solveBySinkhorn(X_s, X_t, weight_s, weight_t)

N = length(weight_s);
M = length(weight_t);

Tquantile = 1-1.3/M;

D = distance(X_s, X_t);
D(D < 0) = 0;
% D = sqrt(D);

tic
lambda = 200/median(D(:));
K = exp(-lambda*D);
UU = K.*D;
[emd, ~, l, m] = sinkhornTransport(weight_s', weight_t', K, UU, lambda);
toc

T = bsxfun(@times, m', (bsxfun(@times, l, K)));
T(T < quantile(T(:), Tquantile)) = 0;
% T(T < min(max(T, [], 2))) = 0;

Tdensity = nnz(T)/N/M;
numFlow = nnz(T);
flow = zeros(numFlow, 3);
k = 1;
for i = 1:N
    for j = 1:M
        if T(i, j) ~= 0
            flow(k, 1) = i-1;
            flow(k, 2) = j-1;
            flow(k, 3) = T(i, j);
            k = k + 1;
        end
    end
end
