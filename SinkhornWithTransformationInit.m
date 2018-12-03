function [emd, O, flow] = SinkhornWithTransformationInit(X_s, X_t, weight_s, weight_t, N, M, O)

fconfig = fopen('matlab.config');
type = textscan(fconfig, '%s');
fclose(fconfig);
type = type{1}{1};

X_s = X_s(:, 1:N);
weight_s = weight_s(1:N);

X_t = X_t(:, 1:M);
weight_t = weight_t(1:M);

X_s = normalizeByColumn(X_s);
X_t = normalizeByColumn(X_t);

weight_s = weight_s/sum(weight_s);
weight_t = weight_t/sum(weight_t);

[dim, ~] = size(X_s);

maxIter = 10;
emds = zeros(maxIter, 1);
for iter = 1:maxIter
    %initial O may not be orthogonal
    if iter == 1
        transformed_X_s = normalizeByColumn(O*X_s);
    else
        transformed_X_s = O*X_s;
    end
    [emd, T, flow] = solveBySinkhorn(transformed_X_s, X_t, weight_s, weight_t);
    emds(iter) = emd;
    
    numFlow = size(flow, 1);
    D_s = zeros(dim, numFlow);
    D_t = zeros(dim, numFlow);
    for i = 1:numFlow
        i_s = flow(i, 1) + 1;
        i_t = flow(i, 2) + 1;
        i_flow = flow(i, 3);
        D_s(:, i) = sqrt(i_flow) * X_s(:, i_s);
        D_t(:, i) = sqrt(i_flow) * X_t(:, i_t);
    end
    [U, S, V] = svd(D_t * D_s');
    O = U*V';
    dlmwrite(sprintf('data/%s/flow.%d', type, iter), flow, ' ');
end
emds
