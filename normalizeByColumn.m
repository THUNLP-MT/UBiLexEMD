function Y = normalizeByColumn(X)

columnNorm = sqrt(sum(X.^2));
Y = bsxfun(@rdivide, X, columnNorm);

end
