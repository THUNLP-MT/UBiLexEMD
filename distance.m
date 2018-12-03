function dist = distance(X, x)

[D, N] = size(X);
if (nargin >= 2)
	[d, n] = size(x);
	if(D ~= d)
		error('Both sets of vectors must have same dimensionality!\n');
	end
	X2 = sum(X.^2, 1);
	x2 = sum(x.^2, 1);
	dist = bsxfun(@plus, X2.', bsxfun(@plus, x2, -2*X.'*x));
else
	[D, N] = size(X);
	s = sum(X.^2, 1);
	dist = bsxfun(@plus, s', bsxfun(@plus, s, -2*X.'*X));
end

