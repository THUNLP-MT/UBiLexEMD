function [emd, O, flow] = SinkhornWithTransformationInitFromData(X_s, X_t, weight_s, weight_t, N, M)

fconfig = fopen('matlab.config');
type = textscan(fconfig, '%s');
fclose(fconfig);
type = type{1}{1};

O = load(['data/', type, '/W']);
O = O';
[emd, O, flow] = SinkhornWithTransformationInit(X_s, X_t, weight_s, weight_t, N, M, O);
