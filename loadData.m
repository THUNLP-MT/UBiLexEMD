fconfig = fopen('matlab.config');
content = textscan(fconfig, '%s');
fclose(fconfig);
type = content{1}{1};
lang1 = content{1}{2};
lang2 = content{1}{3};

X_s = load(['data/', type, '/vec.', lang1]);
X_s = X_s';
X_t = load(['data/', type, '/vec.', lang2]);
X_t = X_t';
weight_s = load(['data/', type, '/count.', lang1]);
weight_t = load(['data/', type, '/count.', lang2]);
weight_s = weight_s';
weight_t = weight_t';
