source config

THEANO_FLAGS=floatX=float32 python src/linear_ae_ff_wgan.py $config $lang1 $lang2 --num-minibatches 10000000 --alt-loss --input-noise-param 0 --hidden-noise-param 0 --Dlr 0.0005 --Glr 0.05 > log
