set -u
set -e

source config

baseDir=data/$config
flow=$baseDir/flow.$1
vocabSrc=$baseDir/vocab.$lang1
vocabTgt=$baseDir/vocab.$lang2
output=$baseDir/result.$1

python3 scripts/processFlow.py $flow $vocabSrc $vocabTgt $output
