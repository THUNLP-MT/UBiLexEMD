import sys

def countLines(file):
	with open(file, 'r', encoding='utf-8') as fin:
		num = len(fin.readlines())
	return num

def sortFlow(flowFile, numWords):
	flowDict = {}
	with open(flowFile, 'r') as fin:
		for line in fin:
			parts = line.split()
			part1 = int(parts[0])
			part2 = int(parts[1])
			part3 = float(parts[2])
			if part1 in flowDict:
				if part2 in flowDict[part1]:
					print('Merging needed!')
					flowDict[part1][part2] += part3
				else:
					flowDict[part1][part2] = part3
			else:
				flowDict[part1] = {part2 : part3}
	sortedFlowList = []
	for i in range(numWords):
		flowToDict = flowDict.get(i, {})
		sortedFlowToDict = sorted(flowToDict.items(), key=lambda kv : kv[1], reverse=True)
		sortedFlowList.append([k for k, _ in sortedFlowToDict])
	return sortedFlowList

def mapID2Word(flowList, sourceVocabFile, targetVocabFile, outputFile):
	with open(targetVocabFile, 'r', encoding='utf-8') as ftv:
		vocab = [line.strip() for line in ftv]
	with open(sourceVocabFile, 'r', encoding='utf-8') as fsv, open(outputFile, 'w', encoding='utf-8') as fout:
		for srcWord, l in zip(fsv, flowList):
			fout.write(srcWord.strip() + '\t' + ' '.join(map(lambda x : vocab[x], l)) + '\n')

if __name__ == '__main__':
	flowFile = sys.argv[1]
	sourceVocabFile = sys.argv[2]
	targetVocabFile = sys.argv[3]
	outputFile = sys.argv[4]
	numWords = countLines(sourceVocabFile)
	sortedFlowList = sortFlow(flowFile, numWords)
	mapID2Word(sortedFlowList, sourceVocabFile, targetVocabFile, outputFile)