import csv

import pandas as pd

from CF import CF

data = pd.read_csv('data/ratings.csv')

train = {}
testId = []
testResult = {}

countUserId = []
for i in range(max(data['userId'])):
	countUserId.append(0)
for i in data['userId']:
	countUserId[i - 1] += 1
num = 0
for i in range(len(countUserId)):
	i += 1
	train.setdefault(i, {})
	testResult.setdefault(i, {})
	testId.append(i)
	dataJ = data[num:num+countUserId[i - 1]].sort_values(by='timestamp')
	num += countUserId[i - 1]
	lenDataJ = len(dataJ)
	for j in range(int(0.2 * lenDataJ)):
		train[i].update({ dataJ.iloc[j]['movieId']: dataJ.iloc[j]['rating'] })
		testResult[i].update({ dataJ.iloc[j]['movieId']: dataJ.iloc[j]['rating'] })
	for j in range(int(0.2 * lenDataJ), lenDataJ):
		testResult[i].update({ dataJ.iloc[j]['movieId']: dataJ.iloc[j]['rating'] })

cf = CF(train)
result = []
correct = 0
error = 0
for i in testId:
	result = cf.recommend(i, way='item')
	if i < 10:
		print("为用户%d推荐电影:" % i, end='')
	for (rating, item) in result:
		if item not in train[i]:
			if item in testResult[i]:
				if testResult[i][item] > 2:
					correct += 1
				else:
					error += 1
			else:
				error += 1
		else:
			error += 1
		if i < 10:
			print("%d" % item, end=' ')
	if i < 10:
		print()
print(correct / (correct + error))
