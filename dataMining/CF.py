from math import sqrt

class CF:
	# data：数据集，为一个字典，第一个key是用户id；value又是一个字典，其key为商品id，value为评分
	# k：表示得出最相近的近邻数
	# metric：表示使用计算相似度的方法
	# n：表示推荐商品的个数
	def __init__(self, data, k=9, metric='pearson', n=9, p=2):
		self.data = data
		self.k = k
		self.n = n
		self.p = p
		if metric == 'pearson':
			self.metric = self.pearson
		elif metric == 'euclidean':
			self.metric = self.euclidean
		elif metric == 'minkowski':
			self.metric = self.minkowski
		elif metric == 'cosine':
			self.metric = self.cosine
		elif metric == 'jaccard':
			self.metric = self.jaccard

	def transformDataFormat(self, data):
		result = {}
		for user in data:
			for item in data[user]:
				result.setdefault(item, {})
				result[item].update({ user: data[user][item] })
		return result

	# 计算欧式距离
	def euclidean(self, score, Id1, Id2):
		# 首先找出两个用户或者两个物品共同评过分的物品
		joint = {}
		for item in score[Id1]:
			if item in score[Id2]:
				joint[item] = 1
		count = len(joint)
		# 如果两个用户或者两个物品没有相似之处，返回0
		if count == 0:
			return 0
		return 1 / (1 + sum([(score[Id1][item] - score[Id2][item]) ** 2 for item in score[Id1] if item in score[Id2]]))

	# 计算闵可夫斯基距离
	def minkowski(self, score, Id1, Id2):
		p = self.p
		# 首先找出两个用户或者两个物品共同评过分的物品
		joint = {}
		for item in score[Id1]:
			if item in score[Id2]:
				joint[item] = 1
		count = len(joint)
		# 如果两个用户或者两个物品没有相似之处，返回0
		if count == 0:
			return 0
		return 1 / (1 + pow(sum([pow((score[Id1][item] - score[Id2][item]), p) for item in score[Id1] if item in score[Id2]]), 1 / p))

	# 余弦距离相关度
	def cosine(self, score, Id1, Id2):
		return sum([score[Id1][item] * score[Id2][item] for item in score[Id1]  if item in score[Id2]]) / (sqrt(sum([score[Id1][item] ** 2 for item in score[Id1]])) * sqrt(sum([score[Id2][item] ** 2 for item in score[Id2]])))

	# 计算皮尔逊相关度
	def pearson(self, score, Id1, Id2):
		'''
		两个目标的皮尔逊距离
		x和y是评分
		'''
		sum_xy = 0
		sum_x = 0
		sum_y = 0
		sum_x2 = 0
		sum_y2 = 0
		# 首先找出两个用户或者两个物品共同评过分的物品
		joint = {}
		for item in score[Id1]:
			if item in score[Id2]:
				joint[item] = 1
		count = len(joint)
		# 如果两个用户或者两个物品没有相似之处，返回0
		if count == 0:
			return 0
		sum_x = sum([score[Id1][item] for item in joint])
		sum_y = sum([score[Id2][item] for item in joint])
		sum_x2 = sum([score[Id1][item] ** 2 for item in joint])
		sum_y2 = sum([score[Id2][item] ** 2 for item in joint])
		sum_xy = sum([score[Id1][item] * score[Id2][item] for item in joint])
		denominator = sqrt(sum_x2 - sum_x ** 2 / count) * sqrt(sum_y2 - sum_y ** 2 / count)
		if denominator == 0:
			return 0
		else:
			return (sum_xy - sum_x * sum_y / count) / denominator

	# 计算杰卡德相关度
	def jaccard(self, score, Id1, Id2):
		Joint = [public for public in score[Id1] if public in score[Id2]]
		return len(Joint) / (len(score[Id1]) + len(score[Id2]) - len(Joint))

	def topMatches(self, score, Id):
		# 计算与用户最相近的k个用户
		scores = [(self.metric(score, Id, other), other) for other in score if other != Id]
		scores.sort()
		for i in scores:
			if i[0] == 0:
				scores.remove(i)
		if len(scores) >= self.k:
			self.realK = self.k
		else:
			self.realK = len(scores)
		scores.reverse()
		return scores[:self.realK]

	def getRecommendationsByUser(self, score, userId):
		totals = {}
		scores = self.topMatches(score, userId)
		for (sim, other) in scores:
			# 不用与自己比较
			if other == userId:
				continue
			# 相似度小于等于0相当于无价值，忽略
			if sim <= 0:
				continue
			for item in score[other]:
				# 对自己的没看过的电影进行评分
				if item not in score[userId]:
					totals.setdefault(item, 0)
					totals[item] += score[other][item] * sim
		rankings = [(total, item) for item, total in totals.items()]
		rankings.sort()
		rankings.reverse()
		self.realN = min(self.n, len(rankings))
		return rankings[:self.realN]

	def calculateItemsSimMatrix(self):
		result = {}
		scoreBasedItem = self.transformDataFormat(self.data)
		for item in scoreBasedItem:
			result.setdefault(item, {})
			result[item] = [(self.metric(scoreBasedItem, item, other), other) for other in scoreBasedItem if other != item]
		return result

	def getRecommendationsByItem(self, score, itemSimMartrix, userId):
		marks = score[userId]
		scores = {}
		for (item, mark) in marks.items():
			for (sim, anotherItem) in itemSimMartrix[item]:
				if sim <= 0:
					continue
				# 对自己的没看过的电影进行评分
				if anotherItem in score[userId]:
					continue
				scores.setdefault(anotherItem, 0)
				scores[anotherItem] += sim * mark
		rankings = [(score, item) for item, score in scores.items()]
		rankings.sort()
		rankings.reverse()
		self.realN = min(self.n, len(rankings))
		return rankings[:self.realN]

	def recommend(self, userId, way='user'):
		if way == 'user':
			return self.getRecommendationsByUser(self.data, userId)
		elif way == 'item':
			return self.getRecommendationsByItem(self.data, self.calculateItemsSimMatrix(), userId)
		else:
			return
