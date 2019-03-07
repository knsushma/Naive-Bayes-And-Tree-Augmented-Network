import json
import numpy as np
import pandas as pd
import itertools
import math


class featureClass:
    def __init__(self, name, values, index):
        self.name = name
        self.values = values
        self.index = index
        self.parent_index = 0


class Bayes:
    def __init__(self, fileName):
        self.classifiers = []
        self.features = []
        self.dataSet = [0][0]
        self.shape = (0, 0)
        self.positiveDataSet = [0][0]
        self.negativeDataSet = [0][0]
        self.posAttrProbability = 0.0
        self.negAttrProbability = 0.0
        self.positiveCPT = [0][0]
        self.negativeCPT = [0][0]
        self.loadAndInitDataSet(fileName)

    def loadAndInitDataSet(self, fileName):
        fileContent = json.load(open(fileName))
        self.features =  np.array(fileContent['metadata']['features'][0:-1])
        self.dataSet = np.array(fileContent['data'])
        self.classifiers = fileContent['metadata']['features'][-1][1]
        self.shape = self.dataSet.shape

    def classifyAndComputeDataValues(self):
        self.positiveDataSet = self.dataSet[np.where(self.dataSet[:,-1] == self.classifiers[0])]
        self.negativeDataSet = self.dataSet[np.where(self.dataSet[:,-1] == self.classifiers[1])]
        self.posAttrProbability = self.findLaplaceEstimation(self.positiveDataSet.shape[0], self.dataSet.shape[0], len(self.classifiers))
        self.negAttrProbability = self.findLaplaceEstimation(self.negativeDataSet.shape[0], self.dataSet.shape[0], len(self.classifiers))

    def computeNBConditionalProbabilityTable(self):
        self.classifyAndComputeDataValues()
        positiveList = []
        negativeList = []
        positiveAttrCount = self.positiveDataSet.shape[0]
        negativeAttrCount = self.negativeDataSet.shape[0]
        for index in range(self.shape[1]-1):
            noOfClassifiers = len(self.features[index][1])
            featureValues = self.features[index][1]
            #print("{} {}".format(self.features[index],noOfClassifiers))
            positiveAttrDict = pd.value_counts(self.positiveDataSet[:, index]).to_dict()
            positiveAttrDict = self.check_for_missing_feature_values(featureValues, positiveAttrDict)
            positiveAttrDict.update((classifier, self.findLaplaceEstimation(count, positiveAttrCount, noOfClassifiers)) for classifier,count in positiveAttrDict.items())
            positiveList.append(positiveAttrDict)
            negativeAttrDict = pd.value_counts(self.negativeDataSet[:, index]).to_dict()
            negativeAttrDict = self.check_for_missing_feature_values(featureValues, negativeAttrDict)
            negativeAttrDict.update((classifier, self.findLaplaceEstimation(count, negativeAttrCount, noOfClassifiers)) for classifier,count in negativeAttrDict.items())
            negativeList.append(negativeAttrDict)
        self.positiveCPT =  np.array(positiveList)
        self.negativeCPT = np.array(negativeList)

    def check_for_missing_feature_values(self, featureValues, attrDict):
        for fValue in featureValues:
            if fValue not in attrDict.keys():
                attrDict[fValue] = 0
        return attrDict

    def findLaplaceEstimation(self, specificAttrCount, totalAttrCount, noOfAttr):
        return (specificAttrCount + 1.0)/(totalAttrCount + noOfAttr)

    def printBayesNwFeatures(self):
        features = self.features[:, 0]
        for f in features:
            print("{0} class".format(f))
        print()

    def printNBProbabilityOnTestDataSet(self, testDataSet):
        self.printBayesNwFeatures()
        corrects = 0
        for rowIndex,testRow in enumerate(testDataSet[:,0:-1]):
            posProbability = self.posAttrProbability
            negProbability = self.negAttrProbability
            for colIndex,feature in enumerate(testRow):
                posProbability *= self.positiveCPT[colIndex][feature] if feature in self.positiveCPT[colIndex] else (1/ (self.positiveDataSet.shape[0]+len(self.features[colIndex][1])+1))
                negProbability *= self.negativeCPT[colIndex][feature] if feature in self.negativeCPT[colIndex] else (1/ (self.negativeDataSet.shape[0]+len(self.features[colIndex][1])+1))
            totalProbabilityOnPosDataSet = (posProbability/(posProbability+negProbability))
            totalProbabilityOnNegDataSet = (negProbability / (posProbability + negProbability))
            if (totalProbabilityOnPosDataSet > totalProbabilityOnNegDataSet):
                print("{0} {1} {2:.12f}".format(self.classifiers[0], testDataSet[rowIndex, -1], totalProbabilityOnPosDataSet))
                if (self.classifiers[0] == testDataSet[rowIndex, -1]):
                    corrects += 1
            else:
                print("{0} {1} {2:.12f}".format(self.classifiers[1], testDataSet[rowIndex, -1], totalProbabilityOnNegDataSet))
                if (self.classifiers[1] == testDataSet[rowIndex, -1]):
                    corrects += 1
        print()
        print(corrects)


    def computeWeights(self):
        features = np.array(self.features[:,-1])
        print(self.features)
        print()
        #print(np.array(np.meshgrid(features[0][0], features[:])).T.reshape(-1,2))
        # print(np.array(np.meshgrid(features, features)).T.reshape(-1, 2)[0])

        result = np.array(np.meshgrid(features, features)).T.reshape(-1, 2)

        #print(list(itertools.combinations(features, 2)))
        #result = [(np.array(np.meshgrid(pair[0], pair[1])).T.reshape(-1,2)) for pair in list(itertools.permutations(features, 2))]
        #print(result)
        xi_index = 0
        xj_index = 0
        adj_matrix = []
        weights = []
        for index, content in enumerate(result):
            if (xi_index == xj_index):
                weights.append(-1.0)
                xj_index += 1
            else:
                if (index != 0 and xj_index % (self.shape[1] - 1) == 0):
                    xi_index += 1
                    xj_index = 0
                    adj_matrix.append(weights)
                    weights = []
                sum = 0.0
                for pair in (np.array(np.meshgrid(content[0], content[1])).T.reshape(-1,2)):
                    n_xi_xj_y = self.dataSet[np.where((self.dataSet[:, xi_index] == pair[0]) & (self.dataSet[:, xj_index] == pair[1]) & (self.dataSet[:, -1] == self.classifiers[0]))].shape[0]
                    n_xi_xj_y_dash = self.dataSet[np.where((self.dataSet[:, xi_index] == pair[0]) & (self.dataSet[:, xj_index] == pair[1]) & (self.dataSet[:, -1] == self.classifiers[1]))].shape[0]

                    p_xi_xj_y = self.findLaplaceEstimation(n_xi_xj_y, self.shape[0], len(content) * 2)
                    p_xi_xj_y_dash = self.findLaplaceEstimation(n_xi_xj_y_dash, self.shape[0], len(content) * 2)

                    p_xi_xj_condition_y = self.findLaplaceEstimation(n_xi_xj_y, self.positiveDataSet.shape[0], len(content))
                    p_xi_xj_condition_y_dash = self.findLaplaceEstimation(n_xi_xj_y_dash, self.negativeDataSet.shape[0], len(content))

                    p_xiy_xjy = 1.0 * self.positiveCPT[xi_index][pair[0]] * self.positiveCPT[xj_index][pair[1]]
                    p_xiy_dash_xjy_dash = 1.0 * self.negativeCPT[xi_index][pair[0]] * self.negativeCPT[xj_index][pair[1]]

                    sum += p_xi_xj_y * math.log((p_xi_xj_condition_y/p_xiy_xjy), 2) + p_xi_xj_y_dash * math.log((p_xi_xj_condition_y_dash/p_xiy_dash_xjy_dash), 2)
                weights.append(sum)
                xj_index += 1
        adj_matrix.append(weights)
        return np.array(adj_matrix)

        # for index in range(len(features)-1):
        #     print(np.array(np.meshgrid(features[:], features[:])).T.reshape(-1, 2))

        # for index in range(len(features)):
        #     perm = np.array(np.meshgrid(features[index], features[:])).T.reshape(-1, 2)
        #     print(perm)
        #     print()
        # for i in range(len(perm[0])):
        #     print(perm[i])

        # for featureIndex in range(len(features)):
        #     featureCombinations = np.array(np.meshgrid(features[featureIndex], features[featureIndex+1:])).T.reshape(-1, 2)
        #     print(featureCombinations.shape)
        #     for index in range(len(featureCombinations)):
        #         count = 1
        #         pairs = np.array(np.meshgrid(featureCombinations[index][0], featureCombinations[index][1])).T.reshape(-1, 2)
        #         print(pairs)
        #         print()
                # for i,p in enumerate(pairs):
                #     print(featureIndex, i, p)
                # print()
        # res = np.array(np.meshgrid(features[0], features[1:])).T.reshape(-1, 2)
        # print(res.shape)


    def getValueByKey(self, inputList, index):
        return inputList[index] if index in inputList else 1.0

    def findMaximumWeightedEdgeUsingPrims(self, adj_matrix, vertices_list):
        V_new = list()
        V_new.append(vertices_list[0])
        E_new = []
        while (len(V_new) < len(vertices_list)):
            candidate_vertices = []
            for u in V_new:
                temp_list = adj_matrix[u].tolist()
                for t in temp_list:
                    if (temp_list.index(t) not in V_new):
                        candidate_vertices.append([temp_list.index(t), u, t])
            candidate_vertices.sort(key=lambda x: x[2])
            # print candidate_vertices
            V_new.append(candidate_vertices[-1][0])
            E_new.append([candidate_vertices[-1][1], candidate_vertices[-1][0]])
        return E_new

if __name__ == '__main__':
    trainingFileName = "./Resources/lymphography_train.json"
    testFileName = "./Resources/lymphography_test.json"
    choice = "n"
    # trainingFileName = "./Resources/tic-tac-toe_train.json"
    # testFileName = "./Resources/tic-tac-toe_test.json"
    # trainingFileName = "./Resources/tic-tac-toe_sub_train.json"
    # testFileName = "./Resources/tic-tac-toe_sub_test.json"

    trainBayesNetwork = Bayes(trainingFileName)
    testBayesNetwork = Bayes(testFileName)

    trainBayesNetwork.computeNBConditionalProbabilityTable()
    if choice == "t":
        trainBayesNetwork.printNBProbabilityOnTestDataSet(testBayesNetwork.dataSet)
    else:
        adj_matrix = trainBayesNetwork.computeWeights()
        print(adj_matrix)
        vertices_list = []
        for i in range(len(trainBayesNetwork.features)):
            vertices_list.append(i)
        edge_matrix = trainBayesNetwork.findMaximumWeightedEdgeUsingPrims(adj_matrix, vertices_list)
        print(edge_matrix)
        # trainBayesNetwork.computeNATConditionalProbabilityTable()
        # trainBayesNetwork.printNATProbabilityOnTestDataSet(testBayesNetwork)

        # Output for Starting part of TAN
        tree_list =[]
        featureDataSet = trainBayesNetwork.features.tolist()
        features = np.array(trainBayesNetwork.features)[:,0].tolist()
        print(featureDataSet[0][0] + " " + "class")
        tree_list.append([0, len(featureDataSet)])
        for attr in features:
            for edge in edge_matrix:
                if (features.index(attr) == edge[1]):
                    print(str(featureDataSet[edge[1]][0]) + " " + str(featureDataSet[edge[0]][0]) + " class")
                    #print(features.index(features[edge[1]][0]), features.index(features[edge[0]][0]))
                    tree_list.append([features.index(features[edge[1]]), features.index(features[edge[0]])])  # tree_list is a collection of rows in the format [child, parent]


        # for edge in tree_list:
        #     features[edge[0]].parent_index = edge[1]
        # features[-1].parent_index = features[-1].index



