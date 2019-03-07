import json
import numpy as np
import pandas as pd
import itertools
import math


class featureClass:
    def __init__(self, index, name, values, parentIndex):
        self.name = name
        self.values = values
        self.index = index
        self.parent_index = parentIndex
        self.positiveCPT = []
        self.negativeCPT = []


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
        self.tanFeatureClass = []
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

    def printNBProbabilityOnTestDataSet(self, testDataSet):
        features = self.features[:, 0]
        for f in features:
            print("{0} class".format(f))
        print()
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
        feature_pairs_array = np.array(np.meshgrid(features, features)).T.reshape(-1, 2)
        xi_index = 0
        xj_index = 0
        adj_matrix = []
        weights = []
        for index, feature_pairs in enumerate(feature_pairs_array):
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
                pairs = (np.array(np.meshgrid(feature_pairs[0], feature_pairs[1])).T.reshape(-1,2)).tolist()
                for pair in pairs:
                    n_xi_xj_y = self.dataSet[np.where((self.dataSet[:, xi_index] == pair[0]) & (self.dataSet[:, xj_index] == pair[1]) & (self.dataSet[:, -1] == self.classifiers[0]))].shape[0]
                    n_xi_xj_y_dash = self.dataSet[np.where((self.dataSet[:, xi_index] == pair[0]) & (self.dataSet[:, xj_index] == pair[1]) & (self.dataSet[:, -1] == self.classifiers[1]))].shape[0]

                    p_xi_xj_y = self.findLaplaceEstimation(n_xi_xj_y, self.shape[0], len(pairs) * 2)
                    p_xi_xj_y_dash = self.findLaplaceEstimation(n_xi_xj_y_dash, self.shape[0], len(pairs) * 2)

                    p_xi_xj_condition_y = self.findLaplaceEstimation(n_xi_xj_y, self.positiveDataSet.shape[0], len(pairs))
                    p_xi_xj_condition_y_dash = self.findLaplaceEstimation(n_xi_xj_y_dash, self.negativeDataSet.shape[0], len(pairs))

                    p_xiy_xjy = 1.0 * self.positiveCPT[xi_index][pair[0]] * self.positiveCPT[xj_index][pair[1]]
                    p_xiy_dash_xjy_dash = 1.0 * self.negativeCPT[xi_index][pair[0]] * self.negativeCPT[xj_index][pair[1]]

                    sum += p_xi_xj_y * math.log((p_xi_xj_condition_y/p_xiy_xjy), 2) + p_xi_xj_y_dash * math.log((p_xi_xj_condition_y_dash/p_xiy_dash_xjy_dash), 2)
                weights.append(sum)
                xj_index += 1
        adj_matrix.append(weights)
        return np.array(adj_matrix)

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


    def compute_tan_cpt(self, dependency_graph):
        tan_feature_class = []
        for class_index,feature_class in enumerate(self.tanFeatureClass):
            # self.tanFeatureClass.positiveCPT[0] = []
            # self.tanFeatureClass.negativeCPT[1] = []
            if (feature_class.parent_index != len(self.features)):
                count_parent_attrs_on_pos_dataset = []
                count_parent_attrs_on_neg_dataset = []
                for attr in self.tanFeatureClass[feature_class.parent_index].values:
                    count_parent_attrs_on_pos_dataset.append(self.positiveDataSet[np.where(self.positiveDataSet[:,feature_class.parent_index] == attr)].shape[0])
                    count_parent_attrs_on_neg_dataset.append(self.negativeDataSet[np.where(self.negativeDataSet[:,feature_class.parent_index] == attr)].shape[0])

                feature_prob_on_pos_dataset = []
                feature_prob_on_neg_dataset = []
                for fIndex, attr in enumerate(feature_class.values):
                    attr_prob_list_on_pos_dataset = []
                    attr_prob_list_on_neg_dataset = []
                    for pIndex, parent_attr in enumerate(self.tanFeatureClass[feature_class.parent_index].values):
                        count_of_attr_on_pos_dataset = self.positiveDataSet[np.where((self.positiveDataSet[:, class_index] == attr) & (self.positiveDataSet[:, feature_class.parent_index] == parent_attr))].shape[0]
                        count_of_attr_on_neg_dataset = self.negativeDataSet[np.where((self.negativeDataSet[:, class_index] == attr) & (self.negativeDataSet[:,feature_class.parent_index] == parent_attr))].shape[0]
                        attr_prob_list_on_pos_dataset.append(self.findLaplaceEstimation(count_of_attr_on_pos_dataset, count_parent_attrs_on_pos_dataset[pIndex], len(feature_class.values)))
                        attr_prob_list_on_neg_dataset.append(self.findLaplaceEstimation(count_of_attr_on_neg_dataset, count_parent_attrs_on_neg_dataset[pIndex], len(feature_class.values)))
                    feature_prob_on_pos_dataset.append(attr_prob_list_on_pos_dataset)
                    feature_prob_on_neg_dataset.append(attr_prob_list_on_neg_dataset)

                feature_class.positiveCPT = feature_prob_on_pos_dataset
                feature_class.negativeCPT = feature_prob_on_neg_dataset
            tan_feature_class.append(feature_class)

    def tan_predict(self, dataSet):
        corrects = 0
        for rowIndex, testRow in enumerate(dataSet[:, 0:-1]):
            posProbability = self.posAttrProbability
            negProbability = self.negAttrProbability
            for colIndex, feature in enumerate(testRow):
                feature_class = self.tanFeatureClass[colIndex]
                if (feature_class.parent_index == len(self.features)):
                    posProbability *= self.positiveCPT[colIndex][feature] if feature in self.positiveCPT[colIndex] else (1 / (self.positiveDataSet.shape[0] + len(self.features[colIndex][1]) + 1))
                    negProbability *= self.negativeCPT[colIndex][feature] if feature in self.negativeCPT[colIndex] else (1 / (self.negativeDataSet.shape[0] + len(self.features[colIndex][1]) + 1))
                else:
                    parent_feature_attr = testRow[feature_class.parent_index]
                    x_index = self.features[colIndex][1].index(feature)
                    y_index = self.features[feature_class.parent_index][1].index(parent_feature_attr)
                    posProbability *= feature_class.positiveCPT[x_index][y_index]
                    negProbability *= feature_class.negativeCPT[x_index][y_index]
            totalProbabilityOnPosDataSet = (posProbability / (posProbability + negProbability))
            totalProbabilityOnNegDataSet = (negProbability / (posProbability + negProbability))
            if (totalProbabilityOnPosDataSet > totalProbabilityOnNegDataSet):
                print("{0} {1} {2:.12f}".format(self.classifiers[0], dataSet[rowIndex, -1], totalProbabilityOnPosDataSet))
                if (self.classifiers[0] == dataSet[rowIndex, -1]):
                    corrects += 1
            else:
                print("{0} {1} {2:.12f}".format(self.classifiers[1], dataSet[rowIndex, -1], totalProbabilityOnNegDataSet))
                if (self.classifiers[1] == dataSet[rowIndex, -1]):
                    corrects += 1
        print()
        print(corrects)

    def created_tan_feature_class(self, dependency_graph):
        for index, feature in enumerate(self.features):
            feature_obj = featureClass(index, feature[0], feature[1], dependency_graph[index])
            self.tanFeatureClass.append(feature_obj)

    def getValueByKey(self, inputList, index):
        return inputList[index] if index in inputList else 1.0

    def check_for_missing_feature_values(self, featureValues, attrDict):
        for fValue in featureValues:
            if fValue not in attrDict.keys():
                attrDict[fValue] = 0
        return attrDict

    def findLaplaceEstimation(self, specificAttrCount, totalAttrCount, noOfAttr):
        return (specificAttrCount + 1.0) / (totalAttrCount + noOfAttr)

if __name__ == '__main__':
    # trainingFileName = "./Resources/lymphography_train.json"
    # testFileName = "./Resources/lymphography_test.json"
    choice = "n"
    # trainingFileName = "./Resources/tic-tac-toe_train.json"
    # testFileName = "./Resources/tic-tac-toe_test.json"
    trainingFileName = "./Resources/tic-tac-toe_sub_train.json"
    testFileName = "./Resources/tic-tac-toe_sub_test.json"

    trainBayesNetwork = Bayes(trainingFileName)
    testBayesNetwork = Bayes(testFileName)

    trainBayesNetwork.computeNBConditionalProbabilityTable()
    if choice == "t":
        trainBayesNetwork.printNBProbabilityOnTestDataSet(testBayesNetwork.dataSet)
    else:
        adj_matrix = trainBayesNetwork.computeWeights()
        #print(adj_matrix)
        vertices_list = []
        for i in range(len(trainBayesNetwork.features)):
            vertices_list.append(i)
        edge_matrix = trainBayesNetwork.findMaximumWeightedEdgeUsingPrims(adj_matrix, vertices_list)
        #print(edge_matrix)
        # trainBayesNetwork.computeNATConditionalProbabilityTable()
        # trainBayesNetwork.printNATProbabilityOnTestDataSet(testBayesNetwork)

        # Output for Starting part of TAN
        dependency_graph = {}
        featureDataSet = trainBayesNetwork.features.tolist()
        features = np.array(trainBayesNetwork.features)[:,0].tolist()
        print(featureDataSet[0][0] + " " + "class")
        dependency_graph[0] = len(featureDataSet)
        for attr in features:
            for edge in edge_matrix:
                if (features.index(attr) == edge[1]):
                    print(str(featureDataSet[edge[1]][0]) + " " + str(featureDataSet[edge[0]][0]) + " class")
                    #print(features.index(features[edge[1]][0]), features.index(features[edge[0]][0]))
                    dependency_graph[features.index(features[edge[1]])] = features.index(features[edge[0]])
        print()
        trainBayesNetwork.created_tan_feature_class(dependency_graph)
        trainBayesNetwork.compute_tan_cpt(dependency_graph)
        trainBayesNetwork.tan_predict(testBayesNetwork.dataSet)
        # for edge in tree_list:
        #     features[edge[0]].parent_index = edge[1]
        # features[-1].parent_index = features[-1].index



