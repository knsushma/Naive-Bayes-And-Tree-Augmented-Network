import json
import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import math
import sys


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
        self.num_of_correct_predictions = 0
        self.positiveCPT = [0][0]
        self.negativeCPT = [0][0]
        self.tanFeatureClass = []
        self.nb_confidence = []
        self.tan_confidence = []
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
        for rowIndex,testRow in enumerate(testDataSet[:,0:-1]):
            posProbability = self.posAttrProbability
            negProbability = self.negAttrProbability
            for colIndex,feature in enumerate(testRow):
                posProbability *= self.positiveCPT[colIndex][feature] if feature in self.positiveCPT[colIndex] else (1/ (self.positiveDataSet.shape[0]+len(self.features[colIndex][1])+1))
                negProbability *= self.negativeCPT[colIndex][feature] if feature in self.negativeCPT[colIndex] else (1/ (self.negativeDataSet.shape[0]+len(self.features[colIndex][1])+1))
            totalProbabilityOnPosDataSet = (posProbability/(posProbability+negProbability))
            totalProbabilityOnNegDataSet = (negProbability / (posProbability + negProbability))
            if (totalProbabilityOnPosDataSet > totalProbabilityOnNegDataSet):
                self.nb_confidence.append(totalProbabilityOnPosDataSet)
                print("{0} {1} {2:.12f}".format(self.classifiers[0], testDataSet[rowIndex, -1], totalProbabilityOnPosDataSet))
                if (self.classifiers[0] == testDataSet[rowIndex, -1]):
                    self.num_of_correct_predictions += 1
            else:
                self.nb_confidence.append(totalProbabilityOnNegDataSet)
                print("{0} {1} {2:.12f}".format(self.classifiers[1], testDataSet[rowIndex, -1], totalProbabilityOnNegDataSet))
                if (self.classifiers[1] == testDataSet[rowIndex, -1]):
                    self.num_of_correct_predictions += 1
        print()
        print(self.num_of_correct_predictions)

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


    def compute_tan_cpt(self):
        tan_feature_class = []
        for class_index,feature_class in enumerate(self.tanFeatureClass):
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
            #self.tan_confidence.append(totalProbabilityOnPosDataSet)
            if (totalProbabilityOnPosDataSet > totalProbabilityOnNegDataSet):
                self.tan_confidence.append(totalProbabilityOnPosDataSet)
                print("{0} {1} {2:.12f}".format(self.classifiers[0], dataSet[rowIndex, -1], totalProbabilityOnPosDataSet))
                if (self.classifiers[0] == dataSet[rowIndex, -1]):
                    self.num_of_correct_predictions += 1
            else:
                self.tan_confidence.append(totalProbabilityOnNegDataSet)
                print("{0} {1} {2:.12f}".format(self.classifiers[1], dataSet[rowIndex, -1], totalProbabilityOnNegDataSet))
                if (self.classifiers[1] == dataSet[rowIndex, -1]):
                    self.num_of_correct_predictions += 1
        print()
        print(self.num_of_correct_predictions)

    def precision_recall_graph(self, dataSet, choice):
        labelMap = {}
        labelMap[self.classifiers[0]] = 1
        labelMap[self.classifiers[1]] = 0

        testLabels = dataSet[:, -1].copy()
        for index, label in enumerate(testLabels):
            testLabels[index,] = labelMap.get(label)
        testLabels = testLabels.astype(int)

        if choice == 'nb':
            rocMatrix = np.column_stack((np.array(testLabels), np.array(self.nb_confidence)))
        else:
            rocMatrix = np.column_stack((np.array(testLabels), np.array(self.tan_confidence)))
        rocMatrixSorted = rocMatrix[np.argsort([-rocMatrix[:, 1]])][0]
        posNegCount = Counter(rocMatrixSorted[:, 0].astype(int))
        posCount = posNegCount.get(1)

        recall_list = []
        precision_list = []
        positives_so_far = 0
        negatives_so_far = 0
        for row in rocMatrixSorted:
            if int(row[0]) == 1:
                positives_so_far += 1
            else:
                negatives_so_far += 1
            precision = (positives_so_far * 1.0) / (positives_so_far + negatives_so_far)
            recall = (positives_so_far * 1.0) / posCount
            recall_list.append(recall)
            precision_list.append(precision)
        return recall_list, precision_list


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

    def plot_pr_curve(self, x1, y1, x2, y2):
        x3 = [0, 1]
        pt = 24.0 / 42
        y3 = [pt, pt]
        x4 = [0, 1]
        y4 = [1, 1]
        x5 = [1, 1]
        y5 = [pt, 1]
        fig, ax = plt.subplots()
        ax.plot(x3, y3, color='grey', ls='--', marker='')
        ax.plot(x4, y4, color='grey', ls='--', marker='')
        ax.plot(x5, y5, color='grey', ls='--', marker='')
        ax.plot(x1, y1, color='blue', marker='', label='Naive Bayes')
        ax.plot(x2, y2, color='red', marker='', label='TAN')

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.2, 0.3))
        # ax.set_ylim(0, 1.2)
        ax.set_title("PR Curve")
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        plt.show()

if __name__ == '__main__':

    if (len(sys.argv)<2):
        print("Please pass 2 arguments. 1) Training File Path, 2) Testing File path ")
        sys.exit(1)

    trainingFileName = sys.argv[1]
    testFileName = sys.argv[2]
    # trainingFileName = "./Resources/tic-tac-toe_train.json"
    # testFileName = "./Resources/tic-tac-toe_test.json"

    trainBayesNetwork = Bayes(trainingFileName)
    testBayesNetwork = Bayes(testFileName)

    trainBayesNetwork.computeNBConditionalProbabilityTable()
    trainBayesNetwork.printNBProbabilityOnTestDataSet(testBayesNetwork.dataSet)
    nb_recall, nb_precision = trainBayesNetwork.precision_recall_graph(testBayesNetwork.dataSet,"nb")

    adj_matrix = trainBayesNetwork.computeWeights()
    vertices_list = []
    for i in range(len(trainBayesNetwork.features)):
        vertices_list.append(i)
    edge_matrix = trainBayesNetwork.findMaximumWeightedEdgeUsingPrims(adj_matrix, vertices_list)
    dependency_graph = {}
    featureDataSet = trainBayesNetwork.features.tolist()
    features = np.array(trainBayesNetwork.features)[:,0].tolist()
    print(featureDataSet[0][0] + " " + "class")
    dependency_graph[0] = len(featureDataSet)
    for attr in features:
        for edge in edge_matrix:
            if (features.index(attr) == edge[1]):
                print(str(featureDataSet[edge[1]][0]) + " " + str(featureDataSet[edge[0]][0]) + " class")
                dependency_graph[features.index(features[edge[1]])] = features.index(features[edge[0]])
    print()
    trainBayesNetwork.created_tan_feature_class(dependency_graph)
    trainBayesNetwork.compute_tan_cpt()
    trainBayesNetwork.tan_predict(testBayesNetwork.dataSet)
    tan_recall,tan_precision = trainBayesNetwork.precision_recall_graph(testBayesNetwork.dataSet, "tan")

    trainBayesNetwork.plot_pr_curve(nb_recall, nb_precision, tan_recall, tan_precision)

# import json
# import numpy as np
# import pandas as pd
# from collections import Counter
# import matplotlib.pyplot as plt
# import math
# import sys
#
#
# class featureClass:
#     def __init__(self, index, name, values, parentIndex):
#         self.name = name
#         self.values = values
#         self.index = index
#         self.parent_index = parentIndex
#         self.positiveCPT = []
#         self.negativeCPT = []
#
#
# class Bayes:
#     def __init__(self, fileName):
#         self.classifiers = []
#         self.features = []
#         self.dataSet = [0][0]
#         self.shape = (0, 0)
#         self.positiveDataSet = [0][0]
#         self.negativeDataSet = [0][0]
#         self.posAttrProbability = 0.0
#         self.negAttrProbability = 0.0
#         self.num_of_correct_predictions = 0
#         self.positiveCPT = [0][0]
#         self.negativeCPT = [0][0]
#         self.tanFeatureClass = []
#         self.nb_confidence = []
#         self.tan_confidence = []
#         self.nb_predicted_class = []
#         self.tan_predicted_class = []
#         self.loadAndInitDataSet(fileName)
#
#     def loadAndInitDataSet(self, fileName):
#         fileContent = json.load(open(fileName))
#         self.features =  np.array(fileContent['metadata']['features'][0:-1])
#         self.dataSet = np.array(fileContent['data'])
#         self.classifiers = fileContent['metadata']['features'][-1][1]
#         self.shape = self.dataSet.shape
#
#     def classifyAndComputeDataValues(self):
#         self.positiveDataSet = self.dataSet[np.where(self.dataSet[:,-1] == self.classifiers[0])]
#         self.negativeDataSet = self.dataSet[np.where(self.dataSet[:,-1] == self.classifiers[1])]
#         self.posAttrProbability = self.findLaplaceEstimation(self.positiveDataSet.shape[0], self.dataSet.shape[0], len(self.classifiers))
#         self.negAttrProbability = self.findLaplaceEstimation(self.negativeDataSet.shape[0], self.dataSet.shape[0], len(self.classifiers))
#
#     def computeNBConditionalProbabilityTable(self):
#         self.classifyAndComputeDataValues()
#         positiveList = []
#         negativeList = []
#         positiveAttrCount = self.positiveDataSet.shape[0]
#         negativeAttrCount = self.negativeDataSet.shape[0]
#         for index in range(self.shape[1]-1):
#             noOfClassifiers = len(self.features[index][1])
#             featureValues = self.features[index][1]
#             #print("{} {}".format(self.features[index],noOfClassifiers))
#             positiveAttrDict = pd.value_counts(self.positiveDataSet[:, index]).to_dict()
#             positiveAttrDict = self.check_for_missing_feature_values(featureValues, positiveAttrDict)
#             positiveAttrDict.update((classifier, self.findLaplaceEstimation(count, positiveAttrCount, noOfClassifiers)) for classifier,count in positiveAttrDict.items())
#             positiveList.append(positiveAttrDict)
#             negativeAttrDict = pd.value_counts(self.negativeDataSet[:, index]).to_dict()
#             negativeAttrDict = self.check_for_missing_feature_values(featureValues, negativeAttrDict)
#             negativeAttrDict.update((classifier, self.findLaplaceEstimation(count, negativeAttrCount, noOfClassifiers)) for classifier,count in negativeAttrDict.items())
#             negativeList.append(negativeAttrDict)
#         self.positiveCPT =  np.array(positiveList)
#         self.negativeCPT = np.array(negativeList)
#
#     def printNBProbabilityOnTestDataSet(self, testDataSet):
#         features = self.features[:, 0]
#         for f in features:
#             print("{0} class".format(f))
#         print()
#         for rowIndex,testRow in enumerate(testDataSet[:,0:-1]):
#             posProbability = self.posAttrProbability
#             negProbability = self.negAttrProbability
#             for colIndex,feature in enumerate(testRow):
#                 posProbability *= self.positiveCPT[colIndex][feature] if feature in self.positiveCPT[colIndex] else (1/ (self.positiveDataSet.shape[0]+len(self.features[colIndex][1])+1))
#                 negProbability *= self.negativeCPT[colIndex][feature] if feature in self.negativeCPT[colIndex] else (1/ (self.negativeDataSet.shape[0]+len(self.features[colIndex][1])+1))
#             totalProbabilityOnPosDataSet = (posProbability/(posProbability+negProbability))
#             totalProbabilityOnNegDataSet = (negProbability / (posProbability + negProbability))
#             if (totalProbabilityOnPosDataSet > totalProbabilityOnNegDataSet):
#                 self.nb_confidence.append(totalProbabilityOnPosDataSet)
#                 self.nb_predicted_class.append(1)
#                 print("{0} {1} {2:.12f}".format(self.classifiers[0], testDataSet[rowIndex, -1], totalProbabilityOnPosDataSet))
#                 if (self.classifiers[0] == testDataSet[rowIndex, -1]):
#                     self.num_of_correct_predictions += 1
#             else:
#                 self.nb_confidence.append(totalProbabilityOnNegDataSet)
#                 self.nb_predicted_class.append(0)
#                 print("{0} {1} {2:.12f}".format(self.classifiers[1], testDataSet[rowIndex, -1], totalProbabilityOnNegDataSet))
#                 if (self.classifiers[1] == testDataSet[rowIndex, -1]):
#                     self.num_of_correct_predictions += 1
#         print()
#         print(self.num_of_correct_predictions)
#
#     def computeWeights(self):
#         features = np.array(self.features[:,-1])
#         feature_pairs_array = np.array(np.meshgrid(features, features)).T.reshape(-1, 2)
#         xi_index = 0
#         xj_index = 0
#         adj_matrix = []
#         weights = []
#         for index, feature_pairs in enumerate(feature_pairs_array):
#             if (xi_index == xj_index):
#                 weights.append(-1.0)
#                 xj_index += 1
#             else:
#                 if (index != 0 and xj_index % (self.shape[1] - 1) == 0):
#                     xi_index += 1
#                     xj_index = 0
#                     adj_matrix.append(weights)
#                     weights = []
#                 sum = 0.0
#                 pairs = (np.array(np.meshgrid(feature_pairs[0], feature_pairs[1])).T.reshape(-1,2)).tolist()
#                 for pair in pairs:
#                     n_xi_xj_y = self.dataSet[np.where((self.dataSet[:, xi_index] == pair[0]) & (self.dataSet[:, xj_index] == pair[1]) & (self.dataSet[:, -1] == self.classifiers[0]))].shape[0]
#                     n_xi_xj_y_dash = self.dataSet[np.where((self.dataSet[:, xi_index] == pair[0]) & (self.dataSet[:, xj_index] == pair[1]) & (self.dataSet[:, -1] == self.classifiers[1]))].shape[0]
#
#                     p_xi_xj_y = self.findLaplaceEstimation(n_xi_xj_y, self.shape[0], len(pairs) * 2)
#                     p_xi_xj_y_dash = self.findLaplaceEstimation(n_xi_xj_y_dash, self.shape[0], len(pairs) * 2)
#
#                     p_xi_xj_condition_y = self.findLaplaceEstimation(n_xi_xj_y, self.positiveDataSet.shape[0], len(pairs))
#                     p_xi_xj_condition_y_dash = self.findLaplaceEstimation(n_xi_xj_y_dash, self.negativeDataSet.shape[0], len(pairs))
#
#                     p_xiy_xjy = 1.0 * self.positiveCPT[xi_index][pair[0]] * self.positiveCPT[xj_index][pair[1]]
#                     p_xiy_dash_xjy_dash = 1.0 * self.negativeCPT[xi_index][pair[0]] * self.negativeCPT[xj_index][pair[1]]
#
#                     sum += p_xi_xj_y * math.log((p_xi_xj_condition_y/p_xiy_xjy), 2) + p_xi_xj_y_dash * math.log((p_xi_xj_condition_y_dash/p_xiy_dash_xjy_dash), 2)
#                 weights.append(sum)
#                 xj_index += 1
#         adj_matrix.append(weights)
#         return np.array(adj_matrix)
#
#     def findMaximumWeightedEdgeUsingPrims(self, adj_matrix, vertices_list):
#         V_new = list()
#         V_new.append(vertices_list[0])
#         E_new = []
#         while (len(V_new) < len(vertices_list)):
#             candidate_vertices = []
#             for u in V_new:
#                 temp_list = adj_matrix[u].tolist()
#                 for t in temp_list:
#                     if (temp_list.index(t) not in V_new):
#                         candidate_vertices.append([temp_list.index(t), u, t])
#             candidate_vertices.sort(key=lambda x: x[2])
#             # print candidate_vertices
#             V_new.append(candidate_vertices[-1][0])
#             E_new.append([candidate_vertices[-1][1], candidate_vertices[-1][0]])
#         return E_new
#
#
#     def compute_tan_cpt(self):
#         tan_feature_class = []
#         for class_index,feature_class in enumerate(self.tanFeatureClass):
#             if (feature_class.parent_index != len(self.features)):
#                 count_parent_attrs_on_pos_dataset = []
#                 count_parent_attrs_on_neg_dataset = []
#                 for attr in self.tanFeatureClass[feature_class.parent_index].values:
#                     count_parent_attrs_on_pos_dataset.append(self.positiveDataSet[np.where(self.positiveDataSet[:,feature_class.parent_index] == attr)].shape[0])
#                     count_parent_attrs_on_neg_dataset.append(self.negativeDataSet[np.where(self.negativeDataSet[:,feature_class.parent_index] == attr)].shape[0])
#
#                 feature_prob_on_pos_dataset = []
#                 feature_prob_on_neg_dataset = []
#                 for fIndex, attr in enumerate(feature_class.values):
#                     attr_prob_list_on_pos_dataset = []
#                     attr_prob_list_on_neg_dataset = []
#                     for pIndex, parent_attr in enumerate(self.tanFeatureClass[feature_class.parent_index].values):
#                         count_of_attr_on_pos_dataset = self.positiveDataSet[np.where((self.positiveDataSet[:, class_index] == attr) & (self.positiveDataSet[:, feature_class.parent_index] == parent_attr))].shape[0]
#                         count_of_attr_on_neg_dataset = self.negativeDataSet[np.where((self.negativeDataSet[:, class_index] == attr) & (self.negativeDataSet[:,feature_class.parent_index] == parent_attr))].shape[0]
#                         attr_prob_list_on_pos_dataset.append(self.findLaplaceEstimation(count_of_attr_on_pos_dataset, count_parent_attrs_on_pos_dataset[pIndex], len(feature_class.values)))
#                         attr_prob_list_on_neg_dataset.append(self.findLaplaceEstimation(count_of_attr_on_neg_dataset, count_parent_attrs_on_neg_dataset[pIndex], len(feature_class.values)))
#                     feature_prob_on_pos_dataset.append(attr_prob_list_on_pos_dataset)
#                     feature_prob_on_neg_dataset.append(attr_prob_list_on_neg_dataset)
#
#                 feature_class.positiveCPT = feature_prob_on_pos_dataset
#                 feature_class.negativeCPT = feature_prob_on_neg_dataset
#             tan_feature_class.append(feature_class)
#
#     def tan_predict(self, dataSet):
#         for rowIndex, testRow in enumerate(dataSet[:, 0:-1]):
#             posProbability = self.posAttrProbability
#             negProbability = self.negAttrProbability
#             for colIndex, feature in enumerate(testRow):
#                 feature_class = self.tanFeatureClass[colIndex]
#                 if (feature_class.parent_index == len(self.features)):
#                     posProbability *= self.positiveCPT[colIndex][feature] if feature in self.positiveCPT[colIndex] else (1 / (self.positiveDataSet.shape[0] + len(self.features[colIndex][1]) + 1))
#                     negProbability *= self.negativeCPT[colIndex][feature] if feature in self.negativeCPT[colIndex] else (1 / (self.negativeDataSet.shape[0] + len(self.features[colIndex][1]) + 1))
#                 else:
#                     parent_feature_attr = testRow[feature_class.parent_index]
#                     x_index = self.features[colIndex][1].index(feature)
#                     y_index = self.features[feature_class.parent_index][1].index(parent_feature_attr)
#                     posProbability *= feature_class.positiveCPT[x_index][y_index]
#                     negProbability *= feature_class.negativeCPT[x_index][y_index]
#             totalProbabilityOnPosDataSet = (posProbability / (posProbability + negProbability))
#             totalProbabilityOnNegDataSet = (negProbability / (posProbability + negProbability))
#             #self.tan_confidence.append(totalProbabilityOnPosDataSet)
#             if (totalProbabilityOnPosDataSet > totalProbabilityOnNegDataSet):
#                 self.tan_confidence.append(totalProbabilityOnPosDataSet)
#                 self.tan_predicted_class.append(1)
#                 print("{0} {1} {2:.12f}".format(self.classifiers[0], dataSet[rowIndex, -1], totalProbabilityOnPosDataSet))
#                 if (self.classifiers[0] == dataSet[rowIndex, -1]):
#                     self.num_of_correct_predictions += 1
#             else:
#                 self.tan_confidence.append(totalProbabilityOnNegDataSet)
#                 self.tan_predicted_class.append(0)
#                 print("{0} {1} {2:.12f}".format(self.classifiers[1], dataSet[rowIndex, -1], totalProbabilityOnNegDataSet))
#                 if (self.classifiers[1] == dataSet[rowIndex, -1]):
#                     self.num_of_correct_predictions += 1
#         print()
#         print(self.num_of_correct_predictions)
#
#     def precision_recall_graph(self, dataSet, choice):
#         labelMap = {}
#         labelMap[self.classifiers[0]] = 1
#         labelMap[self.classifiers[1]] = 0
#
#         testLabels = dataSet[:, -1].copy()
#         for index, label in enumerate(testLabels):
#             testLabels[index,] = labelMap.get(label)
#         testLabels = testLabels.astype(int)
#
#         if choice == 'nb':
#             temp_matrix = np.column_stack((np.array(testLabels), np.array(self.nb_confidence)))
#             prMatrix = np.column_stack((np.array(temp_matrix), np.array(self.nb_predicted_class)))
#         else:
#             temp_matrix = np.column_stack((np.array(testLabels), np.array(self.tan_confidence)))
#             prMatrix = np.column_stack((np.array(temp_matrix), np.array(self.tan_predicted_class)))
#         prMatrixSorted = prMatrix[np.argsort([-prMatrix[:, 1]])][0]
#         posNegCount = Counter(prMatrixSorted[:, 0].astype(int))
#         posCount = posNegCount.get(1)
#         prMatrixSortedPosSelected = prMatrixSorted[np.where(prMatrixSorted[:,-1] == 1.0)]
#
#
#         recall_list = []
#         precision_list = []
#         positives_so_far = 0
#         negatives_so_far = 0
#         for row in prMatrixSortedPosSelected:
#             if int(row[0]) == 1.0:
#                 positives_so_far += 1
#             else:
#                 negatives_so_far += 1
#             precision = (positives_so_far * 1.0) / (positives_so_far + negatives_so_far)
#             recall = (positives_so_far * 1.0) / posCount
#             recall_list.append(recall)
#             precision_list.append(precision)
#         return recall_list, precision_list
#
#
#     def created_tan_feature_class(self, dependency_graph):
#         for index, feature in enumerate(self.features):
#             feature_obj = featureClass(index, feature[0], feature[1], dependency_graph[index])
#             self.tanFeatureClass.append(feature_obj)
#
#     def getValueByKey(self, inputList, index):
#         return inputList[index] if index in inputList else 1.0
#
#     def check_for_missing_feature_values(self, featureValues, attrDict):
#         for fValue in featureValues:
#             if fValue not in attrDict.keys():
#                 attrDict[fValue] = 0
#         return attrDict
#
#     def findLaplaceEstimation(self, specificAttrCount, totalAttrCount, noOfAttr):
#         return (specificAttrCount + 1.0) / (totalAttrCount + noOfAttr)
#
#     def plot_pr_curve(self, x1, y1, x2, y2):
#         x3 = [0, 1]
#         pt = 24.0 / 42
#         y3 = [pt, pt]
#         x4 = [0, 1]
#         y4 = [1, 1]
#         x5 = [1, 1]
#         y5 = [pt, 1]
#         fig, ax = plt.subplots()
#         ax.plot(x3, y3, color='grey', ls='--', marker='')
#         ax.plot(x4, y4, color='grey', ls='--', marker='')
#         ax.plot(x5, y5, color='grey', ls='--', marker='')
#         ax.plot(x1, y1, color='blue', marker='', label='Naive Bayes')
#         ax.plot(x2, y2, color='red', marker='', label='TAN')
#
#         handles, labels = ax.get_legend_handles_labels()
#         ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.2, 0.3))
#         # ax.set_ylim(0, 1.2)
#         ax.set_title("PR Curve")
#         ax.set_xlabel('Recall')
#         ax.set_ylabel('Precision')
#         plt.show()
#
# if __name__ == '__main__':
#
#     if (len(sys.argv)<2):
#         print("Please pass 2 arguments. 1) Training File Path, 2) Testing File path ")
#         sys.exit(1)
#
#     trainingFileName = sys.argv[1]
#     testFileName = sys.argv[2]
#     # trainingFileName = "./Resources/tic-tac-toe_train.json"
#     # testFileName = "./Resources/tic-tac-toe_test.json"
#
#     trainBayesNetwork = Bayes(trainingFileName)
#     testBayesNetwork = Bayes(testFileName)
#
#     trainBayesNetwork.computeNBConditionalProbabilityTable()
#     trainBayesNetwork.printNBProbabilityOnTestDataSet(testBayesNetwork.dataSet)
#     nb_recall, nb_precision = trainBayesNetwork.precision_recall_graph(testBayesNetwork.dataSet,"nb")
#
#     adj_matrix = trainBayesNetwork.computeWeights()
#     vertices_list = []
#     for i in range(len(trainBayesNetwork.features)):
#         vertices_list.append(i)
#     edge_matrix = trainBayesNetwork.findMaximumWeightedEdgeUsingPrims(adj_matrix, vertices_list)
#     dependency_graph = {}
#     featureDataSet = trainBayesNetwork.features.tolist()
#     features = np.array(trainBayesNetwork.features)[:,0].tolist()
#     print(featureDataSet[0][0] + " " + "class")
#     dependency_graph[0] = len(featureDataSet)
#     for attr in features:
#         for edge in edge_matrix:
#             if (features.index(attr) == edge[1]):
#                 print(str(featureDataSet[edge[1]][0]) + " " + str(featureDataSet[edge[0]][0]) + " class")
#                 dependency_graph[features.index(features[edge[1]])] = features.index(features[edge[0]])
#     print()
#     trainBayesNetwork.created_tan_feature_class(dependency_graph)
#     trainBayesNetwork.compute_tan_cpt()
#     trainBayesNetwork.tan_predict(testBayesNetwork.dataSet)
#     tan_recall,tan_precision = trainBayesNetwork.precision_recall_graph(testBayesNetwork.dataSet, "tan")
#
#     trainBayesNetwork.plot_pr_curve(nb_recall, nb_precision, tan_recall, tan_precision)
