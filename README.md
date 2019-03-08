# Naive-Bayes-And-Tree-Augmented-Network


Naive Bayes 
'''
./bayes ../Resources/lymphography_train.json ../Resources/lymphography_test.json n
./bayes ../Resources/tic-tac-toe_train.json ../Resources/tic-tac-toe_test.json n
./bayes ../Resources/tic-tac-toe_sub_train.json ../Resources/tic-tac-toe_sub_test.json n
'''

Tree Augmented Network
'''
./bayes ../Resources/lymphography_train.json ../Resources/lymphography_test.json t
./bayes ../Resources/tic-tac-toe_train.json ../Resources/tic-tac-toe_test.json t
./bayes ../Resources/tic-tac-toe_sub_train.json ../Resources/tic-tac-toe_sub_test.json t
'''

Percision-Recall Curve Plot
'''
python pr_plot.py ../Resources/tic-tac-toe_train.json ../Resources/tic-tac-toe_test.json  
'''

Two paired t-test to compare two models (Naive-Bayes and TAN)
'''
python t_test.py ../Resources/tic-tac-toe.json 
'''
