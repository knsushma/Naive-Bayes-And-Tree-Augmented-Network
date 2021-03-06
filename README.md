## Naive-Bayes-And-Tree-Augmented-Network

This assignment includes:

 Implementation of Naive Bayes and TAN.\
 Evaluating methods through Precision/Recall curves, and understand their differences with ROC curves.\
 Comparing models using a paired t test to see whether one model is significantly better than the other.


### Environment Setup:
```
Follow steps mentioned in python-setup-on-remote/python-setup.pdf
```

Use below commands to run the required step. 
choice n: Naive Bayes execution
choice t: TAN execution

Naive Bayes 
```
./bayes ../Resources/lymphography_train.json ../Resources/lymphography_test.json n
./bayes ../Resources/tic-tac-toe_train.json ../Resources/tic-tac-toe_test.json n
./bayes ../Resources/tic-tac-toe_sub_train.json ../Resources/tic-tac-toe_sub_test.json n
```

Tree Augmented Network
```
./bayes ../Resources/lymphography_train.json ../Resources/lymphography_test.json t
./bayes ../Resources/tic-tac-toe_train.json ../Resources/tic-tac-toe_test.json t
./bayes ../Resources/tic-tac-toe_sub_train.json ../Resources/tic-tac-toe_sub_test.json t
```

Percision-Recall Curve Plot
```
python pr_plot.py ../Resources/tic-tac-toe_train.json ../Resources/tic-tac-toe_test.json  
```

Two paired t-test to compare two models (Naive-Bayes and TAN)
```
python t_test.py ../Resources/tic-tac-toe.json 
```

For more details about implementation, look at hw2.pdf
