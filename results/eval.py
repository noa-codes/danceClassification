import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_performance(true_y, pred_y, print_vals=True):
    """
    Helper function that evaluates a model's performance by outputting accuracy rate and a confusion matrix.
    """
    cnf_matrix = metrics.confusion_matrix(true_y, pred_y)
    
    class_names=['ballet', 'break', 'flamenco', 'foxtrot', 'latin', 'quickstep', 'square', 'swing', 'tango', 'waltz']
    cnf_matrix = pd.DataFrame(cnf_matrix, index = class_names,
                  columns = class_names)
    
    # plot confusion matrix with heatmap
    plt.figure(figsize=(6,4), dpi=200)
    sns.heatmap(cnf_matrix, annot=True, cmap="YlGnBu" ,fmt='g')
    plt.tight_layout()
    plt.title('Confusion matrix', y=1.1)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.show()
    
    if print_vals :
        n_accurate = (true_y == pred_y).sum()
        total = len(true_y)
        acc_rate = n_accurate / total
        print('Accuracy: {}/{} = {:.4f}'.format(n_accurate, total, acc_rate))