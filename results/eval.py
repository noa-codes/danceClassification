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
    plt.figure(figsize=(6,4), dpi=100)
    sns.heatmap(cnf_matrix, annot=True, cmap="YlGnBu" ,fmt='g')
    plt.tight_layout()
    #plt.title('Confusion matrix', y=1.1)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    
    if print_vals :
        n_accurate = (true_y == pred_y).sum()
        total = len(true_y)
        acc_rate = n_accurate / total
        print('Accuracy: {}/{} = {:.4f}'.format(n_accurate, total, acc_rate))


def all_confusion_matrices(lstm_results, lstm_att_results, tcn_results):
    """
    Helper function that prints the confusion matrices of all three models in a row.
    """
    class_names=['ballet', 'break', 'flamenco', 'foxtrot', 'latin', 'quickstep', 'square', 'swing', 'tango', 'waltz']
    
    ### construct the confusion matrices
    # lstm
    lstm_matrix = metrics.confusion_matrix(lstm_results['true_y'], lstm_results['pred_y'])
    lstm_matrix = pd.DataFrame(lstm_matrix, index = class_names,
                  columns = class_names)
    # lstm with attention
    lstm_att_matrix = metrics.confusion_matrix(lstm_att_results['true_y'], lstm_att_results['pred_y'])
    lstm_att_matrix = pd.DataFrame(lstm_att_matrix, index = class_names,
                  columns = class_names)
    # tcn
    tcn_matrix = metrics.confusion_matrix(tcn_results['true_y'], tcn_results['pred_y'])
    tcn_matrix = pd.DataFrame(tcn_matrix, index = class_names,
                  columns = class_names)
    
    ### create joint plot
    fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(15,4), dpi=500) #figsize=(10,8)
    # lstm
    sns.heatmap(lstm_matrix, annot=True, cmap="YlGnBu" ,fmt='g', cbar=False, ax=ax1)
    ax1.set_title('LSTM')
    ax1.set(ylabel='Actual Class', xlabel='Predicted Class')
    # lstm with attention
    sns.heatmap(lstm_att_matrix, annot=True, cmap="YlGnBu" ,fmt='g', cbar=False, ax=ax2)
    ax2.set_title('LSTM with Self-Attention')
    ax2.set(xlabel='Predicted Class')
    # tcn
    sns.heatmap(tcn_matrix, annot=True, cmap="YlGnBu" ,fmt='g', cbar=False, ax=ax3)
    ax3.set_title('Temporal Convolutional Network (TCN)')
    ax3.set(xlabel='Predicted Class')
    # formatting
    plt.tight_layout()
    plt.show()
    