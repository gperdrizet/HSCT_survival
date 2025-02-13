'''Plotting functions for experimentation notebooks.'''

import matplotlib.pyplot as plt
from scipy import stats

def prediction_eval_plot(main_title: str, predictions: list, labels: list) -> plt:
    '''Takes a string title, predictions and labels lists. Plots predictions
    vs label scatter, residuals as a function of prediction and normal QQ plot
    of residuals.'''

    # Plot the results
    fig, axs=plt.subplots(1,3, figsize=(10,4))
    axs=axs.flatten()

    fig.suptitle(main_title)

    axs[0].set_title('Actual vs predicted value')
    axs[0].scatter(labels, predictions, color='black', s=0.2)
    axs[0].set_xlabel('Actual')
    axs[0].set_ylabel('Predicted')

    axs[1].set_title('Fitted value vs fit residual')
    axs[1].scatter(predictions, labels - predictions, color='black', s=0.2)
    axs[1].set_xlabel('Fitted value')
    axs[1].set_ylabel('Fit residual')

    axs[2].set_title('Normal quantiles vs fit residual quantiles')
    stats.probplot(labels - predictions, plot=axs[2])
    axs[2].get_lines()[0].set_markeredgecolor('black')
    axs[2].get_lines()[0].set_markerfacecolor('black')
    axs[2].set_xlabel('Normal quantiles')
    axs[2].set_ylabel('Residual quantiles')

    plt.tight_layout()
    
    return plt