import numpy as np
from sklearn.model_selection import train_test_split
from itertools import product
from perceptron_model import Model


def grid_search(default_params, param_grid, features, labels, validation_split, log=True, top_n=5):
    """
    Grid search over param_grid
    :param default_params: dictionary of default parameter (useless to the model selection)
    :param param_grid: grid of possible parameters that need to be tested
    :param features: matrix of features
    :param labels: matrix of labels
    :param validation_split: ratio validation/total data
    ####:param winner_criteria: 'meanTrainingLoss', 'meanValidationLoss', 'meanLosses' (the last two ... modify)
    :param log: flag, decide whether the log is shown or not
    :param top_n: number of winners
    ####:param log_to_be_returned: log list updated (modify)
    :return: best model parameters (and if required also log)
    """
    list_params = list(param_grid.values())
    winners = []
    all_combinations = list(product(*list_params))
    train_data, validation_data, train_labels, validation_labels = train_test_split(features, labels, test_size=validation_split)
    for index, val in enumerate(all_combinations):
        print("                        {}/{}".format(index, len(all_combinations)), end="\r")
        param = {}
        for index_ in range(len(param_grid)):
            param[list(param_grid.keys())[index_]] = val[index_]
        model = Model(**default_params)
        model.set_params(**param)  #To set_param it's important to inherit from something (object I believe is enough)
        model.fit(train_data, train_labels, validation_data, validation_labels, early_stopping_log=False,
                  coming_from_grid_search=True)
        mean_loss = np.mean(model.losses)
        mean_validation_loss = np.mean(model.validation_losses)
        if model.new_epoch_notification:
            param['epochs'] = model.best_epoch
        temp_log = {
            'params': param,
            'mean_training_loss': mean_loss,
            'mean_validation_loss': mean_validation_loss
        }
        log_to_be_returned = []
        if log:
            log_to_te_returned.append(temp_log)

        if len(winners) < top_n:
            winners.append(temp_log)
        else:
            winners = sorted(winners, key=lambda k: k['mean_validation_loss'])
            if temp_log['mean_validation_loss'] < winners[-1]['mean_validation_loss']:
                winners[-1] = temp_log

    if log:
        return log_to_be_returned, winners
    else:
        return winners