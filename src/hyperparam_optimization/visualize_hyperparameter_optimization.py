import optuna
from optuna.visualization import plot_intermediate_values
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_param_importances
from optuna.trial import TrialState
import os
os.chdir('../results/final/optuna/')

study_name = 'optuna_study_2nd_order_outer_LAIR'
study = optuna.load_study(study_name=study_name, storage='sqlite:///%s.db' % study_name)

pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

print('Study statistics: ')
print('  Number of finished trials: ', len(study.trials))
print('  Number of pruned trials: ', len(pruned_trials))
print('  Number of complete trials: ', len(complete_trials))

best_trial = study.best_trial
print('Best trial:', best_trial.number)


print('  Value: ', best_trial.value)

print('  Params: ')
for key, value in best_trial.params.items():
    print('    {}: {}'.format(key, value))

plot_optimization_history(study).show(renderer='browser')
plot_intermediate_values(study).show(renderer='browser')
plot_param_importances(study).show(renderer='browser')
