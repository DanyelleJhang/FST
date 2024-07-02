# tolerence = 1,2,3,4,5,6,7,8,9,10 .....
n_trials = 10
tolerence = 10 # 1 速度, 3 平衡, 7 準確
if tolerence > 1:
    n_trials = int(n_trials/tolerence**(2/5))
    if n_trials == 0:
        n_trials = 1
    study = optuna.create_study(
        direction="maximize", pruner=SuccessiveHalvingPruner(reduction_factor=tolerence)
    )
else:
    study = optuna.create_study(
        direction="maximize"
    )    
study.optimize(objective, n_trials=n_trials,n_jobs=-1)