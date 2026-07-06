from optuna import Study
from optuna.trial import FrozenTrial
from optuna.trial import TrialState
from optuna.samplers._brute_force import _TreeNode
from optuna.trial import create_trial
from typing import Optional, Sequence
import optuna.samplers

class BruteForceSamplerWithoutStop(optuna.samplers.BruteForceSampler):

    def after_trial(
        self,
        study: Study,
        trial: FrozenTrial,
        state: TrialState,
        values: Optional[Sequence[float]],
    ) -> None:
        trials = study.get_trials(
            deepcopy=False,
            states=(
                TrialState.COMPLETE,
                TrialState.PRUNED,
                TrialState.RUNNING,
                TrialState.FAIL,
            ),
        )
        tree = _TreeNode()
        self._populate_tree(
            tree,
            (
                (
                    t
                    if t.number != trial.number
                    else create_trial(
                        state=state,  # Set current trial as complete.
                        values=values,
                        params=trial.params,
                        distributions=trial.distributions,
                    )
                )
                for t in trials
            ),
            {},
        )