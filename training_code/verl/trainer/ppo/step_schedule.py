"""Pure scheduling after a one-based PPO update has completed."""


def completed_step_actions(step, total_steps, test_freq, save_freq, has_validation=True):
    """Return (finished, validate, save) for the update just completed.

    Final validation runs whenever a validation function exists, even if
    periodic tests are disabled. Nonpositive save_freq disables all saves.
    A final step also divisible by a frequency still triggers only one action.
    """
    if isinstance(step, bool) or isinstance(total_steps, bool) or not isinstance(step, int) or not isinstance(total_steps, int):
        raise ValueError('step and total_steps must be integers')
    if total_steps < 1 or not 1 <= step <= total_steps:
        raise ValueError('Completed step must lie in 1..total_steps')
    finished = step == total_steps
    validate = bool(has_validation and (finished or (test_freq > 0 and step % test_freq == 0)))
    save = bool(save_freq > 0 and (finished or step % save_freq == 0))
    return finished, validate, save
