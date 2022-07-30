import json

class BestValueLogger:
    """
    Logs the best value and step for a dictionary of metrics. 
    """
    def __init__(self, metrics: dict, log_step: bool = True):
        """
        names: dict with keys: metric names, values: bool whether the metric is increasing (larger is better)
        log_step: bool, if True also logs the best step for each metric that is tracked as "best_<metric>_step" 
        Example:
        names = {'val_loss': False, 'val_auroc: True}
        """
        self.metrics = metrics
        self.log_step = log_step
        self.suffices = ['value', 'step'] if log_step else ['value']

        #self.increasing = dict(zip(names, increasing)) 
        self.best_values = {'_'.join(['best', name, suffix]): 
                None for name in metrics.keys() for suffix in self.suffices}

    def __call__(self, values: list, step: int): 
        """
        We assume that values refer to the names at init 
        """
        assert len(values) == len(self.metrics)
        data = dict(zip(self.metrics.keys(), values))

        for name, value in data.items():
            self._update_metric(name, value, step)

    def _update_metric(self, name, value, step):
        """
        Check if current value is better than best before, if yes, update.
        """
        larger_better = int(self.metrics[name]) # convert bool to int
        larger_better = 2*larger_better - 1 # convert [1,0] to [1,-1] 
        old_value = self.best_values[
            '_'.join(['best', name, 'value'])
        ]
        # init if empty at start:
        if old_value is None:
            old_value = 1e10 * (-larger_better)
        if larger_better * (value - old_value) > 0:
            for key_name, key in zip(['value','step'], [value, step]):
                if key_name in self.suffices:
                    key_name = '_'.join(['best', name, key_name]) 
                    self.best_values[key_name] = key
        return self

    def __repr__(self):
        m_dict = json.dumps(self.metrics, indent=4) 
        v_dict = json.dumps(self.best_values, indent=4)
        step_str = ' and steps' if self.log_step else '' 
        rep = [
            '::BestValueLogger:: ',  
            'Tracking the best values of the following metrics:', 
            f'{m_dict}', 
            '(key: metric, value: bool if larger is better)',
            f'Best values{step_str}:', 
            f'{v_dict}'
        ]
        return '\n'.join(rep)
