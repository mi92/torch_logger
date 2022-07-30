# torch_logger 🔥

![PyPI - Python Version](https://img.shields.io/pypi/pyversions/torch_logger)

This minimalist package serves to log best values of performance metrics during the training of PyTorch models.
The idea is to automatically log the best value for each tracked metric such that it can be directly analyzed downstream (e.g. when using wandb) without the need to post-process the raw logged values to identify the overall best values and corresponding steps.

## Usage:  

```
>>> from torch_logger import BestValueLogger
>>> bv_log = BestValueLogger(
        {'val_loss': False, 'val_roc': True} # <-- provide flag if larger is better
    )
```

Log values after each eval step:
```
    ... 
>>> bv_log([val_loss, val_roc], step=0)
    ... 
>>> bv_log([val_loss, val_roc], step=1)
    ...  
>>> bv_log([val_loss, val_roc], step=2)
```

Inspect the logger:
```
>>> bv_log

::BestValueLogger::
Tracking the best values of the following metrics:
{
    "val_loss": false,
    "val_roc": true
}
(key: metric, value: bool if larger is better)
Best values and steps:
{
    "best_val_loss_value": 0.05,
    "best_val_loss_step": 2,
    "best_val_roc_value": 0.8,
    "best_val_roc_step": 1
}
```

Update your experiment logger (e.g. wandb) with best_values at the end of training
```
>>> wandb.log( bv_log.best_values ) 
```

### Logging values without steps

In case you only wish to track values but not the corresponding steps, run: 
```
>>> bvl = BestValueLogger({'val_loss': False, 'val_roc':True}, log_step=False)
```    
Populate logger with metrics: 
```
>>> bvl([0.2,0.8], step=1)
>>> bvl([0.2,0.9], step=2)
```
Inspect:
```
>>> bvl
::BestValueLogger::
Tracking the best values of the following metrics:
{
    "val_loss": false,
    "val_roc": true
}
(key: metric, value: bool if larger is better)
Best values:
{
    "best_val_loss_value": 0.2,
    "best_val_roc_value": 0.9
}
```
