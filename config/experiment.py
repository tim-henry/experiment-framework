import experiments.keep_pct_readout


def get_9_class_keep_pct_readout_args():
    # Training settings
    return {
        "batch_size": 64,
        "test_batch_size": 1000,
        "epochs": 5,
        "lr": 0.003,
        "momentum": 0.5,
        "no_cuda": False,
        "log_interval": 100,
        "save_model": False,
        "keep_pcts": [i / 9 for i in range(1, 10)]
    }


def get_default_keep_pct_readout_args():
    # Training settings
    return {
        "batch_size": 64,
        "test_batch_size": 1000,
        "epochs": 5,
        "lr": 0.003,
        "momentum": 0.5,
        "no_cuda": False,
        "log_interval": 100,
        "save_model": False,
        "keep_pcts": [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    }


options = {
    "keep_pct_readout_default":
        lambda train_loader_fn, test_loader_fn, model:
        experiments.keep_pct_readout.run(train_loader_fn, test_loader_fn, model, get_default_keep_pct_readout_args()),
    "keep_pct_readout_9_class":
        lambda train_loader_fn, test_loader_fn, model:
        experiments.keep_pct_readout.run(train_loader_fn, test_loader_fn, model, get_9_class_keep_pct_readout_args())
}
