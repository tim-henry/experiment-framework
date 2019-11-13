import experiments.keep_pct_readout
import experiments.train
import experiments.run
import experiments.keep_pct_readout_eval
import experiments.keep_pct_readout_3_task
# import experiments.keep_pct_readout_parallel


def get_9_class_keep_pct_readout_args(weighted=False, alpha=0.5):
    # Training settings
    return {
        "alpha": alpha,
        "weighted": weighted,
        "num_classes": (9, 9),
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


def get_3_task_keep_pct_readout_args(weighted=False):
    # Training settings
    return {
        "weighted": weighted,
        "num_classes": (9, 9, 9),
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


def get_default_keep_pct_readout_args(weighted=False, alpha=0.5):
    # Training settings
    return {
        "alpha": alpha,
        "weighted": weighted,
        "batch_size": 64,
        "num_classes": (10, 10),
        "test_batch_size": 1000,
        "epochs": 5,
        "lr": 0.01,
        "momentum": 0.5,
        "no_cuda": False,
        "log_interval": 100,
        "save_model": False,
        "keep_pcts": [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    }


def get_train_args(alpha=0.5):
    # Training settings
    return {
        "alpha": alpha,
        "batch_size": 64,
        "test_batch_size": 1000,
        "epochs": 5,
        "lr": 0.001,
        "weight_decay": 5e-4,
        "momentum": 0.9,
        "no_cuda": False,
        "log_interval": 100,
        "save_model": False,
        "keep_pcts": [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    }


def get_9_class_keep_pct_readout_save_args(weighted=False):
    args = get_9_class_keep_pct_readout_args(weighted=weighted)
    args["save_model"] = True
    return args


def get_default_keep_pct_readout_save_args(weighted=False):
    args = get_default_keep_pct_readout_args(weighted=weighted)
    args["save_model"] = True
    return args


options = {
    "keep_pct_readout_default":
        lambda train_loader_fn, test_loader_fn, model:
        experiments.keep_pct_readout.run(train_loader_fn, test_loader_fn, model, get_default_keep_pct_readout_args()),
    "keep_pct_readout_9_class":
        lambda train_loader_fn, test_loader_fn, model:
        experiments.keep_pct_readout.run(train_loader_fn, test_loader_fn, model, get_9_class_keep_pct_readout_args()),
    "keep_pct_readout_3_task":
        lambda train_loader_fn, test_loader_fn, model:
        experiments.keep_pct_readout_3_task.run(train_loader_fn, test_loader_fn, model,
                                         get_3_task_keep_pct_readout_args()),
    "keep_pct_readout_default_save":
        lambda train_loader_fn, test_loader_fn, model:
        experiments.keep_pct_readout.run(train_loader_fn, test_loader_fn, model,
                                         get_default_keep_pct_readout_save_args()),
    "keep_pct_readout_9_class_save":
        lambda train_loader_fn, test_loader_fn, model:
        experiments.keep_pct_readout.run(train_loader_fn, test_loader_fn, model,
                                         get_9_class_keep_pct_readout_save_args()),
    "train":
        lambda train_loader_fn, test_loader_fn, model:
        experiments.train.run(train_loader_fn, test_loader_fn, model, get_train_args()),
    "run":
        lambda train_loader_fn, test_loader_fn, model:
        experiments.run.run(None, None, model, None),
    "keep_pct_readout_eval":
        lambda train_loader_fn, test_loader_fn, model:
        experiments.keep_pct_readout_eval.run(train_loader_fn, test_loader_fn, model, get_default_keep_pct_readout_args()),
    "keep_pct_readout_eval_loc":
        lambda train_loader_fn, test_loader_fn, model:
        experiments.keep_pct_readout_eval.run(train_loader_fn, test_loader_fn, model,
                                              get_9_class_keep_pct_readout_args()),
    "keep_pct_readout_default_weighted":
        lambda train_loader_fn, test_loader_fn, model:
        experiments.keep_pct_readout.run(train_loader_fn, test_loader_fn, model,
                                         get_default_keep_pct_readout_args(weighted=True)),
    "keep_pct_readout_9_class_weighted":
        lambda train_loader_fn, test_loader_fn, model:
        experiments.keep_pct_readout.run(train_loader_fn, test_loader_fn, model, get_9_class_keep_pct_readout_args(weighted=True)),
    "keep_pct_readout_default_weighted_save":
        lambda train_loader_fn, test_loader_fn, model:
        experiments.keep_pct_readout.run(train_loader_fn, test_loader_fn, model,
                                         get_default_keep_pct_readout_save_args(weighted=True)),
    "keep_pct_readout_9_class_weighted_save":
        lambda train_loader_fn, test_loader_fn, model:
        experiments.keep_pct_readout.run(train_loader_fn, test_loader_fn, model,
                                         get_9_class_keep_pct_readout_save_args(weighted=True)),
}

for i in range(101):
    alpha = i / 100.
    options["keep_pct_readout_default_" + str(i)] = lambda train_loader_fn, test_loader_fn, model, alpha=alpha: \
        experiments.keep_pct_readout.run(train_loader_fn,
                                         test_loader_fn,
                                         model,
                                         get_default_keep_pct_readout_args(alpha=alpha))

    options["keep_pct_readout_9_class_" + str(i)] = lambda train_loader_fn, test_loader_fn, model, alpha=alpha: \
        experiments.keep_pct_readout.run(train_loader_fn,
                                         test_loader_fn,
                                         model,
                                         get_9_class_keep_pct_readout_args(alpha=alpha))
