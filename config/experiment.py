import experiments.keep_pct_readout
import experiments.train
import experiments.run
import experiments.keep_pct_readout_eval
import experiments.keep_pct_readout_3_task
import experiments.keep_pct_readout_v2


def get_3_class_keep_pct_readout_args(weighted=False, alpha=0.5):
    # Training settings
    return {
        "alpha": alpha,
        "weighted": weighted,
        "num_classes": (9, 3),
        "batch_size": 32,
        "test_batch_size": 512,
        "epochs": 5,
        "lr": 0.003,
        "momentum": 0.5,
        "no_cuda": False,
        "log_interval": 100,
        "save_model": True,
        "keep_pcts": list(range(1, 4)),
        'example_pct': 1,
    }


def get_9_class_keep_pct_readout_args(weighted=False, alpha=None):
    # Training settings
    return {
        "alpha": alpha,
        "weighted": weighted,
        "num_classes": (9, 9),
        "batch_size": 32,
        "test_batch_size": 512,
        "epochs": 5,
        "lr": 0.003,
        "momentum": 0.5,
        "no_cuda": False,
        "log_interval": 100,
        "save_model": False,
        "keep_pcts": list(range(1, 10)),
        'example_pct': 1,
        "save_model": True
    }


def get_3_task_keep_pct_readout_args(weighted=False, alpha=None):
    # Training settings
    return {
        "alpha": alpha,
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
        "keep_pcts": list(range(1, 10)),
        'example_pct': 1,
        "save_model": True
    }


def get_default_keep_pct_readout_args(weighted=False, alpha=None):
    # Training settings
    return {
        "alpha": alpha,
        "weighted": weighted,
        "batch_size": 32,
        "num_classes": (10, 10),
        "test_batch_size": 512,
        "epochs": 5,
        "lr": 0.01,
        "momentum": 0.5,
        "no_cuda": False,
        "log_interval": 100,
        "save_model": False,
        "keep_pcts": list(range(1, 11)),
        'example_pct': 1,
        "save_model": True
    }


def get_train_args(alpha=None):
    # Training settings
    return {
        "alpha": alpha,
        "batch_size": 32,
        "test_batch_size": 512,
        "epochs": 5,
        "lr": 0.001,
        "weight_decay": 5e-4,
        "momentum": 0.9,
        "no_cuda": False,
        "log_interval": 100,
        "save_model": False,
        "keep_pcts": list(float(x) for x in range(1, 11)),
        'example_pct': 1,
        "save_model": True
    }


def get_3_class_keep_pct_readout_save_args(weighted=False, alpha=0.5):
    args = get_3_class_keep_pct_readout_args(weighted=weighted, alpha=alpha)
    args["save_model"] = True
    return args


def get_9_class_keep_pct_readout_save_args(weighted=False, alpha=None):
    args = get_9_class_keep_pct_readout_args(weighted=weighted, alpha=alpha)
    args["save_model"] = True
    return args


def get_default_keep_pct_readout_save_args(weighted=False, alpha=None):
    args = get_default_keep_pct_readout_args(weighted=weighted, alpha=alpha)
    args["save_model"] = True
    return args


def get_3_class_half_readout_save_args(weighted=False, alpha=0.5):
    args = get_3_class_keep_pct_readout_args(weighted=weighted, alpha=alpha)
    args["save_model"] = True
    args['example_pct'] = 0.5
    return args


def get_9_class_half_readout_save_args(weighted=False, alpha=None):
    args = get_9_class_keep_pct_readout_args(weighted=weighted, alpha=alpha)
    args["save_model"] = True
    args['example_pct'] = 0.5
    return args


def get_default_half_readout_save_args(weighted=False, alpha=None):
    args = get_default_keep_pct_readout_args(weighted=weighted, alpha=alpha)
    args["save_model"] = True
    args['example_pct'] = 0.5
    return args


def get_3_class_tenth_readout_save_args(weighted=False, alpha=0.5):
    args = get_3_class_keep_pct_readout_args(weighted=weighted, alpha=alpha)
    args["save_model"] = True
    args['example_pct'] = 0.1
    return args


def get_9_class_tenth_readout_save_args(weighted=False, alpha=None):
    args = get_9_class_keep_pct_readout_args(weighted=weighted, alpha=alpha)
    args["save_model"] = True
    args['example_pct'] = 0.1
    return args


def get_default_tenth_readout_save_args(weighted=False, alpha=None):
    args = get_default_keep_pct_readout_args(weighted=weighted, alpha=alpha)
    args["save_model"] = True
    args['example_pct'] = 0.1
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
        experiments.keep_pct_readout_v2.run(train_loader_fn, test_loader_fn, model,
                                         get_9_class_keep_pct_readout_save_args()),
    "train":
        lambda train_loader_fn, test_loader_fn, model:
        experiments.train.run(train_loader_fn, test_loader_fn, model, get_train_args()),
    "run":
        lambda train_loader_fn, test_loader_fn, model:
        experiments.run.run(None, None, model, None),
    "keep_pct_readout_eval":
        lambda train_loader_fn, test_loader_fn, model:
        experiments.keep_pct_readout_eval.run(train_loader_fn, test_loader_fn, model,
                                              get_default_keep_pct_readout_args()),
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
        experiments.keep_pct_readout_v2.run(train_loader_fn, test_loader_fn, model,
                                         get_9_class_keep_pct_readout_args(weighted=True)),
    "keep_pct_readout_default_weighted_save":
        lambda train_loader_fn, test_loader_fn, model:
        experiments.keep_pct_readout.run(train_loader_fn, test_loader_fn, model,
                                         get_default_keep_pct_readout_save_args(weighted=True)),
    "keep_pct_readout_9_class_weighted_save":
        lambda train_loader_fn, test_loader_fn, model:
        experiments.keep_pct_readout.run(train_loader_fn, test_loader_fn, model,
                                         get_9_class_keep_pct_readout_save_args(weighted=True)),
    "keep_pct_readout_default_v2":
        lambda train_loader_fn, test_loader_fn, model:
        experiments.keep_pct_readout_v2.run(train_loader_fn, test_loader_fn, model, get_default_keep_pct_readout_args()),
    "keep_pct_readout_9_class_v2":
        lambda train_loader_fn, test_loader_fn, model:
        experiments.keep_pct_readout_v2.run(train_loader_fn, test_loader_fn, model, get_9_class_keep_pct_readout_args()),
    "keep_pct_readout_default_v2_weighted_save":
        lambda train_loader_fn, test_loader_fn, model:
        experiments.keep_pct_readout_v2.run(train_loader_fn, test_loader_fn, model,
                                         get_default_keep_pct_readout_save_args(weighted=True)),
    "keep_pct_readout_9_class_v2_weighted_save":
        lambda train_loader_fn, test_loader_fn, model:
        experiments.keep_pct_readout_v2.run(train_loader_fn, test_loader_fn, model,
                                            get_9_class_keep_pct_readout_save_args(weighted=True)),
    "keep_pct_readout_3_class_v2_weighted_save":
        lambda train_loader_fn, test_loader_fn, model:
        experiments.keep_pct_readout_v2.run(train_loader_fn, test_loader_fn, model,
                                            get_3_class_keep_pct_readout_save_args(weighted=False)), #TODO get weights, set true
    "keep_pct_readout_default_v2_weighted_save_tenth":
        lambda train_loader_fn, test_loader_fn, model:
        experiments.keep_pct_readout_v2.run(train_loader_fn, test_loader_fn, model,
                                            get_default_tenth_readout_save_args(weighted=True)),
    "keep_pct_readout_9_class_v2_weighted_save_tenth":
        lambda train_loader_fn, test_loader_fn, model:
        experiments.keep_pct_readout_v2.run(train_loader_fn, test_loader_fn, model,
                                            get_9_class_tenth_readout_save_args(weighted=True)),
    "keep_pct_readout_3_class_v2_weighted_save_tenth":
        lambda train_loader_fn, test_loader_fn, model:
        experiments.keep_pct_readout_v2.run(train_loader_fn, test_loader_fn, model,
                                            get_3_class_tenth_readout_save_args(weighted=False)),#TODO get weights, set true
    "keep_pct_readout_default_v2_weighted_save_half":
        lambda train_loader_fn, test_loader_fn, model:
        experiments.keep_pct_readout_v2.run(train_loader_fn, test_loader_fn, model,
                                            get_default_half_readout_save_args(weighted=True)),
    "keep_pct_readout_9_class_v2_weighted_save_half":
        lambda train_loader_fn, test_loader_fn, model:
        experiments.keep_pct_readout_v2.run(train_loader_fn, test_loader_fn, model,
                                            get_9_class_half_readout_save_args(weighted=True)),
    "keep_pct_readout_3_class_v2_weighted_save_half":
        lambda train_loader_fn, test_loader_fn, model:
        experiments.keep_pct_readout_v2.run(train_loader_fn, test_loader_fn, model,
                                            get_3_class_half_readout_save_args(weighted=False)), #todo get weights set true
    "keep_pct_readout_shape_only":
        lambda train_loader_fn, test_loader_fn, model:
        experiments.keep_pct_readout_v2.run(train_loader_fn, test_loader_fn, model,
                                            get_default_keep_pct_readout_args(weighted=True, alpha=1)),
    "keep_pct_readout_color_only":
        lambda train_loader_fn, test_loader_fn, model:
        experiments.keep_pct_readout_v2.run(train_loader_fn, test_loader_fn, model,
                                            get_default_keep_pct_readout_args(weighted=True, alpha=0)),
    "keep_pct_readout_location_only":
        lambda train_loader_fn, test_loader_fn, model:
        experiments.keep_pct_readout_v2.run(train_loader_fn, test_loader_fn, model,
                                            get_9_class_keep_pct_readout_args(weighted=True, alpha=0)),
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

    options["keep_pct_readout_default_v2_" + str(i)] = lambda train_loader_fn, test_loader_fn, model, alpha=alpha: \
        experiments.keep_pct_readout_v2.run(train_loader_fn,
                                            test_loader_fn,
                                            model,
                                            get_default_keep_pct_readout_args(alpha=alpha))

    options["keep_pct_readout_default_v2_half_" + str(i)] = lambda train_loader_fn, test_loader_fn, model, alpha=alpha: \
        experiments.keep_pct_readout_v2.run(train_loader_fn,
                                            test_loader_fn,
                                            model,
                                            get_default_half_readout_save_args(weighted=True, alpha=alpha))

    options["keep_pct_readout_9_class_v2_" + str(i)] = lambda train_loader_fn, test_loader_fn, model, alpha=alpha: \
        experiments.keep_pct_readout_v2.run(train_loader_fn,
                                            test_loader_fn,
                                            model,
                                            get_9_class_keep_pct_readout_args(alpha=alpha))

    options["keep_pct_readout_3_class_v2_" + str(i)] = lambda train_loader_fn, test_loader_fn, model, alpha=alpha: \
        experiments.keep_pct_readout_v2.run(train_loader_fn,
                                            test_loader_fn,
                                            model,
                                            get_3_class_keep_pct_readout_args(alpha=alpha))
