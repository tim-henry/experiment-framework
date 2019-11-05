def get_alpha(model_name, input_name):
    if model_name == "simple_cnn":
        if input_name == "left_out_colored_mnist":
            return 0.5
        elif input_name == "left_out_varied_location_mnist":
            return 0.7
    if model_name == "resnet":
        if input_name == "left_out_colored_mnist":
            return 0.8
        elif input_name == "left_out_varied_location_mnist":
            return 0.9
    elif model_name == "resnet_pretrained":
        if input_name == "left_out_colored_mnist":
            return 0.7
        elif input_name == "left_out_varied_location_mnist":
            return 0.7
    elif model_name == "resnet_pretrained_embeddings":
        if input_name == "left_out_colored_mnist":
            return 0.3
        elif input_name == "left_out_varied_location_mnist":
            return 0.2
    elif model_name == "resnet_no_pool":
        if input_name == "left_out_colored_mnist":
            return 0.9
        elif input_name == "left_out_varied_location_mnist":
            return 0.8

    raise RuntimeError("No loss weight found for model: " + model_name, ", input: " + input_name)
