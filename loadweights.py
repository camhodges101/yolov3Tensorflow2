ptr = 0
i = 0
assign_ops = []
while i < len(var_list) - 1:
    var1 = var_list[i]
    var2 = var_list[i + 1]
    # do something only if we process conv layer
    if 'Conv' in var1.name.split('/')[-2]:
        # check type of next layer
        if 'BatchNorm' in var2.name.split('/')[-2]:
            # load batch norm params
            gamma, beta, mean, var = var_list[i + 1:i + 5]
            batch_norm_vars = [beta, gamma, mean, var]
            for var in batch_norm_vars:
                shape = var.shape.as_list()
                num_params = np.prod(shape)
                var_weights = weights[ptr:ptr + num_params].reshape(shape)
                ptr += num_params
                assign_ops.append(tf.assign(var, var_weights, validate_shape=True))
            # we move the pointer by 4, because we loaded 4 variables
            i += 4
        elif 'Conv' in var2.name.split('/')[-2]:
            # load biases
            bias = var2
            bias_shape = bias.shape.as_list()
            bias_params = np.prod(bias_shape)
            bias_weights = weights[ptr:ptr + bias_params].reshape(bias_shape)
            ptr += bias_params
            assign_ops.append(tf.assign(bias, bias_weights, validate_shape=True))
            # we loaded 2 variables
            i += 1
        # we can load weights of conv layer
        shape = var1.shape.as_list()
        num_params = np.prod(shape)
        var_weights = weights[ptr:ptr + num_params].reshape((shape[3], shape[2], shape[0], shape[1]))
        # remember to transpose to column-major
        var_weights = np.transpose(var_weights, (2, 3, 1, 0))
        ptr += num_params
        assign_ops.append(tf.assign(var1, var_weights, validate_shape=True))
        i += 1
return assign_ops