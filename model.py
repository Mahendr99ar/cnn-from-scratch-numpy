"""
Build a Trainable CNN from Scratch in NumPy

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - argmax_rows
def argmax_rows(matrix):
    # TODO: return the index of the largest element in each row of a 2D array
    return np.argmax(matrix, axis=1)

# Step 2 - row_max
import numpy as np

def row_max(matrix):
    # TODO: return the maximum value of each row of `matrix` with keepdims True for broadcasting.
    return np.max(matrix, axis=1, keepdims=True)

# Step 3 - row_sum
import numpy as np

def row_sum(matrix):
    """Return per-row sums of a 2D array with shape (N, 1)."""
    # TODO: return the sum along axis 1 keeping the reduced dimension
    return np.sum(matrix, axis=1, keepdims=True)

# Step 4 - exp_shifted
import numpy as np

def exp_shifted(logits):
    """Subtract per-row max from logits and exponentiate elementwise."""
    # TODO: shift each row of logits by its max and return elementwise exp
    return np.exp(logits - row_max(logits))

# Step 5 - stable_softmax
def stable_softmax(logits):
    # TODO: Compute a numerically stable softmax row-wise over (N, C) logits.
    exp_vals = exp_shifted(logits)
    return exp_vals / row_sum(exp_vals)

# Step 6 - one_hot
def one_hot(labels, num_classes):
    # TODO: convert integer labels into a (N, num_classes) one-hot float matrix
    return np.eye(num_classes, dtype=float)[labels]

# Step 7 - gather_true_class_probs
def gather_true_class_probs(probs, labels):
    # TODO: return probs[i, labels[i]] for every row i as a 1D length-N array.
    return probs[np.arange(len(labels)), labels]

# Step 8 - cross_entropy_loss
import numpy as np

def cross_entropy_loss(probs, labels, eps=1e-12):
    # TODO: return the mean negative log-likelihood of the true-class probabilities
    true_probs = gather_true_class_probs(probs, labels)
    true_probs = np.clip(true_probs, eps, 1.0)
    return -np.mean(np.log(true_probs))

# Step 9 - accuracy
def accuracy(logits_or_probs, labels):
    # TODO: return the fraction of rows whose argmax matches the integer label.
    preds = argmax_rows(logits_or_probs)
    return np.mean(preds == labels)

# Step 10 - he_std
def he_std(fan_in):
    # TODO: return the He initialization standard deviation sqrt(2 / fan_in).
    return float(np.sqrt(2.0 / fan_in))

# Step 11 - he_init
def he_init(shape, fan_in, seed):
    # TODO: sample a weight tensor from a normal distribution scaled by He std using the seed.
    np.random.seed(seed)
    return np.random.randn(*shape) * he_std(fan_in)

# Step 12 - init_zero_bias
import numpy as np

def init_zero_bias(length):
    # TODO: return a 1D float array of zeros with the given length.
    return np.zeros(length, dtype=np.float64)

# Step 13 - pad_2d
def pad_2d(images, pad):
    # TODO: zero-pad the spatial (H, W) dims of a 4D (N, C, H, W) tensor by `pad` on each side.
    return np.pad(
        images,
        ((0, 0), (0, 0), (pad, pad), (pad, pad)),
        mode="constant"
    )

# Step 14 - output_spatial_size
def output_spatial_size(input_size, kernel, stride, padding):
    # TODO: return the conv/pool output spatial dimension from input_size, kernel, stride, padding
    return int((input_size + 2 * padding - kernel) / stride + 1)

# Step 15 - im2col
def im2col(images, kernel_h, kernel_w, stride, padding):
    # TODO: Unroll overlapping patches of a 4D image tensor into a 2D column matrix.

    images_padded = pad_2d(images, padding)

    N, C, H, W = images_padded.shape

    out_h = output_spatial_size(H - 2 * padding, kernel_h, stride, padding)
    out_w = output_spatial_size(W - 2 * padding, kernel_w, stride, padding)

    cols = []

    for n in range(N):
        for i in range(out_h):
            for j in range(out_w):
                h_start = i * stride
                w_start = j * stride

                patch = images_padded[
                    n,
                    :,
                    h_start:h_start + kernel_h,
                    w_start:w_start + kernel_w
                ]

                cols.append(patch.reshape(-1))

    return np.array(cols)

# Step 16 - col2im
def col2im(cols, input_shape, kernel_h, kernel_w, stride, padding):
    # TODO: re-roll a (N*out_h*out_w, C*kh*kw) column matrix back into a (N, C, H, W) tensor

    N, C, H, W = input_shape

    out_h = output_spatial_size(H, kernel_h, stride, padding)
    out_w = output_spatial_size(W, kernel_w, stride, padding)

    H_padded = H + 2 * padding
    W_padded = W + 2 * padding

    images = np.zeros((N, C, H_padded, W_padded), dtype=np.float64)

    row = 0
    for n in range(N):
        for i in range(out_h):
            for j in range(out_w):
                h_start = i * stride
                w_start = j * stride

                patch = cols[row].reshape(C, kernel_h, kernel_w)

                images[
                    n,
                    :,
                    h_start:h_start + kernel_h,
                    w_start:w_start + kernel_w
                ] += patch

                row += 1

    if padding > 0:
        return images[:, :, padding:-padding, padding:-padding]

    return images

# Step 17 - conv2d_forward
def conv2d_forward(x, weights, bias, stride, padding):
    # TODO: convolve x with weights using im2col, add bias, return output and a backprop cache.

    N, C_in, H, W = x.shape
    C_out, _, kernel_h, kernel_w = weights.shape

    cols = im2col(x, kernel_h, kernel_w, stride, padding)

    W_col = weights.reshape(C_out, -1)

    out_h = output_spatial_size(H, kernel_h, stride, padding)
    out_w = output_spatial_size(W, kernel_w, stride, padding)

    out = cols @ W_col.T
    out += bias

    out = out.reshape(N, out_h, out_w, C_out)
    out = out.transpose(0, 3, 1, 2)

    cache = {
        "x_shape": x.shape,
        "weights": weights,
        "cols": cols,
        "stride": stride,
        "padding": padding,
        "kernel_h": kernel_h,
        "kernel_w": kernel_w,
    }

    return out, cache

# Step 18 - conv2d_grad_input
def conv2d_grad_input(d_out, cache):
    # TODO: backprop d_out through the conv input using col2im

    weights = cache["weights"]
    x_shape = cache["x_shape"]
    stride = cache["stride"]
    padding = cache["padding"]
    kernel_h = cache["kernel_h"]
    kernel_w = cache["kernel_w"]

    C_out = weights.shape[0]

    d_out_flat = d_out.transpose(0, 2, 3, 1).reshape(-1, C_out)

    W_col = weights.reshape(C_out, -1)

    d_cols = d_out_flat @ W_col

    dx = col2im(
        d_cols,
        x_shape,
        kernel_h,
        kernel_w,
        stride,
        padding
    )

    return dx

# Step 19 - conv2d_grad_weights
def conv2d_grad_weights(d_out, cache):
    # TODO: return dL/dW shaped (C_out, C_in, kH, kW) from d_out and the im2col cache.

    cols = cache["cols"]
    weights = cache["weights"]

    C_out, C_in, kernel_h, kernel_w = weights.shape

    d_out_flat = d_out.transpose(0, 2, 3, 1).reshape(-1, C_out)

    dW_col = d_out_flat.T @ cols

    return dW_col.reshape(C_out, C_in, kernel_h, kernel_w)

# Step 20 - conv2d_grad_bias
def conv2d_grad_bias(d_out):
    # TODO: return a length C_out gradient by reducing d_out over batch and spatial axes
    return np.sum(d_out, axis=(0, 2, 3))

# Step 21 - conv2d_backward
def conv2d_backward(d_out, cache):
    # TODO: return (dx, dW, db) using the conv2d gradient helpers and the forward cache

    dx = conv2d_grad_input(d_out, cache)
    dW = conv2d_grad_weights(d_out, cache)
    db = conv2d_grad_bias(d_out)

    return dx, dW, db

# Step 22 - maxpool2d_forward
def maxpool2d_forward(x, kernel, stride):
    # TODO: run 2D max pooling and cache the in-window argmax of each output cell.

    N, C, H, W = x.shape

    out_h = output_spatial_size(H, kernel, stride, 0)
    out_w = output_spatial_size(W, kernel, stride, 0)

    out = np.zeros((N, C, out_h, out_w), dtype=np.float64)
    argmax = np.zeros((N, C, out_h, out_w), dtype=np.int64)

    for n in range(N):
        for c in range(C):
            for i in range(out_h):
                for j in range(out_w):

                    h_start = i * stride
                    w_start = j * stride

                    patch = x[
                        n,
                        c,
                        h_start:h_start + kernel,
                        w_start:w_start + kernel
                    ]

                    out[n, c, i, j] = np.max(patch)
                    argmax[n, c, i, j] = np.argmax(patch)

    cache = {
        "x_shape": x.shape,
        "argmax": argmax,
        "kernel": kernel,
        "stride": stride,
    }

    return out, cache

# Step 23 - scatter_grad_window
import numpy as np

def scatter_grad_window(grad_value, argmax_index, kernel):
    # TODO: place grad_value at the argmax position within a (kernel, kernel) zero array.

    window = np.zeros((kernel, kernel), dtype=np.float64)

    row = argmax_index // kernel
    col = argmax_index % kernel

    window[row, col] = grad_value

    return window

# Step 24 - maxpool2d_backward
def maxpool2d_backward(d_out, cache):
    # TODO: scatter each d_out value to the cached argmax position in its window

    x_shape = cache["x_shape"]
    argmax = cache["argmax"]
    kernel = cache["kernel"]
    stride = cache["stride"]

    N, C, H, W = x_shape

    out_h = d_out.shape[2]
    out_w = d_out.shape[3]

    dx = np.zeros(x_shape, dtype=np.float64)

    for n in range(N):
        for c in range(C):
            for i in range(out_h):
                for j in range(out_w):

                    h_start = i * stride
                    w_start = j * stride

                    grad_patch = scatter_grad_window(
                        d_out[n, c, i, j],
                        argmax[n, c, i, j],
                        kernel
                    )

                    dx[
                        n,
                        c,
                        h_start:h_start + kernel,
                        w_start:w_start + kernel
                    ] += grad_patch

    return dx

# Step 25 - relu_forward
def relu_forward(x):
    # TODO: Compute the elementwise ReLU and cache the input for backprop.

    out = np.maximum(0, x)
    cache = {"x": x}

    return out, cache

# Step 26 - relu_backward
def relu_backward(d_out, cache):
    # TODO: mask the upstream gradient by the positive entries of the cached input.

    x = cache["x"]
    return d_out * (x > 0)

# Step 27 - flatten_forward
def flatten_forward(x):
    # TODO: reshape a 4D feature map into a 2D batch matrix and cache the original shape

    out = x.reshape(x.shape[0], -1)

    cache = {
        "x_shape": x.shape
    }

    return out, cache

# Step 28 - flatten_backward
import numpy as np

def flatten_backward(d_out, cache):
    # TODO: reshape the upstream gradient back to the original 4D feature map shape.
    return d_out.reshape(cache["x_shape"])

# Step 29 - linear_forward
def linear_forward(x, weights, bias):
    # TODO: compute X @ W + b and cache the inputs needed for backprop.

    out = x @ weights + bias

    cache = {
        "x": x,
        "weights": weights
    }

    return out, cache

# Step 30 - linear_grad_input
import numpy as np

def linear_grad_input(d_out, cache):
    """Gradient of a linear layer w.r.t. its input X."""
    # TODO: return dL/dX given d_out (N, D_out) and cache['weights'] (D_in, D_out)

    return d_out @ cache["weights"].T

# Step 31 - linear_grad_weights
import numpy as np

def linear_grad_weights(x, dout):
    """Gradient of loss wrt linear-layer weights W of shape (D_in, D_out)."""
    # TODO: Compute the gradient of a linear layer's loss wrt its weight matrix W.

    return x.T @ dout

# Step 32 - linear_grad_bias
import numpy as np

def linear_grad_bias(dout):
    # TODO: Compute the bias gradient of a linear layer given upstream gradient dout.

    return np.sum(dout, axis=0)

# Step 33 - linear_backward
def linear_backward(dout, cache):
    # TODO: combine input, weight, and bias gradients for a linear layer using the cache

    dx = linear_grad_input(dout, cache)
    dW = linear_grad_weights(cache["x"], dout)
    db = linear_grad_bias(dout)

    return dx, dW, db

# Step 34 - softmax_cross_entropy_forward
def softmax_cross_entropy_forward(logits, y):
    # TODO: return the mean cross-entropy loss for logits (N, C) and integer labels y (N,).

    probs = stable_softmax(logits)
    return abs(float(cross_entropy_loss(probs, y)))

# Step 35 - softmax_cross_entropy_backward
def softmax_cross_entropy_backward(logits, y):
    # TODO: return the fused softmax-cross-entropy gradient of shape (N, C).

    probs = stable_softmax(logits)
    targets = one_hot(y, logits.shape[1])

    return (probs - targets) / logits.shape[0]

# Step 36 - sgd_step
import numpy as np

def sgd_step(param, grad, lr):
    # TODO: return the SGD-updated parameter array (param - lr * grad).

    return param - lr * grad

# Step 37 - adam_update_m
import numpy as np

def adam_update_m(m, grad, beta_one):
    # TODO: return the updated first moment estimate using beta_one and grad.

    return beta_one * m + (1.0 - beta_one) * grad

# Step 38 - adam_update_v
import numpy as np

def adam_update_v(v, grad, beta_two):
    # TODO: return the updated Adam second moment estimate as an EMA of squared gradients.

    return beta_two * v + (1.0 - beta_two) * (grad ** 2)

# Step 39 - adam_bias_correct
def adam_bias_correct(moment, beta, t):
    # TODO: return moment divided by (1 - beta**t) to undo Adam's zero-init bias.

    return moment / (1.0 - beta ** t)

# Step 40 - adam_param_step
import numpy as np

def adam_param_step(param, m_hat, v_hat, lr, eps):
    # TODO: apply one Adam parameter update using bias-corrected moments

    return param - lr * m_hat / (np.sqrt(v_hat) + eps)

# Step 41 - adam_step
import numpy as np

def adam_step(param, grad, m, v, t, lr, beta_one, beta_two, eps):
    # TODO: chain the four Adam helpers and return (new_param, new_m, new_v)

    new_m = adam_update_m(m, grad, beta_one)
    new_v = adam_update_v(v, grad, beta_two)

    m_hat = adam_bias_correct(new_m, beta_one, t)
    v_hat = adam_bias_correct(new_v, beta_two, t)

    new_param = adam_param_step(param, m_hat, v_hat, lr, eps)

    return new_param, new_m, new_v

# Step 42 - init_conv_layer
def init_conv_layer(out_channels, in_channels, kernel_size, seed=0):
    # TODO: Build He-initialized weights and a zero bias for a single conv layer.

    fan_in = in_channels * kernel_size * kernel_size

    W = he_init(
        (out_channels, in_channels, kernel_size, kernel_size),
        fan_in,
        seed
    )

    b = init_zero_bias(out_channels)

    return {
        "W": W,
        "b": b
    }

# Step 43 - init_linear_layer
def init_linear_layer(in_features, out_features, seed=0):
    # TODO: return {'W': He-init matrix (in_features, out_features), 'b': zero bias (out_features,)}

    W = he_init(
        (in_features, out_features),
        in_features,
        seed
    )

    b = init_zero_bias(out_features)

    return {
        "W": W,
        "b": b
    }

# Step 44 - init_lenet
def init_lenet(in_channels, num_classes, seed=0):
    # TODO: build conv1, conv2, fc1, fc2 with the right shapes and return them in a dict.

    return {
        "conv1": init_conv_layer(
            out_channels=6,
            in_channels=in_channels,
            kernel_size=5,
            seed=seed
        ),

        "conv2": init_conv_layer(
            out_channels=16,
            in_channels=6,
            kernel_size=5,
            seed=seed + 1
        ),

        "fc1": init_linear_layer(
            in_features=16 * 4 * 4,
            out_features=120,
            seed=seed + 2
        ),

        "fc2": init_linear_layer(
            in_features=120,
            out_features=num_classes,
            seed=seed + 3
        )
    }

# Step 45 - forward_conv_block
def forward_conv_block(x, W, b, pool_size, stride, pad):
    # TODO: run conv2d -> relu -> maxpool2d and return (out, cache_dict)

    conv_out, conv_cache = conv2d_forward(
        x, W, b,
        stride=stride,
        padding=pad
    )

    relu_out, relu_cache = relu_forward(conv_out)

    out, pool_cache = maxpool2d_forward(
        relu_out,
        kernel=pool_size,
        stride=pool_size
    )

    cache = {
        "conv_cache": conv_cache,
        "relu_cache": relu_cache,
        "pool_cache": pool_cache
    }

    return out, cache

# Step 46 - forward_classifier_block
def forward_classifier_block(x, fc1, fc2):
    # TODO: run flatten -> linear -> relu -> linear and return logits plus a cache dict.

    flat, flatten_cache = flatten_forward(x)

    fc1_out, fc1_cache = linear_forward(
        flat,
        fc1["W"],
        fc1["b"]
    )

    relu_out, relu_cache = relu_forward(fc1_out)

    logits, fc2_cache = linear_forward(
        relu_out,
        fc2["W"],
        fc2["b"]
    )

    cache = {
        "flatten_cache": flatten_cache,
        "fc1_cache": fc1_cache,
        "relu_cache": relu_cache,
        "fc2_cache": fc2_cache
    }

    return logits, cache

# Step 47 - lenet_forward
def lenet_forward(x, params):
    # TODO: run two conv blocks then the classifier block and return (logits, caches).

    out1, block1_cache = forward_conv_block(
        x,
        params["conv1"]["W"],
        params["conv1"]["b"],
        pool_size=2,
        stride=1,
        pad=0
    )

    out2, block2_cache = forward_conv_block(
        out1,
        params["conv2"]["W"],
        params["conv2"]["b"],
        pool_size=2,
        stride=1,
        pad=0
    )

    logits, classifier_cache = forward_classifier_block(
        out2,
        params["fc1"],
        params["fc2"]
    )

    caches = {
        "block1": block1_cache,
        "block2": block2_cache,
        "classifier": classifier_cache
    }

    return logits, caches

# Step 48 - backward_conv_block
def backward_conv_block(dout, cache):
    # TODO: backprop dout through the cached pool, relu, and conv layers in reverse order.

    dout = maxpool2d_backward(dout, cache["pool_cache"])
    dout = relu_backward(dout, cache["relu_cache"])

    dx, dW, db = conv2d_backward(
        dout,
        cache["conv_cache"]
    )

    return dx, dW, db

# Step 49 - backward_classifier_block
def backward_classifier_block(dlogits, cache):
    # TODO: backprop through fc2 -> relu -> fc1 -> flatten using the cached values

    d_relu, dW2, db2 = linear_backward(
        dlogits,
        cache["fc2_cache"]
    )

    d_fc1 = relu_backward(
        d_relu,
        cache["relu_cache"]
    )

    d_flat, dW1, db1 = linear_backward(
        d_fc1,
        cache["fc1_cache"]
    )

    dx = flatten_backward(
        d_flat,
        cache["flatten_cache"]
    )

    return {
        "dx": dx,
        "fc1": {
            "dW": dW1,
            "db": db1
        },
        "fc2": {
            "dW": dW2,
            "db": db2
        }
    }

# Step 50 - lenet_backward
def lenet_backward(dlogits, caches):
    # TODO: walk classifier and conv block caches in reverse to assemble all gradients

    classifier_grads = backward_classifier_block(
        dlogits,
        caches["classifier"]
    )

    dx2, dW2, db2 = backward_conv_block(
        classifier_grads["dx"],
        caches["block2"]
    )

    dx1, dW1, db1 = backward_conv_block(
        dx2,
        caches["block1"]
    )

    return {
        "conv1": {
            "dW": dW1,
            "db": db1
        },
        "conv2": {
            "dW": dW2,
            "db": db2
        },
        "fc1": {
            "dW": classifier_grads["fc1"]["dW"],
            "db": classifier_grads["fc1"]["db"]
        },
        "fc2": {
            "dW": classifier_grads["fc2"]["dW"],
            "db": classifier_grads["fc2"]["db"]
        }
    }

# Step 51 - lenet_predict
def lenet_predict(x, params):
    # TODO: Return the argmax class index per sample from a LeNet forward pass.

    logits, _ = lenet_forward(x, params)
    return argmax_rows(logits)

# Step 52 - build_synthetic_image_dataset
def build_synthetic_image_dataset(num_samples, num_classes, image_size, in_channels=1, seed=0):
    # TODO: Return (x, y) for a reproducible synthetic NCHW image dataset.

    rng = np.random.default_rng(seed)

    y = rng.integers(
        0,
        num_classes,
        size=num_samples
    )

    x = rng.standard_normal(
        (num_samples, in_channels, image_size, image_size)
    )

    shifts = y - (num_classes - 1) / 2

    x = x + shifts[:, None, None, None]

    return x, y

# Step 53 - shuffle_indices
import numpy as np

def shuffle_indices(n, seed=0):
    # TODO: return a reproducible permutation of [0, n) as an int ndarray of shape (n,).

    np.random.seed(seed)
    return np.random.permutation(n)

# Step 54 - train_test_split
def train_test_split(x, y, test_fraction=0.2, seed=0):
    # TODO: partition x and y into train and test halves using a shared shuffled order.

    N = len(y)

    idx = shuffle_indices(N, seed)

    test_size = int(N * test_fraction)

    test_idx = idx[:test_size]
    train_idx = idx[test_size:]

    x_train = x[train_idx]
    y_train = y[train_idx]

    x_test = x[test_idx]
    y_test = y[test_idx]

    return x_train, y_train, x_test, y_test

# Step 55 - iterate_minibatches
def iterate_minibatches(x, y, batch_size, seed=0):
    # TODO: yield shuffled mini-batches of features and labels for one epoch of training.

    N = len(y)

    idx = shuffle_indices(N, seed)

    for start in range(0, N - batch_size + 1, batch_size):
        batch_idx = idx[start:start + batch_size]

        xb = x[batch_idx]
        yb = y[batch_idx]

        yield xb, yb

# Step 56 - train_step
def train_step(params, opt_state, xb, yb, lr, beta_one, beta_two, eps, step):
    # TODO: Run forward + loss + backward and apply one Adam update to every parameter.

    logits, caches = lenet_forward(xb, params)

    loss = softmax_cross_entropy_forward(logits, yb)

    dlogits = softmax_cross_entropy_backward(logits, yb)

    grads = lenet_backward(dlogits, caches)

    new_params = {}
    new_opt_state = {}

    for layer in params:
        new_params[layer] = {}
        new_opt_state[layer] = {}

        for pname in ["W", "b"]:

            param = params[layer][pname]
            grad = grads[layer]["d" + pname]

            m = opt_state[layer][pname]["m"]
            v = opt_state[layer][pname]["v"]

            new_param, new_m, new_v = adam_step(
                param,
                grad,
                m,
                v,
                step,
                lr,
                beta_one,
                beta_two,
                eps
            )

            new_params[layer][pname] = new_param

            new_opt_state[layer][pname] = {
                "m": new_m,
                "v": new_v
            }

    return new_params, new_opt_state, loss

# Step 57 - train_one_epoch
def train_one_epoch(params, opt_state, x, y, batch_size, lr, beta_one, beta_two, eps, step_counter, seed=0):
    # TODO: iterate minibatches and apply one train_step per batch, tracking losses and step_counter.

    losses = []

    for xb, yb in iterate_minibatches(
        x,
        y,
        batch_size,
        seed=seed
    ):
        step_counter += 1

        params, opt_state, loss = train_step(
            params,
            opt_state,
            xb,
            yb,
            lr,
            beta_one,
            beta_two,
            eps,
            step_counter
        )

        losses.append(loss)

    return params, opt_state, step_counter, losses

# Step 58 - train_loop
def train_loop(params, x_train, y_train, num_epochs, batch_size,
               lr=1e-3, beta_one=0.9, beta_two=0.999,
               eps=1e-8, seed=0):
    # TODO: initialize Adam state, loop epochs calling train_one_epoch, return (params, loss_history).

    opt_state = {}

    for layer in params:
        opt_state[layer] = {}

        for pname in ["W", "b"]:
            opt_state[layer][pname] = {
                "m": np.zeros_like(params[layer][pname]),
                "v": np.zeros_like(params[layer][pname]),
            }

    step_counter = 0
    loss_history = []

    for epoch in range(num_epochs):

        params, opt_state, step_counter, losses = train_one_epoch(
            params,
            opt_state,
            x_train,
            y_train,
            batch_size,
            lr,
            beta_one,
            beta_two,
            eps,
            step_counter,
            seed=seed + epoch
        )

        loss_history.extend([float(loss) for loss in losses])

    return params, loss_history

# Step 59 - evaluate
def evaluate(params, x, y):
    # TODO: return the fraction of samples whose predicted class equals the label.

    preds = lenet_predict(x, params)
    return float(np.mean(preds == y))

