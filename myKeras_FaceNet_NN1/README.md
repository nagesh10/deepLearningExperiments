# FaceNet NN1 implementation in Keras

An implementation of the NN1 net from the [FaceNet](https://arxiv.org/pdf/1503.03832.pdf) paper.

## The net

NN1 net consists of 22 layers:
- INPUT_DIM       : LAYER  : EXPLANATION  : OUTPUT_DIM
---
- 220 x 220 x   3 : conv1  : 2D convolution, with 64 7x7x3 filters, stride 2, with reLU activation : 110 x 110 x  64
- 110 x 110 x  64 : pool1  : max pooling, with 64 3x3x64 filters, stride 2 : 55 x 55 x  64
-  55 x  55 x  64 : rnorm1 : (local response normalization) LRN2D : 55 x 55 x  64
-  55 x  55 x  64 : conv2a : 1D convolution, with 64 1x1x64 filters for each pixel (stride 1), with reLU activation : 55 x 55 x  64
-  55 x  55 x  64 : conv2  : 2D convolution, with 192 3x3x64 filters, stride 1, with reLU activation : 55 x 55 x 192
-  55 x  55 x 192 : rnorm2 : (local response normalization) LRN2D : 55 x 55 x 192
-  55 x  55 x 192 : pool2  : max pooling, with 192 3x3x192 filters, stride 2 : 28 x 28 x 192
-  28 x  28 x 192 : conv3a : 1D convolution, with 192 1x1x192 filters for each pixel (stride 1), with reLU activation : 28 x 28 x 192
-  28 x  28 x 192 : conv3  : 2D convolution, with 384 3x3x192 filters, stride 1, with reLU activation :  28 x 28 x 384
-  28 x  28 x 384 : pool3  : max pooling, with 384 3x3x384 filters, stride 2 :  14 x 14 x 384
-  14 x  14 x 384 : conv4a : 1D convolution, with 384 1x1x384 filters for each pixel (stride 1), with reLU activation : 14 x 14 x 384
-  14 x  14 x 384 : conv4  : 2D convolution, with 256 3x3x384 filters, stride 1, with reLU activation :  14 x 14 x 256
-  14 x  14 x 256 : conv5a : 1D convolution, with 256 1x1x256 filters for each pixel (stride 1), with reLU activation : 14 x 14 x 256
-  14 x  14 x 256 : conv5  : 2D convolution, with 256 3x3x256 filters, stride 1, with reLU activation :  14 x 14 x 256
-  14 x  14 x 256 : conv6a : 1D convolution, with 256 1x1x256 filters for each pixel (stride 1), with reLU activation : 14 x 14 x 256
-  14 x  14 x 256 : conv6  : 2D convolution, with 256 3x3x256 filters, stride 1, with reLU activation : 14 x 14 x 256
-  14 x  14 x 256 : pool4  : max pooling, with 384 3x3x256 filters, stride 2    : 7 x 7 x 256
-   7 x   7 x 256 : concat : Flatten : 12544
-           12544 : fc1    : (Fully connected, with maxout) MaxoutDense, with nb_filters 2 : 4096
-            4096 : fc2    : (Fully connected, with maxout) MaxoutDense, with nb_filters 2 : 4096
-            4096 : fc7128 : (Fully connected) Dense : 128
-             128 : L2     : (L2 normalize) Lambda l2_normalize : 128

The last layer, L2 normalize, is to make the output lie on the unit hypersphere of 128 dimensions.

## Pre-requisites

I ran Keras on version 2.0.2 (found in __init__.py), but LRN2D and MaxoutDense were not present in this version.

So I searched for their codes on a previous commit of [Keras](https://github.com/fchollet/keras/), manually added them to their appropriate files - LRN2D in layers/normalization.py, and MaxoutDense in layers/core.py (both installed in /usr/local/Cellar/python/2.7.11/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/keras in my laptop) - and manually compiled them to make new normalization.pyc and core.pyc.

### LRN2D in layers/normalization.py
```
class LRN2D(Layer):
    """
    This code is adapted from pylearn2.
    License at: https://github.com/lisa-lab/pylearn2/blob/master/LICENSE.txt
    """

    def __init__(self, alpha=1e-4, k=2, beta=0.75, n=5):
        if n % 2 == 0:
            raise NotImplementedError("LRN2D only works with odd n. n provided: " + str(n))
        super(LRN2D, self).__init__()
        self.alpha = alpha
        self.k = k
        self.beta = beta
        self.n = n

    def get_output(self, train):
        X = self.get_input(train)
        b, ch, r, c = X.shape
        half_n = self.n // 2
        input_sqr = T.sqr(X)
        extra_channels = T.alloc(0., b, ch + 2*half_n, r, c)
        input_sqr = T.set_subtensor(extra_channels[:, half_n:half_n+ch, :, :], input_sqr)
        scale = self.k
        for i in range(self.n):
            scale += self.alpha * input_sqr[:, i:i+ch, :, :]
        scale = scale ** self.beta
        return X / scale

    def get_config(self):
        return {"name": self.__class__.__name__,
                "alpha": self.alpha,
                "k": self.k,
                "beta": self.beta,
                "n": self.n}
```

### MaxoutDense in layers/core.py
```
class MaxoutDense(Layer):
    '''A dense maxout layer.
    A `MaxoutDense` layer takes the element-wise maximum of
    `nb_feature` `Dense(input_dim, output_dim)` linear layers.
    This allows the layer to learn a convex,
    piecewise linear activation function over the inputs.
    Note that this is a *linear* layer;
    if you wish to apply activation function
    (you shouldn't need to --they are universal function approximators),
    an `Activation` layer must be added after.
    # Arguments
        output_dim: int > 0.
        nb_feature: number of Dense layers to use internally.
        init: name of initialization function for the weights of the layer
            (see [initializations](../initializations.md)),
            or alternatively, Theano function to use for weights
            initialization. This parameter is only relevant
            if you don't pass a `weights` argument.
        weights: list of numpy arrays to set as initial weights.
            The list should have 2 elements, of shape `(input_dim, output_dim)`
            and (output_dim,) for weights and biases respectively.
        W_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the main weights matrix.
        b_regularizer: instance of [WeightRegularizer](../regularizers.md),
            applied to the bias.
        activity_regularizer: instance of [ActivityRegularizer](../regularizers.md),
            applied to the network output.
        W_constraint: instance of the [constraints](../constraints.md) module
            (eg. maxnorm, nonneg), applied to the main weights matrix.
        b_constraint: instance of the [constraints](../constraints.md) module,
            applied to the bias.
        bias: whether to include a bias (i.e. make the layer affine rather than linear).
        input_dim: dimensionality of the input (integer).
            This argument (or alternatively, the keyword argument `input_shape`)
            is required when using this layer as the first layer in a model.
    # Input shape
        2D tensor with shape: `(nb_samples, input_dim)`.
    # Output shape
        2D tensor with shape: `(nb_samples, output_dim)`.
    # References
        - [Maxout Networks](http://arxiv.org/pdf/1302.4389.pdf)
    '''
    def __init__(self, output_dim, nb_feature=4,
                 init='glorot_uniform', weights=None,
                 W_regularizer=None, b_regularizer=None, activity_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, input_dim=None, **kwargs):
        self.output_dim = output_dim
        self.nb_feature = nb_feature
        self.init = initializations.get(init)

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.initial_weights = weights
        self.input_spec = [InputSpec(ndim=2)]

        self.input_dim = input_dim
        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim,)
        super(MaxoutDense, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dim = input_shape[1]
        self.input_spec = [InputSpec(dtype=K.floatx(),
                                     shape=(None, input_dim))]

        self.W = self.init((self.nb_feature, input_dim, self.output_dim),
                           name='{}_W'.format(self.name))
        if self.bias:
            self.b = K.zeros((self.nb_feature, self.output_dim),
                             name='{}_b'.format(self.name))
            self.trainable_weights = [self.W, self.b]
        else:
            self.trainable_weights = [self.W]

        self.regularizers = []
        if self.W_regularizer:
            self.W_regularizer.set_param(self.W)
            self.regularizers.append(self.W_regularizer)

        if self.bias and self.b_regularizer:
            self.b_regularizer.set_param(self.b)
            self.regularizers.append(self.b_regularizer)

        if self.activity_regularizer:
            self.activity_regularizer.set_layer(self)
            self.regularizers.append(self.activity_regularizer)

        self.constraints = {}
        if self.W_constraint:
            self.constraints[self.W] = self.W_constraint
        if self.bias and self.b_constraint:
            self.constraints[self.b] = self.b_constraint

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def get_output_shape_for(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return (input_shape[0], self.output_dim)

    def call(self, x, mask=None):
        # no activation, this layer is only linear.
        output = K.dot(x, self.W)
        if self.bias:
            output += self.b
        output = K.max(output, axis=1)
        return output

    def get_config(self):
        config = {'output_dim': self.output_dim,
                  'init': self.init.__name__,
                  'nb_feature': self.nb_feature,
                  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  'b_regularizer': self.b_regularizer.get_config() if self.b_regularizer else None,
                  'activity_regularizer': self.activity_regularizer.get_config() if self.activity_regularizer else None,
                  'W_constraint': self.W_constraint.get_config() if self.W_constraint else None,
                  'b_constraint': self.b_constraint.get_config() if self.b_constraint else None,
                  'bias': self.bias,
                  'input_dim': self.input_dim}
        base_config = super(MaxoutDense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
```
