import torch
import pysnooper
import pysnooper.utils
import warnings
import numpy
from pkg_resources import get_distribution, DistributionNotFound


try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    # package is not installed
    pass


class TensorFormat:

    def __init__(self, property_name=False, properties=('shape', 'dtype', 'device', 'requires_grad', 'has_nan', 'has_inf')):
        self.properties = properties
        self.properties_name = property_name

    def __call__(self, tensor):
        prefix = 'tensor<'
        suffix = '>'
        properties_str = ''
        for p in self.properties:
            new = ''
            if p == 'shape':
                if self.properties_name:
                    new += 'shape='
                new += str(tuple(tensor.shape))
            elif p == 'dtype':
                dtype_str = str(tensor.dtype)
                dtype_str = dtype_str[len('torch.'):]
                if self.properties_name:
                    new += 'dtype='
                new += dtype_str
            elif p == 'device':
                if self.properties_name:
                    new += 'device='
                new += str(tensor.device)
            elif p == 'requires_grad':
                if self.properties_name:
                    new += 'requires_grad='
                    new += str(tensor.requires_grad)
                else:
                    if tensor.requires_grad:
                        new += 'grad'
            elif p == 'has_nan':
                result = bool(torch.isnan(tensor).any())
                if self.properties_name:
                    new += 'has_nan='
                    new += str(result)
                else:
                    if result:
                        new += 'has_nan'
            elif p == 'has_inf':
                result = bool(torch.isinf(tensor).any())
                if self.properties_name:
                    new += 'has_inf='
                    new += str(result)
                else:
                    if result:
                        new += 'has_inf'
            else:
                raise ValueError('Unknown tensor property')

            if properties_str != '' and len(new) > 0:
                properties_str += ', '
            properties_str += new

        return prefix + properties_str + suffix


default_format = TensorFormat()


class NumpyFormat:

    def __call__(self, x):
        return f'ndarray<{x.shape}, {x.dtype.name}>'


default_numpy_format = NumpyFormat()


class TorchSnooper(pysnooper.tracer.Tracer):

    def __init__(self, *args, tensor_format=default_format, numpy_format=default_numpy_format, **kwargs):
        self.orig_custom_repr = kwargs['custom_repr'] if 'custom_repr' in kwargs else ()
        custom_repr = (lambda x: True, self.compute_repr)
        kwargs['custom_repr'] = (custom_repr,)
        super(TorchSnooper, self).__init__(*args, **kwargs)
        self.tensor_format = tensor_format
        self.numpy_format = numpy_format

    @staticmethod
    def is_return_types(x):
        return type(x).__module__ == 'torch.return_types'

    def return_types_repr(self, x):
        if type(x).__name__ in {'max', 'min', 'median', 'mode', 'sort', 'topk', 'kthvalue'}:
            return type(x).__name__ + '(values=' + self.tensor_format(x.values) + ', indices=' + self.tensor_format(x.indices) + ')'
        if type(x).__name__ == 'svd':
            return 'svd(U=' + self.tensor_format(x.U) + ', S=' + self.tensor_format(x.S) + ', V=' + self.tensor_format(x.V) + ')'
        if type(x).__name__ == 'slogdet':
            return 'slogdet(sign=' + self.tensor_format(x.sign) + ', logabsdet=' + self.tensor_format(x.logabsdet) + ')'
        if type(x).__name__ == 'qr':
            return 'qr(Q=' + self.tensor_format(x.Q) + ', R=' + self.tensor_format(x.R) + ')'
        if type(x).__name__ == 'solve':
            return 'solve(solution=' + self.tensor_format(x.solution) + ', LU=' + self.tensor_format(x.LU) + ')'
        if type(x).__name__ == 'geqrf':
            return 'geqrf(a=' + self.tensor_format(x.a) + ', tau=' + self.tensor_format(x.tau) + ')'
        if type(x).__name__ in {'symeig', 'eig'}:
            return type(x).__name__ + '(eigenvalues=' + self.tensor_format(x.eigenvalues) + ', eigenvectors=' + self.tensor_format(x.eigenvectors) + ')'
        if type(x).__name__ == 'triangular_solve':
            return 'triangular_solve(solution=' + self.tensor_format(x.solution) + ', cloned_coefficient=' + self.tensor_format(x.cloned_coefficient) + ')'
        if type(x).__name__ == 'gels':
            return 'gels(solution=' + self.tensor_format(x.solution) + ', QR=' + self.tensor_format(x.QR) + ')'
        warnings.warn('Unknown return_types encountered, open a bug report!')

    def compute_repr(self, x):
        orig_repr_func = pysnooper.utils.get_repr_function(x, self.orig_custom_repr)
        if torch.is_tensor(x):
            return self.tensor_format(x)
        elif isinstance(x, numpy.ndarray):
            return self.numpy_format(x)
        elif self.is_return_types(x):
            return self.return_types_repr(x)
        elif orig_repr_func is not repr:
            return orig_repr_func(x)
        elif isinstance(x, (list, tuple)):
            content = ''
            for i in x:
                if content != '':
                    content += ', '
                content += self.compute_repr(i)
            if isinstance(x, tuple) and len(x) == 1:
                content += ','
            if isinstance(x, tuple):
                return '(' + content + ')'
            return '[' + content + ']'
        elif isinstance(x, dict):
            content = ''
            for k, v in x.items():
                if content != '':
                    content += ', '
                content += self.compute_repr(k) + ': ' + self.compute_repr(v)
            return '{' + content + '}'
        return repr(x)


snoop = TorchSnooper


def register_snoop(verbose=False, tensor_format=default_format, numpy_format=default_numpy_format):
    import snoop
    import cheap_repr
    import snoop.configuration
    cheap_repr.register_repr(torch.Tensor)(lambda x, _: tensor_format(x))
    cheap_repr.register_repr(numpy.ndarray)(lambda x, _: numpy_format(x))
    cheap_repr.cheap_repr(torch.zeros(6))
    unwanted = {
        snoop.configuration.len_shape_watch,
        snoop.configuration.dtype_watch,
    }
    snoop.config.watch_extras = tuple(x for x in snoop.config.watch_extras if x not in unwanted)
    if verbose:

        class TensorWrap:

            def __init__(self, tensor):
                self.tensor = tensor

            def __repr__(self):
                return self.tensor.__repr__()

        snoop.config.watch_extras += (
            lambda source, value: ('{}.data'.format(source), TensorWrap(value.data)),
        )
