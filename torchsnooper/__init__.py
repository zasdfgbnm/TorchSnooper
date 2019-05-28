import torch
import pysnooper
import warnings
from pkg_resources import get_distribution, DistributionNotFound


try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    # package is not installed
    pass


class TensorFormat:

    def __init__(self, property_name=False, properties=('shape', 'dtype', 'device', 'requires_grad')):
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
            else:
                raise ValueError('Unknown tensor property')

            if properties_str != '' and len(new) > 0:
                properties_str += ', '
            properties_str += new

        return prefix + properties_str + suffix


default_format = TensorFormat()


class TorchSnooper(pysnooper.tracer.Tracer):

    def __init__(self, *args, tensor_format=default_format, **kwargs):
        custom_repr = (self.condition, self.compute_repr)
        if 'custom_repr' in kwargs:
            kwargs['custom_repr'] = (custom_repr, *kwargs['custom_repr'])
        else:
            kwargs['custom_repr'] = (custom_repr,)
        super(TorchSnooper, self).__init__(*args, **kwargs)
        self.tensor_format = tensor_format

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

    def is_list_of_tensors(self, x):
        if not isinstance(x, list):
            return False
        return all([torch.is_tensor(i) for i in x])

    def list_of_tensors_repr(self, x):
        l = ''
        for i in x:
            if l != '':
                l += ', '
            l += self.tensor_format(i)
        return '[' + l + ']'

    def is_tuple_of_tensors(self, x):
        if not isinstance(x, tuple):
            return False
        return all([torch.is_tensor(i) for i in x])

    def tuple_of_tensors_repr(self, x):
        l = ''
        for i in x:
            if l != '':
                l += ', '
            l += self.tensor_format(i)
        if len(x) == 1:
            l += ','
        return '(' + l + ')'

    def is_dict_of_tensors(self, x):
        if not isinstance(x, dict):
            return False
        return all([torch.is_tensor(i) for i in x.values()])

    def dict_of_tensors_repr(self, x):
        l = ''
        for k, v in x.items():
            if l != '':
                l += ', '
            l += repr(k) + ': ' + self.tensor_format(v)
        return '{' + l + '}'

    def condition(self, x):
        return torch.is_tensor(x) or self.is_return_types(x) or \
            self.is_list_of_tensors(x) or self.is_tuple_of_tensors(x) or self.is_dict_of_tensors(x)

    def compute_repr(self, x):
        if torch.is_tensor(x):
            return self.tensor_format(x)
        elif self.is_return_types(x):
            return self.return_types_repr(x)
        elif self.is_list_of_tensors(x):
            return self.list_of_tensors_repr(x)
        elif self.is_tuple_of_tensors(x):
            return self.tuple_of_tensors_repr(x)
        elif self.is_dict_of_tensors(x):
            return self.dict_of_tensors_repr(x)
        raise RuntimeError('Control flow should not reach here, open a bug report!')


snoop = TorchSnooper
