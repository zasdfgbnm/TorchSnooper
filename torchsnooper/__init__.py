import torch
import pysnooper


class TensorFormat:

    def __init__(self, property_name=False, properties=('shape', 'dtype', 'device', 'requires_grad')):
        self.properties = properties
        self.properties_name = property_name

    def __call__(self, tensor):
        prefix = 'Tensor['
        suffix = ']'
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


class TorchSnooper:

    def __init__(self, tensor_format=default_format):
        if tensor_format is not default_format:
            raise NotImplementedError('Formatting is not supported yet')
        self.tensor_format = tensor_format

    @staticmethod
    def is_return_types(x):
        return type(x).__module__ == 'torch.return_types'

    def return_types_repr(self, x):
        if type(x).__name__ == 'min':
            return 'min(values=' + self.tensor_format(x.values) + ', indices=' + self.tensor_format(x.indices) + ')'

    def condition(self, x):
        return torch.is_tensor(x) or self.is_return_types(x)

    def compute_repr(self, x):
        if torch.is_tensor(x):
            return self.tensor_format(x)
        elif self.is_return_types(x):
            return self.return_types_repr(x)
        raise RuntimeError('Control flow should not reach here, open a bug report!')

    def __len__(self):
        return 2

    def __getitem__(self, i):
        return (self.condition, self.compute_repr)[i]

    def snoop(self, *args, **kwargs):
        if 'custom_repr' in kwargs:
            kwargs['custom_repr'] = (self, *kwargs['custom_repr'])
        else:
            kwargs['custom_repr'] = (self,)
        return pysnooper.snoop(*args, **kwargs)


def snoop(*args, **kwargs):
    return TorchSnooper().snoop(*args, **kwargs)
