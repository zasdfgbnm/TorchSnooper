import io
import math
import torch
import numpy
import torchsnooper
from .utils import assert_output, VariableEntry, CallEntry, LineEntry, ReturnEntry, ReturnValueEntry


def test_default_tensor():
    string_io = io.StringIO()

    @torchsnooper.snoop(string_io)
    def my_function():
        x = torch.randn((5, 8), requires_grad=True)
        return x

    my_function()

    output = string_io.getvalue()
    print(output)
    assert_output(
        output,
        (
            CallEntry(),
            LineEntry(),
            VariableEntry('x', 'tensor<(5, 8), float32, cpu, grad>'),
            LineEntry(),
            ReturnEntry(),
            ReturnValueEntry('tensor<(5, 8), float32, cpu, grad>'),
        )
    )


def test_tensor_property_selector():
    string_io = io.StringIO()
    fmt = torchsnooper.TensorFormat(properties=('shape', 'device', 'requires_grad'))

    @torchsnooper.snoop(string_io, tensor_format=fmt)
    def my_function():
        x = torch.randn((5, 8))
        return x

    my_function()

    output = string_io.getvalue()
    print(output)
    assert_output(
        output,
        (
            CallEntry(),
            LineEntry(),
            VariableEntry('x', 'tensor<(5, 8), cpu>'),
            LineEntry(),
            ReturnEntry(),
            ReturnValueEntry('tensor<(5, 8), cpu>'),
        )
    )


def test_tensor_property_name():
    string_io = io.StringIO()
    fmt = torchsnooper.TensorFormat(property_name=True)

    @torchsnooper.snoop(string_io, tensor_format=fmt)
    def my_function():
        x = torch.randn((5, 8))
        return x

    my_function()

    output = string_io.getvalue()
    print(output)
    assert_output(
        output,
        (
            CallEntry(),
            LineEntry(),
            VariableEntry('x', 'tensor<shape=(5, 8), dtype=float32, device=cpu, requires_grad=False>'),
            LineEntry(),
            ReturnEntry(),
            ReturnValueEntry('tensor<shape=(5, 8), dtype=float32, device=cpu, requires_grad=False>'),
        )
    )


def test_tuple_of_tensors():
    string_io = io.StringIO()

    @torchsnooper.snoop(string_io)
    def my_function():
        x = (torch.randn((5, 8)),)
        y = (torch.randn((5, 8)), torch.randn(()))  # noqa: F841
        return x

    my_function()

    output = string_io.getvalue()
    print(output)
    assert_output(
        output,
        (
            CallEntry(),
            LineEntry(),
            VariableEntry('x', '(tensor<(5, 8), float32, cpu>,)'),
            LineEntry(),
            VariableEntry('y', '(tensor<(5, 8), float32, cpu>, tensor<(), float32, cpu>)'),
            LineEntry(),
            ReturnEntry(),
            ReturnValueEntry('(tensor<(5, 8), float32, cpu>,)'),
        )
    )


def test_list_of_tensors():
    string_io = io.StringIO()

    @torchsnooper.snoop(string_io)
    def my_function():
        x = [torch.randn((5, 8))]
        y = [torch.randn((5, 8)), torch.randn(())]  # noqa: F841
        return x

    my_function()

    output = string_io.getvalue()
    print(output)
    assert_output(
        output,
        (
            CallEntry(),
            LineEntry(),
            VariableEntry('x', '[tensor<(5, 8), float32, cpu>]'),
            LineEntry(),
            VariableEntry('y', '[tensor<(5, 8), float32, cpu>, tensor<(), float32, cpu>]'),
            LineEntry(),
            ReturnEntry(),
            ReturnValueEntry('[tensor<(5, 8), float32, cpu>]'),
        )
    )


def test_dict_of_tensors():
    string_io = io.StringIO()

    @torchsnooper.snoop(string_io)
    def my_function():
        x = {'key': torch.randn((5, 8))}
        y = {'key': torch.randn((5, 8)), 'key2': torch.randn(())}  # noqa: F841
        return x

    my_function()

    output = string_io.getvalue()
    print(output)
    assert_output(
        output,
        (
            CallEntry(),
            LineEntry(),
            VariableEntry('x', "{'key': tensor<(5, 8), float32, cpu>}"),
            LineEntry(),
            VariableEntry('y', "{'key': tensor<(5, 8), float32, cpu>, 'key2': tensor<(), float32, cpu>}"),
            LineEntry(),
            ReturnEntry(),
            ReturnValueEntry("{'key': tensor<(5, 8), float32, cpu>}"),
        )
    )


def test_recursive_containers():
    string_io = io.StringIO()

    @torchsnooper.snoop(string_io)
    def my_function():
        return [{'key': torch.zeros(5, 6, 7)}]

    my_function()

    output = string_io.getvalue()
    print(output)
    assert_output(
        output,
        (
            CallEntry(),
            LineEntry(),
            ReturnEntry(),
            ReturnValueEntry("[{'key': tensor<(5, 6, 7), float32, cpu>}]"),
        )
    )


def test_return_types():
    string_io = io.StringIO()

    @torchsnooper.snoop(string_io)
    def my_function():
        x = torch.eye(3)
        y = x.max(dim=0)
        y = x.min(dim=0)
        y = x.median(dim=0)
        y = x.mode(dim=0)
        y = x.kthvalue(dim=0, k=1)
        y = x.sort(dim=0)
        y = x.topk(dim=0, k=1)
        y = x.symeig(eigenvectors=True)
        y = x.eig(eigenvectors=True)
        y = x.qr()
        y = x.geqrf()
        y = x.solve(x)
        y = x.slogdet()
        y = x.gels(x)
        y = x.triangular_solve(x)
        y = x.svd()  # noqa: F841
        return x

    my_function()

    output = string_io.getvalue()
    print(output)
    assert_output(
        output,
        (
            CallEntry(),
            LineEntry(),
            VariableEntry('x', "tensor<(3, 3), float32, cpu>"),
            LineEntry(),
            VariableEntry('y', "max(values=tensor<(3,), float32, cpu>, indices=tensor<(3,), int64, cpu>)"),
            LineEntry(),
            VariableEntry('y', "min(values=tensor<(3,), float32, cpu>, indices=tensor<(3,), int64, cpu>)"),
            LineEntry(),
            VariableEntry('y', "median(values=tensor<(3,), float32, cpu>, indices=tensor<(3,), int64, cpu>)"),
            LineEntry(),
            VariableEntry('y', "mode(values=tensor<(3,), float32, cpu>, indices=tensor<(3,), int64, cpu>)"),
            LineEntry(),
            VariableEntry('y', "kthvalue(values=tensor<(3,), float32, cpu>, indices=tensor<(3,), int64, cpu>)"),
            LineEntry(),
            VariableEntry('y', "sort(values=tensor<(3, 3), float32, cpu>, indices=tensor<(3, 3), int64, cpu>)"),
            LineEntry(),
            VariableEntry('y', "topk(values=tensor<(1, 3), float32, cpu>, indices=tensor<(1, 3), int64, cpu>)"),
            LineEntry(),
            VariableEntry('y', "symeig(eigenvalues=tensor<(3,), float32, cpu>, eigenvectors=tensor<(3, 3), float32, cpu>)"),
            LineEntry(),
            VariableEntry('y', "eig(eigenvalues=tensor<(3, 2), float32, cpu>, eigenvectors=tensor<(3, 3), float32, cpu>)"),
            LineEntry(),
            VariableEntry('y', "qr(Q=tensor<(3, 3), float32, cpu>, R=tensor<(3, 3), float32, cpu>)"),
            LineEntry(),
            VariableEntry('y', "geqrf(a=tensor<(3, 3), float32, cpu>, tau=tensor<(3,), float32, cpu>)"),
            LineEntry(),
            VariableEntry('y', "solve(solution=tensor<(3, 3), float32, cpu>, LU=tensor<(3, 3), float32, cpu>)"),
            LineEntry(),
            VariableEntry('y', "slogdet(sign=tensor<(), float32, cpu>, logabsdet=tensor<(), float32, cpu>)"),
            LineEntry(),
            VariableEntry('y', "gels(solution=tensor<(3, 3), float32, cpu>, QR=tensor<(3, 3), float32, cpu>)"),
            LineEntry(),
            VariableEntry('y', "triangular_solve(solution=tensor<(3, 3), float32... cloned_coefficient=tensor<(3, 3), float32, cpu>)"),
            LineEntry(),
            VariableEntry('y', "svd(U=tensor<(3, 3), float32, cpu>, S=tensor<(3,), float32, cpu>, V=tensor<(3, 3), float32, cpu>)"),
            LineEntry(),
            ReturnEntry(),
            ReturnValueEntry("tensor<(3, 3), float32, cpu>"),
        )
    )


def test_numpy_ndarray():
    string_io = io.StringIO()

    @torchsnooper.snoop(string_io)
    def my_function(x):
        return x

    a = numpy.random.randn(5, 6, 7)
    my_function([a, a])

    output = string_io.getvalue()
    print(output)
    assert_output(
        output,
        (
            CallEntry(),
            LineEntry(),
            ReturnEntry(),
            ReturnValueEntry("[ndarray<(5, 6, 7), float64>, ndarray<(5, 6, 7), float64>]"),
        )
    )


def test_nan_and_inf():
    string_io = io.StringIO()

    @torchsnooper.snoop(string_io)
    def my_function():
        x = torch.tensor(math.inf)  # noqa: F841
        y = torch.tensor(math.nan)  # noqa: F841
        z = torch.tensor(1.0)  # noqa: F841
        t = torch.tensor([1.0, math.nan, math.inf])  # noqa: F841

    my_function()

    output = string_io.getvalue()
    print(output)
    assert_output(
        output,
        (
            CallEntry(),
            LineEntry(),
            VariableEntry('x', "tensor<(), float32, cpu, has_inf>"),
            LineEntry(),
            VariableEntry('y', "tensor<(), float32, cpu, has_nan>"),
            LineEntry(),
            VariableEntry('z', "tensor<(), float32, cpu>"),
            LineEntry(),
            VariableEntry('t', "tensor<(3,), float32, cpu, has_nan, has_inf>"),
            ReturnEntry(),
        )
    )
