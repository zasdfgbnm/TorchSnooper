import torch
import numpy
import math
import sys
import torchsnooper
from python_toolbox import sys_tools
import re
import snoop
import copy


ansi_escape = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')
default_config = copy.copy(snoop.config)


def func():
    x = torch.tensor(math.inf)
    x = torch.tensor(math.nan)
    x = torch.tensor(1.0, requires_grad=True)
    x = torch.tensor([1.0, math.nan, math.inf])
    x = numpy.zeros((2, 2))
    x = (x, x)


verbose_expect = '''
21:43:42.09 >>> Call to func in File "test_snoop.py", line 16
21:43:42.09   16 | def func():
21:43:42.09   17 |     x = torch.tensor(math.inf)
21:43:42.10 .......... x = tensor(inf)
21:43:42.10 .......... x.shape = ()
21:43:42.10 .......... x.dtype = torch.float32
21:43:42.10 .......... x.device = device(type='cpu')
21:43:42.10 .......... x.requires_grad = False
21:43:42.10 .......... x.has_nan = False
21:43:42.10 .......... x.has_inf = True
21:43:42.10   18 |     x = torch.tensor(math.nan)
21:43:42.10 .......... x = tensor(nan)
21:43:42.10 .......... x.has_nan = True
21:43:42.10 .......... x.has_inf = False
21:43:42.10   19 |     x = torch.tensor(1.0, requires_grad=True)
21:43:42.10 .......... x = tensor(1., requires_grad=True)
21:43:42.10 .......... x.requires_grad = True
21:43:42.10 .......... x.has_nan = False
21:43:42.10   20 |     x = torch.tensor([1.0, math.nan, math.inf])
21:43:42.10 .......... x = tensor([1., nan, inf])
21:43:42.10 .......... x.shape = (3,)
21:43:42.10 .......... x.requires_grad = False
21:43:42.10 .......... x.has_nan = True
21:43:42.10 .......... x.has_inf = True
21:43:42.10   21 |     x = numpy.zeros((2, 2))
21:43:42.10 .......... x = array([[0., 0.],
21:43:42.10                       [0., 0.]])
21:43:42.10 .......... x.shape = (2, 2)
21:43:42.10 .......... x.dtype = dtype('float64')
21:43:42.10   22 |     x = (x, x)
21:43:42.10 .......... x = (array([[0., 0.],
21:43:42.10                       [0., 0.]]), array([[0., 0.],
21:43:42.10                       [0., 0.]]))
21:43:42.10 .......... len(x) = 2
21:43:42.10 <<< Return value from func: None
'''.strip()

terse_expect = '''
21:44:09.63 >>> Call to func in File "test_snoop.py", line 16
21:44:09.63   16 | def func():
21:44:09.63   17 |     x = torch.tensor(math.inf)
21:44:09.63 .......... x = tensor<(), float32, cpu, has_inf>
21:44:09.63   18 |     x = torch.tensor(math.nan)
21:44:09.63 .......... x = tensor<(), float32, cpu, has_nan>
21:44:09.63   19 |     x = torch.tensor(1.0, requires_grad=True)
21:44:09.63 .......... x = tensor<(), float32, cpu, grad>
21:44:09.63   20 |     x = torch.tensor([1.0, math.nan, math.inf])
21:44:09.63 .......... x = tensor<(3,), float32, cpu, has_nan, has_inf>
21:44:09.63   21 |     x = numpy.zeros((2, 2))
21:44:09.63 .......... x = ndarray<(2, 2), float64>
21:44:09.63   22 |     x = (x, x)
21:44:09.63 .......... x = (ndarray<(2, 2), float64>, ndarray<(2, 2), float64>)
21:44:09.63 <<< Return value from func: None
'''.strip()


def clean_output(input_):
    lines = input_.splitlines()[1:]
    lines = [x[len('21:14:00.89 '):] for x in lines]
    return '\n'.join(lines)


def assert_output(verbose, expect):
    torchsnooper.register_snoop(verbose=verbose)
    with sys_tools.OutputCapturer(stdout=False, stderr=True) as output_capturer:
        assert sys.gettrace() is None
        snoop(func)()
        assert sys.gettrace() is None
    output = output_capturer.string_io.getvalue()
    output = ansi_escape.sub('', output)
    assert clean_output(output) == clean_output(expect)
    snoop.config = default_config


def test_verbose():
    assert_output(True, verbose_expect)


def test_terse():
    assert_output(False, terse_expect)
