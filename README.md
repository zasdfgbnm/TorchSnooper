# TorchSnooper

Are you having dtype or device errors like `RuntimeError: Expected object of scalar type Double but got scalar type Float`, and feeling it is troublesome to figure out where in you code the mistake starts?

Are you getting output of unexpected shape, but you don't know where in your function went wrong?

TorchSnooper is a PySnooper plugin that helps you debugging these errors.

To use TorchSnooper, you just install it, and use it like using PySnooper, just replace the `pysnooper.snoop` with `torchsnooper.snoop` in your code.

This project is currently in a very early stage. To install, first install my temporary custom version of PySnooper:

```
pip install --upgrade git+https://github.com/zasdfgbnm/PySnooper.git
```

This temporary fork is mainly to be able to use `custom_repr` before https://github.com/cool-RR/PySnooper/pull/126 get approved and merged.

After installing the custom PySnooper, the next step would be to install TorchSnooper.

```
pip install torchsnooper
```

# Example 1

We're writing a simple function:

```python
def myfunc(mask, x):
    y = torch.zeros(6)
    y.masked_scatter_(mask, x)
    return y
```

and use it like below

```python
mask = torch.tensor([0, 1, 0, 1, 1, 0], device='cuda')
source = torch.tensor([1.0, 2.0, 3.0], device='cuda')
y = myfunc(mask, source)
```

The above code seems to be correct, but unfortunately, we are getting the following error:

```
RuntimeError: Expected object of backend CPU but got backend CUDA for argument #2 'mask'
```

What is the problem? Let's snoop it! Decorate your function with `torchsnooper.snoop()`:

```python
import torch
import torchsnooper

@torchsnooper.snoop()
def myfunc(mask, x):
    y = torch.zeros(6)
    y.masked_scatter_(mask, x)
    return y

mask = torch.tensor([0, 1, 0, 1, 1, 0], device='cuda')
source = torch.tensor([1.0, 2.0, 3.0], device='cuda')
y = myfunc(mask, source)
```

Run your script, and you will see:

```
Starting var:.. mask = tensor<(6,), int64, cuda:0>
Starting var:.. x = tensor<(3,), float32, cuda:0>
21:41:42.941668 call         5 def myfunc(mask, x):
21:41:42.941834 line         6     y = torch.zeros(6)
New var:....... y = tensor<(6,), float32, cpu>
21:41:42.943443 line         7     y.masked_scatter_(mask, x)
21:41:42.944404 exception    7     y.masked_scatter_(mask, x)
```

Now pay attention to the devices of tensors, we notice
```
New var:....... y = tensor<(6,), float32, cpu>
```

Now, it's clear that, the problem is because `y` is a tensor on CPU, that is,
we forget to specify the device on `y = torch.zeros(6)`. Changing it to
`y = torch.zeros(6, device='cuda')`, this problem is solved.

But when running the script again we are getting another error:

```
RuntimeError: Expected object of scalar type Byte but got scalar type Long for argument #2 'mask'
```

Look at the trace above again, pay attention to the dtype of variables, we notice

```
Starting var:.. mask = tensor<(6,), int64, cuda:0>
```

OK, the problem is that, we didn't make the `mask` in the input a byte tensor. Changing the line into
```
mask = torch.tensor([0, 1, 0, 1, 1, 0], device='cuda', dtype=torch.uint8)
```
Problem solved.