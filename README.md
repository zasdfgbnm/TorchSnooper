# TorchSnooper

Status:

![PyPI](https://img.shields.io/pypi/v/TorchSnooper.svg)
![PyPI - Downloads](https://img.shields.io/pypi/dm/TorchSnooper.svg)

Checks:

[![Build Status](https://zasdfgbnm.visualstudio.com/TorchSnooper/_apis/build/status/flake8?branchName=master)](https://zasdfgbnm.visualstudio.com/TorchSnooper/_build/latest?definitionId=12&branchName=master)
[![Build Status](https://zasdfgbnm.visualstudio.com/TorchSnooper/_apis/build/status/test?branchName=master)](https://zasdfgbnm.visualstudio.com/TorchSnooper/_build/latest?definitionId=13&branchName=master)
[![Build Status](https://zasdfgbnm.visualstudio.com/TorchSnooper/_apis/build/status/deploy-test-pypi?branchName=master)](https://zasdfgbnm.visualstudio.com/TorchSnooper/_build/latest?definitionId=19&branchName=master)

Deploy (only succeed on tagged commits):

[![Build Status](https://zasdfgbnm.visualstudio.com/TorchSnooper/_apis/build/status/deploy-pypi?branchName=master)](https://zasdfgbnm.visualstudio.com/TorchSnooper/_build/latest?definitionId=14&branchName=master)

Do you want to look at the shape/dtype/etc. of every step of you model, but tired of manually writing prints?

Are you bothered by errors like `RuntimeError: Expected object of scalar type Double but got scalar type Float`, and want to quickly figure out the problem?

TorchSnooper is a [PySnooper](https://github.com/cool-RR/PySnooper) extension that helps you debugging these errors.

To use TorchSnooper, you just use it like using PySnooper. Remember to replace the `pysnooper.snoop` with `torchsnooper.snoop` in your code.

To install:

```
pip install torchsnooper
```

TorchSnooper also support [snoop](https://github.com/alexmojaki/snoop). To use TorchSnooper with snoop, simply execute:
```python
torchsnooper.register_snoop()
```
or
```python
torchsnooper.register_snoop(verbose=True)
```
at the beginning, and use snoop normally.

# Example 1: Monitoring device and dtype

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

What is the problem? Let's snoop it! Decorate our function with `torchsnooper.snoop()`:

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

Run our script, and we will see:

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

# Example 2: Monitoring shape

We are building a linear model

```python
class Model(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(2, 1)

    def forward(self, x):
        return self.layer(x)
```

and we want to fit `y = x1 + 2 * x2 + 3`, so we create a dataset:

```python
x = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
y = torch.tensor([3.0, 5.0, 4.0, 6.0])
```

We train our model on this dataset using SGD optimizer:

```python
model = Model()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
for _ in range(10):
    optimizer.zero_grad()
    pred = model(x)
    squared_diff = (y - pred) ** 2
    loss = squared_diff.mean()
    print(loss.item())
    loss.backward()
    optimizer.step()
```

But unfortunately, the loss does not go down to a low enough number.

What's wrong? Let's snoop it! Putting the training loop inside snoop:

```python
with torchsnooper.snoop():
    for _ in range(100):
        optimizer.zero_grad()
        pred = model(x)
        squared_diff = (y - pred) ** 2
        loss = squared_diff.mean()
        print(loss.item())
        loss.backward()
        optimizer.step()
```

Part of the trace looks like:

```
New var:....... x = tensor<(4, 2), float32, cpu>
New var:....... y = tensor<(4,), float32, cpu>
New var:....... model = Model(  (layer): Linear(in_features=2, out_features=1, bias=True))
New var:....... optimizer = SGD (Parameter Group 0    dampening: 0    lr: 0....omentum: 0    nesterov: False    weight_decay: 0)
22:27:01.024233 line        21     for _ in range(100):
New var:....... _ = 0
22:27:01.024439 line        22         optimizer.zero_grad()
22:27:01.024574 line        23         pred = model(x)
New var:....... pred = tensor<(4, 1), float32, cpu, grad>
22:27:01.026442 line        24         squared_diff = (y - pred) ** 2
New var:....... squared_diff = tensor<(4, 4), float32, cpu, grad>
22:27:01.027369 line        25         loss = squared_diff.mean()
New var:....... loss = tensor<(), float32, cpu, grad>
22:27:01.027616 line        26         print(loss.item())
22:27:01.027793 line        27         loss.backward()
22:27:01.050189 line        28         optimizer.step()
```

We notice that, `y` has shape `(4,)`, but `pred` has shape `(4, 1)`. As a result, `squared_diff` has shape `(4, 4)` due to broadcasting!

This is not the expected behavior, let's fix it: `pred = model(x).squeeze()`, now everything looks good:

```
New var:....... x = tensor<(4, 2), float32, cpu>
New var:....... y = tensor<(4,), float32, cpu>
New var:....... model = Model(  (layer): Linear(in_features=2, out_features=1, bias=True))
New var:....... optimizer = SGD (Parameter Group 0    dampening: 0    lr: 0....omentum: 0    nesterov: False    weight_decay: 0)
22:28:19.778089 line        21     for _ in range(100):
New var:....... _ = 0
22:28:19.778293 line        22         optimizer.zero_grad()
22:28:19.778436 line        23         pred = model(x).squeeze()
New var:....... pred = tensor<(4,), float32, cpu, grad>
22:28:19.780250 line        24         squared_diff = (y - pred) ** 2
New var:....... squared_diff = tensor<(4,), float32, cpu, grad>
22:28:19.781099 line        25         loss = squared_diff.mean()
New var:....... loss = tensor<(), float32, cpu, grad>
22:28:19.781361 line        26         print(loss.item())
22:28:19.781537 line        27         loss.backward()
22:28:19.798983 line        28         optimizer.step()
```

And the final model converge to the desired values.

# Example 3: Monitoring nan and inf

Let's say we have a model that output the likelihood of something. For this example, we will just use a mock:

```python
class MockModel(torch.nn.Module):

    def __init__(self):
        super(MockModel, self).__init__()
        self.unused = torch.nn.Linear(6, 4)

    def forward(self, x):
        return torch.tensor([0.0, 0.25, 0.9, 0.75]) + self.unused(x) * 0.0

model = MockModel()
```

During training, we want to minimize the negative log likelihood, we have code:

```python
for epoch in range(100):
    batch_input = torch.randn(6, 6)
    likelihood = model(batch_input)
    log_likelihood = likelihood.log()
    target = -log_likelihood.mean()
    print(target.item())

    optimizer.zero_grad()
    target.backward()
    optimizer.step()
```

Unfortunately, we first get `inf` then `nan` for our target during training. What's wrong? Let's snoop it:

```python
with torchsnooper.snoop():
    for epoch in range(100):
        batch_input = torch.randn(6, 6)
        likelihood = model(batch_input)
        log_likelihood = likelihood.log()
        target = -log_likelihood.mean()
        print(target.item())

        optimizer.zero_grad()
        target.backward()
        optimizer.step()
```

We will see the part of the output of the snoop looks like:

```
19:30:20.928316 line        18     for epoch in range(100):
New var:....... epoch = 0
19:30:20.928575 line        19         batch_input = torch.randn(6, 6)
New var:....... batch_input = tensor<(6, 6), float32, cpu>
19:30:20.929671 line        20         likelihood = model(batch_input)
New var:....... likelihood = tensor<(6, 4), float32, cpu, grad>
19:30:20.930284 line        21         log_likelihood = likelihood.log()
New var:....... log_likelihood = tensor<(6, 4), float32, cpu, grad, has_inf>
19:30:20.930672 line        22         target = -log_likelihood.mean()
New var:....... target = tensor<(), float32, cpu, grad, has_inf>
19:30:20.931136 line        23         print(target.item())
19:30:20.931508 line        25         optimizer.zero_grad()
19:30:20.931871 line        26         target.backward()
inf
19:30:20.960028 line        27         optimizer.step()
19:30:20.960673 line        18     for epoch in range(100):
Modified var:.. epoch = 1
19:30:20.961043 line        19         batch_input = torch.randn(6, 6)
19:30:20.961423 line        20         likelihood = model(batch_input)
Modified var:.. likelihood = tensor<(6, 4), float32, cpu, grad, has_nan>
19:30:20.961910 line        21         log_likelihood = likelihood.log()
Modified var:.. log_likelihood = tensor<(6, 4), float32, cpu, grad, has_nan>
19:30:20.962302 line        22         target = -log_likelihood.mean()
Modified var:.. target = tensor<(), float32, cpu, grad, has_nan>
19:30:20.962715 line        23         print(target.item())
19:30:20.963089 line        25         optimizer.zero_grad()
19:30:20.963464 line        26         target.backward()
19:30:20.964051 line        27         optimizer.step()
```

Reading the output, we find that, at the first epoch (`epoch = 0`), the `log_likelihood` has `has_inf` flag.
The `has_inf` flag means, your tensor contains `inf` in its value. The same flag appears for `target`.
And at the second epoch, starting from `likelihood`, tensors all have a `has_nan` flag.

From our experience with deep learning, we would guess this is because the first epoch has `inf`, which causes
the gradient to be `nan`, and when parameters are updated, these `nan` propagate to parameters and causing all
future steps to have `nan` result.

Taking a deeper look, we figure out that the `likelihood` contains 0 in it, which leads to `log(0) = -inf`. Changing
the line
```python
log_likelihood = likelihood.log()
```
into
```python
log_likelihood = likelihood.clamp(min=1e-8).log()
```
Problem solved.
