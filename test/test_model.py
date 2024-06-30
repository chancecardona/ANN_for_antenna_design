# imports for examples
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from models.mlp import MLP, loss_fn, predict

from torchtest import assert_vars_change
from torchtest import assert_vars_same
from torchtest import test_suite

inputs = Variable(torch.randn(3, 20))
targets = Variable(torch.randint(-10, 10, (20,100))).long()
batch = [inputs, targets]
model = MLP(len(inputs), 10)

## Do vars change
# what are the variables?
print('Our list of parameters', [ np[0] for np in model.named_parameters() ])
# do they change after a training step?
#  let's run a train step and see
assert_vars_change(
    model=model,
    loss_fn=loss_fn,
    optim=torch.optim.Adam(model.parameters()),
    batch=batch)

""" FAILURE """
# let's try to break this, so the test fails
params_to_train = [ np[1] for np in model.named_parameters() if np[0] is not 'bias' ]
# run test now
assert_vars_change(
    model=model,
    loss_fn=loss_fn,
    optim=torch.optim.Adam(params_to_train),
    batch=batch)
# YES! bias did not change

## Vars don't change
# What if bias is not supposed to change, by design?
#  test to see if bias remains the same after training
assert_vars_same(
    model=model,
    loss_fn=loss_fn,
    optim=torch.optim.Adam(params_to_train),
    batch=batch,
    params=[('bias', model.bias)]
    )
# it does? good. let's move on

## Output Range
# NOTE : bias is fixed (not trainable)
optim = torch.optim.Adam(params_to_train)
#loss_fn=F.cross_entropy
test_suite(model, loss_fn, optim, batch,
    output_range=(-2, 2),
    test_output_range=True
    )
# seems to work

""" FAILURE """
#  let's tweak the model to fail the test
model.bias = nn.Parameter(2 + torch.randn(2, ))
test_suite(
    model,
    loss_fn, optim, batch,
    output_range=(-1, 1),
    test_output_range=True
    )
# as expected, it fails; yay!

## NaN tensors
""" FAILURE """
model.bias = nn.Parameter(float('NaN') * torch.randn(2, ))

test_suite(
    model,
    loss_fn, optim, batch,
    test_nan_vals=True
    )

## Inf tensors
""" FAILURE """
model.bias = nn.Parameter(float('Inf') * torch.randn(2, ))

test_suite(
    model,
    loss_fn, optim, batch,
    test_inf_vals=True
    )
