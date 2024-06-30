# imports for examples
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from models.mlp import MLP, get_device
#from neuro_tf_utils import loss_fn, predict

import pytest
from torchtest import assert_vars_change
from torchtest import assert_vars_same
from torchtest import test_suite as ts2

def loss_poles(actual_coeffs, pred_coeffs):
    #pred_coeffs = pred_coeffs[0]
    loss = torch.norm(actual_coeffs[0] - pred_coeffs[0], p=2) + \
           torch.norm(actual_coeffs[1:] - pred_coeffs[1:], p=2)
    return loss

@pytest.fixture
def setup_tests():
    device = get_device()
    # Batch size of 1
    inputs = Variable(torch.abs(torch.randn(1, 3) * 20))[0]
    target_d = Variable(torch.randn(1) * 100)
    target_poles = Variable(torch.randn(2, 10))
    complex_poles = torch.complex(target_poles[0], target_poles[1])
    complex_targets = torch.cat((target_d, complex_poles), dim=0)
    batch = [inputs, complex_targets]
    model = MLP(3, 10) # FF B of 1x20
    model.eval()
    params_to_train = [ named_p[1] for named_p in model.named_parameters() if "bias" not in named_p[0] ]
    return model, device, batch, params_to_train


## Do vars change after a training step?
def test_vars_change_succeeds(setup_tests):
    model, device, batch, params_to_train = setup_tests
    print("Starting test: VARS CHANGE")
    # what are the variables?
    print('Our list of parameters', [ named_p[0] for named_p in model.named_parameters() ])
    assert_vars_change(
        model=model,
        loss_fn=loss_poles,
        optim=torch.optim.Adam(model.parameters()),
        batch=batch,
        device=device)

def test_vars_change_fails(setup_tests):
    model, device, batch, params_to_train = setup_tests
    with pytest.raises(Exception) as e_info:
        # let's try to break this by not including any bias in the optimizer, so the test fails
        param_names_to_train = [ named_p[0] for named_p in model.named_parameters() if "bias" not in named_p[0] ]
        print('Our list of parameters (should not include bias)', param_names_to_train)
        #params_to_train = [ named_p[1] for named_p in model.named_parameters() if "bias" not in named_p[0] ]
        # run test now
        assert_vars_change(
            model=model,
            loss_fn=loss_poles,
            optim=torch.optim.Adam(params_to_train),
            batch=batch,
            device=device)

## Vars don't change
def test_vars_same_succeeds(setup_tests):
    # What if bias is not supposed to change, by design?
    # test to see if bias remains the same after training
    model, device, batch, params_to_train = setup_tests
    print("Starting test: VARS DONT CHANGE")
    with pytest.raises(Exception) as e_info:
        assert_vars_same(
            model=model,
            loss_fn=loss_poles,
            optim=torch.optim.Adam(params_to_train),
            batch=batch,
            params=[('bias', model.bias)],
            device=device
        )

## Output Range
def test_output_range_succeeds(setup_tests):
    model, device, batch, params_to_train = setup_tests
    with pytest.raises(Exception) as e_info:
        print("Starting test: OUTPUT RANGE")
        # NOTE : bias is fixed (not trainable)
        optim = torch.optim.Adam(params_to_train)
        #loss_fn=F.cross_entropy
        ts2(
            model=model, 
            loss_fn=loss_poles, 
            optim=optim, 
            batch=batch,
            output_range=(-2, 2),
            test_output_range=True,
            device=device
        )

def test_output_range_fails(setup_tests):
    """ FAILURE """
    #  let's tweak the model to fail the test
    model, device, batch, params_to_train = setup_tests
    with pytest.raises(Exception) as e_info:
        optim = torch.optim.Adam(params_to_train)
        model.bias = nn.Parameter(2 + torch.randn(2, ))
        ts2(
            model=model,
            loss_fn=loss_poles, 
            optim=optim, 
            batch=batch,
            output_range=(-1, 1),
            test_output_range=True,
            device=device
        )

## NaN tensors
def test_nan_fails(setup_tests):
    """ FAILURE """
    model, device, batch, params_to_train = setup_tests
    with pytest.raises(Exception):
        print("Starting test: NaN")
        optim = torch.optim.Adam(model.parameters())
        model.layers[0].bias = nn.Parameter(float('NaN') * torch.randn(2, )) 
        ts2(
            model=model,
            loss_fn=loss_poles, 
            optim=optim, 
            batch=batch,
            test_nan_vals=True,
            device=device
        )
   
## Inf tensors
def test_inf_fails(setup_tests):
    """ FAILURE """
    model, device, batch, params_to_train = setup_tests
    with pytest.raises(Exception):
        print("Starting test: Inf")
        optim = torch.optim.Adam(model.parameters())
        model.layers[0].bias = nn.Parameter(float('Inf') * torch.randn(2, ))        
        ts2(
            model=model,
            loss_fn=loss_poles,
            optim=optim,
            batch=batch,
            test_inf_vals=True,
            device=device
        )
