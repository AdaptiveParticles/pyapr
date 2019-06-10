import torch
from torch.autograd.gradcheck import get_analytical_jacobian, get_numerical_jacobian, zero_gradients, make_jacobian


def get_analytical_jacobian_params(output, target):
    """
    Computes the analytical jacobian with respect to all tensors in `target`, which can hold some or all of the
    parameters of a module used to compute `output`.

    output: torch.tensor output from which to backpropagate the gradients
    target: torch.tensor or iterable containing torch.tensor for which to compute the gradients
    """

    jacobian = make_jacobian(target, output.numel())
    grad_output = torch.zeros_like(output)
    flat_grad_output = grad_output.view(-1)

    for i in range(flat_grad_output.numel()):
        flat_grad_output.zero_()
        flat_grad_output[i] = 1

        zero_gradients(target)
        torch.autograd.backward(output, grad_output, retain_graph=True)

        for j in range(len(jacobian)):
            jacobian[j][:, i] = target[j].grad.clone().flatten()

    return jacobian


def gradcheck(m, input, eps=1e-6, atol=1e-5, rtol=1e-3, raise_exception=True):
    """
    Compare analytical gradients of a module to numerical gradients computed via central finite differences.

    Disclaimer: this is a modified version of torch.autograd.gradcheck::gradcheck
    (https://pytorch.org/docs/stable/_modules/torch/autograd/gradcheck.html, 2019-06-04) distributed under license
    https://github.com/pytorch/pytorch/blob/master/LICENSE
    :param m:
    :param input:
    :param eps:
    :param atol:
    :param rtol:
    :param raise_exception:
    :return:
    """
    def fail_test(msg):
        if raise_exception:
            raise RuntimeError(msg)
        return False

    def fn(input):
        return m(*input)

    output = fn(input)

    for i, o in enumerate(output):
        if not o.requires_grad:
            continue

        # compare input gradients
        analytical, reentrant, correct_grad_sizes = get_analytical_jacobian(input, o)
        numerical = get_numerical_jacobian(fn, input, eps=eps)

        if not correct_grad_sizes:
            return fail_test('Analytical gradient has incorrect size')

        for j, (a, n) in enumerate(zip(analytical, numerical)):
            if a.numel() != 0 or n.numel() != 0:
                if not torch.allclose(a, n, rtol, atol):
                    return fail_test('Jacobian mismatch for output %d with respect to input %d,\n'
                                     'numerical:%s\nanalytical:%s\n' % (i, j, n, a))

        if not reentrant:
            return fail_test('Backward is not reentrant, i.e., running backward with same '
                             'input and grad_output multiple times gives different values, '
                             'although analytical gradient matches numerical gradient')

        # compare parameter gradients
        pars = [t for t in m.parameters()]

        if pars:
            numerical = get_numerical_jacobian(fn, input, target=pars)
            analytical = get_analytical_jacobian_params(output, pars)

            for j, (a, n) in enumerate(zip(analytical, numerical)):
                if a.numel() != 0 or n.numel() != 0:
                    if not torch.allclose(a, n, rtol, atol):
                        return fail_test('Jacobian mismatch for output %d with respect to parameter %d,\n'
                                         'numerical:%s\nanalytical:%s\n' % (i, j, n, a))
    return True
