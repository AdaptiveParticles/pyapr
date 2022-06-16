import pytest
import pyapr
import numpy as np
from numbers import Number


PARTICLE_TYPES = [
    np.uint8,
    np.uint16,
    np.uint64,
    np.float32
]


def _generate_particles(dtype, size, val):
    res = pyapr.utils.type_to_particles(dtype)
    res.resize(size)
    res.fill(dtype(val))
    return res


def _add_inplace(a, b):
    a += b
    return a

def _sub_inplace(a, b):
    a -= b
    return a

def _mul_inplace(a, b):
    a *= b
    return a


def _test_inplace_op(p1, p2, op, gt_op):
    dtype1 = pyapr.utils.particles_to_type(p1)
    if isinstance(p2, Number):
        val = dtype1(gt_op(dtype1(p1[0]), p2))
    else:
        dtype2 = pyapr.utils.particles_to_type(p2)
        val = dtype1(gt_op(dtype1(p1[0]), dtype2(p2[0])))
    res = op(p1, p2)
    assert pyapr.utils.particles_to_type(res) is dtype1
    assert np.allclose(np.array(res), val)

def _test_op(p1, p2, op):
    dtype1 = pyapr.utils.particles_to_type(p1)
    if isinstance(p2, Number):
        out_type = dtype1
        val = out_type(op(dtype1(p1[0]), p2))
    else:
        dtype2 = pyapr.utils.particles_to_type(p2)
        # output type should be float32 if one input is FloatParticles, otherwise the largest input integer type
        out_type = np.float32 if np.float32 in (dtype1, dtype2) else type(dtype1(p1[0]) + dtype2(p2[0]))
        val = out_type(op(dtype1(p1[0]), dtype2(p2[0])))
    res = op(p1, p2)
    assert pyapr.utils.particles_to_type(res) is out_type
    assert np.allclose(np.array(res), val)


@pytest.mark.parametrize("p1_type", PARTICLE_TYPES)
@pytest.mark.parametrize("p2_type", PARTICLE_TYPES)
def test_particle_arithmetic(p1_type, p2_type):
    p1 = _generate_particles(p1_type, 17, 5.3)
    p2 = _generate_particles(p2_type, 17, 2.9)

    # compare two ParticleData objects
    assert p1 != p2
    assert p1 == p1.copy()
    assert p2 == p2.copy()

    # compare ParticleData to scalar
    assert np.all(np.array(p1 > 4))  and not np.any(np.array(p1 > p1_type(5.3)))
    assert np.all(np.array(p1 < 7)) and not np.any(np.array(p1 < p1_type(5.3)))
    assert np.all(np.array(p1 == p1_type(5.3)))
    assert np.all(np.array(p1 >= p1_type(5.3)))
    assert np.all(np.array(p1 <= p1_type(5.3)))
    assert np.all(np.array(p1 != 13))

    # in-place arithmetic operations
    _test_inplace_op(p1, p2, _add_inplace, lambda x, y: x + y)
    _test_inplace_op(p1, p2, _sub_inplace, lambda x, y: x - y)
    _test_inplace_op(p1, p2, _mul_inplace, lambda x, y: x * y)
    _test_inplace_op(p1, 4.2, _add_inplace, lambda x, y: x + y)
    _test_inplace_op(p1, 1.9, _sub_inplace, lambda x, y: x - y)
    _test_inplace_op(p1, 1.8, _mul_inplace, lambda x, y: x * y)

    _test_op(p1, p2, lambda x, y: x + y)
    _test_op(p1, p2, lambda x, y: x - y)
    _test_op(p1, p2, lambda x, y: x * y)
    _test_op(p1, 6.7, lambda x, y: x + y)
    _test_op(p1, 5.9, lambda x, y: x - y)
    _test_op(p1, 2.1, lambda x, y: x * y)
