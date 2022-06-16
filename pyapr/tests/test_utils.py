import pytest
import pyapr
from .helpers import load_test_apr
import numpy as np


def test_utils():
    assert isinstance(pyapr.utils.type_to_particles(np.uint8), pyapr.ByteParticles)
    assert isinstance(pyapr.utils.type_to_particles(np.uint16), pyapr.ShortParticles)
    assert isinstance(pyapr.utils.type_to_particles(np.uint64), pyapr.LongParticles)
    assert isinstance(pyapr.utils.type_to_particles(np.float32), pyapr.FloatParticles)

    assert isinstance(pyapr.utils.type_to_lazy_particles(np.uint8), pyapr.LazyDataByte)
    assert isinstance(pyapr.utils.type_to_lazy_particles(np.uint16), pyapr.LazyDataShort)
    assert isinstance(pyapr.utils.type_to_lazy_particles(np.uint64), pyapr.LazyDataLong)
    assert isinstance(pyapr.utils.type_to_lazy_particles(np.float32), pyapr.LazyDataFloat)

    assert pyapr.utils.particles_to_type(pyapr.ByteParticles()) is np.uint8
    assert pyapr.utils.particles_to_type(pyapr.ShortParticles()) is np.uint16
    assert pyapr.utils.particles_to_type(pyapr.LongParticles()) is np.uint64
    assert pyapr.utils.particles_to_type(pyapr.FloatParticles()) is np.float32

    assert pyapr.utils.particles_to_type(pyapr.LazyDataByte()) is np.uint8
    assert pyapr.utils.particles_to_type(pyapr.LazyDataShort()) is np.uint16
    assert pyapr.utils.particles_to_type(pyapr.LazyDataLong()) is np.uint64
    assert pyapr.utils.particles_to_type(pyapr.LazyDataFloat()) is np.float32

    with pytest.raises(TypeError):
        pyapr.utils.particles_to_type(np.zeros(5, dtype=np.uint16))

    with pytest.raises(ValueError):
        pyapr.utils.type_to_particles(np.int32)

    with pytest.raises(ValueError):
        pyapr.utils.type_to_lazy_particles(np.float64)

    cuda_build = pyapr.cuda_enabled()
