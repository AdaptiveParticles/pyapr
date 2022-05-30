import pytest
import pyapr
from .helpers import load_test_apr

APR_SHAPES = [
    (1, 1, 63),
    (1, 63, 63),
    (65, 61, 63)
]

PARTICLE_TYPES = [
    pyapr.ByteParticles,
    pyapr.ShortParticles,
    pyapr.FloatParticles,
    pyapr.LongParticles
]


@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_read_write(tmp_path, ndim: int):
    # read data
    apr, parts = load_test_apr(ndim)
    assert apr.shape() == APR_SHAPES[ndim-1]
    assert len(parts) == apr.total_number_particles()

    tree_parts = pyapr.FloatParticles(apr.total_number_tree_particles())
    tree_parts.fill(1)

    # write data to temporary file
    fpath = str(tmp_path / "tmp.apr")
    pyapr.io.write(fpath, apr, parts, tree_parts=tree_parts)

    # read newly written data
    apr2, parts2 = pyapr.io.read(fpath)
    tree_parts2 = pyapr.io.read_particles(fpath, apr, tree_parts, tree=True)

    assert apr2.shape() == apr.shape()
    assert apr2.total_number_particles() == apr.total_number_particles()
    assert parts2 == parts
    assert tree_parts2 == tree_parts

    # read/write APR only
    pyapr.io.write_apr(fpath, apr)
    apr2 = pyapr.io.read_apr(fpath)
    assert apr2.shape() == apr.shape()
    assert apr2.total_number_particles() == apr.total_number_particles()

    # empty path -> nothing written
    pyapr.io.write('', apr, parts)
    pyapr.io.write_apr('', apr)

    with pytest.raises(ValueError):
        pyapr.io.write(fpath, pyapr.APR(), parts)

    with pytest.raises(ValueError):
        pyapr.io.write(fpath, apr, pyapr.FloatParticles())

    with pytest.raises(ValueError):
        pyapr.io.read('file-does-not-exist.apr')

    with pytest.raises(ValueError):
        pyapr.io.read_apr('file-does-not-exist.apr')



@pytest.mark.parametrize("parts_type", PARTICLE_TYPES)
def test_read_write_particles(tmp_path, parts_type):
    # generate particle data
    parts = parts_type(100)
    parts.fill(113)
    parts2 = parts + 15

    # write to file
    fpath = str(tmp_path / "tmp.apr")
    pyapr.io.write_particles(fpath, parts, append=False)
    pyapr.io.write_particles(fpath, parts2, tree=True, parts_name='non_default_name', append=True)

    # read newly written data
    wparts = pyapr.io.read_particles(fpath)
    wparts2 = pyapr.io.read_particles(fpath, tree=True, parts_name='non_default_name')

    assert wparts == parts
    assert wparts2 == parts2

    # empty path -> nothing written
    pyapr.io.write_particles('', parts)

    # test particle name and type detection
    pname = pyapr.io.get_particle_names(fpath)
    assert len(pname) == 1 and pname[0] == 'particles'
    ptype = pyapr.io.get_particle_type(fpath, parts_name=pname[0])
    assert isinstance(pyapr.utils.type_to_particles(ptype), type(parts))

    pname = pyapr.io.get_particle_names(fpath, tree=True)
    assert len(pname) == 1 and pname[0] == 'non_default_name'
    ptype = pyapr.io.get_particle_type(fpath, parts_name=pname[0], tree=True)
    assert isinstance(pyapr.utils.type_to_particles(ptype), type(parts))

    with pytest.raises(ValueError):
        pyapr.io.write_particles(fpath, pyapr.FloatParticles())

    with pytest.raises(ValueError):
        pyapr.io.read_particles('file-does-not-exist.apr')
