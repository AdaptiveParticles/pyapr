from _pyaprwrapper.data_containers import APR, ByteParticles, ShortParticles, FloatParticles
from _pyaprwrapper.segmentation.graphcut import graphcut as _graphcut, graphcut_tiled, get_terminal_energies
from .._common import _check_input
from typing import Union, Optional, Tuple

__allowed_input_types__ = (ShortParticles, FloatParticles)


def graphcut(apr: APR,
             parts: Union[ShortParticles, FloatParticles],
             alpha: float = 1.0,
             beta: float = 1.0,
             intensity_threshold: float = 0.0,
             min_std: float = 0.0,
             std_window_size: int = 7,
             num_levels: int = 2,
             max_factor: float = 3.0,
             avg_num_neighbors: float = 3.3,
             num_tree_smooth: int = 3,
             num_part_smooth: int = 3,
             push_depth: int = 1,
             z_block_size: Optional[int] = None,
             z_ghost_size: int = 16,
             output: Optional[Union[ByteParticles, ShortParticles]] = None) -> Union[ByteParticles, ShortParticles]:
    """
    Compute a binary segmentation by constructing a particle graph and applying the max-flow algorithm [1]
    using the implementation maxflow-v3.04 [2].

    The graph is formed by linking each particle to its face-side neighbours in each dimension.
    Terminal edge costs are set based on a smoothed local minimum estimate and the local standard deviation, while
    neighbour edge costs are set based on intensity differences, resolution level and the local standard deviation.

    Note: The procedure to compute edge costs is highly experimental, and can likely be improved.

    Parameters
    ----------
    apr: APR
        Input APR data structure.
    parts: ShortParticles or FloatParticles
        Input particle values.
    alpha: float
        Scaling factor for terminal edge costs.
    beta: float
        Scaling factor for neighbor edge costs.
    intensity_threshold: float
        Lower threshold on absolute intensity. Particles with intensity below this threshold are considered background.
        (default: 0)
    min_std: float
        Regions where the local standard deviation is lower than this value are considered flat, and both
        terminal energies are set to 0 such that the result depends solely on neighbor information. (default: 0)
    std_window_size: int
        Size of the window used to compute local standard deviation. (default: 7)
    num_levels: int
        Terminal costs are only set for particles at the ``num_levels`` finest resolution levels. (default: 2)
    max_factor: float
        Particles brighter than ``max_factor`` (local) standard deviations from the local minimum estimate
        are considered foreground. (default: 3.0)
    avg_num_neighbors: float
        Controls the amount of memory initially allocated for edges. If memory is being reallocated,
        consider increasing this value. (default: 3.3)
    num_tree_smooth: int
        Number of smoothing iterations to perform on tree particles to compute the local minimum estimate. (default: 3)
    num_part_smooth: int
        Number of smoothing iterations to perform on the APR particles to compute the local minimum estimate. (default: 3)
    push_depth: int
        The local minimum estimate is coarsened by `push_depth` levels. (default: 1)
    z_block_size: int, optional
        (optional) If provided, the operation is applied tile-wise to overlapping blocks in the z dimension to reduce
        memory requirements. The amount of overlap is controlled by ``z_ghost_size``. (default: None)
    z_ghost_size: int
        If ``z_block_size`` is provided, the blocks are padded with ``z_ghost_size`` slices in both directions
        to reduce boundary artifacts between blocks.
    output: ByteParticles, ShortParticles, optional
        (optional) Particle object to which the resulting mask is written. If None, a new ByteParticles object
        is generated.

    Returns
    -------
    output: ByteParticles or ShortParticles
        The binary segmentation mask resulting from the minimum cut.

    References
    ----------
    [1] Yuri Boykov and Vladimir Kolmogorov. "An experimental comparison of min-cut/max-flow algorithms for
    energy minimization in vision." IEEE transactions on pattern analysis and machine intelligence 26.9
    (2004): 1124-1137.

    [2] https://pub.ist.ac.at/~vnk/software.html
    """
    _check_input(apr, parts, __allowed_input_types__)
    if output is None:
        output = ByteParticles()
    assert isinstance(output, (ByteParticles, ShortParticles)), TypeError(f'output must be ByteParticles or '
                                                                          f'ShortParticles, got {type(output)}.')
    if z_block_size is not None and z_block_size > 0:
        graphcut_tiled(apr, parts, output, alpha, beta, avg_num_neighbors, z_block_size, z_ghost_size,
                       num_tree_smooth, num_part_smooth, push_depth, intensity_threshold, min_std,
                       std_window_size, max_factor, num_levels)
    else:
        _graphcut(apr, parts, output, alpha, beta, avg_num_neighbors, num_tree_smooth, num_part_smooth,
                  push_depth, intensity_threshold, min_std, std_window_size, max_factor, num_levels)
    return output


def compute_terminal_costs(apr: APR,
                           parts: Union[ShortParticles, FloatParticles],
                           alpha: float = 1.0,
                           intensity_threshold: float = 0.0,
                           min_std: float = 0.0,
                           std_window_size: int = 7,
                           num_levels: int = 2,
                           max_factor: float = 3.0,
                           num_tree_smooth: int = 3,
                           num_part_smooth: int = 3,
                           push_depth: int = 1) \
        -> Tuple[FloatParticles, FloatParticles]:
    """
    Compute the terminal costs as used in `pyapr.segmentation.graphcut`. Useful for debugging and parameter tuning.

    Parameters
    ----------
    apr: APR
        Input APR data structure.
    parts: ShortParticles or FloatParticles
        Input particle values.
    alpha: float
        Scaling factor for terminal edge costs.
    intensity_threshold: float
        Lower threshold on absolute intensity. Particles with intensity below this threshold are considered background.
        (default: 0)
    min_std: float
        Regions where the local standard deviation is lower than this value are considered flat, and both
        terminal energies are set to 0 such that the result depends solely on neighbor information. (default: 0)
    std_window_size: int
        Size of the window used to compute local standard deviation. (default: 7)
    num_levels: int
        Terminal costs are only set for particles at the ``num_levels`` finest resolution levels. (default: 2)
    max_factor: float
        Particles brighter than ``max_factor`` (local) standard deviations from the local minimum estimate
        are considered foreground. (default: 3.0)
    num_tree_smooth: int
        Number of smoothing iterations to perform on tree particles to compute the local minimum estimate. (default: 3)
    num_part_smooth: int
        Number of smoothing iterations to perform on the APR particles to compute the local minimum estimate. (default: 3)
    push_depth: int
        The local minimum estimate is coarsened by `push_depth` levels. (default: 1)

    Returns
    -------
    foreground: FloatParticles
        The cost on the edge from each particle to the "foreground" terminal.
    background: FloatParticles
        The cost on the edge from each particle to the "background" terminal.
    """

    _check_input(apr, parts, __allowed_input_types__)
    foreground = FloatParticles()
    background = FloatParticles()

    get_terminal_energies(apr, parts, foreground, background, alpha, num_tree_smooth, num_part_smooth, push_depth,
                          intensity_threshold, min_std, std_window_size, max_factor, num_levels)
    return foreground, background
