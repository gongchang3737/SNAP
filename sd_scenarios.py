# -*- coding: utf-8 -*-
"""Independent delay-scenario sampling for SD search and training.

This module deliberately accepts only an adjacency matrix, its saved hop
distance matrix, a predeclared prior, and explicit random-stream identifiers.
It has no data-loader dependency and cannot inspect a final test realization,
its SD metadata, a propagation source, or observed arrival times.

The ``scale`` values below are the scale parameter of the *untruncated*
normal distribution.  They are not the standard deviation of the resulting
truncated distribution.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.stats import truncnorm

from graph_indexing import (
    validate_adjacency_hop_alignment,
    validate_delay_distance,
)


TRAIN_STREAM = 'train'
SEARCH_STREAM = 'search'
VALIDATION_STREAM = 'validation'
FINAL_TEST_STREAM = 'final-test'
TEST_SMOKE_STREAM = 'test-smoke'


@dataclass(frozen=True)
class DelayBand:
    """One right-closed prior interval for an untruncated-normal scale."""

    name: str
    lower: float
    upper: float
    weight: float = 1.0

    def validate(self, *, mu: float) -> None:
        if not self.name or not self.name.strip():
            raise ValueError('Delay-band name must be non-empty')
        values = np.asarray(
            [self.lower, self.upper, self.weight], dtype=float
        )
        if not np.isfinite(values).all():
            raise ValueError(f'Delay band {self.name!r} contains NaN or Inf')
        if self.lower < 0.0 or self.upper <= self.lower:
            raise ValueError(
                f'Delay band {self.name!r} must satisfy '
                f'0 <= lower < upper; got ({self.lower}, {self.upper}]'
            )
        if self.weight <= 0.0:
            raise ValueError(
                f'Delay band {self.name!r} must have positive weight'
            )
        # A zero lower truncation bound is allowed because a continuous
        # truncated normal assigns exactly zero probability to that endpoint.
        # Negative edge delays are never allowed.
        if mu - 2.0 * self.upper < 0.0:
            raise ValueError(
                f'Delay band {self.name!r} permits scale={self.upper}, whose '
                f'truncation interval starts below zero for mu={mu}'
            )


def _default_bands() -> tuple[DelayBand, ...]:
    return (
        DelayBand('low', 0.0, 0.1),
        DelayBand('medium', 0.1, 0.3),
        DelayBand('high', 0.3, 0.5),
    )


@dataclass(frozen=True)
class DelayPrior:
    """A fixed, equally stratified delay prior declared before evaluation."""

    name: str = 'equal_lmh'
    mu: float = 1.0
    bands: tuple[DelayBand, ...] = field(default_factory=_default_bands)

    def validate(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError('Delay-prior name must be non-empty')
        if not np.isfinite(self.mu) or self.mu <= 0.0:
            raise ValueError(f'Delay-prior mu must be positive; got {self.mu}')
        if not self.bands:
            raise ValueError('Delay prior must contain at least one band')
        names = [band.name for band in self.bands]
        if len(set(names)) != len(names):
            raise ValueError(f'Delay-band names must be unique; got {names}')
        for band in self.bands:
            band.validate(mu=self.mu)
        weights = np.asarray([band.weight for band in self.bands], dtype=float)
        if not np.allclose(weights, weights[0], rtol=0.0, atol=1e-12):
            raise ValueError(
                'The current stratified sampler requires equal band weights'
            )

    def fingerprint(self) -> str:
        self.validate()
        parts = [self.name, float(self.mu).hex()]
        for band in self.bands:
            parts.extend((
                band.name,
                float(band.lower).hex(),
                float(band.upper).hex(),
                float(band.weight).hex(),
            ))
        return hashlib.sha256('|'.join(parts).encode('utf-8')).hexdigest()


DEFAULT_DELAY_PRIOR = DelayPrior()

CONDITIONAL_DELAY_BANDS = {
    band.name: band for band in _default_bands()
}


def conditional_delay_prior(
    sd_level: str,
    *,
    mu: float = 1.0,
) -> DelayPrior:
    """Return the mutually exclusive prior declared for one known SD level.

    Only the categorical level is accepted.  In particular, this interface
    cannot consume the actual SD value or delay matrix of a test realization.
    """
    normalized = str(sd_level).strip().lower()
    if normalized not in CONDITIONAL_DELAY_BANDS:
        raise ValueError(
            f'Unknown conditional SD level {sd_level!r}; expected one of '
            f'{sorted(CONDITIONAL_DELAY_BANDS)}'
        )
    band = CONDITIONAL_DELAY_BANDS[normalized]
    return DelayPrior(
        name=f'conditional_{normalized}',
        mu=float(mu),
        bands=(band,),
    )


@dataclass(frozen=True)
class DelayScenarioPool:
    """An immutable-by-contract set of independently simulated scenarios."""

    delay_distances: tuple[np.ndarray, ...]
    edge_weight_matrices: tuple[np.ndarray, ...]
    raw_normal_scales: tuple[float, ...]
    band_names: tuple[str, ...]
    topology_id: str
    topology_fingerprint: str
    prior_name: str
    prior_fingerprint: str
    stream: str
    base_seed: int
    pool_fingerprint: str
    generation_seconds: float
    no_disturbance: bool = False

    @property
    def count(self) -> int:
        return len(self.delay_distances)


@dataclass(frozen=True)
class SearchScoreContext:
    """Inputs allowed to influence an SD-SGBS deployment score.

    The context intentionally contains no final-test realization, observed
    arrival time, actual source, or test-realization metadata.
    """

    hop_distance: np.ndarray
    source_indices: tuple[int, ...]
    scenario_pool: DelayScenarioPool
    spread_method: str
    locate_strategy: str
    config_fingerprint: str

    @property
    def cache_scope(self) -> tuple[str, str]:
        return self.scenario_pool.pool_fingerprint, self.config_fingerprint


def _readonly_copy(values: np.ndarray) -> np.ndarray:
    copied = np.array(values, dtype=float, copy=True, order='C')
    copied.setflags(write=False)
    return copied


def _topology_fingerprint(adjacency: np.ndarray) -> str:
    topology = np.asarray(adjacency > 0, dtype=np.uint8, order='C')
    digest = hashlib.sha256()
    digest.update(str(topology.shape).encode('ascii'))
    digest.update(topology.tobytes(order='C'))
    return digest.hexdigest()


def _stable_generator(*parts: object) -> np.random.Generator:
    identity = '|'.join(str(part) for part in parts)
    digest = hashlib.sha256(identity.encode('utf-8')).digest()
    seed = int.from_bytes(digest[:16], byteorder='little', signed=False)
    return np.random.default_rng(seed)


def _sample_right_closed_uniform(
    rng: np.random.Generator,
    lower: float,
    upper: float,
) -> float:
    # 1-random() lies in (0, 1], matching the declared (lower, upper] band.
    return float(lower + (upper - lower) * (1.0 - rng.random()))


def _sample_one_scenario(
    adjacency: np.ndarray,
    *,
    mu: float,
    scale: float,
    rng: np.random.Generator,
    context: str,
) -> tuple[np.ndarray, np.ndarray]:
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f'{context}: normal scale must be positive; got {scale}')
    lower = mu - 2.0 * scale
    upper = mu + 2.0 * scale
    if lower < 0.0 or not np.isfinite([lower, upper]).all():
        raise ValueError(
            f'{context}: invalid truncation interval [{lower}, {upper}]'
        )

    rows, columns = np.nonzero(np.triu(adjacency > 0, k=1))
    if rows.size == 0:
        raise ValueError(f'{context}: topology has no edges')
    delays = np.asarray(
        truncnorm.rvs(
            (lower - mu) / scale,
            (upper - mu) / scale,
            loc=mu,
            scale=scale,
            size=rows.size,
            random_state=rng,
        ),
        dtype=float,
    )
    if not np.isfinite(delays).all() or np.any(delays <= 0.0):
        raise ValueError(
            f'{context}: sampled edge delays must be finite and strictly positive'
        )
    if np.any(delays < lower) or np.any(delays > upper):
        raise ValueError(f'{context}: sampled delay escaped truncation bounds')

    edge_weights = np.zeros_like(adjacency, dtype=float)
    edge_weights[rows, columns] = delays
    edge_weights[columns, rows] = delays
    delay_distance = np.asarray(
        dijkstra(
            csr_matrix(edge_weights),
            directed=False,
            return_predecessors=False,
        ),
        dtype=float,
    )
    if not np.isfinite(delay_distance).all():
        raise ValueError(f'{context}: generated delay distances are not finite')
    return edge_weights, delay_distance


def sample_delay_scenarios(
    adjacency: np.ndarray,
    hop_distance: np.ndarray,
    *,
    topology_id: str,
    count: int,
    seed: int,
    stream: str,
    prior: DelayPrior = DEFAULT_DELAY_PRIOR,
    no_disturbance: bool = False,
) -> DelayScenarioPool:
    """Generate a reproducible scenario pool without touching global RNG.

    ``topology_id`` is a stable logical identifier (for example
    ``BA/20_nodes/graph_7``).  The binary adjacency fingerprint is included as
    an additional guard, so reusing the identifier for a different topology
    still produces a separate stream.
    """
    if not isinstance(topology_id, str) or not topology_id.strip():
        raise ValueError('topology_id must be a non-empty string')
    if not isinstance(stream, str) or not stream.strip():
        raise ValueError('stream must be a non-empty string')
    if isinstance(count, bool) or not isinstance(count, (int, np.integer)):
        raise TypeError(f'count must be an integer; got {type(count).__name__}')
    count = int(count)
    if count <= 0:
        raise ValueError(f'count must be positive; got {count}')
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError(f'seed must be an integer; got {type(seed).__name__}')
    seed = int(seed)

    adjacency = np.asarray(adjacency, dtype=float)
    hop_distance = np.asarray(hop_distance, dtype=float)
    context = f'SD scenarios for {topology_id}'
    validate_adjacency_hop_alignment(
        adjacency,
        hop_distance,
        context=context,
    )
    prior.validate()
    topology_fingerprint = _topology_fingerprint(adjacency)
    prior_fingerprint = prior.fingerprint()

    if not no_disturbance and count % len(prior.bands) != 0:
        raise ValueError(
            f'count must be a multiple of {len(prior.bands)} for equal '
            f'stratification; got {count}'
        )

    started = time.perf_counter()
    distances: list[np.ndarray] = []
    edge_matrices: list[np.ndarray] = []
    scales: list[float] = []
    band_names: list[str] = []

    if no_disturbance:
        unit_edges = np.asarray(adjacency > 0, dtype=float)
        for _ in range(count):
            distances.append(_readonly_copy(hop_distance))
            edge_matrices.append(_readonly_copy(unit_edges))
            scales.append(0.0)
            band_names.append('none')
    else:
        scenarios_per_band = count // len(prior.bands)
        scenario_index = 0
        for band in prior.bands:
            for _ in range(scenarios_per_band):
                rng = _stable_generator(
                    'snap-sd-scenario-v1',
                    seed,
                    stream,
                    topology_id,
                    topology_fingerprint,
                    prior_fingerprint,
                    scenario_index,
                )
                scale = _sample_right_closed_uniform(
                    rng, band.lower, band.upper
                )
                edge_weights, delay_distance = _sample_one_scenario(
                    adjacency,
                    mu=prior.mu,
                    scale=scale,
                    rng=rng,
                    context=f'{context} scenario {scenario_index}',
                )
                validate_delay_distance(
                    delay_distance,
                    hop_distance,
                    sd=scale,
                    mu=prior.mu,
                    context=f'{context} scenario {scenario_index}',
                )
                edge_matrices.append(_readonly_copy(edge_weights))
                distances.append(_readonly_copy(delay_distance))
                scales.append(scale)
                band_names.append(band.name)
                scenario_index += 1

    pool_identity: Sequence[object] = (
        'snap-sd-pool-v1',
        topology_id,
        topology_fingerprint,
        prior_fingerprint,
        stream,
        seed,
        count,
        int(no_disturbance),
        *(float(scale).hex() for scale in scales),
    )
    pool_digest = hashlib.sha256(
        '|'.join(str(part) for part in pool_identity).encode('utf-8')
    )
    # Include the realized matrices so the identifier remains safe as a cache
    # scope even if a numerical dependency changes its sampling details.
    for matrix in distances:
        pool_digest.update(np.asarray(matrix, dtype='<f8').tobytes(order='C'))
    pool_fingerprint = pool_digest.hexdigest()
    return DelayScenarioPool(
        delay_distances=tuple(distances),
        edge_weight_matrices=tuple(edge_matrices),
        raw_normal_scales=tuple(scales),
        band_names=tuple(band_names),
        topology_id=topology_id,
        topology_fingerprint=topology_fingerprint,
        prior_name=prior.name,
        prior_fingerprint=prior_fingerprint,
        stream=stream,
        base_seed=seed,
        pool_fingerprint=pool_fingerprint,
        generation_seconds=time.perf_counter() - started,
        no_disturbance=no_disturbance,
    )


def delay_prior_from_preset(
    name: str = 'equal_lmh',
    *,
    mu: float = 1.0,
) -> DelayPrior:
    """Resolve a declared prior without consulting any test metadata."""
    if name != 'equal_lmh':
        raise ValueError(
            f'Unknown delay-prior preset {name!r}; expected "equal_lmh"'
        )
    return DelayPrior(name=name, mu=float(mu))


def build_search_score_context(
    adjacency: np.ndarray,
    hop_distance: np.ndarray,
    *,
    topology_id: str,
    count: int = 12,
    seed: int = 42,
    prior: DelayPrior = DEFAULT_DELAY_PRIOR,
    spread_method: str = 'snowball',
    locate_strategy: str = 'TRBS',
) -> SearchScoreContext:
    """Build one shared independent scenario pool for one topology search."""
    if spread_method != 'snowball' or locate_strategy != 'TRBS':
        raise ValueError(
            'SD-SGBS Monte Carlo scoring currently requires snowball/TRBS'
        )
    pool = sample_delay_scenarios(
        adjacency,
        hop_distance,
        topology_id=topology_id,
        count=count,
        seed=seed,
        stream=SEARCH_STREAM,
        prior=prior,
    )
    hop_copy = _readonly_copy(hop_distance)
    sources = tuple(range(hop_copy.shape[0]))
    digest = hashlib.sha256()
    digest.update(pool.pool_fingerprint.encode('ascii'))
    digest.update(spread_method.encode('utf-8'))
    digest.update(locate_strategy.encode('utf-8'))
    digest.update(np.asarray(sources, dtype='<i8').tobytes(order='C'))
    digest.update(np.asarray(hop_copy, dtype='<f8').tobytes(order='C'))
    return SearchScoreContext(
        hop_distance=hop_copy,
        source_indices=sources,
        scenario_pool=pool,
        spread_method=spread_method,
        locate_strategy=locate_strategy,
        config_fingerprint=digest.hexdigest(),
    )

