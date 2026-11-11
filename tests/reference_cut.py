"""Reference tables and a cut-aware oracle for the viscosity cuts of chunk c4b
(NOTES/chunks/c4b-boundaries-across-jumps/01-plan.md section 9.0).

A viscosity cut is a viscosity jump inside one interval of a fixed node table at a
traced radius ``c`` (``Layers(..., motion="cut")``, NOTES/docs/equations.md section 19).
The cut interval ``(r_a, r_(a+1))`` has the viscosity of the layer below the cut between
``r_a`` and ``c``, the viscosity of the layer above it between ``c`` and ``r_(a+1)``, and
the material (``rho0`` and, under TALA, ``kappa_h``) of the whole interval.

Every builder here takes only the INPUT table of a test (radius, ``rho0`` and the
scalars, as numpy arrays), the boundary radii (with every cut boundary at its cut radius
``c``) and the layer viscosities, and builds a concrete node table with numpy. No builder
reads a profile that ``Layers`` returns, so a reference cannot share an error with the
map under test. Each concrete table equals the cut model exactly for the formulations
and the positions of its row in the plan (section 9.0):

``swapped_pair_table``
    A cut strictly between two nodes, incompressible (FULL or NO_SELF_GRAVITY, either
    background, either boundary class): a pair at ``c`` whose lower member carries
    ``rho0`` of the upper interval node and whose upper member carries ``rho0`` of the
    lower interval node. Both sub-intervals then have the interval-mean density of the
    whole interval, the two shell masses of the gravity integral add up to the shell of
    the interval, and the pair step is the identity.
``node_split_table``
    A cut exactly on an input node that is not a pair: the node duplicated, the lower
    copy with the viscosity below, the upper copy with the viscosity above (exact under
    every formulation: the zero-width step between two copies of one ``rho0`` is the
    identity also under TALA).
``pair_table``
    A cut exactly on a jump pair: the input table, the lower member with the viscosity
    below and the upper member with the viscosity above (the "M-mean" model: the TALA
    pair step takes the mean of the two). Exact under an incompressible formulation,
    where every pair step is the identity.
``triple_table``
    A cut exactly on a jump pair under any formulation: a copy of the upper member
    inserted above it (``side="up"``, the members with the viscosity below, the copy with
    the viscosity above: the cut at ``c = J``, "M-below"), or a copy of the lower member
    inserted above it (``side="lo"``, the lower member below, the copy and the upper
    member above: the limit ``c -> J-``, "M-above"). The package refuses three equal
    radii, so these tables go to the public oracle only.
``geometric_pair_table``
    A cut strictly between two nodes under TALA with NO_SELF_GRAVITY and a uniform
    background: a pair at ``c`` whose members both carry
    ``rho_c = rho_a (rho_b / rho_a)^f_lo`` with ``f_lo = ln(c / r_a) / ln(r_b / r_a)``.
    The two ``kappa_h`` of the sub-intervals are then the fractions of the interval
    value, the pair step is the identity (no density jump), and the interval density
    enters neither the exponent (no self-gravity) nor the gravity (uniform).

The one combination without a concrete table is TALA with self-gravity or hydrostatic
gravity and a cut strictly between nodes. There ``cut_oracle_kernels`` is the reference:
the public mpmath oracle with the step of the cut interval replaced by the product of
its two exact sub-steps (a subclass of the private ``reference_mpmath._Model``; the
oracle module itself is not changed, its sha256 is in the golden manifest). Its node
table comes from ``as_node_table`` of the numpy table without a cut, never from the
profile under test.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass(frozen=True, eq=False)
class RefTable:
    """A concrete node table of numpy arrays and the scalars of the input table.

    Attributes
    ----------
    radius, rho0, eta
        ``(N,)`` float64: radii in m (CMB first), reference density in kg/m^3, viscosity
        in Pa s.
    core_density, core_mass, outer_density, surface_density
        The scalars of the input table (``core_mass`` and ``surface_density`` can be
        ``None``).
    """

    radius: np.ndarray
    rho0: np.ndarray
    eta: np.ndarray
    core_density: float
    core_mass: float | None
    outer_density: float
    surface_density: float | None

    @classmethod
    def from_profile(cls, profile) -> "RefTable":
        """The input table of a concrete ``RadialProfile`` (the profile given to
        ``Layers``, never one that ``Layers`` returns)."""
        f = lambda v: None if v is None else float(v)
        return cls(radius=np.asarray(profile.radius, dtype=np.float64).copy(),
                   rho0=np.asarray(profile.rho0, dtype=np.float64).copy(),
                   eta=np.asarray(profile.eta, dtype=np.float64).copy(),
                   core_density=float(profile.core_density), core_mass=f(profile.core_mass),
                   outer_density=float(profile.outer_density), surface_density=f(profile.surface_density))

    def with_arrays(self, radius, rho0, eta) -> "RefTable":
        """The same scalars with new node arrays."""
        return RefTable(radius=np.asarray(radius, dtype=np.float64), rho0=np.asarray(rho0, dtype=np.float64),
                        eta=np.asarray(eta, dtype=np.float64), core_density=self.core_density,
                        core_mass=self.core_mass, outer_density=self.outer_density,
                        surface_density=self.surface_density)


def layer_viscosity(radius, boundaries, cut_index, eta_layers) -> np.ndarray:
    """The node viscosity of the reference rule of plan 9.0, ``(N,)`` Pa s.

    Parameters
    ----------
    radius
        ``(N,)`` node radii in m of a concrete table, CMB first; two equal neighbours are a
        pair.
    boundaries
        ``(B,)`` boundary radii in m, CMB to surface, with every cut boundary at its cut
        radius.
    cut_index
        The indices of the cut boundaries in ``boundaries``.
    eta_layers
        ``(B - 1,)`` layer viscosities in Pa s, CMB upward.

    Rule: a node strictly between two boundaries takes the viscosity of that layer. At a
    FIXED boundary the table has a pair at its radius: the lower member takes the layer
    below, the upper member the layer above. For a CUT boundary at ``c`` a node is above
    it when ``r_n > c`` (a node on the cut is below it). A fixed boundary on a radius
    that is not a pair of the table raises ``ValueError`` (the reference tables of the
    tests put every fixed boundary on a pair).
    """
    r = np.asarray(radius, dtype=np.float64)
    b = np.asarray(boundaries, dtype=np.float64)
    cuts = set(int(k) for k in cut_index)
    layer = np.zeros(r.size, dtype=int)
    for k in range(1, b.size - 1):
        if k in cuts:
            layer += (r > b[k]).astype(int)
            continue
        at = np.flatnonzero(r == b[k])
        if at.size != 2 or at[1] != at[0] + 1:
            raise ValueError(f"fixed boundary {k} at {b[k]} m is not a pair of the table")
        above = r > b[k]
        above[at[1]] = True                       # the upper member opens the layer above
        layer += above.astype(int)
    return np.asarray(eta_layers, dtype=np.float64)[layer]


def _cut_interval(radius, c) -> int:
    """The 0-based interval ``a`` with ``r_a <= c < r_(a+1)`` (the last node at or below c)."""
    return int(np.flatnonzero(np.asarray(radius) <= c)[-1])


def _insert(table: RefTable, eta, at: int, radii, rho, etas) -> tuple[RefTable, np.ndarray]:
    """Insert nodes before index ``at``; returns the table and ``keep``, the indices of the
    original nodes in it."""
    k = len(radii)
    rad = np.insert(table.radius, at, radii)
    rh = np.insert(table.rho0, at, rho)
    et = np.insert(np.asarray(eta, dtype=np.float64), at, etas)
    keep = np.concatenate([np.arange(at), np.arange(at + k, rad.size)])
    return table.with_arrays(rad, rh, et), keep


def _apply_cuts(table: RefTable, boundaries, eta_layers, cut_index, one_cut) -> tuple[RefTable, np.ndarray]:
    """Apply ``one_cut(table, eta, c, k)`` for every cut boundary ``k``, from the top cut
    down (so that an insertion does not shift the indices of a cut below it), and track
    the indices of the input nodes. ``eta`` is the node viscosity of the input table by
    the rule of ``layer_viscosity``."""
    b = np.asarray(boundaries, dtype=np.float64)
    eta = layer_viscosity(table.radius, b, cut_index, eta_layers)
    keep = np.arange(table.radius.size)
    cur = table.with_arrays(table.radius, table.rho0, eta)
    for k in sorted((int(k) for k in cut_index), reverse=True):
        cur, kk = one_cut(cur, cur.eta, float(b[k]), k)
        keep = kk[keep]
    return cur, keep


def swapped_pair_table(table: RefTable, boundaries, eta_layers, cut_index=(1,)) -> tuple[RefTable, np.ndarray]:
    """The swapped-pair table of every cut strictly between two nodes (module docstring).
    Exact for an incompressible formulation. Returns ``(table, keep)``."""
    def one(cur, eta, c, k):
        a = _cut_interval(cur.radius, c)
        if cur.radius[a] == c:
            raise ValueError(f"cut {k} at {c} m lies on a node; use node_split_table or pair_table")
        rho = cur.rho0
        return _insert(cur, eta, a + 1, [c, c], [rho[a + 1], rho[a]], [eta[a], eta[a + 1]])
    return _apply_cuts(table, boundaries, eta_layers, cut_index, one)


def geometric_pair_table(table: RefTable, boundaries, eta_layers, cut_index=(1,)) -> tuple[RefTable, np.ndarray]:
    """The geometric-pair table of every cut strictly between two nodes (module
    docstring). Exact under TALA with NO_SELF_GRAVITY and a uniform background."""
    def one(cur, eta, c, k):
        a = _cut_interval(cur.radius, c)
        r, rho = cur.radius, cur.rho0
        if r[a] == c:
            raise ValueError(f"cut {k} at {c} m lies on a node")
        f_lo = np.log1p((c - r[a]) / r[a]) / np.log1p((r[a + 1] - r[a]) / r[a])
        rho_c = rho[a] * (rho[a + 1] / rho[a]) ** f_lo
        return _insert(cur, eta, a + 1, [c, c], [rho_c, rho_c], [eta[a], eta[a + 1]])
    return _apply_cuts(table, boundaries, eta_layers, cut_index, one)


def node_split_table(table: RefTable, boundaries, eta_layers, cut_index=(1,)) -> tuple[RefTable, np.ndarray]:
    """The node-split table of every cut exactly on an input node that is not a pair:
    the node duplicated, the lower copy (the input node) with the viscosity below, the
    upper copy with the viscosity above. Exact under every formulation."""
    def one(cur, eta, c, k):
        at = np.flatnonzero(cur.radius == c)
        if at.size != 1:
            raise ValueError(f"cut {k} at {c} m is not on a single node ({at.size} nodes at its radius)")
        n = int(at[0])
        return _insert(cur, eta, n + 1, [c], [cur.rho0[n]], [eta[n + 1]])
    return _apply_cuts(table, boundaries, eta_layers, cut_index, one)


def pair_table(table: RefTable, boundaries, eta_layers, cut_index=(1,)) -> tuple[RefTable, np.ndarray]:
    """The pair table ("M-mean") of every cut exactly on a jump pair: the input table, the
    lower member with the viscosity below and the upper member with the viscosity above.
    Exact for an incompressible formulation."""
    def one(cur, eta, c, k):
        at = np.flatnonzero(cur.radius == c)
        if at.size != 2:
            raise ValueError(f"cut {k} at {c} m is not on a pair ({at.size} nodes at its radius)")
        et = np.asarray(eta, dtype=np.float64).copy()
        et[at[1]] = et[at[1] + 1]                    # the upper member opens the layer above
        return cur.with_arrays(cur.radius, cur.rho0, et), np.arange(cur.radius.size)
    return _apply_cuts(table, boundaries, eta_layers, cut_index, one)


def triple_table(table: RefTable, boundaries, eta_layers, cut_index=(1,),
                 side: Literal["up", "lo"] = "up") -> tuple[RefTable, np.ndarray]:
    """The triple table of every cut exactly on a jump pair (module docstring):
    ``side="up"`` is the cut at the pair ("M-below"), ``side="lo"`` its limit from below
    ("M-above"). Public oracle only (the package refuses three equal radii)."""
    def one(cur, eta, c, k):
        at = np.flatnonzero(cur.radius == c)
        if at.size != 2:
            raise ValueError(f"cut {k} at {c} m is not on a pair ({at.size} nodes at its radius)")
        lo_m, up_m = int(at[0]), int(at[1])
        e_below, e_above = eta[lo_m], eta[up_m + 1]
        if side == "up":
            # Both members below the cut, a copy of the upper member above it.
            et = np.asarray(eta, dtype=np.float64).copy()
            et[lo_m] = et[up_m] = e_below
            return _insert(cur, et, up_m + 1, [c], [cur.rho0[up_m]], [e_above])
        if side == "lo":
            # The lower member below the cut; a copy of the lower member and the upper
            # member above it.
            et = np.asarray(eta, dtype=np.float64).copy()
            et[lo_m], et[up_m] = e_below, e_above
            return _insert(cur, et, lo_m + 1, [c], [cur.rho0[lo_m]], [e_above])
        raise ValueError(f"side must be 'up' or 'lo', got {side!r}")
    return _apply_cuts(table, boundaries, eta_layers, cut_index, one)


def cut_table(table: RefTable, boundaries, eta_layers, cut_index=(1,)) -> tuple[RefTable, list[tuple[int, float]]]:
    """The numpy table WITHOUT a cut (the input table with the node viscosities of the
    reference rule) and the oracle descriptions ``(interval, c)`` of its cuts, for
    ``cut_oracle_kernels``: ``interval`` is the oracle's interval index ``a + 1`` (between
    the nodes ``a`` and ``a + 1``, ``r_a <= c < r_(a+1)``)."""
    b = np.asarray(boundaries, dtype=np.float64)
    eta = layer_viscosity(table.radius, b, cut_index, eta_layers)
    cuts = [(_cut_interval(table.radius, float(b[k])) + 1, float(b[k])) for k in sorted(int(k) for k in cut_index)]
    return table.with_arrays(table.radius, table.rho0, eta), cuts


def as_profile(ref: RefTable):
    """``sj.RadialProfile`` of a reference table (the package solver refuses a triple)."""
    import stokesjx as sj
    return sj.RadialProfile(ref.radius, ref.rho0, ref.eta, core_density=ref.core_density, core_mass=ref.core_mass,
                            outer_density=ref.outer_density, surface_density=ref.surface_density)


def as_node_table(ref: RefTable):
    """``reference_mpmath.NodeTable`` of a reference table: decimal strings from
    ``repr(float)``, so the oracle reads exactly the float64 values of the table."""
    import reference_mpmath as ro
    s = lambda v: tuple(repr(float(x)) for x in np.asarray(v))
    return ro.NodeTable(radius=s(ref.radius), rho0=s(ref.rho0), eta=s(ref.eta),
                        core_density=repr(float(ref.core_density)),
                        core_mass=None if ref.core_mass is None else repr(float(ref.core_mass)),
                        outer_density=repr(float(ref.outer_density)))


def _cut_model_class():
    """The ``reference_mpmath._Model`` subclass with cut intervals (built on first use,
    so that importing this module needs no mpmath)."""
    import mpmath as mp
    import reference_mpmath as ro

    class CutModel(ro._Model):
        """``_Model`` with the step of every cut interval ``i`` (between the nodes ``i - 1``
        and ``i``) replaced by ``P_hi P_lo``: the exact exponentials of the two parts with
        the interval material (interval-mean ``rho0``; under TALA ``kappa_h`` in
        proportion to the widths in ``ln r``), the viscosity of node ``i - 1`` below the cut
        and of node ``i`` above it, each by the exact viscosity similarity (eq. 4.5)."""

        def __init__(self, table, physics, disc, l, cuts):
            self._cuts = tuple(cuts)
            super().__init__(table, physics, disc, l)

        def _steps(self):
            super()._steps()
            for i, cut in self._cuts:
                c = mp.mpf(cut)
                r = self.r
                assert r[i - 1] <= c < r[i] and not self.pair[i], (i, cut)
                h_lo, h_hi = mp.log(c / r[i - 1]), mp.log(r[i] / c)
                rho_rel = self.rho_int[i] / self.rho_s

                def sub(h, e):
                    arg = ro.matrix_a_rel(self.l, 1, rho_rel, self.full) * h
                    if self.kappa_h[i] != 0:
                        arg += ro.matrix_k_rel(1) * (self.kappa_h[i] * h / self.h[i])
                    E = mp.expm(arg, method="taylor")
                    for a_ in range(6):
                        for b_ in range(6):
                            da = 1 if a_ < 2 else e
                            db = 1 if b_ < 2 else e
                            if da is not db:
                                E[a_, b_] = E[a_, b_] * da / db
                    return E
                self.steps[i] = sub(h_hi, self.eta[i] / self.eta_s) * sub(h_lo, self.eta[i - 1] / self.eta_s)
            # _Model.__init__ builds the propagator products after this method, from
            # the steps set here.

    return CutModel


def cut_oracle_kernels(table, physics, cuts, degrees, *, sources=None) -> dict:
    """Kernels of the cut model by the oracle at the degrees ``degrees`` and the source
    nodes ``sources`` (default every node).

    Parameters
    ----------
    table
        ``reference_mpmath.NodeTable`` of the table without a cut (``as_node_table`` of
        ``cut_table``).
    physics
        ``reference_mpmath.Physics``.
    cuts
        ``((interval, c), ...)``: the oracle interval index (``a + 1``) and the cut radius
        in m (a float, converted with ``repr``) of every cut (``cut_table``).
    degrees
        The degrees.

    Returns
    -------
    ``{output: (len(degrees), n_sources)}`` float64 for the four outputs of
    ``reference_mpmath.SI_OUTPUTS``. The oracle's private names (``_Model``,
    ``_solve_local``, ``_outputs``, ``_frame_for``) are used; a change of the oracle
    internals breaks this helper with an attribute error, not with a wrong number.
    """
    import mpmath as mp
    import reference_mpmath as ro
    CutModel = _cut_model_class()
    cuts = tuple((int(i), repr(float(c))) for i, c in cuts)
    sources = tuple(range(table.n_nodes)) if sources is None else tuple(sources)
    res = {k: np.zeros((len(degrees), len(sources))) for k in ro.SI_OUTPUTS}
    for a_, l in enumerate(degrees):
        dps = ro.required_dps(l, table.radius[0], table.radius[-1])
        with mp.workdps(dps):
            m = CutModel(table, physics, ro.Discretisation(), l, cuts)
            fr = ro._frame_for(table, physics, l)
            for b_, j in enumerate(sources):
                sol = ro._solve_local(m, j, fr)
                out = ro._outputs(m, sol, fr, j)
                for k in ro.SI_OUTPUTS:
                    res[k][a_, b_] = float(out[k])
    return res
