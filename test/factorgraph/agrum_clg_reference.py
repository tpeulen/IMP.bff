"""aGrUM's linear-Gaussian (CLG) inference, transcribed without pyagrum.

The A/B reference for `IMP.bff.InferenceCanonicalForm`,
`IMP.bff.InferenceGaussianElimination` and the ``"weighted"`` elimination
order of `IMP.bff.InferenceFactorGraph` (okf/prds/prd-151.md). pyagrum is
installed nowhere here, and the CLG package imports it at module level, so the
computation is transcribed as literally as a pyagrum-free rewrite allows, with
line references into ../chisurf/junk/aGrUM at 9f2905b60:

* `CanonicalForm` -- wrappers/pyagrum/pyLibs/clg/canonicalForm.py, whole class:
  `fromCLG` 92-138, `toGaussian` 140-155, `_sort` 200-210, `augment` 212-259,
  `__mul__` 261-287, `__truediv__` 289-314, `marginalize` 327-369,
  `reduce` 371-415. Transcribed verbatim; only the numpy imports are shared.
* the CLG's factors -- clg/CLG.py 640-653 (`_build_canonical_forms`): one
  `fromCLG` per node over the node and its parents, `sigma` a standard
  deviation.
* `CLGVariableElimination.canonicalPosterior` -- clg/variableElimination.py
  124-168, with `_sum_product_ve` 192-225 and `_sum_product_eliminate_var`
  227-259, including their in-place mutation of the factor list.
* the elimination order -- `JunctionTreeGenerator.eliminationOrder(dag)`
  (wrappers/pyagrum/extensions/JunctionTreeGenerator.h 67-69, 127-141): the
  DAG's moral graph, every domain size 2, `DefaultTriangulation`, whose
  sequence is `DefaultEliminationSequenceStrategy::nextNodeToEliminate`
  (defaultEliminationSequenceStrategy.cpp 159-189) over a `SimplicialSet`
  (simplicialSet.cpp): simplicial nodes first (any weight), then almost
  simplicial nodes whose log weight is <= log treewidth + log(1 + 0.0), then
  the minimum log weight (Kjaerulff). The log weight of a node is the sum of the
  log domain sizes over the node and its neighbours (`_initialize_` 554-569);
  the log treewidth starts at the minimum weight and becomes the maximum weight
  eliminated (`eraseClique` 325). The quasi-simplicial list is never populated:
  `_updateList_` 463 divides two `Size`s, which is >= 0.99 only for a simplicial
  node, handled on 438. Rewritten non-incrementally, since only the decisions
  matter. **Ties** are broken on the node id here; aGrUM's follow its hash
  tables and priority queue, which no transcription can reproduce.

Brute force, for the exactness half: the joint of a linear-Gaussian BN is
`x = (I - B)^-1 (mu + e)`, `e ~ N(0, diag(sigma^2))`, so its mean and
covariance are closed-form, and every posterior is a textbook Gaussian
conditional of that joint.

Deliberately NOT importing IMP.bff (prototyping rule, tpeulen 2026-09-14): this
is the reference, the C++ is what is checked against it.

Run with the arm64 env's python; writes
test/factorgraph/data/clg_agrum_fixture.json.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from numpy.linalg import det, inv


# --- canonicalForm.py, verbatim --------------------------------------------


class CanonicalForm:
    def __init__(self, scope=[], K=[], h=[], g=0):  # 55-70
        self._scope = scope
        self._size = len(self._scope)
        self._K = np.reshape(K, (len(K), len(K)))
        self._h = np.reshape(h, (len(h), 1))
        self._g = g
        self._sort()

    def __contains__(self, item):  # 78-79
        return item in self._scope

    @classmethod
    def fromCLG(cls, variable, parents, mu, sigma, B):  # 92-138
        gamma = sigma**2
        if len(parents) == 0:
            scope = [variable]
            h = [mu / gamma]
            K = [[1 / gamma]]
        else:
            scope = [variable] + parents
            B = np.reshape(B, (len(B), 1))
            h = mu / gamma * np.insert(-B, 0, 1, axis=0)
            K = 1 / gamma * np.block([[1, -B.T], [-B, np.dot(B, B.T)]])
        g = -(mu**2 / gamma + np.log(2 * np.pi * gamma)) / 2
        cf = cls(scope, K, h, g)
        cf._sort()
        return cf

    def toGaussian(self):  # 140-155
        Sigma = inv(self._K)
        mu = np.dot(Sigma, self._h)
        return self._scope, mu, Sigma

    @staticmethod
    def _permute_matrix(matrix, permutation):  # 157-177
        matrix[:, :] = matrix[permutation, :]
        matrix[:, :] = matrix[:, permutation]
        return matrix

    @staticmethod
    def _permute_vector(vector, permutation):  # 179-198
        vector[:, :] = vector[permutation, :]
        return vector

    def _sort(self):  # 200-210
        sorted_scope = sorted(self._scope)
        if self._scope != sorted_scope:
            permutation = [self._scope.index(e) for e in sorted_scope]
            self._K = CanonicalForm._permute_matrix(self._K, permutation)
            self._h = CanonicalForm._permute_vector(self._h, permutation)
            self._scope = sorted_scope

    def augment(self, variables):  # 212-259
        new_scope = list(set(self._scope + variables))
        new_scope.sort()
        new_variables = list(set(new_scope) - set(self._scope))
        new_variables.sort()
        if len(self._scope) == 0:
            augmented_K = np.zeros((len(new_variables), len(new_variables)))
            augmented_h = np.zeros((len(new_variables), 1))
        else:
            augmented_K = self._K
            augmented_h = self._h
            positions = [new_scope.index(v) for v in new_variables]
            fv_new_position = new_scope.index(self._scope[0])
            for position in positions:
                if position < fv_new_position:
                    augmented_K = np.concatenate((np.zeros((len(augmented_K), 1)), augmented_K), axis=1)
                    augmented_K = np.concatenate((np.zeros((1, len(augmented_K) + 1)), augmented_K), axis=0)
                    augmented_h = np.concatenate(([[0]], augmented_h), axis=0)
                elif position < len(augmented_K):
                    augmented_K = np.insert(augmented_K, [position], np.zeros((len(augmented_K), 1)), axis=1)
                    augmented_K = np.insert(augmented_K, [position], np.zeros((1, len(augmented_K) + 1)), axis=0)
                    augmented_h = np.insert(augmented_h, position, 0, axis=0)
                else:
                    augmented_K = np.concatenate((augmented_K, np.zeros((len(augmented_K), 1))), axis=1)
                    augmented_K = np.concatenate((augmented_K, np.zeros((1, len(augmented_K) + 1))), axis=0)
                    augmented_h = np.concatenate((augmented_h, [[0]]), axis=0)
        return CanonicalForm(new_scope, augmented_K, augmented_h, self._g)

    def __mul__(self, other):  # 261-287
        scope = list(set(self._scope + other._scope))
        scope.sort()
        augmented_self = self.augment(other._scope)
        augmented_other = other.augment(self._scope)
        K = augmented_self._K + augmented_other._K
        h = augmented_self._h + augmented_other._h
        g = augmented_self._g + augmented_other._g
        return CanonicalForm(scope, K, h, g)

    def __truediv__(self, other):  # 289-314
        scope = list(set(self._scope + other._scope))
        scope.sort()
        augmented_self = self.augment(other._scope)
        augmented_other = other.augment(self._scope)
        K = augmented_self._K - augmented_other._K
        h = augmented_self._h - augmented_other._h
        g = augmented_self._g - augmented_other._g
        return CanonicalForm(scope, K, h, g)

    def marginalize(self, variables):  # 327-369
        if len(variables) == 0:
            return self
        if not all(variable in self._scope for variable in variables):
            raise ValueError("All the variables to marginalize are not in the scope of the canonical form")
        new_scope = list(set(self._scope) - set(variables))
        variables.sort()
        new_scope.sort()
        ns_positions = [self._scope.index(v) for v in new_scope]
        v_positions = [self._scope.index(v) for v in variables]
        K_xx = self._K[np.ix_(ns_positions, ns_positions)]
        K_yy = self._K[np.ix_(v_positions, v_positions)]
        K_xy = self._K[np.ix_(ns_positions, v_positions)]
        K_yx = self._K[np.ix_(v_positions, ns_positions)]
        h_x = self._h[ns_positions]
        h_y = self._h[v_positions]
        K_yy_inv = inv(K_yy)
        K = K_xx - np.dot(np.dot(K_xy, K_yy_inv), K_yx)
        h = h_x - np.dot(np.dot(K_xy, K_yy_inv), h_y)
        g = self._g + 0.5 * (
            len(variables) * np.log(2 * np.pi) - np.log(det(K_yy)) + np.dot(np.dot(np.transpose(h_y), K_yy_inv), h_y)
        )
        return CanonicalForm(new_scope, K, h, g[0][0])

    def reduce(self, evidences):  # 371-415
        evidences = {var: evidences[var] for var in self._scope if var in evidences}
        if len(evidences) == 0:
            return self
        variables = list(evidences.keys())
        values = list(evidences.values())
        variables, values = map(list, zip(*sorted(zip(variables, values))))
        values = np.reshape(values, (len(values), 1))
        new_scope = list(set(self._scope) - set(variables))
        new_scope.sort()
        ns_positions = [self._scope.index(v) for v in new_scope]
        v_positions = [self._scope.index(v) for v in variables]
        K_xx = self._K[np.ix_(ns_positions, ns_positions)]
        K_yy = self._K[np.ix_(v_positions, v_positions)]
        K_xy = self._K[np.ix_(ns_positions, v_positions)]
        h_x = self._h[ns_positions]
        h_y = self._h[v_positions]
        h = h_x - np.dot(K_xy, values)
        g = self._g + np.dot(np.transpose(h_y), values) - 0.5 * np.dot(np.dot(np.transpose(values), K_yy), values)
        return CanonicalForm(new_scope, K_xx, h, g[0][0])


# --- the elimination order (JunctionTreeGenerator -> SimplicialSet) --------


def elimination_order(n, edges, log_domain_sizes):
    """Node ids in the order aGrUM's default triangulation eliminates them."""
    adj = [set() for _ in range(n)]
    for a, b in edges:
        if a != b:
            adj[a].add(b)
            adj[b].add(a)
    alive = set(range(n))

    def weight(v):  # simplicialSet.cpp 559-561
        return log_domain_sizes[v] + sum(log_domain_sizes[u] for u in adj[v])

    def adjacent_pairs(v):  # _nb_adjacent_neighbours_
        nb = sorted(adj[v])
        return sum(1 for i, a in enumerate(nb) for b in nb[i + 1:] if b in adj[a])

    def is_simplicial(v):  # _updateList_ 438
        k = len(adj[v])
        return adjacent_pairs(v) == k * (k - 1) // 2

    def is_almost_simplicial(v):  # _updateList_ 448-450
        k = len(adj[v])
        nb_almost = (k - 1) * (k - 2) // 2
        nb = adjacent_pairs(v)
        return any(nb_almost == nb - len(adj[u] & adj[v]) for u in adj[v])

    log_tree_width = min((weight(v) for v in alive), default=0.0)  # 561-568
    log_threshold = math.log(1.0 + 0.0)  # GUM_WEIGHT_THRESHOLD
    order = []
    while alive:
        ranked = sorted(alive, key=lambda v: (weight(v), v))
        simplicial = [v for v in ranked if is_simplicial(v)]
        almost = [v for v in ranked if not is_simplicial(v) and is_almost_simplicial(v)]
        if simplicial:
            best = simplicial[0]
        elif almost and weight(almost[0]) <= log_tree_width + log_threshold:
            best = almost[0]
        else:
            best = ranked[0]  # Kjaerulff: the minimum log weight
        log_tree_width = max(log_tree_width, weight(best))  # eraseClique 325
        nb = list(adj[best])
        for i, a in enumerate(nb):  # makeClique
            for b in nb[i + 1:]:
                adj[a].add(b)
                adj[b].add(a)
        for u in nb:
            adj[u].discard(best)
        adj[best].clear()
        alive.discard(best)
        order.append(best)
    return order


# --- CLG and its variable elimination --------------------------------------


class CLG:
    """The part of clg/CLG.py inference reads: nodes, parents, coefficients."""

    def __init__(self, nodes, arcs):
        self.names = [n["name"] for n in nodes]
        self.mu = [float(n["mu"]) for n in nodes]
        self.sigma = [float(n["sigma"]) for n in nodes]
        self.arcs = [(self.names.index(a["parent"]), self.names.index(a["child"]), float(a["coef"])) for a in arcs]

    def parents(self, node):
        return [p for p, c, _ in self.arcs if c == node]

    def coef(self, parent, child):
        return next(w for p, c, w in self.arcs if p == parent and c == child)

    def _build_canonical_forms(self):  # CLG.py 640-653
        cf_dict = {}
        for node in range(len(self.names)):
            parents = self.parents(node)
            if len(parents) == 0:
                cf = CanonicalForm.fromCLG(node, [], self.mu[node], self.sigma[node], [])
            else:
                B = [self.coef(p, node) for p in parents]
                cf = CanonicalForm.fromCLG(node, parents, self.mu[node], self.sigma[node], B)
            cf_dict[node] = cf
        return cf_dict

    def moral_edges(self):  # DAG::moralGraph
        edges = {(min(p, c), max(p, c)) for p, c, _ in self.arcs}
        for node in range(len(self.names)):
            ps = sorted(self.parents(node))
            edges |= {(a, b) for i, a in enumerate(ps) for b in ps[i + 1:]}
        return sorted(edges)

    def joint(self):
        """Brute force: the joint mean and covariance of every node."""
        n = len(self.names)
        B = np.zeros((n, n))
        for p, c, w in self.arcs:
            B[c, p] = w
        A = inv(np.eye(n) - B)
        return A @ np.asarray(self.mu), A @ np.diag(np.square(self.sigma)) @ A.T


def canonical_posterior(clg, evidence, targets, normalized=True):
    """variableElimination.py 124-168, _sum_product_ve 192-225 and 227-259."""
    cf_dict = clg._build_canonical_forms()
    for t in targets:
        if t in evidence:
            raise ValueError(f"The variable {t} is observed.")
    variables = [clg.names.index(v) for v in targets]
    elimination = elimination_order(len(clg.names), clg.moral_edges(), [math.log(2.0)] * len(clg.names))
    removed = [v for v in elimination if v not in variables]
    kept = [v for v in elimination if v in variables]

    def sum_product_ve(order, cf_list, ev):
        ev = {clg.names.index(k): v for k, v in ev.items()}
        if len(ev) != 0:
            for i, cf in enumerate(cf_list):
                cf_list[i] = cf.reduce(ev)
        for variable in order:
            cf_list = eliminate(cf_list, variable)
        return cf_list

    def eliminate(cf_list, variable):
        contain, ids = [], []
        for i, cf in enumerate(cf_list):
            if variable in cf:
                contain.append(cf)
                ids.append(i)
        if len(contain) == 0:
            return cf_list
        for i in reversed(ids):
            cf_list.pop(i)
        product = np.prod(contain)
        cf_list.append(product.marginalize([variable]))
        return cf_list

    cf_list = sum_product_ve(removed, list(cf_dict.values()), dict(evidence))
    posterior = np.prod(cf_list)
    if normalized:
        normalization = sum_product_ve(kept, cf_list, {})
        posterior = posterior / np.prod(normalization)
    return posterior, elimination


def brute_force(clg, evidence, targets):
    mean, cov = clg.joint()
    b = [clg.names.index(k) for k in evidence]
    a = [clg.names.index(t) for t in targets]
    v = np.array([evidence[clg.names[i]] for i in b])
    if b:
        s_ab, s_bb = cov[np.ix_(a, b)], cov[np.ix_(b, b)]
        m = mean[a] + s_ab @ np.linalg.solve(s_bb, v - mean[b])
        c = cov[np.ix_(a, a)] - s_ab @ np.linalg.solve(s_bb, cov[np.ix_(b, a)])
        # the evidence density p(x_B = v), which the unnormalised posterior integrates to
        d = v - mean[b]
        log_z = -0.5 * (len(b) * math.log(2 * math.pi) + math.log(det(s_bb)) + d @ np.linalg.solve(s_bb, d))
    else:
        m, c, log_z = mean[a], cov[np.ix_(a, a)], 0.0
    return m, c, log_z


def form_json(cf, names=None):
    scope = [names[i] for i in cf._scope] if names is not None else list(cf._scope)
    return {"scope": scope, "K": np.asarray(cf._K).tolist(), "h": np.asarray(cf._h).ravel().tolist(), "g": float(cf._g)}


# --- the fixture ------------------------------------------------------------

NETWORKS = {
    # A diamond with a tail: A -> B, A -> C, (B, C) -> D -> E. D's two parents
    # are married by moralisation, so the graph has a 3-clique.
    "diamond": {
        "nodes": [
            {"name": "A", "mu": 1.0, "sigma": 2.0},
            {"name": "B", "mu": -0.5, "sigma": 1.5},
            {"name": "C", "mu": 2.0, "sigma": 0.7},
            {"name": "D", "mu": 0.3, "sigma": 1.1},
            {"name": "E", "mu": -1.2, "sigma": 0.4},
        ],
        "arcs": [
            {"parent": "A", "child": "B", "coef": 0.8},
            {"parent": "A", "child": "C", "coef": -1.3},
            {"parent": "B", "child": "D", "coef": 2.0},
            {"parent": "C", "child": "D", "coef": 0.5},
            {"parent": "D", "child": "E", "coef": -0.9},
        ],
        "queries": [
            {"evidence": {}, "targets": ["E"]},
            {"evidence": {"D": 1.5}, "targets": ["A"]},
            {"evidence": {"E": -0.3, "B": 2.0}, "targets": ["C"]},
            {"evidence": {"E": 0.7}, "targets": ["A", "D"]},
            {"evidence": {"A": 0.1, "E": 0.2}, "targets": ["B", "C"]},
        ],
    },
    # Explaining away: X and Y are independent until their common child Z is
    # observed, after which they are (negatively) correlated.
    "collider": {
        "nodes": [
            {"name": "X", "mu": 0.0, "sigma": 1.0},
            {"name": "Y", "mu": 1.0, "sigma": 2.0},
            {"name": "Z", "mu": 0.5, "sigma": 0.3},
            {"name": "W", "mu": -2.0, "sigma": 1.2},
        ],
        "arcs": [
            {"parent": "X", "child": "Z", "coef": 1.0},
            {"parent": "Y", "child": "Z", "coef": 1.0},
            {"parent": "Y", "child": "W", "coef": 0.25},
        ],
        "queries": [
            {"evidence": {}, "targets": ["X", "Y"]},
            {"evidence": {"Z": 3.0}, "targets": ["X", "Y"]},
            {"evidence": {"Z": 3.0, "W": -1.0}, "targets": ["X"]},
        ],
    },
    # A chain of six, the case a star-shaped elimination handles in O(n).
    "chain": {
        "nodes": [{"name": f"S{i}", "mu": 0.2 * i, "sigma": 0.5 + 0.1 * i} for i in range(6)],
        "arcs": [{"parent": f"S{i}", "child": f"S{i + 1}", "coef": 0.9 - 0.2 * i} for i in range(5)],
        "queries": [
            {"evidence": {"S5": 1.0}, "targets": ["S0"]},
            {"evidence": {"S0": -1.0, "S5": 2.0}, "targets": ["S2", "S3"]},
        ],
    },
}


def singular_cases():
    """Where K is singular: informative, and where aGrUM's inv() refuses."""
    out = []
    # A factor that only constrains the sum a + b (rank 1). Integrating b out
    # leaves a with K = 0: the data say nothing about a on its own.
    f = CanonicalForm([0, 1], [[1.0, 1.0], [1.0, 1.0]], [2.0, 2.0], -0.5)
    marg = f.marginalize([1])
    try:
        marg.toGaussian()
        raised = False
    except np.linalg.LinAlgError:
        raised = True
    red = f.reduce({1: 0.5})
    _, rm, rc = red.toGaussian()
    prior = CanonicalForm.fromCLG(0, [], 3.0, 0.5, [])
    prod = f * prior
    _, pm, pc = prod.toGaussian()
    out.append({
        "name": "sum_only",
        "form": form_json(f),
        "marginalize_1": form_json(marg),
        "marginal_to_gaussian_raises": raised,
        "reduce_1": form_json(red),
        "reduce_1_mean": rm.ravel().tolist(),
        "reduce_1_cov": rc.tolist(),
        "prior_0": form_json(prior),
        "product_with_prior": form_json(prod),
        "product_mean": pm.ravel().tolist(),
        "product_cov": pc.tolist(),
    })
    # Integrating out a direction nothing constrains diverges: K_yy = 0.
    g = CanonicalForm([0, 1], [[2.0, 0.0], [0.0, 0.0]], [1.0, 0.0], 0.0)
    try:
        g.marginalize([1])
        raised = False
    except np.linalg.LinAlgError:
        raised = True
    out.append({"name": "unconstrained", "form": form_json(g), "marginalize_1_raises": raised})
    return out


def form_operations(rng):
    """product, division, reduce and marginalize on random forms with overlapping scopes."""
    def random_form(scope):
        d = len(scope)
        a = rng.normal(size=(d, d))
        return CanonicalForm(list(scope), a @ a.T + d * np.eye(d), rng.normal(size=d), float(rng.normal()))

    cases = []
    for scopes in (([0, 2], [1, 2]), ([3, 1, 0], [2, 0]), ([4], [0, 1, 2, 3]), ([2, 0, 1], [1, 2, 0])):
        f, g = random_form(scopes[0]), random_form(scopes[1])
        prod, quot = f * g, f / g
        both = sorted(set(prod._scope))
        cases.append({
            "f": form_json(f), "g": form_json(g),
            "product": form_json(prod), "division": form_json(quot),
            "reduce": {"evidence": {str(v): 0.3 * (v + 1) for v in both[:1]}, "result": form_json(prod.reduce({v: 0.3 * (v + 1) for v in both[:1]}))},
            "marginalize": {"variables": both[-1:], "result": form_json(prod.marginalize(list(both[-1:])))},
            "reduce_out_of_scope": form_json(f.reduce({99: 1.0})),
        })
    return cases


def order_cases(rng):
    """Random undirected graphs with free dimensions as log domain sizes."""
    cases = []
    for n, p, sized in ((6, 0.4, False), (9, 0.35, False), (12, 0.3, False), (10, 0.3, True), (14, 0.25, True)):
        edges = [(i, j) for i in range(n) for j in range(i + 1, n) if rng.uniform() < p]
        sizes = [int(rng.integers(1, 5)) if sized else 1 for _ in range(n)]
        cases.append({"n": n, "edges": edges, "sizes": sizes, "order": elimination_order(n, edges, [float(s) for s in sizes])})
    return cases


def main():
    fixture = {"networks": {}, "singular": singular_cases()}
    for name, spec in NETWORKS.items():
        clg = CLG(spec["nodes"], spec["arcs"])
        entry = {
            "nodes": spec["nodes"], "arcs": spec["arcs"],
            "factors": {clg.names[k]: form_json(cf, clg.names) for k, cf in clg._build_canonical_forms().items()},
            "moral_edges": [[clg.names[a], clg.names[b]] for a, b in clg.moral_edges()],
            "queries": [],
        }
        for q in spec["queries"]:
            for normalized in (True, False):
                post, order = canonical_posterior(clg, q["evidence"], q["targets"], normalized)
                row = {"evidence": q["evidence"], "targets": q["targets"], "normalized": normalized,
                       "posterior": form_json(post, clg.names),
                       "elimination_order": [clg.names[v] for v in order]}
                if normalized:
                    _, mu, sigma = post.toGaussian()
                    bm, bc, log_z = brute_force(clg, q["evidence"], [clg.names[i] for i in post._scope])
                    row.update(mean=mu.ravel().tolist(), covariance=sigma.tolist(),
                               brute_mean=bm.tolist(), brute_covariance=bc.tolist(), brute_log_evidence=log_z)
                entry["queries"].append(row)
        fixture["networks"][name] = entry
    rng = np.random.default_rng(151)
    fixture["form_operations"] = form_operations(rng)
    fixture["orders"] = order_cases(rng)
    out = Path(__file__).parent / "data" / "clg_agrum_fixture.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(fixture, indent=1) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
