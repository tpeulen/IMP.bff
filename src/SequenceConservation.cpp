/**
 * \file SequenceConservation.cpp
 * \brief Per-site evolutionary rates (Rate4Site's method) and ConSurf grades.
 *
 * Written from the published method and Rate4Site's documented behaviour;
 * see the header for the references. The JTT exchangeabilities and
 * frequencies are the published data of Jones, Taylor & Thornton (1992).
 *
 * Copyright 2007-2026 IMP Inventors. All rights reserved.
 */
#include <IMP/bff/SequenceConservation.h>
#include <IMP/bff/internal/ProbePairKernels.h>

#include <Eigen/Dense>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <functional>
#include <limits>
#include <map>
#include <sstream>

IMPBFF_BEGIN_NAMESPACE

namespace {

const int kA = 20;
const double kInf = std::numeric_limits<double>::infinity();

// ---------------------------------------------------------------------------
// JTT: Jones, Taylor & Thornton, CABIOS 8:275 (1992). Exchangeabilities as a
// lower triangle, row by row, and equilibrium frequencies, both in the order
// A R N D C Q E G H I L K M F P S T W Y V.
// ---------------------------------------------------------------------------
const char kJttOrder[] = "ARNDCQEGHILKMFPSTWYV";
const double kJttExchange[190] = {
    58,
    54, 45,
    81, 16, 528,
    56, 113, 34, 10,
    57, 310, 86, 49, 9,
    105, 29, 58, 767, 5, 323,
    179, 137, 81, 130, 59, 26, 119,
    27, 328, 391, 112, 69, 597, 26, 23,
    36, 22, 47, 11, 17, 9, 12, 6, 16,
    30, 38, 12, 7, 23, 72, 9, 6, 56, 229,
    35, 646, 263, 26, 7, 292, 181, 27, 45, 21, 14,
    54, 44, 30, 15, 31, 43, 18, 14, 33, 479, 388, 65,
    15, 5, 10, 4, 78, 4, 5, 5, 40, 89, 248, 4, 43,
    194, 74, 15, 15, 14, 164, 18, 24, 115, 10, 102, 21, 16, 17,
    378, 101, 503, 59, 223, 53, 30, 201, 73, 40, 59, 47, 29, 92, 285,
    475, 64, 232, 38, 42, 51, 32, 33, 46, 245, 25, 103, 226, 12, 118, 477,
    9, 126, 8, 4, 115, 18, 10, 55, 8, 9, 52, 10, 24, 53, 6, 35, 12,
    11, 20, 70, 46, 209, 24, 7, 8, 573, 32, 24, 8, 18, 536, 10, 63, 21, 71,
    298, 17, 16, 31, 62, 20, 45, 47, 11, 961, 180, 14, 323, 62, 23, 38, 112, 25, 16};
const double kJttFrequencies[20] = {
    0.076748, 0.051691, 0.042645, 0.051544, 0.019803, 0.040752, 0.061830,
    0.073152, 0.022944, 0.053761, 0.091904, 0.058676, 0.023826, 0.040126,
    0.050901, 0.068765, 0.058565, 0.014261, 0.032102, 0.066005};

//! A reversible amino-acid model in SequenceMSA's order, with P(t) from the
//! eigen-decomposition of its symmetrised rate matrix.
struct Model {
    double pi[kA];
    double lambda[kA];
    double left[kA * kA];   // left[i*20+k]
    double right[kA * kA];  // right[k*20+j]

    Model() {
        const std::string ours = get_sequence_alphabet();
        int to_ours[kA];
        for (int i = 0; i < kA; ++i) {
            to_ours[i] = static_cast<int>(ours.find(kJttOrder[i]));
        }
        double s[kA][kA] = {{0}};
        int k = 0;
        for (int i = 1; i < kA; ++i) {
            for (int j = 0; j < i; ++j, ++k) {
                s[to_ours[i]][to_ours[j]] = s[to_ours[j]][to_ours[i]] = kJttExchange[k];
            }
        }
        for (int i = 0; i < kA; ++i) pi[to_ours[i]] = kJttFrequencies[i];
        // Q = S pi, scaled to one expected substitution per unit time.
        double q[kA][kA];
        double scale = 0.0;
        for (int i = 0; i < kA; ++i) {
            double row = 0.0;
            for (int j = 0; j < kA; ++j) {
                q[i][j] = i == j ? 0.0 : s[i][j] * pi[j];
                row += q[i][j];
            }
            q[i][i] = -row;
            scale += pi[i] * row;
        }
        Eigen::MatrixXd m(kA, kA);
        for (int i = 0; i < kA; ++i) {
            for (int j = 0; j < kA; ++j) {
                m(i, j) = q[i][j] / scale * std::sqrt(pi[i] / pi[j]);
            }
        }
        m = 0.5 * (m + m.transpose()).eval();
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(m);
        for (int kk = 0; kk < kA; ++kk) {
            lambda[kk] = es.eigenvalues()(kk);
            for (int i = 0; i < kA; ++i) {
                left[i * kA + kk] = es.eigenvectors()(i, kk) / std::sqrt(pi[i]);
                right[kk * kA + i] = es.eigenvectors()(i, kk) * std::sqrt(pi[i]);
            }
        }
    }

    //! dP/dt, row-major, without clamping.
    void transition_derivative(double t, double* dp) const {
        double e[kA];
        for (int k = 0; k < kA; ++k) e[k] = lambda[k] * std::exp(lambda[k] * t);
        for (int i = 0; i < kA; ++i) {
            for (int j = 0; j < kA; ++j) {
                double v = 0.0;
                for (int k = 0; k < kA; ++k) v += left[i * kA + k] * e[k] * right[k * kA + j];
                dp[i * kA + j] = v;
            }
        }
    }

    //! P(t), row-major P[a*20+b] = P(b at the end | a at the start).
    void transition(double t, double* p) const {
        if (t <= 0.0) {
            for (int i = 0; i < kA * kA; ++i) p[i] = 0.0;
            for (int i = 0; i < kA; ++i) p[i * kA + i] = 1.0;
            return;
        }
        double e[kA];
        for (int k = 0; k < kA; ++k) e[k] = std::exp(lambda[k] * t);
        for (int i = 0; i < kA; ++i) {
            double li[kA];
            for (int k = 0; k < kA; ++k) li[k] = left[i * kA + k] * e[k];
            for (int j = 0; j < kA; ++j) {
                double v = 0.0;
                for (int k = 0; k < kA; ++k) v += li[k] * right[k * kA + j];
                if (v < 0.0) v = 0.0;
                if (v > 1.0) v = 1.0;
                if (v == 0.0 && i != j) v = DBL_EPSILON;
                p[i * kA + j] = v;
            }
        }
    }
};

const Model& jtt() {
    static const Model model;
    return model;
}

// ---------------------------------------------------------------------------
// Discrete gamma (Yang 1994), category means, shape alpha, mean one.
// ---------------------------------------------------------------------------

//! Regularised lower incomplete gamma P(a, x).
double gamma_p(double a, double x) {
    if (x <= 0.0) return 0.0;
    if (!std::isfinite(x)) return 1.0;
    return x < a + 1.0 ? internal::gamma_p_series(a, x)
                       : 1.0 - internal::gamma_q_continued_fraction(a, x);
}

std::vector<double> gamma_rates(double alpha, int k) {
    if (k <= 1) return std::vector<double>(1, 1.0);
    alpha = std::max(alpha, 0.05);
    // quantiles z_i of the standard gamma(alpha) at i/k, by bisection to
    // 1e-5 in probability over [0, 99999] from 5000 (Rate4Site's search)
    std::vector<double> z(static_cast<std::size_t>(k) + 1, 0.0);
    z[static_cast<std::size_t>(k)] = kInf;
    for (int i = 1; i < k; ++i) {
        const double target = static_cast<double>(i) / k;
        double lo = 0.0, hi = 99999.0, x = 5000.0;
        for (int it = 0; it < 200; ++it) {
            const double v = gamma_p(alpha, x);
            if (std::fabs(v - target) < 1e-5) break;
            if (v < target) lo = x; else hi = x;
            x = 0.5 * (lo + hi);
        }
        z[static_cast<std::size_t>(i)] = x;
    }
    std::vector<double> r(static_cast<std::size_t>(k));
    for (int i = 0; i < k; ++i) {
        r[static_cast<std::size_t>(i)] =
                k * (gamma_p(alpha + 1.0, z[static_cast<std::size_t>(i) + 1]) -
                     gamma_p(alpha + 1.0, z[static_cast<std::size_t>(i)]));
    }
    return r;
}

// ---------------------------------------------------------------------------
// Brent's minimiser (golden section with parabolic steps), bracket a < b < c,
// stopping width tol |x| + 1e-10.
// ---------------------------------------------------------------------------
double brent_minimise(const std::function<double(double)>& f, double ax, double bx,
                      double cx, double tol, double* fmin) {
    const double cgold = 0.3819660, zeps = 1e-10;
    double a = std::min(ax, cx), b = std::max(ax, cx);
    double x = bx, w = bx, v = bx;
    double fx = f(x), fw = fx, fv = fx;
    double d = 0.0, e = 0.0;
    for (int iter = 0; iter < 100; ++iter) {
        const double xm = 0.5 * (a + b);
        const double tol1 = tol * std::fabs(x) + zeps, tol2 = 2.0 * tol1;
        if (std::fabs(x - xm) <= tol2 - 0.5 * (b - a)) break;
        if (std::fabs(e) > tol1) {
            double r = (x - w) * (fx - fv);
            double q = (x - v) * (fx - fw);
            double p = (x - v) * q - (x - w) * r;
            q = 2.0 * (q - r);
            if (q > 0.0) p = -p;
            q = std::fabs(q);
            const double etemp = e;
            e = d;
            if (std::fabs(p) >= std::fabs(0.5 * q * etemp) || p <= q * (a - x) ||
                p >= q * (b - x)) {
                e = x >= xm ? a - x : b - x;
                d = cgold * e;
            } else {
                d = p / q;
                const double u = x + d;
                if (u - a < tol2 || b - u < tol2) d = xm - x >= 0 ? tol1 : -tol1;
            }
        } else {
            e = x >= xm ? a - x : b - x;
            d = cgold * e;
        }
        const double u = std::fabs(d) >= tol1 ? x + d : x + (d >= 0 ? tol1 : -tol1);
        const double fu = f(u);
        if (fu <= fx) {
            if (u >= x) a = x; else b = x;
            v = w; w = x; x = u;
            fv = fw; fw = fx; fx = fu;
        } else {
            if (u < x) a = u; else b = u;
            if (fu <= fw || w == x) {
                v = w; w = u; fv = fw; fw = fu;
            } else if (fu <= fv || v == x || v == w) {
                v = u; fv = fu;
            }
        }
    }
    if (fmin) *fmin = fx;
    return x;
}

//! Brent's method with first derivatives: secant steps on the derivative
//! from the two previous points, kept inside the bracket and pointing
//! downhill, else bisection towards the downhill side; stopping width
//! tol |x| + 1e-10. \p f returns the value and writes the derivative.
double brent_minimise_derivative(const std::function<double(double, double*)>& f, double ax,
                                 double bx, double cx, double tol) {
    const double zeps = 1e-10;
    double a = std::min(ax, cx), b = std::max(ax, cx);
    double x = bx, w = bx, v = bx;
    double dx = 0.0;
    double fx = f(x, &dx);
    double fw = fx, fv = fx, dw = dx, dv = dx;
    double d = 0.0, e = 0.0;
    for (int iter = 0; iter < 100; ++iter) {
        const double xm = 0.5 * (a + b);
        const double tol1 = tol * std::fabs(x) + zeps, tol2 = 2.0 * tol1;
        if (std::fabs(x - xm) <= tol2 - 0.5 * (b - a)) return x;
        bool bisect = true;
        if (std::fabs(e) > tol1) {
            double d1 = 2.0 * (b - a), d2 = d1;
            if (dw != dx) d1 = (w - x) * dx / (dx - dw);
            if (dv != dx) d2 = (v - x) * dx / (dx - dv);
            const double u1 = x + d1, u2 = x + d2;
            const bool ok1 = (a - u1) * (u1 - b) > 0.0 && dx * d1 <= 0.0;
            const bool ok2 = (a - u2) * (u2 - b) > 0.0 && dx * d2 <= 0.0;
            const double olde = e;
            e = d;
            if (ok1 || ok2) {
                if (ok1 && ok2) d = std::fabs(d1) < std::fabs(d2) ? d1 : d2;
                else d = ok1 ? d1 : d2;
                if (std::fabs(d) <= std::fabs(0.5 * olde)) {
                    const double u = x + d;
                    if (u - a < tol2 || b - u < tol2) d = xm - x >= 0.0 ? tol1 : -tol1;
                    bisect = false;
                }
            }
        }
        if (bisect) {
            e = dx >= 0.0 ? a - x : b - x;
            d = 0.5 * e;
        }
        double u, fu, du = 0.0;
        if (std::fabs(d) >= tol1) {
            u = x + d;
            fu = f(u, &du);
        } else {
            u = x + (d >= 0.0 ? tol1 : -tol1);
            fu = f(u, &du);
            if (fu > fx) return x;   // the smallest step is uphill: done
        }
        if (fu <= fx) {
            if (u >= x) a = x; else b = x;
            v = w; fv = fw; dv = dw;
            w = x; fw = fx; dw = dx;
            x = u; fx = fu; dx = du;
        } else {
            if (u < x) a = u; else b = u;
            if (fu <= fw || w == x) {
                v = w; fv = fw; dv = dw;
                w = u; fw = fu; dw = du;
            } else if (fu < fv || v == x || v == w) {
                v = u; fv = fu; dv = du;
            }
        }
    }
    return x;
}

// ---------------------------------------------------------------------------
// Tree: nodes 0..n-1 are the sequences; the root is trifurcating.
// ---------------------------------------------------------------------------
struct Tree {
    int n_leaves = 0;
    int root = -1;
    std::vector<int> parent;
    std::vector<std::vector<int> > children;
    std::vector<double> length;  // branch to the parent
    std::vector<int> postorder;  // children before parents, root last

    int add_node() {
        parent.push_back(-1);
        children.push_back(std::vector<int>());
        length.push_back(0.0);
        return static_cast<int>(parent.size()) - 1;
    }
    void finish() {
        postorder.clear();
        std::vector<std::pair<int, bool> > stack(1, std::make_pair(root, false));
        while (!stack.empty()) {
            const std::pair<int, bool> top = stack.back();
            stack.pop_back();
            if (top.second) {
                postorder.push_back(top.first);
                continue;
            }
            stack.push_back(std::make_pair(top.first, true));
            const std::vector<int>& ch = children[static_cast<std::size_t>(top.first)];
            for (std::size_t k = ch.size(); k-- > 0;) stack.push_back(std::make_pair(ch[k], false));
        }
    }
    std::string newick(const std::vector<std::string>& names) const {
        std::ostringstream out;
        std::function<void(int)> write = [&](int node) {
            const std::vector<int>& ch = children[static_cast<std::size_t>(node)];
            if (ch.empty()) {
                out << names[static_cast<std::size_t>(node)];
            } else {
                out << "(";
                for (std::size_t k = 0; k < ch.size(); ++k) {
                    if (k) out << ",";
                    write(ch[k]);
                }
                out << ")";
            }
            if (node != root) {
                char buffer[64];
                std::snprintf(buffer, sizeof(buffer), ":%.6f", length[static_cast<std::size_t>(node)]);
                out << buffer;
            }
        };
        write(root);
        out << ";";
        return out.str();
    }
};

//! Neighbour joining with the Studier-Keppler criterion; ties to the first pair.
Tree neighbour_joining(const std::vector<double>& dist, int n) {
    Tree tree;
    tree.n_leaves = n;
    for (int i = 0; i < n; ++i) tree.add_node();
    tree.root = tree.add_node();
    const int cap = 2 * n + 2;
    std::vector<double> d(static_cast<std::size_t>(cap) * cap, 0.0);
    auto at = [&](int i, int j) -> double& { return d[static_cast<std::size_t>(i) * cap + j]; };
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) at(i, j) = dist[static_cast<std::size_t>(i) * n + j];
    }
    std::vector<int> active;
    for (int i = 0; i < n; ++i) active.push_back(i);
    const double floor = 1e-6;
    std::vector<int> joined_under(static_cast<std::size_t>(cap), -1);
    while (active.size() > 3) {
        const int m = static_cast<int>(active.size());
        std::vector<double> r(static_cast<std::size_t>(m), 0.0);
        for (int a = 0; a < m; ++a) {
            for (int b = 0; b < m; ++b) r[static_cast<std::size_t>(a)] += at(active[a], active[b]);
        }
        int bi = 0, bj = 1;
        double best = kInf;
        for (int a = 0; a < m; ++a) {
            for (int b = a + 1; b < m; ++b) {
                const double q = at(active[a], active[b]) -
                        (r[static_cast<std::size_t>(a)] + r[static_cast<std::size_t>(b)]) / (m - 2);
                if (q < best) { best = q; bi = a; bj = b; }
            }
        }
        const int i = active[bi], j = active[bj];
        const double dij = at(i, j);
        double li = 0.5 * dij + (r[static_cast<std::size_t>(bi)] - r[static_cast<std::size_t>(bj)]) / (2.0 * (m - 2));
        double lj = dij - li;
        const int u = tree.add_node();
        tree.children[static_cast<std::size_t>(u)] = {i, j};
        tree.parent[static_cast<std::size_t>(i)] = u;
        tree.parent[static_cast<std::size_t>(j)] = u;
        tree.length[static_cast<std::size_t>(i)] = std::max(li, floor);
        tree.length[static_cast<std::size_t>(j)] = std::max(lj, floor);
        for (int k : active) {
            if (k == i || k == j) continue;
            at(u, k) = at(k, u) = 0.5 * (at(i, k) + at(j, k) - dij);
        }
        std::vector<int> next;
        for (int k : active) if (k != i && k != j) next.push_back(k);
        next.push_back(u);
        active.swap(next);
    }
    // the last three hang from the root
    std::vector<double> r(active.size(), 0.0);
    for (std::size_t a = 0; a < active.size(); ++a) {
        for (std::size_t b = 0; b < active.size(); ++b) r[a] += at(active[a], active[b]);
    }
    std::vector<double> l(active.size(), floor);
    if (active.size() == 3) {
        l[0] = 0.5 * at(active[0], active[1]) + 0.5 * (r[0] - r[1]);
        l[1] = 0.5 * at(active[0], active[1]) + 0.5 * (r[1] - r[0]);
        l[2] = 0.5 * at(active[0], active[2]) + 0.5 * (r[2] - r[0]);
    } else if (active.size() == 2) {
        l[0] = l[1] = 0.5 * at(active[0], active[1]);
    }
    for (std::size_t a = 0; a < active.size(); ++a) {
        tree.children[static_cast<std::size_t>(tree.root)].push_back(active[a]);
        tree.parent[static_cast<std::size_t>(active[a])] = tree.root;
        tree.length[static_cast<std::size_t>(active[a])] = std::max(l[a], floor);
    }
    tree.finish();
    return tree;
}

//! A Newick tree over the alignment's names; missing lengths become 0.05.
Tree parse_newick(const std::string& text, const std::vector<std::string>& names) {
    Tree tree;
    const int n = static_cast<int>(names.size());
    tree.n_leaves = n;
    for (int i = 0; i < n; ++i) tree.add_node();
    std::map<std::string, int> by_name;
    for (int i = 0; i < n; ++i) by_name[names[static_cast<std::size_t>(i)]] = i;
    std::vector<char> used(static_cast<std::size_t>(n), 0);
    std::size_t pos = 0;
    auto skip = [&]() { while (pos < text.size() && std::isspace(static_cast<unsigned char>(text[pos]))) ++pos; };
    std::function<int()> subtree = [&]() -> int {
        skip();
        int node;
        if (pos < text.size() && text[pos] == '(') {
            ++pos;
            node = tree.add_node();
            while (true) {
                const int child = subtree();
                tree.children[static_cast<std::size_t>(node)].push_back(child);
                tree.parent[static_cast<std::size_t>(child)] = node;
                skip();
                if (pos < text.size() && text[pos] == ',') { ++pos; continue; }
                if (pos < text.size() && text[pos] == ')') { ++pos; break; }
                IMP_THROW("Newick: expected ',' or ')' at " << pos, ValueException);
            }
            // an internal label, if any, is ignored
            while (pos < text.size() && text[pos] != ':' && text[pos] != ',' && text[pos] != ')' &&
                   text[pos] != ';') ++pos;
        } else {
            std::string label;
            while (pos < text.size() && text[pos] != ':' && text[pos] != ',' && text[pos] != ')' &&
                   text[pos] != ';') label.push_back(text[pos++]);
            while (!label.empty() && std::isspace(static_cast<unsigned char>(label.back()))) label.pop_back();
            while (!label.empty() && std::isspace(static_cast<unsigned char>(label.front()))) label.erase(0, 1);
            const auto it = by_name.find(label);
            if (it == by_name.end()) IMP_THROW("Newick: unknown sequence '" << label << "'", ValueException);
            node = it->second;
            used[static_cast<std::size_t>(node)] = 1;
        }
        double length = 0.05;
        skip();
        if (pos < text.size() && text[pos] == ':') {
            ++pos;
            std::size_t used_chars = 0;
            length = std::stod(text.substr(pos), &used_chars);
            pos += used_chars;
        }
        tree.length[static_cast<std::size_t>(node)] = length;
        return node;
    };
    tree.root = subtree();
    for (int i = 0; i < n; ++i) {
        if (!used[static_cast<std::size_t>(i)]) {
            IMP_THROW("Newick: sequence '" << names[static_cast<std::size_t>(i)] << "' is not in the tree",
                      ValueException);
        }
    }
    tree.length[static_cast<std::size_t>(tree.root)] = 0.0;
    tree.finish();
    return tree;
}

// ---------------------------------------------------------------------------
// Likelihood over site patterns, scaled.
// ---------------------------------------------------------------------------
class Engine {
public:
    Engine(const SequenceMSA& msa, const Tree& tree) : model_(jtt()), tree_(tree) {
        n_seq_ = msa.get_n_sequences();
        const int n_col = msa.get_n_columns();
        std::map<std::string, int> index;
        pattern_of_column_.resize(static_cast<std::size_t>(n_col));
        for (int c = 0; c < n_col; ++c) {
            std::string key(static_cast<std::size_t>(n_seq_), '\0');
            for (int m = 0; m < n_seq_; ++m) key[static_cast<std::size_t>(m)] = static_cast<char>(msa.get_state(m, c));
            const auto it = index.find(key);
            if (it == index.end()) {
                const int p = static_cast<int>(counts_.size());
                index[key] = p;
                counts_.push_back(1.0);
                for (int m = 0; m < n_seq_; ++m) states_.push_back(msa.get_state(m, c));
                pattern_of_column_[static_cast<std::size_t>(c)] = p;
            } else {
                counts_[static_cast<std::size_t>(it->second)] += 1.0;
                pattern_of_column_[static_cast<std::size_t>(c)] = it->second;
            }
        }
        n_pat_ = static_cast<int>(counts_.size());
    }

    Tree& tree() { return tree_; }
    const Tree& tree() const { return tree_; }
    int n_patterns() const { return n_pat_; }
    int pattern_of_column(int c) const { return pattern_of_column_[static_cast<std::size_t>(c)]; }

    //! log L_c(pattern) for every category rate, row-major n_pat x K.
    std::vector<double> category_log_likelihoods(const std::vector<double>& rates) const {
        const int k = static_cast<int>(rates.size());
        std::vector<double> out(static_cast<std::size_t>(n_pat_) * k);
        for (int c = 0; c < k; ++c) {
            const std::vector<double> p = branch_matrices(rates[static_cast<std::size_t>(c)]);
            std::vector<double> msg, ls;
            up_pass(p, &msg, &ls, nullptr, nullptr);
            for (int s = 0; s < n_pat_; ++s) out[static_cast<std::size_t>(s) * k + c] = root_log_likelihood(msg, ls, s);
        }
        return out;
    }

    //! Total log-likelihood under equally weighted categories.
    double log_likelihood(const std::vector<double>& rates,
                          std::vector<double>* site = nullptr) const {
        const int k = static_cast<int>(rates.size());
        const std::vector<double> lc = category_log_likelihoods(rates);
        double total = 0.0;
        if (site) site->assign(static_cast<std::size_t>(n_pat_), 0.0);
        for (int s = 0; s < n_pat_; ++s) {
            double mx = -kInf;
            for (int c = 0; c < k; ++c) mx = std::max(mx, lc[static_cast<std::size_t>(s) * k + c]);
            double sum = 0.0;
            for (int c = 0; c < k; ++c) sum += std::exp(lc[static_cast<std::size_t>(s) * k + c] - mx);
            const double l = mx + std::log(sum / k);
            if (site) (*site)[static_cast<std::size_t>(s)] = l;
            total += counts_[static_cast<std::size_t>(s)] * l;
        }
        return total;
    }

    //! Expected transition counts per branch, category and (parent, child)
    //! state pair: row-major nodes x K x 20 x 20.
    std::vector<double> expected_counts(const std::vector<double>& rates,
                                        const std::vector<double>& site_ll) const {
        const int k = static_cast<int>(rates.size());
        const int n_nodes = static_cast<int>(tree_.parent.size());
        std::vector<double> counts(static_cast<std::size_t>(n_nodes) * k * kA * kA, 0.0);
        for (int c = 0; c < k; ++c) {
            const std::vector<double> p = branch_matrices(rates[static_cast<std::size_t>(c)]);
            std::vector<double> msg, ls_msg, up, ls_up;
            up_pass(p, &msg, &ls_msg, &up, &ls_up);
            std::vector<double> out, ls_out;
            down_pass(p, msg, ls_msg, &out, &ls_out);
            for (int node = 0; node < n_nodes; ++node) {
                if (node == tree_.root) continue;
                const double* pm = &p[static_cast<std::size_t>(node) * kA * kA];
                double* acc = &counts[(static_cast<std::size_t>(node) * k + c) * kA * kA];
                for (int s = 0; s < n_pat_; ++s) {
                    const double* o = &out[(static_cast<std::size_t>(node) * n_pat_ + s) * kA];
                    const double w = counts_[static_cast<std::size_t>(s)] *
                            std::exp(ls_out[static_cast<std::size_t>(node) * n_pat_ + s] +
                                     up_scale(node, s, ls_up) - site_ll[static_cast<std::size_t>(s)]) / k;
                    if (w == 0.0) continue;
                    if (node < n_seq_) {
                        const int x = states_[static_cast<std::size_t>(s) * n_seq_ + node];
                        if (x > 0) {
                            const int a = x - 1;
                            for (int b = 0; b < kA; ++b) acc[b * kA + a] += w * o[b] * pm[b * kA + a];
                        } else {
                            for (int b = 0; b < kA; ++b) {
                                for (int a = 0; a < kA; ++a) acc[b * kA + a] += w * o[b] * pm[b * kA + a];
                            }
                        }
                    } else {
                        const double* u = &up[(static_cast<std::size_t>(node) * n_pat_ + s) * kA];
                        for (int b = 0; b < kA; ++b) {
                            const double ob = w * o[b];
                            for (int a = 0; a < kA; ++a) acc[b * kA + a] += ob * pm[b * kA + a] * u[a];
                        }
                    }
                }
            }
        }
        return counts;
    }

    const Model& model() const { return model_; }

private:
    //! P(t r) for every branch, nodes x 400.
    std::vector<double> branch_matrices(double rate) const {
        const int n_nodes = static_cast<int>(tree_.parent.size());
        std::vector<double> p(static_cast<std::size_t>(n_nodes) * kA * kA, 0.0);
        for (int node = 0; node < n_nodes; ++node) {
            if (node == tree_.root) continue;
            model_.transition(tree_.length[static_cast<std::size_t>(node)] * rate,
                              &p[static_cast<std::size_t>(node) * kA * kA]);
        }
        return p;
    }

    double up_scale(int node, int s, const std::vector<double>& ls_up) const {
        return node < n_seq_ ? 0.0 : ls_up[static_cast<std::size_t>(node) * n_pat_ + s];
    }

    //! Messages to the parent, M_node(b) = sum_a P(b -> a) U_node(a), and
    //! (optionally) the conditional likelihoods U of internal nodes; each
    //! scaled to a maximum of one with the log scale kept.
    void up_pass(const std::vector<double>& p, std::vector<double>* msg,
                 std::vector<double>* ls_msg, std::vector<double>* up_out,
                 std::vector<double>* ls_up_out) const {
        const int n_nodes = static_cast<int>(tree_.parent.size());
        msg->assign(static_cast<std::size_t>(n_nodes) * n_pat_ * kA, 0.0);
        ls_msg->assign(static_cast<std::size_t>(n_nodes) * n_pat_, 0.0);
        std::vector<double> up_local, ls_local;
        std::vector<double>& up = up_out ? *up_out : up_local;
        std::vector<double>& ls_up = ls_up_out ? *ls_up_out : ls_local;
        up.assign(static_cast<std::size_t>(n_nodes) * n_pat_ * kA, 0.0);
        ls_up.assign(static_cast<std::size_t>(n_nodes) * n_pat_, 0.0);
        for (int node : tree_.postorder) {
            if (node < n_seq_) {
                if (node == tree_.root) continue;
                const double* pm = &p[static_cast<std::size_t>(node) * kA * kA];
                for (int s = 0; s < n_pat_; ++s) {
                    double* m = &(*msg)[(static_cast<std::size_t>(node) * n_pat_ + s) * kA];
                    const int x = states_[static_cast<std::size_t>(s) * n_seq_ + node];
                    if (x > 0) {
                        for (int b = 0; b < kA; ++b) m[b] = pm[b * kA + x - 1];
                    } else {
                        for (int b = 0; b < kA; ++b) m[b] = 1.0;
                    }
                }
                continue;
            }
            const std::vector<int>& ch = tree_.children[static_cast<std::size_t>(node)];
            for (int s = 0; s < n_pat_; ++s) {
                double* u = &up[(static_cast<std::size_t>(node) * n_pat_ + s) * kA];
                double ls = 0.0;
                for (int b = 0; b < kA; ++b) u[b] = 1.0;
                for (int child : ch) {
                    const double* m = &(*msg)[(static_cast<std::size_t>(child) * n_pat_ + s) * kA];
                    for (int b = 0; b < kA; ++b) u[b] *= m[b];
                    ls += (*ls_msg)[static_cast<std::size_t>(child) * n_pat_ + s];
                }
                double mx = 0.0;
                for (int b = 0; b < kA; ++b) mx = std::max(mx, u[b]);
                if (mx > 0.0) {
                    for (int b = 0; b < kA; ++b) u[b] /= mx;
                    ls += std::log(mx);
                }
                ls_up[static_cast<std::size_t>(node) * n_pat_ + s] = ls;
                if (node == tree_.root) continue;
                const double* pm = &p[static_cast<std::size_t>(node) * kA * kA];
                double* m = &(*msg)[(static_cast<std::size_t>(node) * n_pat_ + s) * kA];
                for (int b = 0; b < kA; ++b) {
                    double v = 0.0;
                    for (int a = 0; a < kA; ++a) v += pm[b * kA + a] * u[a];
                    m[b] = v;
                }
                (*ls_msg)[static_cast<std::size_t>(node) * n_pat_ + s] = ls;
            }
        }
    }

    double root_log_likelihood(const std::vector<double>& msg, const std::vector<double>& ls_msg,
                               int s) const {
        double v[kA];
        for (int b = 0; b < kA; ++b) v[b] = model_.pi[b];
        double ls = 0.0;
        for (int child : tree_.children[static_cast<std::size_t>(tree_.root)]) {
            const double* m = &msg[(static_cast<std::size_t>(child) * n_pat_ + s) * kA];
            for (int b = 0; b < kA; ++b) v[b] *= m[b];
            ls += ls_msg[static_cast<std::size_t>(child) * n_pat_ + s];
        }
        double sum = 0.0;
        for (int b = 0; b < kA; ++b) sum += v[b];
        return std::log(sum) + ls;
    }

    //! Outside likelihoods O_node(b): the data outside the node's subtree
    //! with its parent in state b (the root's pi included), scaled.
    void down_pass(const std::vector<double>& p, const std::vector<double>& msg,
                   const std::vector<double>& ls_msg, std::vector<double>* out,
                   std::vector<double>* ls_out) const {
        const int n_nodes = static_cast<int>(tree_.parent.size());
        out->assign(static_cast<std::size_t>(n_nodes) * n_pat_ * kA, 0.0);
        ls_out->assign(static_cast<std::size_t>(n_nodes) * n_pat_, 0.0);
        for (std::size_t k = tree_.postorder.size(); k-- > 0;) {
            const int f = tree_.postorder[k];
            const std::vector<int>& ch = tree_.children[static_cast<std::size_t>(f)];
            if (ch.empty()) continue;
            for (int s = 0; s < n_pat_; ++s) {
                // the part that does not depend on which child: data above f
                double above[kA];
                double ls_above = 0.0;
                if (f == tree_.root) {
                    for (int b = 0; b < kA; ++b) above[b] = model_.pi[b];
                } else {
                    const double* o = &(*out)[(static_cast<std::size_t>(f) * n_pat_ + s) * kA];
                    const double* pm = &p[static_cast<std::size_t>(f) * kA * kA];
                    for (int b = 0; b < kA; ++b) {
                        double v = 0.0;
                        for (int y = 0; y < kA; ++y) v += o[y] * pm[y * kA + b];
                        above[b] = v;
                    }
                    ls_above = (*ls_out)[static_cast<std::size_t>(f) * n_pat_ + s];
                }
                for (int child : ch) {
                    double* o = &(*out)[(static_cast<std::size_t>(child) * n_pat_ + s) * kA];
                    double ls = ls_above;
                    for (int b = 0; b < kA; ++b) o[b] = above[b];
                    for (int sib : ch) {
                        if (sib == child) continue;
                        const double* m = &msg[(static_cast<std::size_t>(sib) * n_pat_ + s) * kA];
                        for (int b = 0; b < kA; ++b) o[b] *= m[b];
                        ls += ls_msg[static_cast<std::size_t>(sib) * n_pat_ + s];
                    }
                    double mx = 0.0;
                    for (int b = 0; b < kA; ++b) mx = std::max(mx, o[b]);
                    if (mx > 0.0) {
                        for (int b = 0; b < kA; ++b) o[b] /= mx;
                        ls += std::log(mx);
                    }
                    (*ls_out)[static_cast<std::size_t>(child) * n_pat_ + s] = ls;
                }
            }
        }
    }

    const Model& model_;
    Tree tree_;
    int n_seq_ = 0, n_pat_ = 0;
    std::vector<int> states_;       // n_pat x n_seq
    std::vector<double> counts_;    // columns per pattern
    std::vector<int> pattern_of_column_;
};

//! Pairwise ML distances under JTT without gamma; Jukes-Cantor start.
std::vector<double> ml_distances(const SequenceMSA& msa) {
    const Model& model = jtt();
    const int n = msa.get_n_sequences();
    const int n_col = msa.get_n_columns();
    std::vector<double> d(static_cast<std::size_t>(n) * n, 0.0);
    const int n_pairs = n * (n - 1) / 2;
#pragma omp parallel for schedule(dynamic)
    for (int pair = 0; pair < n_pairs; ++pair) {
        int i = 0, rest = pair;
        while (rest >= n - 1 - i) { rest -= n - 1 - i; ++i; }
        const int j = i + 1 + rest;
        double nxy[kA * kA] = {0};
        int compared = 0, mismatch = 0;
        for (int c = 0; c < n_col; ++c) {
            const int x = msa.get_state(i, c), y = msa.get_state(j, c);
            if (x > 0 && y > 0) {
                ++compared;
                mismatch += x != y;
                nxy[(x - 1) * kA + (y - 1)] += 1.0;
            }
        }
        const double pdiff = compared ? static_cast<double>(mismatch) / compared : 1.0;
        const double arg = 1.0 - 20.0 * pdiff / 19.0;
        double start = arg > 0.0 ? -19.0 / 20.0 * std::log(arg) : 15.0;
        if (!std::isfinite(start)) start = 1.0;
        if (start <= 0.0) start = 1e-6;
        double lo = 1e-7, hi = 5.0;
        if (start > hi) { hi = 15.0; start = 5.0; }
        auto negll = [&](double t, double* grad) {
            double pm[kA * kA], dpm[kA * kA];
            model.transition(t, pm);
            model.transition_derivative(t, dpm);
            double ll = 0.0, dll = 0.0;
            for (int k = 0; k < kA * kA; ++k) {
                if (nxy[k] > 0.0) {
                    ll += nxy[k] * std::log(pm[k]);
                    dll += nxy[k] * dpm[k] / pm[k];
                }
            }
            *grad = -dll;
            return -ll;
        };
        double value = start;
        if (compared > 0) {
            value = brent_minimise_derivative(negll, lo, std::min(std::max(start, lo), hi), hi, 0.01);
        }
        d[static_cast<std::size_t>(i) * n + j] = d[static_cast<std::size_t>(j) * n + i] = value;
    }
    return d;
}

//! One expectation-maximisation run over the branch lengths.
double branch_lengths_em(Engine& engine, const std::vector<double>& rates, int max_iter,
                         double epsilon, double tolerance) {
    Tree& tree = engine.tree();
    const int k = static_cast<int>(rates.size());
    const int n_nodes = static_cast<int>(tree.parent.size());
    double previous = -kInf;
    std::vector<double> saved = tree.length;
    for (int iter = 0; iter < max_iter; ++iter) {
        std::vector<double> site;
        const double ll = engine.log_likelihood(rates, &site);
        if (ll < previous + epsilon) {
            if (ll < previous) {
                tree.length = saved;
                return previous;
            }
            return ll;
        }
        saved = tree.length;
        previous = ll;
        const std::vector<double> counts = engine.expected_counts(rates, site);
        for (int node = 0; node < n_nodes; ++node) {
            if (node == tree.root) continue;
            const double* n_abc = &counts[static_cast<std::size_t>(node) * k * kA * kA];
            auto objective = [&](double t, double* grad) {
                double pm[kA * kA], dpm[kA * kA];
                double value = 0.0, slope = 0.0;
                for (int c = 0; c < k; ++c) {
                    const double r = rates[static_cast<std::size_t>(c)];
                    engine.model().transition(t * r, pm);
                    engine.model().transition_derivative(t * r, dpm);
                    const double* nn = n_abc + static_cast<std::size_t>(c) * kA * kA;
                    for (int q = 0; q < kA * kA; ++q) {
                        if (nn[q] > 0.0) {
                            if (pm[q] < 1e-10) {
                                value += nn[q] * std::log(1e-10);
                            } else {
                                value += nn[q] * std::log(pm[q]);
                                slope += nn[q] * r * dpm[q] / pm[q];
                            }
                        }
                    }
                }
                *grad = -slope;
                return -value;
            };
            const double start = std::min(std::max(tree.length[static_cast<std::size_t>(node)], 1e-7), 10.0);
            tree.length[static_cast<std::size_t>(node)] =
                    brent_minimise_derivative(objective, 1e-7, start, 10.0, tolerance);
        }
    }
    const double ll = engine.log_likelihood(rates);
    if (ll < previous) {
        tree.length = saved;
        return previous;
    }
    return ll;
}

//! Brent over the gamma shape on a fixed tree; returns the log-likelihood.
double optimise_alpha(const Engine& engine, int k, double start, double* alpha) {
    auto negll = [&](double a) {
        return -engine.log_likelihood(gamma_rates(std::max(a, 0.05), k));
    };
    double fmin = 0.0;
    const double best = brent_minimise(negll, 0.0, start, 5.0, 0.1, &fmin);
    *alpha = std::max(best, 0.05);
    return -fmin;
}

std::string format_g(double v, int width) {
    char buffer[64];
    std::snprintf(buffer, sizeof(buffer), "%*.4g", width, v);
    return buffer;
}

}  // namespace

// ---------------------------------------------------------------------------

SequenceConservationOptions SequenceConservationOptions::rate4site() {
    SequenceConservationOptions o;
    o.n_categories = 16;
    o.n_posterior_categories = 16;
    o.branch_lengths = "gamma";
    o.max_rounds = 2;
    o.rate_inference = "bayes";
    return o;
}

class SequenceConservationBuilder {
public:
    static SequenceConservation run(const SequenceMSA& msa, const SequenceConservationOptions& o) {
        const int n = msa.get_n_sequences();
        if (n < 3) IMP_THROW("compute_sequence_conservation: needs at least three sequences",
                             ValueException);
        if (o.n_categories < 1 || o.n_categories > 50 || o.n_posterior_categories < 1 ||
            o.n_posterior_categories > 50) {
            IMP_THROW("compute_sequence_conservation: categories must be 1..50", ValueException);
        }
        if (o.branch_lengths != "gamma" && o.branch_lengths != "homogeneous" &&
            o.branch_lengths != "none") {
            IMP_THROW("compute_sequence_conservation: branch_lengths is gamma, homogeneous or none",
                      ValueException);
        }
        if (o.rate_inference != "bayes" && o.rate_inference != "ml") {
            IMP_THROW("compute_sequence_conservation: rate_inference is bayes or ml", ValueException);
        }
        Tree tree = o.tree.empty() ? neighbour_joining(ml_distances(msa), n)
                                   : parse_newick(o.tree, msa.get_names());
        Engine engine(msa, tree);

        // the shape and the branch lengths, on every column
        double alpha = o.alpha > 0.0 ? std::max(o.alpha, 0.05) : 1.0;
        double ll = 0.0;
        const bool estimate = !(o.alpha > 0.0);
        if (o.branch_lengths == "gamma") {
            if (estimate) {
                double previous = -kInf;
                for (int round = 0; round < std::max(o.max_rounds, 1); ++round) {
                    ll = optimise_alpha(engine, o.n_categories, 1.5, &alpha);
                    if (ll < previous + 0.1) break;
                    const double after = branch_lengths_em(engine, gamma_rates(alpha, o.n_categories),
                                                           5, 0.1, 0.001);
                    const bool small = after < ll + 0.1;
                    ll = after;
                    if (small) break;
                    previous = ll;
                }
            } else {
                ll = branch_lengths_em(engine, gamma_rates(alpha, o.n_categories), 5, 0.1, 0.1);
            }
        } else if (o.branch_lengths == "homogeneous") {
            ll = branch_lengths_em(engine, std::vector<double>(1, 1.0), 5, 0.1, 0.1);
            if (estimate) ll = optimise_alpha(engine, o.n_categories, 1.0, &alpha);
            else ll = engine.log_likelihood(gamma_rates(alpha, o.n_categories));
        } else {
            if (estimate) ll = optimise_alpha(engine, o.n_categories, 1.0, &alpha);
            else ll = engine.log_likelihood(gamma_rates(alpha, o.n_categories));
        }

        SequenceConservation out;
        out.n_seq_ = n;
        out.n_categories_ = o.n_categories;
        out.alpha_ = alpha;
        out.log_likelihood_ = ll;
        out.tree_ = engine.tree().newick(msa.get_names());
        out.bayes_ = o.rate_inference == "bayes";
        const std::string alphabet = get_sequence_alphabet();
        const int ref = msa.get_reference();
        for (int c = 0; c < msa.get_n_columns(); ++c) {
            const int x = msa.get_state(ref, c);
            if (x <= 0) continue;
            out.column_.push_back(c);
            out.residues_.push_back(alphabet[static_cast<std::size_t>(x - 1)]);
            out.n_data_.push_back(get_n_with_data(msa, c));
        }
        const int np = static_cast<int>(out.column_.size());
        if (np < 2) IMP_THROW("compute_sequence_conservation: the reference has fewer than two residues",
                              ValueException);

        std::vector<double> rate(static_cast<std::size_t>(np)), lo(rate), hi(rate), sd(rate);
        if (out.bayes_) {
            const int kp = o.n_posterior_categories;
            const std::vector<double> r = gamma_rates(alpha, kp);
            const std::vector<double> lc = engine.category_log_likelihoods(r);
            for (int i = 0; i < np; ++i) {
                const int s = engine.pattern_of_column(out.column_[static_cast<std::size_t>(i)]);
                double mx = -kInf;
                for (int c = 0; c < kp; ++c) mx = std::max(mx, lc[static_cast<std::size_t>(s) * kp + c]);
                std::vector<double> post(static_cast<std::size_t>(kp));
                double z = 0.0;
                for (int c = 0; c < kp; ++c) {
                    post[static_cast<std::size_t>(c)] = std::exp(lc[static_cast<std::size_t>(s) * kp + c] - mx);
                    z += post[static_cast<std::size_t>(c)];
                }
                double mean = 0.0, second = 0.0;
                for (int c = 0; c < kp; ++c) {
                    post[static_cast<std::size_t>(c)] /= z;
                    mean += post[static_cast<std::size_t>(c)] * r[static_cast<std::size_t>(c)];
                    second += post[static_cast<std::size_t>(c)] * r[static_cast<std::size_t>(c)] *
                              r[static_cast<std::size_t>(c)];
                }
                double var = second - mean * mean;
                if (var < 0.0 && var >= -1e-4) var = 0.0;
                // the 25-75 % interval of the discrete posterior, no interpolation:
                // the category before the one crossing 25 %, and the one crossing 75 %
                double cdf = 0.0;
                int kl = -1, ku = -1;
                for (int c = 0; c < kp; ++c) {
                    cdf += post[static_cast<std::size_t>(c)];
                    if (kl < 0 && cdf > 0.25) kl = c;
                    if (kl >= 0 && ku < 0 && cdf > 0.75) { ku = c; break; }
                }
                if (kl < 0) kl = kp - 1;
                if (ku < 0) ku = kp - 1;
                rate[static_cast<std::size_t>(i)] = mean;
                sd[static_cast<std::size_t>(i)] = std::sqrt(var);
                lo[static_cast<std::size_t>(i)] = kl == 0 ? 0.0 : r[static_cast<std::size_t>(kl - 1)];
                hi[static_cast<std::size_t>(i)] = r[static_cast<std::size_t>(ku)];
            }
        } else {
            // Maximum-likelihood rate per site, homogeneous, every branch
            // scaled by the rate, over [1e-5, 20]. All sites share one pass
            // per rate on a log grid (121 points, 11 % apart); each site's
            // optimum is then refined by the parabola through its best grid
            // point and its neighbours, in log rate.
            const int n_grid = 121;
            const double lmin = std::log(1e-5), lmax = std::log(20.0);
            std::vector<double> grid(static_cast<std::size_t>(n_grid));
            std::vector<double> table(static_cast<std::size_t>(n_grid) * engine.n_patterns());
            for (int g = 0; g < n_grid; ++g) {
                grid[static_cast<std::size_t>(g)] = lmin + (lmax - lmin) * g / (n_grid - 1);
                const std::vector<double> lc = engine.category_log_likelihoods(
                        std::vector<double>(1, std::exp(grid[static_cast<std::size_t>(g)])));
                std::copy(lc.begin(), lc.end(), table.begin() + static_cast<std::ptrdiff_t>(g) * engine.n_patterns());
            }
            for (int i = 0; i < np; ++i) {
                const int s = engine.pattern_of_column(out.column_[static_cast<std::size_t>(i)]);
                auto at = [&](int g) { return table[static_cast<std::size_t>(g) * engine.n_patterns() + s]; };
                int best = 0;
                for (int g = 1; g < n_grid; ++g) if (at(g) > at(best)) best = g;
                double x = grid[static_cast<std::size_t>(best)];
                if (best > 0 && best < n_grid - 1) {
                    const double f0 = at(best - 1), f1 = at(best), f2 = at(best + 1);
                    const double denom = f0 - 2.0 * f1 + f2;
                    if (denom < 0.0) x += 0.5 * (f0 - f2) / denom * (grid[1] - grid[0]);
                }
                rate[static_cast<std::size_t>(i)] = lo[static_cast<std::size_t>(i)] =
                        hi[static_cast<std::size_t>(i)] = std::exp(x);
                sd[static_cast<std::size_t>(i)] = 0.0;
            }
        }
        out.raw_ = rate;
        double sum = 0.0, sum2 = 0.0;
        for (double v : rate) { sum += v; sum2 += v * v; }
        double mean = sum / np;
        double sdev = std::sqrt((sum2 - sum * sum / np) / (np - 1));
        if (std::fabs(mean) < 1e-9) mean = 0.0;
        if (std::fabs(sdev - 1.0) < 1e-9) sdev = 1.0;
        if (!(sdev > 0.0)) IMP_THROW("compute_sequence_conservation: every rate is the same",
                                     ValueException);
        out.mean_ = mean;
        out.sd_ = sdev;
        for (int i = 0; i < np; ++i) {
            const std::size_t k = static_cast<std::size_t>(i);
            out.score_.push_back((rate[k] - mean) / sdev);
            out.lower_.push_back((lo[k] - mean) / sdev);
            out.upper_.push_back((hi[k] - mean) / sdev);
            out.std_.push_back(sd[k] / sdev);
        }
        return out;
    }
};

SequenceConservation compute_sequence_conservation(const SequenceMSA& msa,
                                                   const SequenceConservationOptions& options) {
    return SequenceConservationBuilder::run(msa, options);
}

std::string SequenceConservation::get_rate4site_table() const {
    std::ostringstream out;
    char buffer[256];
    if (bayes_) {
        out << "#Rates were calculated using the expectation of the posterior rate distribution\n"
            << "#Prior distribution is Gamma with " << n_categories_ << " discrete categories\n\n"
            << "#SEQ: the amino acid in the reference sequence in one letter code.\n"
            << "#SCORE: The conservation scores. lower value = higher conservation.\n"
            << "#QQ-INTERVAL: the confidence interval for the rate estimates. The default interval is 25-75 percentiles\n"
            << "#STD: the standard deviation of the posterior rate distribution.\n"
            << "#MSA DATA: The number of aligned sequences having an amino acid (non-gapped) from the overall number of sequences at each position.\n\n"
            << "#POS SEQ  SCORE    QQ-INTERVAL     STD      MSA DATA\n";
        std::snprintf(buffer, sizeof(buffer), "#The alpha parameter %g\n", alpha_);
        out << buffer << "#The likelihood of the data given alpha and the tree is: \n";
        std::snprintf(buffer, sizeof(buffer), "#LL=%g\n", log_likelihood_);
        out << buffer;
        for (int i = 0; i < get_n_positions(); ++i) {
            const std::size_t k = static_cast<std::size_t>(i);
            std::snprintf(buffer, sizeof(buffer), "%5d %5c %s   [%s,%s] %s %4d/%d\n", i + 1,
                          residues_[k], format_g(score_[k], 7).c_str(),
                          format_g(lower_[k], 6).c_str(), format_g(upper_[k], 6).c_str(),
                          format_g(std_[k], 7).c_str(), n_data_[k], n_seq_);
            out << buffer;
        }
    } else {
        out << "#Rates were calculated using Maximim Likelihood\n"
            << "#The likelihood of the data given the tree is: \n";
        std::snprintf(buffer, sizeof(buffer), "#LL=%g\n", log_likelihood_);
        out << buffer
            << "#SEQ: the amino acid in the reference sequence in one letter code.\n"
            << "#SCORE: The conservation scores. lower value = higher conservation.\n"
            << "#MSA DATA: The number of aligned sequences having an amino acid (non-gapped) from the overall number of sequences at each position.\n\n"
            << "#POS SEQ  SCORE     MSA DATA\n";
        for (int i = 0; i < get_n_positions(); ++i) {
            const std::size_t k = static_cast<std::size_t>(i);
            std::snprintf(buffer, sizeof(buffer), "%5d %5c %s %4d/%d\n", i + 1, residues_[k],
                          format_g(score_[k], 7).c_str(), n_data_[k], n_seq_);
            out << buffer;
        }
    }
    out << "#Average = 0\n#Standard Deviation = 1\n";
    return out.str();
}

std::vector<int> get_consurf_grades(const std::vector<double>& scores,
                                    const std::vector<double>& lower,
                                    const std::vector<double>& upper,
                                    const std::vector<int>& n_data) {
    const std::size_t n = scores.size();
    if (lower.size() != n || upper.size() != n || n_data.size() != n) {
        IMP_THROW("get_consurf_grades: one bound pair and count per score", ValueException);
    }
    std::vector<int> out(4 * n, 0);
    if (n == 0) return out;
    // Nine equal bins from the lowest (most conserved) score m up to -m; a
    // value at or above -m is in the last bin, one below m in the first.
    const double m = *std::min_element(scores.begin(), scores.end());
    const double unit = m < 0.0 ? -m / 4.5 : m;
    auto bin = [&](double x) {
        for (int i = 0; i < 9; ++i) {
            if (x < m + i * unit) return i == 0 ? 0 : i - 1;
            if (x < m + (i + 1) * unit) return i;
        }
        return 8;
    };
    for (std::size_t i = 0; i < n; ++i) {
        const int b = bin(scores[i]);
        const int bl = bin(lower[i]);
        const int bu = bin(upper[i]);
        out[4 * i] = 9 - b;
        out[4 * i + 1] = 9 - bl;
        out[4 * i + 2] = 9 - bu;
        out[4 * i + 3] = (bu - bl > 3 || n_data[i] <= 5) ? 1 : 0;
    }
    return out;
}

std::vector<int> get_consurf_grades(const SequenceConservation& conservation) {
    return get_consurf_grades(conservation.get_scores(), conservation.get_lower(),
                              conservation.get_upper(), conservation.get_n_data());
}

IMPBFF_END_NAMESPACE
