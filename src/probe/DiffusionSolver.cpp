/**
 * \file DiffusionSolver.cpp
 * \brief Explicit propagation of an excited-state density on an AV grid.
 *
 * Copyright 2007-2026 IMP Inventors. All rights reserved.
 */
#include <IMP/bff/probe/DiffusionSolver.h>
#include <IMP/bff/util/ComputeBackend.h>
#include <IMP/bff/internal/OutputView.h>
// The solver and its adjoint are tttrlib's LatticeDiffusion.h, vendored
// verbatim (test/test_vendored_headers.py keeps the copy identical). This file
// is the IMP-facing shell: the enum names IMP.bff has always exported, the
// numpy out_view returns, nothing numerical.
#ifndef TTTRLIB_LATTICE_NAMESPACE
#define TTTRLIB_LATTICE_NAMESPACE IMP::bff::internal
#endif
#include <IMP/bff/internal/LatticeDiffusion.h>

#include <algorithm>
#include <stdexcept>


IMPBFF_BEGIN_NAMESPACE

void diffusion_step(
        const std::vector<double>& cur, const std::vector<double>& d,
        const std::vector<double>& decay, const std::vector<double>& bounds,
        int ng, int flux_form, double** out_view, int* n_out_view) {
    std::vector<double> nxt(cur.size(), 0.0);
    internal::lattice_sweep_form(cur, d, decay, bounds,
                                 internal::lattice_root_weight(std::vector<double>(), bounds),
                                 static_cast<std::size_t>(ng), flux_form, nxt);
    internal::copy_to_view(nxt, out_view, n_out_view);
}

namespace internal {

namespace {
int lattice_form(int flux_form) {
    switch (flux_form) {
        case FLUX_SMOLUCHOWSKI: return LATTICE_FLUX_SMOLUCHOWSKI;
        case FLUX_ITO: return LATTICE_FLUX_ITO;
        case FLUX_DETAILED_BALANCE: return LATTICE_FLUX_DETAILED_BALANCE;
    }
    IMP_THROW("flux_form must be FLUX_SMOLUCHOWSKI, FLUX_ITO or "
              "FLUX_DETAILED_BALANCE, not " << flux_form, IMP::ValueException);
}
//! lattice_root_weight()'s refusal, as the exception the bindings translate.
void check_weight(const std::vector<double>& weight,
                  const std::vector<double>& bounds) {
    try {
        lattice_root_weight(weight, bounds);
    } catch (const std::invalid_argument& e) {
        IMP_THROW(e.what(), IMP::ValueException);
    }
}
}  // namespace

void lattice_propagate_any(
        const std::vector<double>& cur, const std::vector<double>& d,
        const std::vector<double>& decay, const std::vector<double>& bounds,
        const std::vector<double>& weight, int ng, int flux_form, int n_steps,
        int n_out, std::vector<double>& fluorescence,
        std::vector<double>& density) {
    const int form = lattice_form(flux_form);
    const bool weighted = form == LATTICE_FLUX_DETAILED_BALANCE && !weight.empty();
    // Validated here, before a backend sees it: the GPU and the CPU must
    // refuse the same weight field.
    if (weighted) check_weight(weight, bounds);
    const std::vector<double>& checked = weight;
    // An accelerator gets the whole propagation or none of it (ComputeBackend.h).
    // It may decline per call -- too small to be worth the transfer, or a
    // precision it cannot hold -- and then this falls through to the CPU
    // below with nothing written.
    const ImpBffComputeBackend* backend = get_compute_backend();
    if (backend && backend->propagate && n_steps > 0 && n_out > 0) {
        const std::size_t n_voxel = static_cast<std::size_t>(ng) * ng * ng;
        const int n_reported = n_steps / n_out + 1;
        std::vector<double> fl(static_cast<std::size_t>(n_reported), 0.0);
        std::vector<double> dens(n_voxel, 0.0);
        const int rc = backend->propagate(
                cur.data(), d.data(), decay.data(), bounds.data(),
                weighted ? checked.data() : nullptr, ng, flux_form,
                n_steps, n_out, fl.data(), n_reported, dens.data());
        if (rc == 0) {
            fluorescence.swap(fl);
            density.swap(dens);
            return;
        }
    }
    density = lattice_propagate(cur, d, decay, bounds, ng, form, n_steps, n_out,
                                fluorescence, checked);
}

void lattice_propagate_adjoint_any(
        const std::vector<double>& cur, const std::vector<double>& d,
        const std::vector<double>& decay, const std::vector<double>& bounds,
        const std::vector<double>& weight, int ng, int flux_form, int n_steps,
        int n_out, const std::vector<double>& dL_dF,
        std::vector<double>& dL_dd, std::vector<double>& dL_ddecay,
        std::vector<double>& dL_dcur) {
    if (flux_form == FLUX_DETAILED_BALANCE) check_weight(weight, bounds);
    lattice_propagate_adjoint(cur, d, decay, bounds, ng, lattice_form(flux_form),
                              n_steps, n_out, dL_dF, dL_dd, dL_ddecay, dL_dcur,
                              weight);
}

}  // namespace internal

void diffusion_propagate(
        const std::vector<double>& cur, const std::vector<double>& d,
        const std::vector<double>& decay, const std::vector<double>& bounds,
        int ng, int flux_form, int n_steps, int n_out,
        double** out_fluorescence, int* n_out_fluorescence,
        double** out_view, int* n_out_view) {
    std::vector<double> fluorescence, final_density;
    internal::lattice_propagate_any(cur, d, decay, bounds, std::vector<double>(),
                                    ng, flux_form, n_steps, n_out, fluorescence,
                                    final_density);
    internal::copy_to_view(fluorescence, out_fluorescence, n_out_fluorescence);
    // One memcpy, against ng^3 x ~35-40 ns of marshalling.
    internal::copy_to_view(final_density, out_view, n_out_view);
}

void diffusion_propagate_adjoint(
        const std::vector<double>& cur, const std::vector<double>& d,
        const std::vector<double>& decay, const std::vector<double>& bounds,
        int ng, int flux_form, int n_steps, int n_out,
        const std::vector<double>& dL_dF,
        std::vector<double>& dL_dd, std::vector<double>& dL_ddecay,
        std::vector<double>& dL_dcur) {
    internal::lattice_propagate_adjoint_any(cur, d, decay, bounds,
                                            std::vector<double>(), ng, flux_form,
                                            n_steps, n_out, dL_dF, dL_dd,
                                            dL_ddecay, dL_dcur);
}

void diffusion_propagate_adjoint(
        const std::vector<double>& cur, const std::vector<double>& d,
        const std::vector<double>& decay, const std::vector<double>& bounds,
        int ng, int flux_form, int n_steps, int n_out,
        const std::vector<double>& dL_dF,
        double** out_view, int* n_out_view) {
    std::vector<double> gd, gdec, gcur;
    diffusion_propagate_adjoint(cur, d, decay, bounds, ng, flux_form, n_steps, n_out,
                                dL_dF, gd, gdec, gcur);
    const std::size_t nv = gd.size();
    double* buf = internal::new_double_view(3 * nv, out_view, n_out_view);
    if (buf == nullptr) return;
    std::copy(gd.begin(), gd.end(), buf);
    std::copy(gdec.begin(), gdec.end(), buf + nv);
    std::copy(gcur.begin(), gcur.end(), buf + 2 * nv);
}

IMPBFF_END_NAMESPACE
