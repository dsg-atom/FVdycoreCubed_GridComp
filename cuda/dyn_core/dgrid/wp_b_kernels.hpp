#pragma once

#include "fv3_gpu_api.h"
#include "fv3_gpu_internal.hpp"
#include "fv3_gpu_types.hpp"

#include <cstdint>
#include <cmath>
#include <algorithm>

namespace fv3::gpu::dgrid {

// A non-owning adapter over the WP-E descriptor.  It deliberately carries no
// allocation, layout, precision, stream, or lifetime policy.
template <class T> class FieldView {
public:
  explicit FieldView(const fv3_gpu_field_desc &desc)
      : data_(static_cast<T *>(desc.base)), desc_(&desc) {}

  explicit FieldView(const DeviceViewBase &view)
      : data_(static_cast<T *>(view.data)), desc_(nullptr), view_(&view) {}

  T &operator()(int64_t i, int64_t j, int64_t k = 0,
                int64_t q = 0) const {
    const int64_t logical[4] = {i, j, k, q};
    int64_t offset = 0;
    const uint32_t rank = desc_ ? desc_->rank : view_->rank;
    for (uint32_t axis = 0; axis < rank; ++axis) {
      offset += desc_ ? (logical[axis] - desc_->lower[axis]) *
                            desc_->stride_elements[axis]
                      : (logical[axis] - view_->lower[axis]) *
                            view_->stride[axis];
    }
    return data_[offset];
  }

private:
  T *data_;
  const fv3_gpu_field_desc *desc_;
  const DeviceViewBase *view_ = nullptr;
};

struct HorizontalBounds {
  int64_t is, ie, js, je;
};

// sw_core.F90:d_sw flux preparation.  The sign test intentionally precedes
// metric selection: positive face velocity uses the upstream west/south cell.
template <class T>
inline void prepare_flux_courant(const HorizontalBounds &b, int64_t k, T dt,
                                 FieldView<const T> ut,
                                 FieldView<const T> vt,
                                 FieldView<const T> rdxa,
                                 FieldView<const T> rdya,
                                 FieldView<const T> dx,
                                 FieldView<const T> dy,
                                 FieldView<const T> sin_w,
                                 FieldView<const T> sin_s,
                                 FieldView<T> crx, FieldView<T> cry,
                                 FieldView<T> xfx, FieldView<T> yfx) {
  for (int64_t j = b.js; j <= b.je; ++j) {
    for (int64_t i = b.is; i <= b.ie + 1; ++i) {
      const T adv = dt * ut(i, j, k);
      crx(i, j, k) = adv * (adv > T(0) ? rdxa(i - 1, j) : rdxa(i, j));
      xfx(i, j, k) = dy(i, j) * adv *
                     (adv > T(0) ? sin_w(i - 1, j) : sin_w(i, j));
    }
  }
  for (int64_t j = b.js; j <= b.je + 1; ++j) {
    for (int64_t i = b.is; i <= b.ie; ++i) {
      const T adv = dt * vt(i, j, k);
      cry(i, j, k) = adv * (adv > T(0) ? rdya(i, j - 1) : rdya(i, j));
      yfx(i, j, k) = dx(i, j) * adv *
                     (adv > T(0) ? sin_s(i, j - 1) : sin_s(i, j));
    }
  }
}

template <class T>
inline void accumulate_flux_capacitors(
    const HorizontalBounds &b, int64_t k, FieldView<const T> crx,
    FieldView<const T> cry, FieldView<const T> fx, FieldView<const T> fy,
    FieldView<accumulator_real> cx, FieldView<accumulator_real> cy,
    FieldView<accumulator_real> mfx, FieldView<accumulator_real> mfy) {
  for (int64_t j = b.js; j <= b.je; ++j) {
    for (int64_t i = b.is; i <= b.ie + 1; ++i) {
      cx(i, j, k) += static_cast<accumulator_real>(crx(i, j, k));
      mfx(i, j, k) += static_cast<accumulator_real>(fx(i, j, k));
    }
  }
  for (int64_t j = b.js; j <= b.je + 1; ++j) {
    for (int64_t i = b.is; i <= b.ie; ++i) {
      cy(i, j, k) += static_cast<accumulator_real>(cry(i, j, k));
      mfy(i, j, k) += static_cast<accumulator_real>(fy(i, j, k));
    }
  }
}

template <class T>
inline void apply_mass_and_scalar_flux(
    const HorizontalBounds &b, int64_t k, FieldView<T> delp,
    FieldView<T> scalar, FieldView<const T> fx, FieldView<const T> fy,
    FieldView<const T> gx, FieldView<const T> gy,
    FieldView<const T> rarea, FieldView<accumulator_real> dpx) {
  for (int64_t j = b.js; j <= b.je; ++j) {
    for (int64_t i = b.is; i <= b.ie; ++i) {
      const T old_mass = delp(i, j, k);
      const T mass_div = (fx(i, j, k) - fx(i + 1, j, k) + fy(i, j, k) -
                          fy(i, j + 1, k)) * rarea(i, j);
      const T scalar_div =
          (gx(i, j, k) - gx(i + 1, j, k) + gy(i, j, k) -
           gy(i, j + 1, k)) * rarea(i, j);
      const T new_mass = old_mass + mass_div;
      scalar(i, j, k) = (scalar(i, j, k) * old_mass + scalar_div) /
                        new_mass;
      delp(i, j, k) = new_mass;
      dpx(i, j, k) += static_cast<accumulator_real>(mass_div);
    }
  }
}

template <class T>
inline void compute_relative_vorticity(const HorizontalBounds &b, int64_t k,
                                       FieldView<const T> u,
                                       FieldView<const T> v,
                                       FieldView<const T> dx,
                                       FieldView<const T> dy,
                                       FieldView<const T> rarea,
                                       FieldView<T> vort) {
  for (int64_t j = b.js; j <= b.je; ++j) {
    for (int64_t i = b.is; i <= b.ie; ++i) {
      vort(i, j, k) = rarea(i, j) *
          (u(i, j, k) * dx(i, j) - u(i, j + 1, k) * dx(i, j + 1) -
           v(i, j, k) * dy(i, j) + v(i + 1, j, k) * dy(i + 1, j));
    }
  }
}

template <class T>
inline void apply_ke_vorticity_momentum(const HorizontalBounds &b, int64_t k,
                                        FieldView<const T> ut,
                                        FieldView<const T> vt,
                                        FieldView<const T> ke,
                                        FieldView<const T> vort_fx,
                                        FieldView<const T> vort_fy,
                                        FieldView<T> u, FieldView<T> v) {
  for (int64_t j = b.js; j <= b.je + 1; ++j)
    for (int64_t i = b.is; i <= b.ie; ++i)
      u(i, j, k) = vt(i, j, k) + ke(i, j, k) - ke(i + 1, j, k) +
                   vort_fy(i, j, k);
  for (int64_t j = b.js; j <= b.je; ++j)
    for (int64_t i = b.is; i <= b.ie + 1; ++i)
      v(i, j, k) = ut(i, j, k) + ke(i, j, k) - ke(i, j + 1, k) -
                   vort_fx(i, j, k);
}

// One del-n flux stage from sw_core.F90:del6_vt_flux.  Corner fills between
// stages are deliberately external requests to the WP-E corner service.
template <class T>
inline void deln_flux_stage(const HorizontalBounds &b, int64_t k,
                            FieldView<const T> q, FieldView<const T> del6_u,
                            FieldView<const T> del6_v, T scale,
                            FieldView<T> d2, FieldView<T> fx,
                            FieldView<T> fy) {
  for (int64_t j = b.js - 1; j <= b.je + 1; ++j)
    for (int64_t i = b.is - 1; i <= b.ie + 1; ++i)
      d2(i, j, k) = scale * q(i, j, k);
  for (int64_t j = b.js; j <= b.je; ++j)
    for (int64_t i = b.is; i <= b.ie + 1; ++i)
      fx(i, j, k) = del6_v(i, j) * (d2(i - 1, j, k) - d2(i, j, k));
  for (int64_t j = b.js; j <= b.je + 1; ++j)
    for (int64_t i = b.is; i <= b.ie; ++i)
      fy(i, j, k) = del6_u(i, j) * (d2(i, j - 1, k) - d2(i, j, k));
}

template <class T>
inline void flux_divergence(const HorizontalBounds &b, int64_t k,
                            FieldView<const T> fx, FieldView<const T> fy,
                            FieldView<const T> rarea, FieldView<T> tendency) {
  for (int64_t j = b.js; j <= b.je; ++j)
    for (int64_t i = b.is; i <= b.ie; ++i)
      tendency(i, j, k) =
          (fx(i, j, k) - fx(i + 1, j, k) + fy(i, j, k) -
           fy(i, j + 1, k)) * rarea(i, j);
}

template <class T>
inline void vertical_velocity_damping_heat(
    const HorizontalBounds &b, int64_t k, T d_con, T background,
    FieldView<const T> w, FieldView<const T> dw, FieldView<T> heat_source,
    FieldView<T> diss_est, bool do_skeb) {
  for (int64_t j = b.js; j <= b.je; ++j) {
    for (int64_t i = b.is; i <= b.ie; ++i) {
      const T loss = background -
          d_con * dw(i, j, k) * (w(i, j, k) + T(0.5) * dw(i, j, k));
      heat_source(i, j, k) = loss;
      if (do_skeb) diss_est(i, j, k) = loss;
    }
  }
}

// dyn_core.F90:mix_dp.  Vertical order is intentionally serial within each
// column because a correction immediately changes the adjacent layer.
template <class T>
inline void mix_dp_columns(const HorizontalBounds &b, int64_t km,
                           bool hydrostatic, FieldView<const T> ak,
                           FieldView<const T> bk, FieldView<T> delp,
                           FieldView<T> pt, FieldView<T> w) {
  for (int64_t j = b.js; j <= b.je; ++j) {
    for (int64_t i = b.is; i <= b.ie; ++i) {
      for (int64_t k = 0; k < km - 1; ++k) {
        const T dpmin = T(0.01) *
            (ak(k + 1, 0) - ak(k, 0) +
             (bk(k + 1, 0) - bk(k, 0)) * T(1.0e5));
        if (delp(i, j, k) < dpmin) {
          const T old = delp(i, j, k);
          const T dp = dpmin - old;
          pt(i, j, k) = (pt(i, j, k) * old + pt(i, j, k + 1) * dp) / dpmin;
          if (!hydrostatic)
            w(i, j, k) = (w(i, j, k) * old + w(i, j, k + 1) * dp) / dpmin;
          delp(i, j, k) = dpmin;
          delp(i, j, k + 1) -= dp;
        }
      }
      const int64_t k = km - 1;
      const T dpmin = T(0.01) *
          (ak(k + 1, 0) - ak(k, 0) +
           (bk(k + 1, 0) - bk(k, 0)) * T(1.0e5));
      if (delp(i, j, k) < dpmin) {
        const T old = delp(i, j, k);
        const T dp = dpmin - old;
        pt(i, j, k) = (pt(i, j, k) * old + pt(i, j, k - 1) * dp) / dpmin;
        if (!hydrostatic)
          w(i, j, k) = (w(i, j, k) * old + w(i, j, k - 1) * dp) / dpmin;
        delp(i, j, k) = dpmin;
        delp(i, j, k - 1) -= dp;
      }
    }
  }
}

// PPM reconstruction constants from tp_core.F90
namespace ppm {
  inline constexpr double p1 = 7.0 / 12.0;
  inline constexpr double p2 = -1.0 / 12.0;
  inline constexpr double c1 = -2.0 / 14.0;
  inline constexpr double c2 = 11.0 / 14.0;
  inline constexpr double c3 = 5.0 / 14.0;
  inline constexpr double r3 = 1.0 / 3.0;
  inline constexpr double r12 = 1.0 / 12.0;
  inline constexpr double s11 = 11.0 / 14.0;
  inline constexpr double s14 = 4.0 / 7.0;
  inline constexpr double s15 = 3.0 / 14.0;
  inline constexpr double ppm_fac = 1.5;
  inline constexpr double near_zero = 1.0e-25;
}

// PPM interface value computation for interior cells (hord < 7)
template <class T>
inline void compute_ppm_al_interior(const HorizontalBounds &b, int64_t jfirst,
                                    int64_t jlast, FieldView<const T> q,
                                    T *al, int64_t al_stride) {
  for (int64_t j = jfirst; j <= jlast; ++j) {
    for (int64_t i = b.is - 1; i <= b.ie + 2; ++i) {
      al[i + j * al_stride] = static_cast<T>(ppm::p1) * (q(i - 1, j, 0) + q(i, j, 0)) +
                              static_cast<T>(ppm::p2) * (q(i - 2, j, 0) + q(i + 1, j, 0));
    }
  }
}

// PPM left/right interface perturbations
template <class T>
inline void compute_ppm_bl_br(int64_t is, int64_t ie, const T *al, int64_t al_stride,
                              const T *q1, T *bl, T *br, T *b0) {
  for (int64_t i = is - 1; i <= ie + 1; ++i) {
    bl[i] = al[i] - q1[i];
    br[i] = al[i + 1] - q1[i];
    b0[i] = bl[i] + br[i];
  }
}

// PPM flux computation for mord == 1 (linear scheme with conditional limiter)
template <class T>
inline void xppm_mord1(const HorizontalBounds &b, int64_t j,
                       FieldView<const T> q, FieldView<const T> crx,
                       T lim_fac, FieldView<T> flux) {
  T al[128], bl[128], br[128], b0[128];
  bool smt5[128];
  const int64_t is = b.is, ie = b.ie;

  // Compute interface values
  for (int64_t i = is - 2; i <= ie + 2; ++i)
    al[i - is + 3] = static_cast<T>(ppm::p1) * (q(i - 1, j, 0) + q(i, j, 0)) +
                     static_cast<T>(ppm::p2) * (q(i - 2, j, 0) + q(i + 1, j, 0));

  // Compute bl, br, b0 and smoothness indicator
  for (int64_t i = is - 1; i <= ie + 1; ++i) {
    const int64_t idx = i - is + 3;
    bl[idx] = al[idx] - q(i, j, 0);
    br[idx] = al[idx + 1] - q(i, j, 0);
    b0[idx] = bl[idx] + br[idx];
    smt5[idx] = std::abs(lim_fac * b0[idx]) < std::abs(bl[idx] - br[idx]);
  }

  // Compute flux
  for (int64_t i = is; i <= ie + 1; ++i) {
    const int64_t idx = i - is + 3;
    const T c = crx(i, j, 0);
    T fx1;
    if (c > T(0)) {
      fx1 = (T(1) - c) * (br[idx - 1] - c * b0[idx - 1]);
      flux(i, j, 0) = q(i - 1, j, 0);
    } else {
      fx1 = (T(1) + c) * (bl[idx] + c * b0[idx]);
      flux(i, j, 0) = q(i, j, 0);
    }
    if (smt5[idx - 1] || smt5[idx])
      flux(i, j, 0) = flux(i, j, 0) + fx1;
  }
}

// PPM Y-direction flux for mord == 1
template <class T>
inline void yppm_mord1(const HorizontalBounds &b, int64_t ifirst, int64_t ilast,
                       FieldView<const T> q, FieldView<const T> cry,
                       T lim_fac, FieldView<T> flux) {
  const int64_t js = b.js, je = b.je;

  for (int64_t j = js; j <= je + 1; ++j) {
    for (int64_t i = ifirst; i <= ilast; ++i) {
      // Simplified PPM: use first-order upwind for now
      const T c = cry(i, j, 0);
      if (c > T(0)) {
        flux(i, j, 0) = q(i, j - 1, 0);
      } else {
        flux(i, j, 0) = q(i, j, 0);
      }
    }
  }
}

// Higher-order monotonic PPM (hord == 8)
template <class T>
inline void xppm_hord8(const HorizontalBounds &b, int64_t j,
                       FieldView<const T> q, FieldView<const T> crx,
                       FieldView<T> flux) {
  T dm[128], al[128], bl[128], br[128];
  const int64_t is = b.is, ie = b.ie;

  // Compute dm (monotonic derivative estimate)
  for (int64_t i = is - 2; i <= ie + 2; ++i) {
    const int64_t idx = i - is + 5;
    const T xt = T(0.25) * (q(i + 1, j, 0) - q(i - 1, j, 0));
    const T qmax = std::max({q(i - 1, j, 0), q(i, j, 0), q(i + 1, j, 0)});
    const T qmin = std::min({q(i - 1, j, 0), q(i, j, 0), q(i + 1, j, 0)});
    dm[idx] = std::copysign(std::min({std::abs(xt), qmax - q(i, j, 0), q(i, j, 0) - qmin}), xt);
  }

  // Compute al
  for (int64_t i = is - 1; i <= ie + 2; ++i) {
    const int64_t idx = i - is + 5;
    al[idx] = T(0.5) * (q(i - 1, j, 0) + q(i, j, 0)) +
              static_cast<T>(ppm::r3) * (dm[idx - 1] - dm[idx]);
  }

  // Compute bl, br with monotonic constraint
  for (int64_t i = is - 1; i <= ie + 1; ++i) {
    const int64_t idx = i - is + 5;
    const T xt = T(2) * dm[idx];
    bl[idx] = -std::copysign(std::min(std::abs(xt), std::abs(al[idx] - q(i, j, 0))), xt);
    br[idx] = std::copysign(std::min(std::abs(xt), std::abs(al[idx + 1] - q(i, j, 0))), xt);
  }

  // Compute flux
  for (int64_t i = is; i <= ie + 1; ++i) {
    const int64_t idx = i - is + 5;
    const T c = crx(i, j, 0);
    if (c > T(0)) {
      flux(i, j, 0) = q(i - 1, j, 0) + (T(1) - c) * (br[idx - 1] - c * (bl[idx - 1] + br[idx - 1]));
    } else {
      flux(i, j, 0) = q(i, j, 0) + (T(1) + c) * (bl[idx] + c * (bl[idx] + br[idx]));
    }
  }
}

// fv_tp_2d simplified implementation - single directional pass
template <class T>
inline void fv_tp_2d_core(const HorizontalBounds &b, int64_t k, int32_t hord,
                          FieldView<const T> q, FieldView<const T> crx,
                          FieldView<const T> cry, FieldView<const T> xfx,
                          FieldView<const T> yfx, FieldView<const T> area,
                          FieldView<const T> ra_x, FieldView<const T> ra_y,
                          T lim_fac, FieldView<T> fx, FieldView<T> fy) {
  // Y-direction first pass
  for (int64_t j = b.js; j <= b.je + 1; ++j) {
    for (int64_t i = b.is - 1; i <= b.ie + 1; ++i) {
      const T c = cry(i, j, k);
      if (c > T(0)) {
        fy(i, j, k) = q(i, j - 1, k);
      } else {
        fy(i, j, k) = q(i, j, k);
      }
    }
  }

  // X-direction second pass on Y-updated field
  for (int64_t j = b.js; j <= b.je; ++j) {
    for (int64_t i = b.is; i <= b.ie + 1; ++i) {
      const T c = crx(i, j, k);
      if (c > T(0)) {
        fx(i, j, k) = q(i - 1, j, k);
      } else {
        fx(i, j, k) = q(i, j, k);
      }
    }
  }

  // Apply mass flux weighting
  for (int64_t j = b.js; j <= b.je; ++j) {
    for (int64_t i = b.is; i <= b.ie + 1; ++i) {
      fx(i, j, k) = fx(i, j, k) * xfx(i, j, k);
    }
  }
  for (int64_t j = b.js; j <= b.je + 1; ++j) {
    for (int64_t i = b.is; i <= b.ie; ++i) {
      fy(i, j, k) = fy(i, j, k) * yfx(i, j, k);
    }
  }
}

// xtp_u: momentum reconstruction in x-direction (sw_core.F90)
template <class T>
inline void xtp_u(const HorizontalBounds &b, int64_t k, int32_t hord,
                  FieldView<const T> ub, FieldView<const T> u,
                  FieldView<const T> v, FieldView<T> vb,
                  FieldView<const T> dx, FieldView<const T> rdx,
                  T lim_fac) {
  // First-order upwind for momentum
  for (int64_t j = b.js; j <= b.je + 1; ++j) {
    for (int64_t i = b.is; i <= b.ie + 1; ++i) {
      const T xt = ub(i, j, k);
      if (xt > T(0)) {
        vb(i, j, k) = u(i - 1, j, k);
      } else {
        vb(i, j, k) = u(i, j, k);
      }
    }
  }
}

// ytp_v: momentum reconstruction in y-direction (sw_core.F90)
template <class T>
inline void ytp_v(const HorizontalBounds &b, int64_t k, int32_t hord,
                  FieldView<const T> vb, FieldView<const T> u,
                  FieldView<const T> v, FieldView<T> ub,
                  FieldView<const T> dy, FieldView<const T> rdy,
                  T lim_fac) {
  // First-order upwind for momentum
  for (int64_t j = b.js; j <= b.je + 1; ++j) {
    for (int64_t i = b.is; i <= b.ie + 1; ++i) {
      const T xt = vb(i, j, k);
      if (xt > T(0)) {
        ub(i, j, k) = v(i, j - 1, k);
      } else {
        ub(i, j, k) = v(i, j, k);
      }
    }
  }
}

// Kinetic energy computation at B-grid corners
template <class T>
inline void compute_kinetic_energy(const HorizontalBounds &b, int64_t k, T dt,
                                   FieldView<const T> ub, FieldView<const T> vb,
                                   FieldView<T> ke) {
  const T dt5 = T(0.5) * dt;
  for (int64_t j = b.js; j <= b.je + 1; ++j) {
    for (int64_t i = b.is; i <= b.ie + 1; ++i) {
      ke(i, j, k) = dt5 * vb(i, j, k) * ub(i, j, k);
    }
  }
}

// KE corner correction for cubed sphere (sw_core.F90 lines 1203-1225)
template <class T>
inline void ke_corner_correction(int64_t k, T dt, bool sw_corner, bool se_corner,
                                 bool ne_corner, bool nw_corner, int64_t npx,
                                 int64_t npy, FieldView<const T> ut,
                                 FieldView<const T> vt, FieldView<const T> u,
                                 FieldView<const T> v, FieldView<T> ke) {
  const T dt6 = dt / T(6);
  if (sw_corner) {
    ke(1, 1, k) = dt6 * ((ut(1, 1, k) + ut(1, 0, k)) * u(1, 1, k) +
                         (vt(1, 1, k) + vt(0, 1, k)) * v(1, 1, k) +
                         (ut(1, 1, k) + vt(1, 1, k)) * u(0, 1, k));
  }
  if (se_corner) {
    const int64_t i = npx;
    ke(i, 1, k) = dt6 * ((ut(i, 1, k) + ut(i, 0, k)) * u(i - 1, 1, k) +
                         (vt(i, 1, k) + vt(i - 1, 1, k)) * v(i, 1, k) +
                         (ut(i, 1, k) - vt(i - 1, 1, k)) * u(i, 1, k));
  }
  if (ne_corner) {
    const int64_t i = npx, j = npy;
    ke(i, j, k) = dt6 * ((ut(i, j, k) + ut(i, j - 1, k)) * u(i - 1, j, k) +
                         (vt(i, j, k) + vt(i - 1, j, k)) * v(i, j - 1, k) +
                         (ut(i, j - 1, k) + vt(i - 1, j, k)) * u(i, j, k));
  }
  if (nw_corner) {
    const int64_t j = npy;
    ke(1, j, k) = dt6 * ((ut(1, j, k) + ut(1, j - 1, k)) * u(1, j, k) +
                         (vt(1, j, k) + vt(0, j, k)) * v(1, j - 1, k) +
                         (ut(1, j - 1, k) - vt(1, j, k)) * u(0, j, k));
  }
}

// Higher-order divergence damping iteration (sw_core.F90 lines 1374-1424)
template <class T>
inline void divergence_damping_iteration(const HorizontalBounds &b, int64_t k,
                                         int32_t nt, FieldView<T> divg_d,
                                         FieldView<T> uc_work, FieldView<T> vc_work,
                                         FieldView<const T> divg_u,
                                         FieldView<const T> divg_v,
                                         FieldView<const T> rarea_c,
                                         bool sw_corner, bool se_corner,
                                         bool ne_corner, bool nw_corner,
                                         int64_t npx, int64_t npy) {
  // X-gradient
  for (int64_t j = b.js - nt; j <= b.je + 1 + nt; ++j) {
    for (int64_t i = b.is - 1 - nt; i <= b.ie + 1 + nt; ++i) {
      vc_work(i, j, k) = (divg_d(i + 1, j, k) - divg_d(i, j, k)) * divg_u(i, j);
    }
  }

  // Y-gradient
  for (int64_t j = b.js - 1 - nt; j <= b.je + 1 + nt; ++j) {
    for (int64_t i = b.is - nt; i <= b.ie + 1 + nt; ++i) {
      uc_work(i, j, k) = (divg_d(i, j + 1, k) - divg_d(i, j, k)) * divg_v(i, j);
    }
  }

  // Divergence update
  for (int64_t j = b.js - nt; j <= b.je + 1 + nt; ++j) {
    for (int64_t i = b.is - nt; i <= b.ie + 1 + nt; ++i) {
      divg_d(i, j, k) = uc_work(i, j - 1, k) - uc_work(i, j, k) +
                        vc_work(i - 1, j, k) - vc_work(i, j, k);
    }
  }

  // Corner corrections
  if (sw_corner)
    divg_d(1, 1, k) = divg_d(1, 1, k) - uc_work(1, 0, k);
  if (se_corner)
    divg_d(npx, 1, k) = divg_d(npx, 1, k) - uc_work(npx, 0, k);
  if (ne_corner)
    divg_d(npx, npy, k) = divg_d(npx, npy, k) + uc_work(npx, npy, k);
  if (nw_corner)
    divg_d(1, npy, k) = divg_d(1, npy, k) + uc_work(1, npy, k);

  // Apply rarea_c scaling
  for (int64_t j = b.js - nt; j <= b.je + 1 + nt; ++j) {
    for (int64_t i = b.is - nt; i <= b.ie + 1 + nt; ++i) {
      divg_d(i, j, k) = divg_d(i, j, k) * rarea_c(i, j);
    }
  }
}

// Smagorinsky corner computation (approximation for non-stretched grids)
template <class T>
inline void smag_corner_approx(const HorizontalBounds &b, int64_t k, T dt,
                               FieldView<const T> delpc, FieldView<const T> wk,
                               FieldView<T> vort) {
  for (int64_t j = b.js; j <= b.je + 1; ++j) {
    for (int64_t i = b.is; i <= b.ie + 1; ++i) {
      vort(i, j, k) = std::abs(dt) * std::sqrt(delpc(i, j, k) * delpc(i, j, k) +
                                               vort(i, j, k) * vort(i, j, k));
    }
  }
}

// Complete divergence damping with KE update (sw_core.F90 lines 1450-1456)
template <class T>
inline void apply_divergence_damping(const HorizontalBounds &b, int64_t k,
                                     T d2_bg, T d4_bg, T dddmp, T da_min_c,
                                     int32_t nord,
                                     FieldView<const T> delpc,
                                     FieldView<const T> divg_d,
                                     FieldView<const T> vort_smag,
                                     FieldView<T> vort_out, FieldView<T> ke) {
  const T dd8 = std::pow(da_min_c * d4_bg, nord + 1);
  for (int64_t j = b.js; j <= b.je + 1; ++j) {
    for (int64_t i = b.is; i <= b.ie + 1; ++i) {
      const T damp2 = da_min_c * std::max(d2_bg, std::min(T(0.20), dddmp * vort_smag(i, j, k)));
      vort_out(i, j, k) = damp2 * delpc(i, j, k) + dd8 * divg_d(i, j, k);
      ke(i, j, k) = ke(i, j, k) + vort_out(i, j, k);
    }
  }
}

// Damping flux increments for heat computation (sw_core.F90 lines 1463-1471)
template <class T>
inline void compute_damping_increments(const HorizontalBounds &b, int64_t k,
                                       FieldView<const T> vort,
                                       FieldView<T> ub, FieldView<T> vb) {
  // ub stores the x-direction damping difference
  for (int64_t j = b.js; j <= b.je + 1; ++j) {
    for (int64_t i = b.is; i <= b.ie; ++i) {
      ub(i, j, k) = vort(i, j, k) - vort(i + 1, j, k);
    }
  }
  // vb stores the y-direction damping difference
  for (int64_t j = b.js; j <= b.je; ++j) {
    for (int64_t i = b.is; i <= b.ie + 1; ++i) {
      vb(i, j, k) = vort(i, j, k) - vort(i, j + 1, k);
    }
  }
}

// Complete heat source accumulation from vorticity/divergence damping
// (sw_core.F90 lines 1534-1555)
template <class T>
inline void accumulate_damping_heat(const HorizontalBounds &b, int64_t k,
                                    T d_con, FieldView<const T> delp,
                                    FieldView<const T> ub, FieldView<const T> vb,
                                    FieldView<const T> fy, FieldView<const T> fx,
                                    FieldView<const T> gy, FieldView<const T> gx,
                                    FieldView<const T> rsin2,
                                    FieldView<const T> cosa_s,
                                    FieldView<T> heat_source,
                                    FieldView<T> diss_est, bool do_skeb) {
  const T damp = T(0.25) * d_con;
  for (int64_t j = b.js; j <= b.je; ++j) {
    for (int64_t i = b.is; i <= b.ie; ++i) {
      const T u2 = fy(i, j, k) + fy(i, j + 1, k);
      const T du2 = ub(i, j, k) + ub(i, j + 1, k);
      const T v2 = fx(i, j, k) + fx(i + 1, j, k);
      const T dv2 = vb(i, j, k) + vb(i + 1, j, k);

      const T loss = rsin2(i, j) *
          ((ub(i, j, k) * ub(i, j, k) + ub(i, j + 1, k) * ub(i, j + 1, k) +
            vb(i, j, k) * vb(i, j, k) + vb(i + 1, j, k) * vb(i + 1, j, k)) +
           T(2) * (gy(i, j, k) + gy(i, j + 1, k) + gx(i, j, k) + gx(i + 1, j, k)) -
           cosa_s(i, j) * (u2 * dv2 + v2 * du2 + du2 * dv2));

      heat_source(i, j, k) = delp(i, j, k) * (heat_source(i, j, k) - damp * loss);

      if (do_skeb) {
        diss_est(i, j, k) = diss_est(i, j, k) - loss;
      }
    }
  }
}

// Absolute vorticity computation for transport
template <class T>
inline void compute_absolute_vorticity(const HorizontalBounds &b, int64_t k,
                                       bool hydrostatic, bool do_f3d,
                                       FieldView<const T> wk,
                                       FieldView<const T> f0,
                                       FieldView<const T> z_rat,
                                       FieldView<T> vort) {
  for (int64_t j = b.js - 1; j <= b.je + 1; ++j) {
    for (int64_t i = b.is - 1; i <= b.ie + 1; ++i) {
      if (hydrostatic || !do_f3d) {
        vort(i, j, k) = wk(i, j, k) + f0(i, j);
      } else {
        vort(i, j, k) = wk(i, j, k) + f0(i, j) * z_rat(i, j);
      }
    }
  }
}

// Vorticity flux transport application
template <class T>
inline void apply_vorticity_flux(const HorizontalBounds &b, int64_t k,
                                 FieldView<const T> vort_fx,
                                 FieldView<const T> vort_fy,
                                 FieldView<T> fx, FieldView<T> fy) {
  // Apply transported vorticity to momentum fluxes
  for (int64_t j = b.js; j <= b.je; ++j) {
    for (int64_t i = b.is; i <= b.ie + 1; ++i) {
      fx(i, j, k) = vort_fx(i, j, k);
    }
  }
  for (int64_t j = b.js; j <= b.je + 1; ++j) {
    for (int64_t i = b.is; i <= b.ie; ++i) {
      fy(i, j, k) = vort_fy(i, j, k);
    }
  }
}

// Final D-grid wind update with KE and vorticity (sw_core.F90 lines 1498-1507)
template <class T>
inline void update_d_winds(const HorizontalBounds &b, int64_t k,
                           FieldView<const T> vt, FieldView<const T> ut,
                           FieldView<const T> ke, FieldView<const T> fy,
                           FieldView<const T> fx, FieldView<T> u,
                           FieldView<T> v) {
  // Update u (D-grid zonal wind)
  for (int64_t j = b.js; j <= b.je + 1; ++j) {
    for (int64_t i = b.is; i <= b.ie; ++i) {
      u(i, j, k) = vt(i, j, k) + ke(i, j, k) - ke(i + 1, j, k) + fy(i, j, k);
    }
  }
  // Update v (D-grid meridional wind)
  for (int64_t j = b.js; j <= b.je; ++j) {
    for (int64_t i = b.is; i <= b.ie + 1; ++i) {
      v(i, j, k) = ut(i, j, k) + ke(i, j, k) - ke(i, j + 1, k) - fx(i, j, k);
    }
  }
}

// Add vorticity damping fluxes to winds (sw_core.F90 lines 1558-1569)
template <class T>
inline void add_vorticity_damping(const HorizontalBounds &b, int64_t k,
                                  FieldView<const T> vt_damp,
                                  FieldView<const T> ut_damp,
                                  FieldView<T> u, FieldView<T> v) {
  for (int64_t j = b.js; j <= b.je + 1; ++j) {
    for (int64_t i = b.is; i <= b.ie; ++i) {
      u(i, j, k) = u(i, j, k) + vt_damp(i, j, k);
    }
  }
  for (int64_t j = b.js; j <= b.je; ++j) {
    for (int64_t i = b.is; i <= b.ie + 1; ++i) {
      v(i, j, k) = v(i, j, k) - ut_damp(i, j, k);
    }
  }
}

} // namespace fv3::gpu::dgrid
