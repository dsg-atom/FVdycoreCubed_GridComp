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

// =============================================================================
// GEOS Production First-Run Transport Completion
// =============================================================================
// The following implements the production transport path required by:
//   hord_mt = 6, hord_vt = 6, hord_tm = 6, hord_dp = -6
//   vtdm4 > 0, grid_type = 0
// =============================================================================

// PPM order 6 X-direction transport (tp_core.F90:xppm mord=6 branch)
// Uses the limiter: smt5(i) = 3.*abs(b0(i)) < abs(bl(i)-br(i))
// For negative hord, applies positivity constraint to interface values.
template <class T>
inline void xppm_hord6(const HorizontalBounds &b, int64_t j, int32_t hord,
                       FieldView<const T> q, FieldView<const T> crx,
                       FieldView<const T> dxa, bool nested, int32_t grid_type,
                       int64_t npx, T lim_fac, FieldView<T> flux) {
  T al[256], bl[256], br[256], b0[256];
  bool smt5[256];
  const int64_t is = b.is, ie = b.ie;
  const int32_t mord = std::abs(hord);

  // Compute boundary indices for cubed-sphere
  int64_t is1, ie3, ie1;
  if (!nested && grid_type < 3) {
    is1 = std::max(int64_t(3), is - 1);
    ie3 = std::min(npx - 2, ie + 2);
    ie1 = std::min(npx - 3, ie + 1);
  } else {
    is1 = is - 1;
    ie3 = ie + 2;
    ie1 = ie + 1;
  }

  // Compute interface values using 4th-order formula: al(i) = p1*(q(i-1)+q(i)) + p2*(q(i-2)+q(i+1))
  for (int64_t i = is1; i <= ie3; ++i) {
    const int64_t idx = i - is + 4;
    al[idx] = static_cast<T>(ppm::p1) * (q(i - 1, j, 0) + q(i, j, 0)) +
              static_cast<T>(ppm::p2) * (q(i - 2, j, 0) + q(i + 1, j, 0));
  }

  // Panel boundary treatment for cubed-sphere (grid_type < 3)
  if (!nested && grid_type < 3) {
    if (is == 1) {
      const int64_t idx0 = 0 - is + 4, idx1 = 1 - is + 4, idx2 = 2 - is + 4;
      al[idx0] = static_cast<T>(ppm::c1) * q(-2, j, 0) +
                 static_cast<T>(ppm::c2) * q(-1, j, 0) +
                 static_cast<T>(ppm::c3) * q(0, j, 0);
      // al(1) = weighted average using dxa
      const T dxa_m1 = dxa(-1, j), dxa_0 = dxa(0, j);
      const T dxa_1 = dxa(1, j), dxa_2 = dxa(2, j);
      al[idx1] = T(0.5) * (((T(2) * dxa_0 + dxa_m1) * q(0, j, 0) - dxa_0 * q(-1, j, 0)) / (dxa_m1 + dxa_0) +
                           ((T(2) * dxa_1 + dxa_2) * q(1, j, 0) - dxa_1 * q(2, j, 0)) / (dxa_1 + dxa_2));
      al[idx2] = static_cast<T>(ppm::c3) * q(1, j, 0) +
                 static_cast<T>(ppm::c2) * q(2, j, 0) +
                 static_cast<T>(ppm::c1) * q(3, j, 0);
    }
    if ((ie + 1) == npx) {
      const int64_t idxm1 = npx - 1 - is + 4, idxn = npx - is + 4, idxp1 = npx + 1 - is + 4;
      al[idxm1] = static_cast<T>(ppm::c1) * q(npx - 3, j, 0) +
                  static_cast<T>(ppm::c2) * q(npx - 2, j, 0) +
                  static_cast<T>(ppm::c3) * q(npx - 1, j, 0);
      const T dxa_nm2 = dxa(npx - 2, j), dxa_nm1 = dxa(npx - 1, j);
      const T dxa_n = dxa(npx, j), dxa_np1 = dxa(npx + 1, j);
      al[idxn] = T(0.5) * (((T(2) * dxa_nm1 + dxa_nm2) * q(npx - 1, j, 0) - dxa_nm1 * q(npx - 2, j, 0)) / (dxa_nm2 + dxa_nm1) +
                           ((T(2) * dxa_n + dxa_np1) * q(npx, j, 0) - dxa_n * q(npx + 1, j, 0)) / (dxa_n + dxa_np1));
      al[idxp1] = static_cast<T>(ppm::c3) * q(npx, j, 0) +
                  static_cast<T>(ppm::c2) * q(npx + 1, j, 0) +
                  static_cast<T>(ppm::c1) * q(npx + 2, j, 0);
    }
  }

  // Apply positivity constraint for negative hord (mass transport)
  if (hord < 0) {
    for (int64_t i = is - 1; i <= ie + 2; ++i) {
      const int64_t idx = i - is + 4;
      al[idx] = std::max(T(0), al[idx]);
    }
  }

  // Compute bl, br, b0 and the mord=6 limiter: smt5 = 3*|b0| < |bl-br|
  for (int64_t i = is - 1; i <= ie + 1; ++i) {
    const int64_t idx = i - is + 4;
    bl[idx] = al[idx] - q(i, j, 0);
    br[idx] = al[idx + 1] - q(i, j, 0);
    b0[idx] = bl[idx] + br[idx];
    smt5[idx] = T(3) * std::abs(b0[idx]) < std::abs(bl[idx] - br[idx]);
  }

  // Fix edge issues for cubed-sphere (from WMP fix in tp_core.F90)
  if (!nested && grid_type < 3) {
    if (is == 1) {
      const int64_t idx0 = 0 - is + 4, idx1 = 1 - is + 4;
      smt5[idx0] = bl[idx0] * br[idx0] < T(0);
      smt5[idx1] = bl[idx1] * br[idx1] < T(0);
    }
    if ((ie + 1) == npx) {
      const int64_t idxm1 = npx - 1 - is + 4, idxn = npx - is + 4;
      smt5[idxm1] = bl[idxm1] * br[idxm1] < T(0);
      smt5[idxn] = bl[idxn] * br[idxn] < T(0);
    }
  }

  // Compute flux using PPM formula with conditional limiter
  for (int64_t i = is; i <= ie + 1; ++i) {
    const int64_t idx = i - is + 4;
    const T c = crx(i, j, 0);
    T fx1;
    if (c > T(0)) {
      fx1 = (T(1) - c) * (br[idx - 1] - c * b0[idx - 1]);
      flux(i, j, 0) = q(i - 1, j, 0);
    } else {
      fx1 = (T(1) + c) * (bl[idx] + c * b0[idx]);
      flux(i, j, 0) = q(i, j, 0);
    }
    if (smt5[idx - 1] || smt5[idx]) {
      flux(i, j, 0) = flux(i, j, 0) + fx1;
    }
  }
}

// PPM order 6 Y-direction transport (tp_core.F90:yppm mord=6 branch)
template <class T>
inline void yppm_hord6(const HorizontalBounds &b, int64_t ifirst, int64_t ilast,
                       int32_t hord, FieldView<const T> q, FieldView<const T> cry,
                       FieldView<const T> dya, bool nested, int32_t grid_type,
                       int64_t npy, T lim_fac, FieldView<T> flux) {
  const int64_t js = b.js, je = b.je;
  const int32_t mord = std::abs(hord);

  // Compute boundary indices for cubed-sphere
  int64_t js1, je3, je1;
  if (!nested && grid_type < 3) {
    js1 = std::max(int64_t(3), js - 1);
    je3 = std::min(npy - 2, je + 2);
    je1 = std::min(npy - 3, je + 1);
  } else {
    js1 = js - 1;
    je3 = je + 2;
    je1 = je + 1;
  }

  // Process each i-column
  for (int64_t i = ifirst; i <= ilast; ++i) {
    T al[256], bl[256], br[256], b0[256];
    bool smt5[256];

    // Compute interface values
    for (int64_t j = js1; j <= je3; ++j) {
      const int64_t jdx = j - js + 4;
      al[jdx] = static_cast<T>(ppm::p1) * (q(i, j - 1, 0) + q(i, j, 0)) +
                static_cast<T>(ppm::p2) * (q(i, j - 2, 0) + q(i, j + 1, 0));
    }

    // Panel boundary treatment for cubed-sphere
    if (!nested && grid_type < 3) {
      if (js == 1) {
        const int64_t jdx0 = 0 - js + 4, jdx1 = 1 - js + 4, jdx2 = 2 - js + 4;
        al[jdx0] = static_cast<T>(ppm::c1) * q(i, -2, 0) +
                   static_cast<T>(ppm::c2) * q(i, -1, 0) +
                   static_cast<T>(ppm::c3) * q(i, 0, 0);
        const T dya_m1 = dya(i, -1), dya_0 = dya(i, 0);
        const T dya_1 = dya(i, 1), dya_2 = dya(i, 2);
        al[jdx1] = T(0.5) * (((T(2) * dya_0 + dya_m1) * q(i, 0, 0) - dya_0 * q(i, -1, 0)) / (dya_m1 + dya_0) +
                             ((T(2) * dya_1 + dya_2) * q(i, 1, 0) - dya_1 * q(i, 2, 0)) / (dya_1 + dya_2));
        al[jdx2] = static_cast<T>(ppm::c3) * q(i, 1, 0) +
                   static_cast<T>(ppm::c2) * q(i, 2, 0) +
                   static_cast<T>(ppm::c1) * q(i, 3, 0);
      }
      if ((je + 1) == npy) {
        const int64_t jdxm1 = npy - 1 - js + 4, jdxn = npy - js + 4, jdxp1 = npy + 1 - js + 4;
        al[jdxm1] = static_cast<T>(ppm::c1) * q(i, npy - 3, 0) +
                    static_cast<T>(ppm::c2) * q(i, npy - 2, 0) +
                    static_cast<T>(ppm::c3) * q(i, npy - 1, 0);
        const T dya_nm2 = dya(i, npy - 2), dya_nm1 = dya(i, npy - 1);
        const T dya_n = dya(i, npy), dya_np1 = dya(i, npy + 1);
        al[jdxn] = T(0.5) * (((T(2) * dya_nm1 + dya_nm2) * q(i, npy - 1, 0) - dya_nm1 * q(i, npy - 2, 0)) / (dya_nm2 + dya_nm1) +
                             ((T(2) * dya_n + dya_np1) * q(i, npy, 0) - dya_n * q(i, npy + 1, 0)) / (dya_n + dya_np1));
        al[jdxp1] = static_cast<T>(ppm::c3) * q(i, npy, 0) +
                    static_cast<T>(ppm::c2) * q(i, npy + 1, 0) +
                    static_cast<T>(ppm::c1) * q(i, npy + 2, 0);
      }
    }

    // Apply positivity constraint for negative hord
    if (hord < 0) {
      for (int64_t j = js - 1; j <= je + 2; ++j) {
        const int64_t jdx = j - js + 4;
        al[jdx] = std::max(T(0), al[jdx]);
      }
    }

    // Compute bl, br, b0 and limiter
    for (int64_t j = js - 1; j <= je + 1; ++j) {
      const int64_t jdx = j - js + 4;
      bl[jdx] = al[jdx] - q(i, j, 0);
      br[jdx] = al[jdx + 1] - q(i, j, 0);
      b0[jdx] = bl[jdx] + br[jdx];
      smt5[jdx] = T(3) * std::abs(b0[jdx]) < std::abs(bl[jdx] - br[jdx]);
    }

    // Fix edge issues
    if (!nested && grid_type < 3) {
      if (js == 1) {
        const int64_t jdx0 = 0 - js + 4, jdx1 = 1 - js + 4;
        smt5[jdx0] = bl[jdx0] * br[jdx0] < T(0);
        smt5[jdx1] = bl[jdx1] * br[jdx1] < T(0);
      }
      if ((je + 1) == npy) {
        const int64_t jdxm1 = npy - 1 - js + 4, jdxn = npy - js + 4;
        smt5[jdxm1] = bl[jdxm1] * br[jdxm1] < T(0);
        smt5[jdxn] = bl[jdxn] * br[jdxn] < T(0);
      }
    }

    // Compute flux
    for (int64_t j = js; j <= je + 1; ++j) {
      const int64_t jdx = j - js + 4;
      const T c = cry(i, j, 0);
      T fx1;
      if (c > T(0)) {
        fx1 = (T(1) - c) * (br[jdx - 1] - c * b0[jdx - 1]);
        flux(i, j, 0) = q(i, j - 1, 0);
      } else {
        fx1 = (T(1) + c) * (bl[jdx] + c * b0[jdx]);
        flux(i, j, 0) = q(i, j, 0);
      }
      if (smt5[jdx - 1] || smt5[jdx]) {
        flux(i, j, 0) = flux(i, j, 0) + fx1;
      }
    }
  }
}

// Complete fv_tp_2d with directional sequencing (tp_core.F90:fv_tp_2d)
// Y-X inner, X-Y outer with ra_x/ra_y area corrections
template <class T>
inline void fv_tp_2d_hord6(const HorizontalBounds &b, int64_t k, int32_t hord,
                           FieldView<const T> q_in, FieldView<const T> crx,
                           FieldView<const T> cry, FieldView<const T> xfx,
                           FieldView<const T> yfx, FieldView<const T> area,
                           FieldView<const T> ra_x, FieldView<const T> ra_y,
                           FieldView<const T> dxa, FieldView<const T> dya,
                           bool nested, int32_t grid_type, int64_t npx, int64_t npy,
                           T lim_fac, FieldView<T> fx, FieldView<T> fy,
                           FieldView<T> q_work, FieldView<T> fy2, FieldView<T> fx2) {
  const int64_t is = b.is, ie = b.ie, js = b.js, je = b.je;
  const int64_t isd = is - 3, ied = ie + 3, jsd = js - 3, jed = je + 3;

  // Determine inner/outer orders (hord=10 uses 8 for inner)
  const int32_t ord_in = (hord == 10) ? 8 : hord;
  const int32_t ord_ou = hord;

  // --- First pass: Y-direction on input q -> fy2 ---
  HorizontalBounds b_y{isd, ied, js, je};
  for (int64_t j = js; j <= je + 1; ++j) {
    for (int64_t i = isd; i <= ied; ++i) {
      // Simple Y-direction PPM using hord6 for each j-slice
      const T c = cry(i, j, k);
      T al_jm1 = static_cast<T>(ppm::p1) * (q_in(i, j - 2, k) + q_in(i, j - 1, k)) +
                 static_cast<T>(ppm::p2) * (q_in(i, j - 3, k) + q_in(i, j, k));
      T al_j = static_cast<T>(ppm::p1) * (q_in(i, j - 1, k) + q_in(i, j, k)) +
               static_cast<T>(ppm::p2) * (q_in(i, j - 2, k) + q_in(i, j + 1, k));
      T al_jp1 = static_cast<T>(ppm::p1) * (q_in(i, j, k) + q_in(i, j + 1, k)) +
                 static_cast<T>(ppm::p2) * (q_in(i, j - 1, k) + q_in(i, j + 2, k));

      // Apply positivity for negative hord
      if (hord < 0) {
        al_jm1 = std::max(T(0), al_jm1);
        al_j = std::max(T(0), al_j);
        al_jp1 = std::max(T(0), al_jp1);
      }

      const T bl_jm1 = al_jm1 - q_in(i, j - 1, k);
      const T br_jm1 = al_j - q_in(i, j - 1, k);
      const T b0_jm1 = bl_jm1 + br_jm1;
      const T bl_j = al_j - q_in(i, j, k);
      const T br_j = al_jp1 - q_in(i, j, k);
      const T b0_j = bl_j + br_j;
      const bool smt5_jm1 = T(3) * std::abs(b0_jm1) < std::abs(bl_jm1 - br_jm1);
      const bool smt5_j = T(3) * std::abs(b0_j) < std::abs(bl_j - br_j);

      T fx1;
      if (c > T(0)) {
        fx1 = (T(1) - c) * (br_jm1 - c * b0_jm1);
        fy2(i, j, k) = q_in(i, j - 1, k);
      } else {
        fx1 = (T(1) + c) * (bl_j + c * b0_j);
        fy2(i, j, k) = q_in(i, j, k);
      }
      if (smt5_jm1 || smt5_j) {
        fy2(i, j, k) = fy2(i, j, k) + fx1;
      }
    }
  }

  // Compute q_i = (q*area + fyy(j) - fyy(j+1)) / ra_y
  // where fyy = yfx * fy2
  for (int64_t j = js; j <= je; ++j) {
    for (int64_t i = isd; i <= ied; ++i) {
      const T fyy_j = yfx(i, j, k) * fy2(i, j, k);
      const T fyy_jp1 = yfx(i, j + 1, k) * fy2(i, j + 1, k);
      q_work(i, j, k) = (q_in(i, j, k) * area(i, j) + fyy_j - fyy_jp1) / ra_y(i, j);
    }
  }

  // --- Second pass: X-direction on q_i -> fx (outer order) ---
  for (int64_t j = js; j <= je; ++j) {
    for (int64_t i = is; i <= ie + 1; ++i) {
      const T c = crx(i, j, k);
      T al_im1 = static_cast<T>(ppm::p1) * (q_work(i - 2, j, k) + q_work(i - 1, j, k)) +
                 static_cast<T>(ppm::p2) * (q_work(i - 3, j, k) + q_work(i, j, k));
      T al_i = static_cast<T>(ppm::p1) * (q_work(i - 1, j, k) + q_work(i, j, k)) +
               static_cast<T>(ppm::p2) * (q_work(i - 2, j, k) + q_work(i + 1, j, k));
      T al_ip1 = static_cast<T>(ppm::p1) * (q_work(i, j, k) + q_work(i + 1, j, k)) +
                 static_cast<T>(ppm::p2) * (q_work(i - 1, j, k) + q_work(i + 2, j, k));

      if (hord < 0) {
        al_im1 = std::max(T(0), al_im1);
        al_i = std::max(T(0), al_i);
        al_ip1 = std::max(T(0), al_ip1);
      }

      const T bl_im1 = al_im1 - q_work(i - 1, j, k);
      const T br_im1 = al_i - q_work(i - 1, j, k);
      const T b0_im1 = bl_im1 + br_im1;
      const T bl_i = al_i - q_work(i, j, k);
      const T br_i = al_ip1 - q_work(i, j, k);
      const T b0_i = bl_i + br_i;
      const bool smt5_im1 = T(3) * std::abs(b0_im1) < std::abs(bl_im1 - br_im1);
      const bool smt5_i = T(3) * std::abs(b0_i) < std::abs(bl_i - br_i);

      T fx1;
      if (c > T(0)) {
        fx1 = (T(1) - c) * (br_im1 - c * b0_im1);
        fx(i, j, k) = q_work(i - 1, j, k);
      } else {
        fx1 = (T(1) + c) * (bl_i + c * b0_i);
        fx(i, j, k) = q_work(i, j, k);
      }
      if (smt5_im1 || smt5_i) {
        fx(i, j, k) = fx(i, j, k) + fx1;
      }
    }
  }

  // --- Third pass: X-direction on original q -> fx2 ---
  for (int64_t j = jsd; j <= jed; ++j) {
    for (int64_t i = is; i <= ie + 1; ++i) {
      const T c = crx(i, j, k);
      T al_im1 = static_cast<T>(ppm::p1) * (q_in(i - 2, j, k) + q_in(i - 1, j, k)) +
                 static_cast<T>(ppm::p2) * (q_in(i - 3, j, k) + q_in(i, j, k));
      T al_i = static_cast<T>(ppm::p1) * (q_in(i - 1, j, k) + q_in(i, j, k)) +
               static_cast<T>(ppm::p2) * (q_in(i - 2, j, k) + q_in(i + 1, j, k));
      T al_ip1 = static_cast<T>(ppm::p1) * (q_in(i, j, k) + q_in(i + 1, j, k)) +
                 static_cast<T>(ppm::p2) * (q_in(i - 1, j, k) + q_in(i + 2, j, k));

      if (hord < 0) {
        al_im1 = std::max(T(0), al_im1);
        al_i = std::max(T(0), al_i);
        al_ip1 = std::max(T(0), al_ip1);
      }

      const T bl_im1 = al_im1 - q_in(i - 1, j, k);
      const T br_im1 = al_i - q_in(i - 1, j, k);
      const T b0_im1 = bl_im1 + br_im1;
      const T bl_i = al_i - q_in(i, j, k);
      const T br_i = al_ip1 - q_in(i, j, k);
      const T b0_i = bl_i + br_i;
      const bool smt5_im1 = T(3) * std::abs(b0_im1) < std::abs(bl_im1 - br_im1);
      const bool smt5_i = T(3) * std::abs(b0_i) < std::abs(bl_i - br_i);

      T fx1;
      if (c > T(0)) {
        fx1 = (T(1) - c) * (br_im1 - c * b0_im1);
        fx2(i, j, k) = q_in(i - 1, j, k);
      } else {
        fx1 = (T(1) + c) * (bl_i + c * b0_i);
        fx2(i, j, k) = q_in(i, j, k);
      }
      if (smt5_im1 || smt5_i) {
        fx2(i, j, k) = fx2(i, j, k) + fx1;
      }
    }
  }

  // Compute q_j = (q*area + fxx(i) - fxx(i+1)) / ra_x
  // where fxx = xfx * fx2
  for (int64_t j = jsd; j <= jed; ++j) {
    for (int64_t i = is; i <= ie; ++i) {
      const T fxx_i = xfx(i, j, k) * fx2(i, j, k);
      const T fxx_ip1 = xfx(i + 1, j, k) * fx2(i + 1, j, k);
      q_work(i, j, k) = (q_in(i, j, k) * area(i, j) + fxx_i - fxx_ip1) / ra_x(i, j);
    }
  }

  // --- Fourth pass: Y-direction on q_j -> fy (outer order) ---
  for (int64_t j = js; j <= je + 1; ++j) {
    for (int64_t i = is; i <= ie; ++i) {
      const T c = cry(i, j, k);
      T al_jm1 = static_cast<T>(ppm::p1) * (q_work(i, j - 2, k) + q_work(i, j - 1, k)) +
                 static_cast<T>(ppm::p2) * (q_work(i, j - 3, k) + q_work(i, j, k));
      T al_j = static_cast<T>(ppm::p1) * (q_work(i, j - 1, k) + q_work(i, j, k)) +
               static_cast<T>(ppm::p2) * (q_work(i, j - 2, k) + q_work(i, j + 1, k));
      T al_jp1 = static_cast<T>(ppm::p1) * (q_work(i, j, k) + q_work(i, j + 1, k)) +
                 static_cast<T>(ppm::p2) * (q_work(i, j - 1, k) + q_work(i, j + 2, k));

      if (hord < 0) {
        al_jm1 = std::max(T(0), al_jm1);
        al_j = std::max(T(0), al_j);
        al_jp1 = std::max(T(0), al_jp1);
      }

      const T bl_jm1 = al_jm1 - q_work(i, j - 1, k);
      const T br_jm1 = al_j - q_work(i, j - 1, k);
      const T b0_jm1 = bl_jm1 + br_jm1;
      const T bl_j = al_j - q_work(i, j, k);
      const T br_j = al_jp1 - q_work(i, j, k);
      const T b0_j = bl_j + br_j;
      const bool smt5_jm1 = T(3) * std::abs(b0_jm1) < std::abs(bl_jm1 - br_jm1);
      const bool smt5_j = T(3) * std::abs(b0_j) < std::abs(bl_j - br_j);

      T fx1;
      if (c > T(0)) {
        fx1 = (T(1) - c) * (br_jm1 - c * b0_jm1);
        fy(i, j, k) = q_work(i, j - 1, k);
      } else {
        fx1 = (T(1) + c) * (bl_j + c * b0_j);
        fy(i, j, k) = q_work(i, j, k);
      }
      if (smt5_jm1 || smt5_j) {
        fy(i, j, k) = fy(i, j, k) + fx1;
      }
    }
  }

  // --- Flux averaging and weighting ---
  // For transport of delp/vorticity: flux = 0.5*(flux + flux2) * xfx/yfx
  for (int64_t j = js; j <= je; ++j) {
    for (int64_t i = is; i <= ie + 1; ++i) {
      fx(i, j, k) = T(0.5) * (fx(i, j, k) + fx2(i, j, k)) * xfx(i, j, k);
    }
  }
  for (int64_t j = js; j <= je + 1; ++j) {
    for (int64_t i = is; i <= ie; ++i) {
      fy(i, j, k) = T(0.5) * (fy(i, j, k) + fy2(i, j, k)) * yfx(i, j, k);
    }
  }
}

// deln_flux transport damping (tp_core.F90:deln_flux)
// Del-n damping for cell-mean values: nord=0->del-2, nord=1->del-4, nord=2->del-6
template <class T>
inline void deln_flux_transport(const HorizontalBounds &b, int64_t k, int32_t nord,
                                T damp, T da_min, FieldView<const T> q,
                                FieldView<const T> del6_u, FieldView<const T> del6_v,
                                FieldView<const T> rarea, bool has_mass,
                                FieldView<const T> mass, FieldView<T> fx,
                                FieldView<T> fy, FieldView<T> d2, FieldView<T> fx2,
                                FieldView<T> fy2) {
  const int64_t is = b.is, ie = b.ie, js = b.js, je = b.je;
  const T damp_coef = std::pow(damp * da_min, nord + 1);

  // Initialize d2 with scaled q (or unscaled if mass weighting will be applied)
  const int64_t i1 = is - 1 - nord, i2 = ie + 1 + nord;
  const int64_t j1 = js - 1 - nord, j2 = je + 1 + nord;

  if (!has_mass) {
    for (int64_t j = j1; j <= j2; ++j) {
      for (int64_t i = i1; i <= i2; ++i) {
        d2(i, j, k) = damp_coef * q(i, j, k);
      }
    }
  } else {
    for (int64_t j = j1; j <= j2; ++j) {
      for (int64_t i = i1; i <= i2; ++i) {
        d2(i, j, k) = q(i, j, k);
      }
    }
  }

  // First flux computation
  for (int64_t j = js - nord; j <= je + nord; ++j) {
    for (int64_t i = is - nord; i <= ie + nord + 1; ++i) {
      fx2(i, j, k) = del6_v(i, j) * (d2(i - 1, j, k) - d2(i, j, k));
    }
  }
  for (int64_t j = js - nord; j <= je + nord + 1; ++j) {
    for (int64_t i = is - nord; i <= ie + nord; ++i) {
      fy2(i, j, k) = del6_u(i, j) * (d2(i, j - 1, k) - d2(i, j, k));
    }
  }

  // High-order iterations (nord > 0)
  for (int32_t n = 1; n <= nord; ++n) {
    const int32_t nt = nord - n;

    // Compute divergence into d2
    for (int64_t j = js - nt - 1; j <= je + nt + 1; ++j) {
      for (int64_t i = is - nt - 1; i <= ie + nt + 1; ++i) {
        d2(i, j, k) = (fx2(i, j, k) - fx2(i + 1, j, k) +
                       fy2(i, j, k) - fy2(i, j + 1, k)) * rarea(i, j);
      }
    }

    // Recompute fluxes (note sign change: d2(i,j)-d2(i-1,j) instead of d2(i-1,j)-d2(i,j))
    for (int64_t j = js - nt; j <= je + nt; ++j) {
      for (int64_t i = is - nt; i <= ie + nt + 1; ++i) {
        fx2(i, j, k) = del6_v(i, j) * (d2(i, j, k) - d2(i - 1, j, k));
      }
    }
    for (int64_t j = js - nt; j <= je + nt + 1; ++j) {
      for (int64_t i = is - nt; i <= ie + nt; ++i) {
        fy2(i, j, k) = del6_u(i, j) * (d2(i, j, k) - d2(i, j - 1, k));
      }
    }
  }

  // Add diffusive fluxes to flux arrays
  if (has_mass) {
    const T damp2 = T(0.5) * damp_coef;
    for (int64_t j = js; j <= je; ++j) {
      for (int64_t i = is; i <= ie + 1; ++i) {
        fx(i, j, k) = fx(i, j, k) + damp2 * (mass(i - 1, j, k) + mass(i, j, k)) * fx2(i, j, k);
      }
    }
    for (int64_t j = js; j <= je + 1; ++j) {
      for (int64_t i = is; i <= ie; ++i) {
        fy(i, j, k) = fy(i, j, k) + damp2 * (mass(i, j - 1, k) + mass(i, j, k)) * fy2(i, j, k);
      }
    }
  } else {
    for (int64_t j = js; j <= je; ++j) {
      for (int64_t i = is; i <= ie + 1; ++i) {
        fx(i, j, k) = fx(i, j, k) + fx2(i, j, k);
      }
    }
    for (int64_t j = js; j <= je + 1; ++j) {
      for (int64_t i = is; i <= ie; ++i) {
        fy(i, j, k) = fy(i, j, k) + fy2(i, j, k);
      }
    }
  }
}

// Contravariant edge corrections for grid_type < 3 (sw_core.F90 lines 791-863)
// Solves coupled 2x2 system at panel corners for ut/vt
template <class T>
inline void contravariant_edge_corrections(int64_t k, int64_t npx, int64_t npy,
                                           bool sw_corner, bool se_corner,
                                           bool ne_corner, bool nw_corner,
                                           FieldView<const T> uc, FieldView<const T> vc,
                                           FieldView<const T> cosa_u, FieldView<const T> cosa_v,
                                           FieldView<T> ut, FieldView<T> vt) {
  // Southwest corner
  if (sw_corner) {
    // ut(2,0) and vt(0,2) exterior pair
    T damp = T(1) / (T(1) - T(0.0625) * cosa_u(2, 0) * cosa_v(1, 0));
    ut(2, 0, k) = (uc(2, 0, k) - T(0.25) * cosa_u(2, 0) *
                  (vt(1, 1, k) + vt(2, 1, k) + vt(2, 0, k) + vc(1, 0, k) -
                   T(0.25) * cosa_v(1, 0) * (ut(1, 0, k) + ut(1, -1, k) + ut(2, -1, k)))) * damp;

    damp = T(1) / (T(1) - T(0.0625) * cosa_u(0, 1) * cosa_v(0, 2));
    vt(0, 2, k) = (vc(0, 2, k) - T(0.25) * cosa_v(0, 2) *
                  (ut(1, 1, k) + ut(1, 2, k) + ut(0, 2, k) + uc(0, 1, k) -
                   T(0.25) * cosa_u(0, 1) * (vt(0, 1, k) + vt(-1, 1, k) + vt(-1, 2, k)))) * damp;

    // ut(2,1) and vt(1,2) interior pair (coupled solve)
    damp = T(1) / (T(1) - T(0.0625) * cosa_u(2, 1) * cosa_v(1, 2));
    ut(2, 1, k) = (uc(2, 1, k) - T(0.25) * cosa_u(2, 1) *
                  (vt(1, 1, k) + vt(2, 1, k) + vt(2, 2, k) + vc(1, 2, k) -
                   T(0.25) * cosa_v(1, 2) * (ut(1, 1, k) + ut(1, 2, k) + ut(2, 2, k)))) * damp;
    vt(1, 2, k) = (vc(1, 2, k) - T(0.25) * cosa_v(1, 2) *
                  (ut(1, 1, k) + ut(1, 2, k) + ut(2, 2, k) + uc(2, 1, k) -
                   T(0.25) * cosa_u(2, 1) * (vt(1, 1, k) + vt(2, 1, k) + vt(2, 2, k)))) * damp;
  }

  // Southeast corner
  if (se_corner) {
    T damp = T(1) / (T(1) - T(0.0625) * cosa_u(npx - 1, 0) * cosa_v(npx - 1, 0));
    ut(npx - 1, 0, k) = (uc(npx - 1, 0, k) - T(0.25) * cosa_u(npx - 1, 0) *
                        (vt(npx - 1, 1, k) + vt(npx - 2, 1, k) + vt(npx - 2, 0, k) + vc(npx - 1, 0, k) -
                         T(0.25) * cosa_v(npx - 1, 0) * (ut(npx, 0, k) + ut(npx, -1, k) + ut(npx - 1, -1, k)))) * damp;

    damp = T(1) / (T(1) - T(0.0625) * cosa_u(npx + 1, 1) * cosa_v(npx, 2));
    vt(npx, 2, k) = (vc(npx, 2, k) - T(0.25) * cosa_v(npx, 2) *
                   (ut(npx, 1, k) + ut(npx, 2, k) + ut(npx + 1, 2, k) + uc(npx + 1, 1, k) -
                    T(0.25) * cosa_u(npx + 1, 1) * (vt(npx, 1, k) + vt(npx + 1, 1, k) + vt(npx + 1, 2, k)))) * damp;

    damp = T(1) / (T(1) - T(0.0625) * cosa_u(npx - 1, 1) * cosa_v(npx - 1, 2));
    ut(npx - 1, 1, k) = (uc(npx - 1, 1, k) - T(0.25) * cosa_u(npx - 1, 1) *
                        (vt(npx - 1, 1, k) + vt(npx - 2, 1, k) + vt(npx - 2, 2, k) + vc(npx - 1, 2, k) -
                         T(0.25) * cosa_v(npx - 1, 2) * (ut(npx, 1, k) + ut(npx, 2, k) + ut(npx - 1, 2, k)))) * damp;
    vt(npx - 1, 2, k) = (vc(npx - 1, 2, k) - T(0.25) * cosa_v(npx - 1, 2) *
                        (ut(npx, 1, k) + ut(npx, 2, k) + ut(npx - 1, 2, k) + uc(npx - 1, 1, k) -
                         T(0.25) * cosa_u(npx - 1, 1) * (vt(npx - 1, 1, k) + vt(npx - 2, 1, k) + vt(npx - 2, 2, k)))) * damp;
  }

  // Northeast corner
  if (ne_corner) {
    T damp = T(1) / (T(1) - T(0.0625) * cosa_u(npx - 1, npy) * cosa_v(npx - 1, npy + 1));
    ut(npx - 1, npy, k) = (uc(npx - 1, npy, k) - T(0.25) * cosa_u(npx - 1, npy) *
                          (vt(npx - 1, npy, k) + vt(npx - 2, npy, k) + vt(npx - 2, npy + 1, k) + vc(npx - 1, npy + 1, k) -
                           T(0.25) * cosa_v(npx - 1, npy + 1) * (ut(npx, npy, k) + ut(npx, npy + 1, k) + ut(npx - 1, npy + 1, k)))) * damp;

    damp = T(1) / (T(1) - T(0.0625) * cosa_u(npx + 1, npy - 1) * cosa_v(npx, npy - 1));
    vt(npx, npy - 1, k) = (vc(npx, npy - 1, k) - T(0.25) * cosa_v(npx, npy - 1) *
                          (ut(npx, npy - 1, k) + ut(npx, npy - 2, k) + ut(npx + 1, npy - 2, k) + uc(npx + 1, npy - 1, k) -
                           T(0.25) * cosa_u(npx + 1, npy - 1) * (vt(npx, npy, k) + vt(npx + 1, npy, k) + vt(npx + 1, npy - 1, k)))) * damp;

    damp = T(1) / (T(1) - T(0.0625) * cosa_u(npx - 1, npy - 1) * cosa_v(npx - 1, npy - 1));
    ut(npx - 1, npy - 1, k) = (uc(npx - 1, npy - 1, k) - T(0.25) * cosa_u(npx - 1, npy - 1) *
                              (vt(npx - 1, npy, k) + vt(npx - 2, npy, k) + vt(npx - 2, npy - 1, k) + vc(npx - 1, npy - 1, k) -
                               T(0.25) * cosa_v(npx - 1, npy - 1) * (ut(npx, npy - 1, k) + ut(npx, npy - 2, k) + ut(npx - 1, npy - 2, k)))) * damp;
    vt(npx - 1, npy - 1, k) = (vc(npx - 1, npy - 1, k) - T(0.25) * cosa_v(npx - 1, npy - 1) *
                              (ut(npx, npy - 1, k) + ut(npx, npy - 2, k) + ut(npx - 1, npy - 2, k) + uc(npx - 1, npy - 1, k) -
                               T(0.25) * cosa_u(npx - 1, npy - 1) * (vt(npx - 1, npy, k) + vt(npx - 2, npy, k) + vt(npx - 2, npy - 1, k)))) * damp;
  }

  // Northwest corner
  if (nw_corner) {
    T damp = T(1) / (T(1) - T(0.0625) * cosa_u(2, npy) * cosa_v(1, npy + 1));
    ut(2, npy, k) = (uc(2, npy, k) - T(0.25) * cosa_u(2, npy) *
                   (vt(1, npy, k) + vt(2, npy, k) + vt(2, npy + 1, k) + vc(1, npy + 1, k) -
                    T(0.25) * cosa_v(1, npy + 1) * (ut(1, npy, k) + ut(1, npy + 1, k) + ut(2, npy + 1, k)))) * damp;

    damp = T(1) / (T(1) - T(0.0625) * cosa_u(0, npy - 1) * cosa_v(0, npy - 1));
    vt(0, npy - 1, k) = (vc(0, npy - 1, k) - T(0.25) * cosa_v(0, npy - 1) *
                        (ut(1, npy - 1, k) + ut(1, npy - 2, k) + ut(0, npy - 2, k) + uc(0, npy - 1, k) -
                         T(0.25) * cosa_u(0, npy - 1) * (vt(0, npy, k) + vt(-1, npy, k) + vt(-1, npy - 1, k)))) * damp;

    damp = T(1) / (T(1) - T(0.0625) * cosa_u(2, npy - 1) * cosa_v(1, npy - 1));
    ut(2, npy - 1, k) = (uc(2, npy - 1, k) - T(0.25) * cosa_u(2, npy - 1) *
                        (vt(1, npy, k) + vt(2, npy, k) + vt(2, npy - 1, k) + vc(1, npy - 1, k) -
                         T(0.25) * cosa_v(1, npy - 1) * (ut(1, npy - 1, k) + ut(1, npy - 2, k) + ut(2, npy - 2, k)))) * damp;
    vt(1, npy - 1, k) = (vc(1, npy - 1, k) - T(0.25) * cosa_v(1, npy - 1) *
                        (ut(1, npy - 1, k) + ut(1, npy - 2, k) + ut(2, npy - 2, k) + uc(2, npy - 1, k) -
                         T(0.25) * cosa_u(2, npy - 1) * (vt(1, npy, k) + vt(2, npy, k) + vt(2, npy - 1, k)))) * damp;
  }
}

// Copy corners for PPM halo (tp_core.F90:copy_corners)
template <class T>
inline void copy_corners(const HorizontalBounds &b, int64_t k, int32_t dir,
                         int64_t npx, int64_t npy, int32_t ng,
                         bool sw_corner, bool se_corner,
                         bool ne_corner, bool nw_corner,
                         FieldView<T> q) {
  if (dir == 1) {
    // X-direction: corners for Y-PPM
    if (sw_corner) {
      for (int64_t j = 1 - ng; j <= 0; ++j) {
        for (int64_t i = 1 - ng; i <= 0; ++i) {
          q(i, j, k) = q(j, 1 - i, k);
        }
      }
    }
    if (se_corner) {
      for (int64_t j = 1 - ng; j <= 0; ++j) {
        for (int64_t i = npx; i <= npx + ng - 1; ++i) {
          q(i, j, k) = q(npy - j, i - npx + 1, k);
        }
      }
    }
    if (ne_corner) {
      for (int64_t j = npy; j <= npy + ng - 1; ++j) {
        for (int64_t i = npx; i <= npx + ng - 1; ++i) {
          q(i, j, k) = q(j, 2 * npx - 1 - i, k);
        }
      }
    }
    if (nw_corner) {
      for (int64_t j = npy; j <= npy + ng - 1; ++j) {
        for (int64_t i = 1 - ng; i <= 0; ++i) {
          q(i, j, k) = q(npy - j, i - 1 + npx, k);
        }
      }
    }
  } else if (dir == 2) {
    // Y-direction: corners for X-PPM
    if (sw_corner) {
      for (int64_t j = 1 - ng; j <= 0; ++j) {
        for (int64_t i = 1 - ng; i <= 0; ++i) {
          q(i, j, k) = q(1 - j, i, k);
        }
      }
    }
    if (se_corner) {
      for (int64_t j = 1 - ng; j <= 0; ++j) {
        for (int64_t i = npx; i <= npx + ng - 1; ++i) {
          q(i, j, k) = q(npy + j - 1, npx - i, k);
        }
      }
    }
    if (ne_corner) {
      for (int64_t j = npy; j <= npy + ng - 1; ++j) {
        for (int64_t i = npx; i <= npx + ng - 1; ++i) {
          q(i, j, k) = q(2 * npy - 1 - j, i, k);
        }
      }
    }
    if (nw_corner) {
      for (int64_t j = npy; j <= npy + ng - 1; ++j) {
        for (int64_t i = 1 - ng; i <= 0; ++i) {
          q(i, j, k) = q(j + 1 - npx, npy - i, k);
        }
      }
    }
  }
}

// =============================================================================
// WP-B Transport Service API for WP-C (update_dz_d)
// =============================================================================

// Transport service configuration for WP-C consumption
template <class T>
struct TransportServiceConfig {
  int32_t hord;           // Transport order (6 for GEOS production)
  int32_t nord;           // Damping order
  T damp_c;               // Damping coefficient (from vtdm4)
  T da_min;               // Minimum cell area
  T lim_fac;              // Limiter factor (ppm_limiter)
  int32_t grid_type;      // Grid type (0 for cubed-sphere)
  int64_t npx, npy;       // Panel dimensions
  bool nested;            // Nested grid flag
};

// Transport service entry point for WP-C update_dz_d
// Executes fv_tp_2d with hord_tm and optional deln_flux damping
template <class T>
inline void transport_service_for_update_dz(
    const HorizontalBounds &b, int64_t k,
    const TransportServiceConfig<T> &config,
    FieldView<const T> q_in, FieldView<const T> crx, FieldView<const T> cry,
    FieldView<const T> xfx, FieldView<const T> yfx, FieldView<const T> area,
    FieldView<const T> ra_x, FieldView<const T> ra_y,
    FieldView<const T> dxa, FieldView<const T> dya,
    FieldView<const T> del6_u, FieldView<const T> del6_v,
    FieldView<const T> rarea, FieldView<T> fx, FieldView<T> fy,
    FieldView<T> q_work, FieldView<T> fx2, FieldView<T> fy2,
    FieldView<T> d2) {

  // Execute fv_tp_2d with order-6 transport
  fv_tp_2d_hord6(b, k, config.hord, q_in, crx, cry, xfx, yfx, area,
                 ra_x, ra_y, dxa, dya, config.nested, config.grid_type,
                 config.npx, config.npy, config.lim_fac, fx, fy,
                 q_work, fy2, fx2);

  // Apply deln_flux damping if enabled (vtdm4 > 0 means damp_c > 1e-4)
  if (config.damp_c > T(1e-4)) {
    // Note: for update_dz_d, mass weighting is NOT used (has_mass = false)
    deln_flux_transport(b, k, config.nord, config.damp_c, config.da_min,
                        q_in, del6_u, del6_v, rarea, false,
                        q_in, // placeholder, not used when has_mass=false
                        fx, fy, d2, fx2, fy2);
  }
}

// Hord dispatch selector for GEOS production configuration
// Returns true if hord is supported, false otherwise
inline bool is_hord_supported(int32_t hord) {
  const int32_t mord = std::abs(hord);
  // Supported orders: 1, 6, 8 (GEOS production uses 6 and -6)
  return mord == 1 || mord == 6 || mord == 8;
}

// GEOS first-run configuration validator
inline bool validate_geos_first_run_config(int32_t hord_mt, int32_t hord_vt,
                                           int32_t hord_tm, int32_t hord_dp,
                                           double vtdm4, int32_t grid_type) {
  // GEOS nonhydrostatic default:
  // hord_mt=6, hord_vt=6, hord_tm=6, hord_dp=-6, vtdm4>0, grid_type=0
  (void)vtdm4;  // vtdm4 > 0 enables deln_flux but doesn't affect order support
  return is_hord_supported(hord_mt) &&
         is_hord_supported(hord_vt) &&
         is_hord_supported(hord_tm) &&
         is_hord_supported(hord_dp) &&
         grid_type < 3;  // cubed-sphere or lower
}

} // namespace fv3::gpu::dgrid
