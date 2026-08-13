#pragma once

#include "fv3_gpu_api.h"
#include "fv3_gpu_internal.hpp"
#include "fv3_gpu_types.hpp"

#include <cstdint>

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

} // namespace fv3::gpu::dgrid
