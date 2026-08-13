#include "fv3_cgrid_view.hpp"

#include <cassert>

using fv3::gpu::cgrid::FieldView;

int main() {
  // Fortran i-fastest allocation with nonzero/negative logical lower bounds.
  FieldView<double, 3> a{};
  a.lower[0] = -2;
  a.lower[1] = 4;
  a.lower[2] = 1;
  a.stride[0] = 1;
  a.stride[1] = 9;
  a.stride[2] = 63;
  assert(a.offset(-2, 4, 1) == 0);
  assert(a.offset(-1, 4, 1) == 1);
  assert(a.offset(-2, 5, 1) == 9);
  assert(a.offset(-2, 4, 2) == 63);
  assert(a.offset(3, 6, 4) == 212);

  FieldView<float, 2> pe_like{};
  pe_like.lower[0] = 1;
  pe_like.lower[1] = 0;
  pe_like.stride[0] = 1;
  pe_like.stride[1] = 17;
  assert(pe_like.offset(1, 0) == 0);
  assert(pe_like.offset(4, 2) == 37);
}
