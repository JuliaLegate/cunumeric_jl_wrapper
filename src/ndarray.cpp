#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <string_view>
#include <vector>

#include <cupynumeric.h>
#include <cupynumeric/operators.h>
#include <cupynumeric/ndarray.h>
#include <cupynumeric/runtime.h>

#include <legate.h>
#include <legate/mapping/mapping.h>

#include "ndarray_c_api.h"

constexpr uint64_t KiB = 1024ull;
constexpr uint64_t MiB = KiB * 1024ull;
constexpr uint64_t GiB = MiB * 1024ull;

// Find "--key=" and parse the decimal that follows. Returns default_val if not
// found / bad.
static uint64_t parse_flag_mb(std::string_view cfg, std::string_view key,
                              uint64_t default_val) {
  // pattern to look for: "--key="
  std::string pattern = "--";
  pattern += key;
  pattern += '=';

  size_t pos = cfg.find(pattern);
  if (pos == std::string_view::npos) return default_val;

  pos += pattern.size();
  size_t end = cfg.find_first_of(" \t", pos);
  std::string_view numstr = cfg.substr(
      pos, end == std::string_view::npos ? cfg.size() - pos : end - pos);
  if (numstr.empty()) return default_val;

  uint64_t v = 0;
  for (char c : numstr) {
    if (c < '0' || c > '9') return default_val;  // reject non-digit
    v = v * 10 + uint64_t(c - '0');
  }
  return v;  // still in MB
}

static uint64_t compute_total_fb_bytes_from_env() {
  const char* env = std::getenv("LEGATE_CONFIG");
  if (!env) return 0;

  std::string_view cfg(env);

  // Legate convention: numeric values are MB
  uint64_t fbmem_mb = parse_flag_mb(cfg, "fbmem", 0);  // per GPU
  uint64_t gpus = parse_flag_mb(cfg, "gpus", 1);       // total # GPUs requested

  if (fbmem_mb == 0) return 0;

  // aggregate across GPUs (change if you want per‑GPU instead)
  uint64_t total = fbmem_mb * gpus * MiB;
  return total;
}

extern "C" {

using cupynumeric::full;
using cupynumeric::NDArray;
using cupynumeric::random;
using cupynumeric::zeros;

using legate::Scalar;

struct CN_NDArray {
  NDArray obj;
};

struct CN_Type {
  legate::Type obj;
};

struct CN_Store {
  legate::LogicalStore obj;
};

uint64_t nda_query_device_memory() {
  uint64_t total = compute_total_fb_bytes_from_env();
  if (total == 0) total = 8ull * GiB;
  return total;
}

CN_NDArray* nda_zeros_array(int32_t dim, const uint64_t* shape, CN_Type type) {
  std::vector<uint64_t> shp(shape, shape + dim);
  NDArray result = zeros(shp, type.obj);
  return new CN_NDArray{NDArray(std::move(result))};
}

CN_NDArray* nda_full_array(int32_t dim, const uint64_t* shape, CN_Type type,
                           const void* value) {
  std::vector<uint64_t> shp(shape, shape + dim);
  Scalar s(type.obj, value, true);
  NDArray result = full(shp, s);
  return new CN_NDArray{NDArray(std::move(result))};
}

void nda_random(CN_NDArray* arr, int32_t code) { arr->obj.random(code); }

CN_NDArray* nda_random_array(int32_t dim, const uint64_t* shape) {
  std::vector<uint64_t> shp(shape, shape + dim);
  NDArray result = random(shp);
  return new CN_NDArray{NDArray(std::move(result))};
}

CN_NDArray* nda_reshape_array(CN_NDArray* arr, int32_t dim,
                              const uint64_t* shape) {
  std::vector<int64_t> shp(shape, shape + dim);
  NDArray result = cupynumeric::reshape(arr->obj, shp, "C");
  return new CN_NDArray{NDArray(std::move(result))};
}

CN_NDArray* nda_from_scalar(CN_Type type, const void* value) {
  Scalar s(type.obj, value, true);
  auto runtime = cupynumeric::CuPyNumericRuntime::get_runtime();
  auto scalar_store = runtime->create_scalar_store(s);
  return new CN_NDArray{cupynumeric::as_array(scalar_store)};
  // return new CN_NDArray{NDArray(std::move(scalar_store))};
}

// CN_NDArray* nda_from_scalar_0D(CN_Type type, const void* value) {
//   Scalar s(type.obj, value, true);
//   return new CN_NDArray{
//       legate::Runtime::get_runtime()->create_store(s, legate::Shape{})};
// }

CN_NDArray* nda_astype(CN_NDArray* arr, CN_Type type) {
  NDArray result = arr->obj.as_type(type.obj);
  return new CN_NDArray{NDArray(std::move(result))};
}

void nda_fill_array(CN_NDArray* arr, CN_Type type, const void* value) {
  Scalar s(type.obj, value, true);
  arr->obj.fill(s);
}

void nda_multiply(CN_NDArray* rhs1, CN_NDArray* rhs2, CN_NDArray* out) {
  cupynumeric::multiply(rhs1->obj, rhs2->obj, out->obj);
}

void nda_add(CN_NDArray* rhs1, CN_NDArray* rhs2, CN_NDArray* out) {
  cupynumeric::add(rhs1->obj, rhs2->obj, out->obj);
}

CN_NDArray* nda_multiply_scalar(CN_NDArray* rhs1, CN_Type type,
                                const void* value) {
  Scalar s(type.obj, value, true);
  NDArray result = rhs1->obj * s;
  return new CN_NDArray{NDArray(std::move(result))};
}

CN_NDArray* nda_add_scalar(CN_NDArray* rhs1, CN_Type type, const void* value) {
  Scalar s(type.obj, value, true);
  NDArray result = rhs1->obj + s;
  return new CN_NDArray{NDArray(std::move(result))};
}

CN_NDArray* nda_dot(CN_NDArray* rhs1, CN_NDArray* rhs2) {
  NDArray result = cupynumeric::dot(rhs1->obj, rhs2->obj);
  return new CN_NDArray{NDArray(std::move(result))};
}

void nda_three_dot_arg(CN_NDArray* rhs1, CN_NDArray* rhs2, CN_NDArray* out) {
  out->obj.dot(rhs1->obj, rhs2->obj);
}

CN_NDArray* nda_copy(CN_NDArray* arr) {
  NDArray result = arr->obj.copy();
  return new CN_NDArray{NDArray(std::move(result))};
}

void nda_assign(CN_NDArray* arr, CN_NDArray* other) {
  arr->obj.assign(other->obj);
}

void nda_move(CN_NDArray* dst, CN_NDArray* src) {
  dst->obj.operator=(std::move(src->obj));
}

void nda_destroy_array(CN_NDArray* arr) {
  if (arr != NULL) {
    delete arr;
  }
}

int32_t nda_array_dim(const CN_NDArray* arr) { return arr->obj.dim(); }

uint64_t nda_array_size(const CN_NDArray* arr) { return arr->obj.size(); }

int32_t nda_array_type_code(const CN_NDArray* arr) {
  return static_cast<int32_t>(arr->obj.type().code());
}

CN_Type* nda_array_type(const CN_NDArray* arr) {
  return new CN_Type{arr->obj.type()};
}

uint64_t nda_nbytes(CN_NDArray* arr) {
  return (uint64_t)nda_array_type(arr)->obj.size() * nda_array_size(arr);
}

void nda_array_shape(const CN_NDArray* arr, uint64_t* out_shape) {
  const auto& shp = arr->obj.shape();
  for (size_t i = 0; i < shp.size(); ++i) out_shape[i] = shp[i];
}

void nda_binary_op(CN_NDArray* out, CuPyNumericBinaryOpCode op_code,
                   const CN_NDArray* rhs1, const CN_NDArray* rhs2) {
  out->obj.binary_op(op_code, rhs1->obj, rhs2->obj);
}

void nda_binary_reduction(CN_NDArray* out, CuPyNumericBinaryOpCode op_code,
                          const CN_NDArray* rhs1, const CN_NDArray* rhs2) {
  out->obj.binary_reduction(op_code, rhs1->obj, rhs2->obj);
}

CN_NDArray* nda_array_equal(const CN_NDArray* rhs1, const CN_NDArray* rhs2) {
  return new CN_NDArray{cupynumeric::array_equal(rhs1->obj, rhs2->obj)};
}

void nda_unary_op(CN_NDArray* out, CuPyNumericUnaryOpCode op_code,
                  CN_NDArray* input) {
  out->obj.unary_op(op_code, input->obj);
}

void nda_unary_reduction(CN_NDArray* out, CuPyNumericUnaryRedCode op_code,
                         CN_NDArray* input) {
  out->obj.unary_reduction(op_code, input->obj);
}

NDArray get_slice(NDArray arr, std::vector<legate::Slice> slices) {
  switch (slices.size()) {
    case 1: {
      std::initializer_list<legate::Slice> slice_list = {slices[0]};
      return arr[slice_list];
    }
    case 2: {
      std::initializer_list<legate::Slice> slice_list = {slices[0], slices[1]};
      return arr[slice_list];
    }
    default: {
      assert(0 && "dim gteq 3 not supported yet\b");
    }
  };
  assert(0 && "you should not enter here\n");
}

CN_NDArray* nda_get_slice(CN_NDArray* arr, const CN_Slice* slices,
                          int32_t ndim) {
  std::vector<legate::Slice> slice_vec;
  slice_vec.reserve(ndim);
  for (int i = 0; i < ndim; ++i) {
    std::optional<int64_t> start = slices[i].has_start
                                       ? std::optional<int64_t>{slices[i].start}
                                       : std::nullopt;
    std::optional<int64_t> stop = slices[i].has_stop
                                      ? std::optional<int64_t>{slices[i].stop}
                                      : std::nullopt;
    slice_vec.emplace_back(legate::Slice(start, stop));
  }
  NDArray result = get_slice(arr->obj, slice_vec);
  return new CN_NDArray{NDArray(std::move(result))};
}

CN_NDArray* nda_attach_external(const void* ptr, size_t size, int dim, const uint64_t* shape, CN_Type type) {
  std::vector<uint64_t> shp_vec(shape, shape + dim);
  legate::Shape shp = legate::Shape(shp_vec);

  legate::ExternalAllocation alloc = legate::ExternalAllocation::create_sysmem(ptr, size);
  legate::mapping::DimOrdering ordering = legate::mapping::DimOrdering::fortran_order();

  auto store = legate::Runtime::get_runtime()->create_store(shp, type.obj, alloc, ordering);
  return new CN_NDArray{cupynumeric::as_array(store)};
}
}  // extern "C"
