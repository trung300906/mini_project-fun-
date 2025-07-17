```cpp
#pragma once
#ifndef SIMD_ELEMENT_WISE_HPP
#define SIMD_ELEMENT_WISE_HPP
#include "../based/header.hpp"
// element-wise operations
template <typename data_type>
void simd_elem_add(data_type *A, size_t shape, const data_type &scalor); // element-wise addition

template <typename data_type>
void simd_elem_sub(data_type *A, size_t shape, const data_type &scalor); // element-wise subtraction

template <typename data_type>
void simd_elem_mul(data_type *A, size_t shape, const data_type &scalor); // element-wise multiplication

template <typename data_type>
void simd_elem_div(data_type *A, size_t shape, const data_type &scalor); // element-wise division

template <typename data_type>
void simd_elem_power(data_type *A, size_t shape, const data_type &scalor); // element-wise power

// boolean operations
template <typename data_type>
bool simd_elem_eq(const data_type *A, size_t shape, const data_type &scalor); // element-wise equal (==)

template <typename data_type>
bool simd_elem_larger(const data_type *A, size_t shape, const data_type &scalor); // element-wise larger (>)

template <typename data_type>
bool simd_elem_smaller(const data_type *A, size_t shape, const data_type &scalor); // element-wise smaller (<)
#include "element_wise.tpp"
#endif
```
include into[[simd_index.hpp]]
linked into [[element_wise.tpp]]