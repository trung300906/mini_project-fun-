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
void simd_elem_sub(const data_type &scalor, data_type *A, size_t shape); // element-wise subtraction

template <typename data_type>
void simd_elem_mul(data_type *A, size_t shape, const data_type &scalor); // element-wise multiplication

template <typename data_type>
void simd_elem_div(data_type *A, size_t shape, const data_type &scalor); // element-wise division

template <typename data_type>
void simd_elem_div(const data_type &scalor, data_type *A, size_t shape); // element-wise division

template <typename data_type>
void simd_elem_power(data_type *A, size_t shape, const data_type &scalor); // element-wise power

template <typename data_type>
void simd_elem_power(const data_type &scalor, data_type *A, size_t shape); // element-wise scalor power matrix

template <typename data_type>
void simd_elem_mod(data_type *A, size_t shape, const data_type &scalor); // element-wise mod

template <typename data_type>
void simd_elem_mod(const data_type &scalor, data_type *A, size_t shape); // element-wise mod

// boolean operations
template <typename data_type>
bool simd_elem_eq(const data_type *A, size_t shape, const data_type &scalor); // element-wise equal (==)

template <typename data_type>
bool simd_elem_larger(const data_type *A, size_t shape, const data_type &scalor); // element-wise larger (>)

template <typename data_type>
bool simd_elem_smaller(const data_type *A, size_t shape, const data_type &scalor); // element-wise smaller (<)

// for log SIMD
template <typename data_type>
void simd_elem_log(data_type *A, size_t shape, const data_type &scalor); // element-wise log

template <typename data_type>
void simd_elem_sqrt(const data_type &scalor, data_type *A, size_t shape);

template <typename data_type>
void simd_elem_xor(const data_type &scalor, data_type *A, size_t shape);
#include "element_wise.tpp"
#endif