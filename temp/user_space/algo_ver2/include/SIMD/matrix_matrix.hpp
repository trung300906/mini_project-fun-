#pragma once
#ifndef MATRIX_MATRIX_HPP
#define MATRIX_MATRIX_HPP
#include "../based/header.hpp"
// matrix-matrix operations
template <typename data_type>
void simd_add(const data_type *A, const data_type *B, data_type *C, size_t shape); // add 2 matrix(+)

template <typename data_type>
void simd_sub(const data_type *A, const data_type *B, data_type *C, size_t shape); // sub 2 matrix (-)

template <typename data_type>
void simd_mul(const data_type *A, const data_type *B, data_type *C, size_t shape); // multiply 2 matrix (*)

template <typename data_type>
void simd_div(const data_type *A, const data_type *B, data_type *C, size_t shape); // divide 2 matrix (/)

template <typename data_type>
void simd_power(const data_type *A, const data_type *B, data_type *C, size_t shape); // power of matrix

template <typename data_type>
void simd_mod(const data_type *A, const data_type *B, data_type *C, size_t shape); // mod of matrix
// boolean operations
template <typename data_type>
bool simd_eq(const data_type *a_ptr, const data_type *b_ptr, size_t size); //(==)

template <typename data_type>
bool simd_larger(const data_type *a_ptr, const data_type *b_ptr, size_t size); // for > operator

template <typename data_type>
bool simd_smaller(const data_type *a_ptr, const data_type *b_ptr, size_t size); // for < operator
// Note that the operators < and > are only relative, so we will define them as a constant, and divide them left and right, left will be smaller with operator <, and left will be larger with operator >

// for log function
template <typename data_type>
void simd_log(const data_type *A, const data_type *B, data_type *C, size_t shape); // log of matrix

template <typename data_type>
void simd_sqrt(const data_type *A, const data_type *B, data_type *C, size_t shape); // square root of matrix

template <typename data_type>
void simd_xor(const data_type *A, const data_type *B, data_type *C, size_t shape); // xor of matrix
#include "matrix_matrix.tpp"
#endif