#pragma once
#ifndef SEFL_VECTOR_HPP
#define SEFL_VECTOR_HPP
#include "../based/header.hpp"

// another simd function for self-vector

template <typename data_type>
void sum_avx512(const data_type *A, size_t shape, data_type &answer); // sum of vector // mark

template <typename SrcType, typename DstType>
void simd_cast(SrcType *data, size_t shape); // casting function  // mark

template <typename data_type>
void simd_sin(data_type *A, size_t shape); // sin function // mark

template <typename data_type>
void simd_cos(data_type *A, size_t shape); // cos function  // mark

template <typename data_type>
void simd_tan(data_type *A, size_t shape); // tan function //mark

template <typename data_type>
void simd_arcsin(data_type *A, size_t shape); // arcsin function // mark

template <typename data_type>
void simd_arccos(data_type *A, size_t shape); // arccos function // mark

template <typename data_type>
void simd_arctan(data_type *A, size_t shape); // arctan function // mark

template <typename data_type>
void simd_cotan(data_type *A, size_t shape); // cotan function //mark

template <typename data_type>
void simd_sinh(data_type *A, size_t shape); // sinh function // mark

template <typename data_type>
void simd_cosh(data_type *A, size_t shape); // cosh function // mark

template <typename data_type>
void simd_tanh(data_type *A, size_t shape); // tanh function // mark

template <typename data_type>
void simd_arcsinh(data_type *A, size_t shape); // arcsinh function // mark

template <typename data_type>
void simd_arccosh(data_type *A, size_t shape); // arccosh function // mark

template <typename data_type>
void simd_arctanh(data_type *A, size_t shape); // arctanh function // mark

template <typename data_type>
void simd_sec(data_type *A, size_t shape); // sec function // mark

template <typename data_type>
void simd_cosec(data_type *A, size_t shape); // cosec function // mark

template <typename data_type>
void simd_arcsec(data_type *A, size_t shape); // arc sec function // mark

template <typename data_type>
void simd_arccosec(data_type *A, size_t shape); // arc cosec function
#include "self_vector.tpp"
#endif