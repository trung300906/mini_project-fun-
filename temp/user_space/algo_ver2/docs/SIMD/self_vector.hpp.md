```cpp
#pragma once
#ifndef SEFL_VECTOR_HPP
#define SEFL_VECTOR_HPP
#include "../based/header.hpp"

// another simd function for self-vector

template <typename data_type>
void sum_avx512(const data_type *A, size_t shape, data_type &answer); // sum of vector

#include "self_vector.tpp"
#endif
```
include into[[simd_index.hpp]]
linked into [[self_vector.tpp]]