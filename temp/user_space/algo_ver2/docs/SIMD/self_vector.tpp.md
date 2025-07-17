include into [[self_vector.hpp]]
```cpp
#pragma once

// Explicit template instantiations
template void sum_avx512<float>(const float *, size_t, float &);
template void sum_avx512<double>(const double *, size_t, double &);
template void sum_avx512<int>(const int *, size_t, int &);
template void sum_avx512<long>(const long *, size_t, long &);

template <typename data_type>
void sum_avx512(const data_type *A, size_t size, data_type &answer)
{
    constexpr size_t vec_size = 64 / sizeof(data_type); // Kích thước vector AVX-512
    alignas(64) data_type buffer[vec_size] = {};        // Bộ nhớ căn chỉnh 64-byte
    answer = 0;                                         // Đảm bảo kết quả khởi tạo đúng

    size_t i = 0;

    if constexpr (std::is_same_v<data_type, float>)
    { // SIMD cho float (16 phần tử mỗi lần)
        __m512 sum_vec = _mm512_setzero_ps();

        for (; i + vec_size <= size; i += vec_size)
        {
            __m512 vec = _mm512_loadu_ps(&A[i]); // Load 16 float
            sum_vec = _mm512_add_ps(sum_vec, vec);
        }

        // Performance option 1: Use store and loop
        _mm512_store_ps(buffer, sum_vec); // Lưu vào buffer căn chỉnh
        for (size_t j = 0; j < vec_size; ++j)
            answer += buffer[j];

        // Performance option 2 (alternative): Use AVX-512 reduction
        // answer += _mm512_reduce_add_ps(sum_vec);
    }
    else if constexpr (std::is_same_v<data_type, double>)
    { // SIMD cho double (8 phần tử mỗi lần)
        __m512d sum_vec = _mm512_setzero_pd();

        for (; i + vec_size <= size; i += vec_size)
        {
            __m512d vec = _mm512_loadu_pd(&A[i]); // Load 8 double
            sum_vec = _mm512_add_pd(sum_vec, vec);
        }

        // Performance option 1: Use store and loop
        _mm512_store_pd(buffer, sum_vec);
        for (size_t j = 0; j < vec_size; ++j)
            answer += buffer[j];

        // Performance option 2 (alternative): Use AVX-512 reduction
        // answer += _mm512_reduce_add_pd(sum_vec);
    }
    else if constexpr (std::is_integral_v<data_type>)
    {                                         // SIMD cho số nguyên
        if constexpr (sizeof(data_type) == 4) // int32_t
        {
            __m512i sum_vec = _mm512_setzero_epi32();

            for (; i + vec_size <= size; i += vec_size) // Use vec_size instead of hardcoded 16
            {
                __m512i vec = _mm512_loadu_epi32(&A[i]); // Load 16 int32_t
                sum_vec = _mm512_add_epi32(sum_vec, vec);
            }

            _mm512_store_epi32(buffer, sum_vec);
            for (size_t j = 0; j < vec_size; ++j) // Use vec_size instead of hardcoded 16
                answer += buffer[j];
        }
        else if constexpr (sizeof(data_type) == 8) // int64_t
        {
            __m512i sum_vec = _mm512_setzero_si512();

            for (; i + vec_size <= size; i += vec_size) // Use vec_size instead of hardcoded 8
            {
                __m512i vec = _mm512_loadu_epi64(&A[i]); // Load 8 int64_t
                sum_vec = _mm512_add_epi64(sum_vec, vec);
            }

            _mm512_store_epi64(buffer, sum_vec);
            for (size_t j = 0; j < vec_size; ++j) // Use vec_size instead of hardcoded 8
                answer += buffer[j];
        }
    }

    // Cộng nốt phần dư
    for (; i < size; ++i)
    {
        answer += A[i];
    }
}
```
