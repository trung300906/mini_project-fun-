include into [[element_wise.hpp]]
```cpp
#pragma once

// Explicit template instantiations
template void simd_elem_add<float>(float *, size_t, const float &);
template void simd_elem_add<double>(double *, size_t, const double &);
template void simd_elem_add<int>(int *, size_t, const int &);
template void simd_elem_add<long>(long *, size_t, const long &);

template void simd_elem_sub<float>(float *, size_t, const float &);
template void simd_elem_sub<double>(double *, size_t, const double &);
template void simd_elem_sub<int>(int *, size_t, const int &);
template void simd_elem_sub<long>(long *, size_t, const long &);

template void simd_elem_mul<float>(float *, size_t, const float &);
template void simd_elem_mul<double>(double *, size_t, const double &);
template void simd_elem_mul<int>(int *, size_t, const int &);
template void simd_elem_mul<long>(long *, size_t, const long &);

template void simd_elem_div<float>(float *, size_t, const float &);
template void simd_elem_div<double>(double *, size_t, const double &);
template void simd_elem_div<int>(int *, size_t, const int &);
template void simd_elem_div<long>(long *, size_t, const long &);

template void simd_elem_power<float>(float *, size_t, const float &);
template void simd_elem_power<double>(double *, size_t, const double &);
template void simd_elem_power<int>(int *, size_t, const int &);
template void simd_elem_power<long>(long *, size_t, const long &);

template <typename data_type>
void simd_elem_add(data_type *A, size_t shape, const data_type &scalor)
{
    size_t i = 0;
    if constexpr (std::is_same_v<data_type, float>)
    {
        __m512 scalar_vec = _mm512_set1_ps(scalor); // Broadcast scalar to all elements of the vector
        for (; i + 16 <= shape; i += 16)
        {
            __m512 a = _mm512_loadu_ps(&A[i]); // Load 16 floats from memory
            a = _mm512_add_ps(a, scalar_vec);  // Perform element-wise addition
            _mm512_storeu_ps(&A[i], a);        // Store the result back to memory
        }
    }
    else if constexpr (std::is_same_v<data_type, double>)
    {
        __m512d scalar_vec = _mm512_set1_pd(scalor); // Broadcast scalar to all elements of the vector
        for (; i + 8 <= shape; i += 8)
        {
            __m512d a = _mm512_loadu_pd(&A[i]); // Load 8 doubles from memory
            a = _mm512_add_pd(a, scalar_vec);   // Perform element-wise addition
            _mm512_storeu_pd(&A[i], a);         // Store the result back to memory
        }
    }
    else if constexpr (std::is_integral_v<data_type>)
    {
        if constexpr (sizeof(data_type) == 4) // 32-bit integers
        {
            __m512i scalar_vec = _mm512_set1_epi32(scalor); // Broadcast scalar to all elements of the vector
            for (; i + 16 <= shape; i += 16)
            {
                __m512i a = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(&A[i])); // Load 16 integers
                a = _mm512_add_epi32(a, scalar_vec);                                      // Perform element-wise addition
                _mm512_storeu_si512(reinterpret_cast<__m512i *>(&A[i]), a);               // Store the result back to memory
            }
        }
        else if constexpr (sizeof(data_type) == 8) // 64-bit integers
        {
            __m512i scalar_vec = _mm512_set1_epi64(scalor); // Broadcast scalar to all elements of the vector
            for (; i + 8 <= shape; i += 8)
            {
                __m512i a = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(&A[i])); // Load 8 integers
                a = _mm512_add_epi64(a, scalar_vec);                                      // Perform element-wise addition
                _mm512_storeu_si512(reinterpret_cast<__m512i *>(&A[i]), a);               // Store the result back to memory
            }
        }
    }
    // Process remaining elements
    for (; i < shape; i++)
    {
        A[i] += scalor;
    }
}

template <typename data_type>
void simd_elem_sub(data_type *A, size_t shape, const data_type &scalor)
{
    size_t i = 0;
    if constexpr (std::is_same_v<data_type, float>)
    {
        __m512 scalar_vec = _mm512_set1_ps(scalor);
        for (; i + 16 <= shape; i += 16)
        {
            __m512 a = _mm512_loadu_ps(&A[i]);
            a = _mm512_sub_ps(a, scalar_vec);
            _mm512_storeu_ps(&A[i], a);
        }
    }
    else if constexpr (std::is_same_v<data_type, double>)
    {
        __m512d scalar_vec = _mm512_set1_pd(scalor);
        for (; i + 8 <= shape; i += 8)
        {
            __m512d a = _mm512_loadu_pd(&A[i]);
            a = _mm512_sub_pd(a, scalar_vec);
            _mm512_storeu_pd(&A[i], a);
        }
    }
    else if constexpr (std::is_integral_v<data_type>)
    {
        if constexpr (sizeof(data_type) == 4) // 32-bit integers
        {
            __m512i scalar_vec = _mm512_set1_epi32(scalor);
            for (; i + 16 <= shape; i += 16)
            {
                __m512i a = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(&A[i]));
                a = _mm512_sub_epi32(a, scalar_vec);
                _mm512_storeu_si512(reinterpret_cast<__m512i *>(&A[i]), a);
            }
        }
        else if constexpr (sizeof(data_type) == 8) // 64-bit integers
        {
            __m512i scalar_vec = _mm512_set1_epi64(scalor);
            for (; i + 8 <= shape; i += 8)
            {
                __m512i a = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(&A[i]));
                a = _mm512_sub_epi64(a, scalar_vec);
                _mm512_storeu_si512(reinterpret_cast<__m512i *>(&A[i]), a);
            }
        }
    }
    // Process remaining elements
    for (; i < shape; i++)
    {
        A[i] -= scalor;
    }
}

template <typename data_type>
void simd_elem_mul(data_type *A, size_t shape, const data_type &scalor)
{
    size_t i = 0;
    if constexpr (std::is_same_v<data_type, float>)
    {
        __m512 scalar_vec = _mm512_set1_ps(scalor);
        for (; i + 16 <= shape; i += 16)
        {
            __m512 a = _mm512_loadu_ps(&A[i]);
            a = _mm512_mul_ps(a, scalar_vec);
            _mm512_storeu_ps(&A[i], a);
        }
    }
    else if constexpr (std::is_same_v<data_type, double>)
    {
        __m512d scalar_vec = _mm512_set1_pd(scalor);
        for (; i + 8 <= shape; i += 8)
        {
            __m512d a = _mm512_loadu_pd(&A[i]);
            a = _mm512_mul_pd(a, scalar_vec);
            _mm512_storeu_pd(&A[i], a);
        }
    }
    else if constexpr (std::is_integral_v<data_type>)
    {
        if constexpr (sizeof(data_type) == 4) // 32-bit integers
        {
            __m512i scalar_vec = _mm512_set1_epi32(scalor);
            for (; i + 16 <= shape; i += 16)
            {
                __m512i a = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(&A[i]));
                a = _mm512_mullo_epi32(a, scalar_vec);
                _mm512_storeu_si512(reinterpret_cast<__m512i *>(&A[i]), a);
            }
        }
        else if constexpr (sizeof(data_type) == 8) // 64-bit integers
        {
            __m512i scalar_vec = _mm512_set1_epi64(scalor);
            for (; i + 8 <= shape; i += 8)
            {
                __m512i a = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(&A[i]));
                a = _mm512_mullo_epi64(a, scalar_vec);
                _mm512_storeu_si512(reinterpret_cast<__m512i *>(&A[i]), a);
            }
        }
    }
    // Process remaining elements
    for (; i < shape; i++)
    {
        A[i] *= scalor;
    }
}

template <typename data_type>
void simd_elem_div(data_type *A, size_t shape, const data_type &scalor)
{
    size_t i = 0;
    if constexpr (std::is_same_v<data_type, float>)
    {
        __m512 scalar_vec = _mm512_set1_ps(scalor);
        for (; i + 16 <= shape; i += 16)
        {
            __m512 a = _mm512_loadu_ps(&A[i]);
            a = _mm512_div_ps(a, scalar_vec);
            _mm512_storeu_ps(&A[i], a);
        }
    }
    else if constexpr (std::is_same_v<data_type, double>)
    {
        __m512d scalar_vec = _mm512_set1_pd(scalor);
        for (; i + 8 <= shape; i += 8)
        {
            __m512d a = _mm512_loadu_pd(&A[i]);
            a = _mm512_div_pd(a, scalar_vec);
            _mm512_storeu_pd(&A[i], a);
        }
    }
    else if constexpr (std::is_integral_v<data_type>)
    {
        if constexpr (sizeof(data_type) == 4) // 32-bit integers
        {
            for (; i + 16 <= shape; i += 16)
            {
                __m512i vec = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(&A[i]));
                alignas(64) data_type temp[16];
                _mm512_store_si512(reinterpret_cast<__m512i *>(temp), vec);
                for (size_t j = 0; j < 16; j++)
                {
                    temp[j] /= scalor;
                }
                vec = _mm512_load_si512(reinterpret_cast<const __m512i *>(temp));
                _mm512_storeu_si512(reinterpret_cast<__m512i *>(&A[i]), vec);
            }
        }
        else if constexpr (sizeof(data_type) == 8) // 64-bit integers
        {
            for (; i + 8 <= shape; i += 8)
            {
                __m512i vec = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(&A[i]));
                alignas(64) data_type temp[8];
                _mm512_store_si512(reinterpret_cast<__m512i *>(temp), vec);
                for (size_t j = 0; j < 8; j++)
                {
                    temp[j] /= scalor;
                }
                vec = _mm512_load_si512(reinterpret_cast<const __m512i *>(temp));
                _mm512_storeu_si512(reinterpret_cast<__m512i *>(&A[i]), vec);
            }
        }
    }
    // Xử lý các phần tử còn lại
    for (; i < shape; i++)
    {
        A[i] /= scalor;
    }
}

template <typename data_type>
void simd_elem_power(data_type *A, size_t shape, const data_type &scalor)
{
    size_t i = 0;
    if constexpr (std::is_same_v<data_type, float>)
    {
        for (; i + 16 <= shape; i += 16)
        {
            __m512 a = _mm512_loadu_ps(&A[i]);

#if 0
            // Handle special cases with masks
            __mmask16 zero_mask = _mm512_cmpeq_ps_mask(a, _mm512_setzero_ps());
            __mmask16 neg_mask = _mm512_cmplt_ps_mask(a, _mm512_setzero_ps());
#endif
            // Use temporary arrays instead of non-existent _mm512_log_ps and _mm512_exp_ps
            alignas(64) float temp_in[16], temp_out[16];
            _mm512_storeu_ps(temp_in, a);

            for (int j = 0; j < 16; j++)
            {
                // Handle special cases
                if (temp_in[j] == 0.0f)
                {
                    temp_out[j] = (scalor == 0.0f) ? 1.0f : 0.0f; // 0^0 = 1, 0^n = 0
                }
                else if (temp_in[j] < 0.0f)
                {
                    // For negative bases, check if exponent is integer and odd/even
                    if (std::floor(scalor) == scalor)
                    { // Integer exponent
                        if (static_cast<int>(scalor) % 2 == 0)
                        { // Even exponent
                            temp_out[j] = std::pow(std::abs(temp_in[j]), scalor);
                        }
                        else
                        { // Odd exponent
                            temp_out[j] = -std::pow(std::abs(temp_in[j]), scalor);
                        }
                    }
                    else
                    {
                        // Negative base with non-integer exponent results in complex number
                        // Set to NaN for simplicity
                        temp_out[j] = std::numeric_limits<float>::quiet_NaN();
                    }
                }
                else
                {
                    // Normal case: positive base
                    temp_out[j] = std::pow(temp_in[j], scalor);
                }
            }

            // Load results back to SIMD register
            __m512 result = _mm512_loadu_ps(temp_out);
            _mm512_storeu_ps(&A[i], result);
        }
    }
    else if constexpr (std::is_same_v<data_type, double>)
    {
        for (; i + 8 <= shape; i += 8)
        {
            __m512d a = _mm512_loadu_pd(&A[i]);

#if 0
            // Handle special cases with masks
            __mmask8 zero_mask = _mm512_cmpeq_pd_mask(a, _mm512_setzero_pd());
            __mmask8 neg_mask = _mm512_cmplt_pd_mask(a, _mm512_setzero_pd());
#endif
            // Use temporary arrays instead of non-existent _mm512_log_pd and _mm512_exp_pd
            alignas(64) double temp_in[8], temp_out[8];
            _mm512_storeu_pd(temp_in, a);

            for (int j = 0; j < 8; j++)
            {
                // Handle special cases
                if (temp_in[j] == 0.0)
                {
                    temp_out[j] = (scalor == 0.0) ? 1.0 : 0.0; // 0^0 = 1, 0^n = 0
                }
                else if (temp_in[j] < 0.0)
                {
                    // For negative bases, check if exponent is integer and odd/even
                    if (std::floor(scalor) == scalor)
                    { // Integer exponent
                        if (static_cast<int>(scalor) % 2 == 0)
                        { // Even exponent
                            temp_out[j] = std::pow(std::abs(temp_in[j]), scalor);
                        }
                        else
                        { // Odd exponent
                            temp_out[j] = -std::pow(std::abs(temp_in[j]), scalor);
                        }
                    }
                    else
                    {
                        // Negative base with non-integer exponent results in complex number
                        temp_out[j] = std::numeric_limits<double>::quiet_NaN();
                    }
                }
                else
                {
                    // Normal case: positive base
                    temp_out[j] = std::pow(temp_in[j], scalor);
                }
            }

            // Load results back to SIMD register
            __m512d result = _mm512_loadu_pd(temp_out);
            _mm512_storeu_pd(&A[i], result);
        }
    }
    else if constexpr (std::is_integral_v<data_type>)
    {
        // Your integer implementation looks good as is
        if (scalor >= 0 && scalor <= 20) // Limit to avoid overflow
        {
            for (; i < shape; i++)
            {
                data_type result = 1;
                data_type base = A[i];
                data_type exponent = scalor;

                // Exponentiation by squaring
                while (exponent > 0)
                {
                    if (exponent % 2 == 1)
                    {
                        result *= base;
                    }
                    base *= base;
                    exponent /= 2;
                }

                A[i] = result;
            }
            return; // Early return to avoid the second loop
        }
    }

    // Process remaining elements
    for (; i < shape; i++)
    {
        if constexpr (std::is_same_v<data_type, float>)
            A[i] = __builtin_powf(A[i], scalor);
        else if constexpr (std::is_same_v<data_type, double>)
            A[i] = __builtin_pow(A[i], scalor);
        else if constexpr (std::is_integral_v<data_type>)
            A[i] = static_cast<data_type>(__builtin_pow(static_cast<double>(A[i]), static_cast<double>(scalor)));
    }
}

template <typename data_type>
bool simd_elem_eq(const data_type *A, size_t shape, const data_type &scalor)
{
    size_t i = 0;
    if constexpr (std::is_same_v<data_type, float>)
    {
        __m512 scalar_vec = _mm512_set1_ps(scalor);
        for (; i + 16 <= shape; i += 16)
        {
            __m512 a = _mm512_loadu_ps(&A[i]);
            __mmask16 mask = _mm512_cmp_ps_mask(a, scalar_vec, _CMP_EQ_OQ);
            if (mask != 0xFFFF)
                return false;
        }
    }
    else if constexpr (std::is_same_v<data_type, double>)
    {
        __m512d scalar_vec = _mm512_set1_pd(scalor);
        for (; i + 8 <= shape; i += 8)
        {
            __m512d a = _mm512_loadu_pd(&A[i]);
            __mmask8 mask = _mm512_cmp_pd_mask(a, scalar_vec, _CMP_EQ_OQ);
            if (mask != 0xFF)
                return false;
        }
    }
    else if constexpr (std::is_integral_v<data_type>)
    {
        if constexpr (sizeof(data_type) == 4) // 32-bit integers
        {
            __m512i scalar_vec = _mm512_set1_epi32(scalor);
            for (; i + 16 <= shape; i += 16)
            {
                __m512i a = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(&A[i]));
                __mmask16 mask = _mm512_cmpeq_epi32_mask(a, scalar_vec);
                if (mask != 0xFFFF)
                    return false;
            }
        }
        else if constexpr (sizeof(data_type) == 8) // 64-bit integers
        {
            __m512i scalar_vec = _mm512_set1_epi64(scalor);
            for (; i + 8 <= shape; i += 8)
            {
                __m512i a = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(&A[i]));
                __mmask8 mask = _mm512_cmpeq_epi64_mask(a, scalar_vec);
                if (mask != 0xFF)
                    return false;
            }
        }
    }
    // Process remaining elements
    for (; i < shape; i++)
    {
        if (A[i] != scalor)
            return false;
    }
    return true;
}

template <typename data_type>
bool simd_elem_larger(const data_type *A, size_t shape, const data_type &scalor)
{
    size_t i = 0;
    if constexpr (std::is_same_v<data_type, float>)
    {
        __m512 scalar_vec = _mm512_set1_ps(scalor);
        for (; i + 16 <= shape; i += 16)
        {
            __m512 a = _mm512_loadu_ps(&A[i]);
            __mmask16 mask = _mm512_cmp_ps_mask(a, scalar_vec, _CMP_GT_OQ);
            if (mask != 0x0000)
                return false;
        }
    }
    else if constexpr (std::is_same_v<data_type, double>)
    {
        __m512d scalar_vec = _mm512_set1_pd(scalor);
        for (; i + 8 <= shape; i += 8)
        {
            __m512d a = _mm512_loadu_pd(&A[i]);
            __mmask8 mask = _mm512_cmp_pd_mask(a, scalar_vec, _CMP_GT_OQ);
            if (mask != 0x00)
                return false;
        }
    }
    else if constexpr (std::is_integral_v<data_type>)
    {
        if constexpr (sizeof(data_type) == 4) // 32-bit integers
        {
            __m512i scalar_vec = _mm512_set1_epi32(scalor);
            for (; i + 16 <= shape; i += 16)
            {
                __m512i a = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(&A[i]));
                __mmask16 mask = _mm512_cmpgt_epi32_mask(a, scalar_vec);
                if (mask != 0x0000)
                    return false;
            }
        }
        else if constexpr (sizeof(data_type) == 8) // 64-bit integers
        {
            __m512i scalar_vec = _mm512_set1_epi64(scalor);
            for (; i + 8 <= shape; i += 8)
            {
                __m512i a = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(&A[i]));
                __mmask8 mask = _mm512_cmpgt_epi64_mask(a, scalar_vec);
                if (mask != 0x00)
                    return false;
            }
        }
    }
    // Process remaining elements
    for (; i < shape; i++)
    {
        if (A[i] <= scalor)
            return false;
    }
    return true;
}

template <typename data_type>
bool simd_elem_smaller(const data_type *A, size_t shape, const data_type &scalor)
{
    size_t i = 0;
    if constexpr (std::is_same_v<data_type, float>)
    {
        __m512 scalar_vec = _mm512_set1_ps(scalor);
        for (; i + 16 <= shape; i += 16)
        {
            __m512 a = _mm512_loadu_ps(&A[i]);
            __mmask16 mask = _mm512_cmp_ps_mask(a, scalar_vec, _CMP_LT_OQ);
            if (mask != 0x0000)
                return false;
        }
    }
    else if constexpr (std::is_same_v<data_type, double>)
    {
        __m512d scalar_vec = _mm512_set1_pd(scalor);
        for (; i + 8 <= shape; i += 8)
        {
            __m512d a = _mm512_loadu_pd(&A[i]);
            __mmask8 mask = _mm512_cmp_pd_mask(a, scalar_vec, _CMP_LT_OQ);
            if (mask != 0x00)
                return false;
        }
    }
    else if constexpr (std::is_integral_v<data_type>)
    {
        if constexpr (sizeof(data_type) == 4) // 32-bit integers
        {
            __m512i scalar_vec = _mm512_set1_epi32(scalor);
            for (; i + 16 <= shape; i += 16)
            {
                __m512i a = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(&A[i]));
                __mmask16 mask = _mm512_cmplt_epi32_mask(a, scalar_vec);
                if (mask != 0x0000)
                    return false;
            }
        }
        else if constexpr (sizeof(data_type) == 8) // 64-bit integers
        {
            __m512i scalar_vec = _mm512_set1_epi64(scalor);
            for (; i + 8 <= shape; i += 8)
            {
                __m512i a = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(&A[i]));
                __mmask8 mask = _mm512_cmplt_epi64_mask(a, scalar_vec);
                if (mask != 0x00)
                    return false;
            }
        }
    }
    // Process remaining elements
    for (; i < shape; i++)
    {
        if (A[i] >= scalor)
            return false;
    }
    return true;
}
```
