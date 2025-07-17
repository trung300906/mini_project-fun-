include into [[matrix_matrix.hpp]]
```cpp
#pragma once

// Explicit template instantiations
template void simd_add<float>(const float *, const float *, float *, size_t);
template void simd_add<double>(const double *, const double *, double *, size_t);
template void simd_add<int>(const int *, const int *, int *, size_t);
template void simd_add<long>(const long *, const long *, long *, size_t);

template void simd_sub<float>(const float *, const float *, float *, size_t);
template void simd_sub<double>(const double *, const double *, double *, size_t);
template void simd_sub<int>(const int *, const int *, int *, size_t);
template void simd_sub<long>(const long *, const long *, long *, size_t);

template void simd_mul<float>(const float *, const float *, float *, size_t);
template void simd_mul<double>(const double *, const double *, double *, size_t);
template void simd_mul<int>(const int *, const int *, int *, size_t);
template void simd_mul<long>(const long *, const long *, long *, size_t);

template void simd_div<float>(const float *, const float *, float *, size_t);
template void simd_div<double>(const double *, const double *, double *, size_t);
template void simd_div<int>(const int *, const int *, int *, size_t);
template void simd_div<long>(const long *, const long *, long *, size_t);

template void simd_power<float>(const float *, const float *, float *, size_t);
template void simd_power<double>(const double *, const double *, double *, size_t);
template void simd_power<int>(const int *, const int *, int *, size_t);
template void simd_power<long>(const long *, const long *, long *, size_t);

// boolean operations for matrix_matrix
// explicit template instantiations
template bool simd_eq<float>(const float *, const float *, size_t);
template bool simd_eq<double>(const double *, const double *, size_t);
template bool simd_eq<int>(const int *, const int *, size_t);
template bool simd_eq<long>(const long *, const long *, size_t);

template bool simd_larger<float>(const float *, const float *, size_t);
template bool simd_larger<double>(const double *, const double *, size_t);
template bool simd_larger<int>(const int *, const int *, size_t);
template bool simd_larger<long>(const long *, const long *, size_t);

template bool simd_smaller<float>(const float *, const float *, size_t);
template bool simd_smaller<double>(const double *, const double *, size_t);
template bool simd_smaller<int>(const int *, const int *, size_t);
template bool simd_smaller<long>(const long *, const long *, size_t);

//==========================================================================//
template <typename data_type>
void simd_add(const data_type *A, const data_type *B, data_type *C, size_t shape)
{
    size_t i = 0;
    if constexpr (std::is_same_v<data_type, float>)
    {
        for (; i + 16 <= shape; i += 16)
        {
            __m512 a = _mm512_loadu_ps(&A[i]);
            __m512 b = _mm512_loadu_ps(&B[i]);
            __m512 c = _mm512_add_ps(a, b);
            _mm512_storeu_ps(&C[i], c);
        }
    }
    else if constexpr (std::is_same_v<data_type, double>)
    {
        for (; i + 8 <= shape; i += 8)
        {
            __m512d a = _mm512_loadu_pd(&A[i]);
            __m512d b = _mm512_loadu_pd(&B[i]);
            __m512d c = _mm512_add_pd(a, b);
            _mm512_storeu_pd(&C[i], c);
        }
    }
    else if constexpr (std::is_integral_v<data_type>)
    {
        using int_type = std::conditional_t<(sizeof(data_type) == 8), __m512i, __m512i>;
        for (; i + 16 <= shape; i += 16)
        {
            int_type a = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(&A[i]));
            int_type b = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(&B[i]));
            int_type c = _mm512_add_epi32(a, b);
            _mm512_storeu_si512(reinterpret_cast<__m512i *>(&C[i]), c);
        }
    }
    // process data haven't finished yet
    for (; i < shape; i++)
    {
        C[i] = A[i] + B[i];
    }
}

template <typename data_type>
void simd_sub(const data_type *A, const data_type *B, data_type *C, size_t shape)
{
    size_t i = 0;
    if constexpr (std::is_same_v<data_type, float>)
    {
        for (; i + 16 <= shape; i += 16)
        {
            __m512 a = _mm512_loadu_ps(&A[i]);
            __m512 b = _mm512_loadu_ps(&B[i]);
            __m512 c = _mm512_sub_ps(a, b);
            _mm512_storeu_ps(&C[i], c);
        }
    }
    else if constexpr (std::is_same_v<data_type, double>)
    {
        for (; i + 8 <= shape; i += 8)
        {
            __m512d a = _mm512_loadu_pd(&A[i]);
            __m512d b = _mm512_loadu_pd(&B[i]);
            __m512d c = _mm512_sub_pd(a, b);
            _mm512_storeu_pd(&C[i], c);
        }
    }
    else if constexpr (std::is_integral_v<data_type>)
    {
        using int_type = std::conditional_t<(sizeof(data_type) == 8), __m512i, __m512i>;
        for (; i + 16 <= shape; i += 16)
        {
            int_type a = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(&A[i]));
            int_type b = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(&B[i]));
            int_type c = _mm512_sub_epi32(a, b);
            _mm512_storeu_si512(reinterpret_cast<__m512i *>(&C[i]), c);
        }
    }
    // process data haven't finished yet
    for (; i < shape; i++)
    {
        C[i] = A[i] - B[i];
    }
}

template <typename data_type>
void simd_mul(const data_type *A, const data_type *B, data_type *C, size_t shape)
{
    size_t i = 0;
    if constexpr (std::is_same_v<data_type, float>)
    {
        for (; i + 16 <= shape; i += 16)
        {
            __m512 a = _mm512_loadu_ps(&A[i]);
            __m512 b = _mm512_loadu_ps(&B[i]);
            __m512 c = _mm512_mul_ps(a, b);
            _mm512_storeu_ps(&C[i], c);
        }
    }
    else if constexpr (std::is_same_v<data_type, double>)
    {
        for (; i + 8 <= shape; i += 8)
        {
            __m512d a = _mm512_loadu_pd(&A[i]);
            __m512d b = _mm512_loadu_pd(&B[i]);
            __m512d c = _mm512_mul_pd(a, b);
            _mm512_storeu_pd(&C[i], c);
        }
    }
    else if constexpr (std::is_integral_v<data_type>)
    {
        using int_type = std::conditional_t<(sizeof(data_type) == 8), __m512i, __m512i>;
        for (; i + 16 <= shape; i += 16)
        {
            int_type a = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(&A[i]));
            int_type b = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(&B[i]));
            int_type c = _mm512_mullo_epi32(a, b);
            _mm512_storeu_si512(reinterpret_cast<__m512i *>(&C[i]), c);
        }
    }
    // process data haven't finished yet
    for (; i < shape; i++)
    {
        C[i] = A[i] * B[i];
    }
}

template <typename data_type>
void simd_div(const data_type *A, const data_type *B, data_type *C, size_t shape)
{
    size_t i = 0;
    if constexpr (std::is_same_v<data_type, float>)
    {
        for (; i + 16 <= shape; i += 16)
        {
            __m512 a = _mm512_loadu_ps(&A[i]);
            __m512 b = _mm512_loadu_ps(&B[i]);
            __m512 c = _mm512_div_ps(a, b);
            _mm512_storeu_ps(&C[i], c);
        }
    }
    else if constexpr (std::is_same_v<data_type, double>)
    {
        for (; i + 8 <= shape; i += 8)
        {
            __m512d a = _mm512_loadu_pd(&A[i]);
            __m512d b = _mm512_loadu_pd(&B[i]);
            __m512d c = _mm512_div_pd(a, b);
            _mm512_storeu_pd(&C[i], c);
        }
    }
    else if constexpr (std::is_integral_v<data_type>)
    {
        if constexpr (sizeof(data_type) == 4) // 32-bit integers
        {
            for (; i + 16 <= shape; i += 16)
            {
                __m512i a = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(&A[i]));
                __m512i b = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(&B[i]));

                // Store to temporary arrays for scalar division
                alignas(64) int32_t temp_a[16], temp_b[16], temp_c[16];
                _mm512_storeu_si512(reinterpret_cast<__m512i *>(temp_a), a);
                _mm512_storeu_si512(reinterpret_cast<__m512i *>(temp_b), b);

                // Perform scalar division
                for (int j = 0; j < 16; j++)
                {
                    // Check for division by zero
                    if (temp_b[j] != 0)
                        temp_c[j] = temp_a[j] / temp_b[j];
                    else
                        temp_c[j] = 0; // Or set to a defined value for division by zero
                }

                // Load results back to SIMD register and store
                __m512i c = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(temp_c));
                _mm512_storeu_si512(reinterpret_cast<__m512i *>(&C[i]), c);
            }
        }
        else if constexpr (sizeof(data_type) == 8) // 64-bit integers
        {
            for (; i + 8 <= shape; i += 8)
            {
                __m512i a = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(&A[i]));
                __m512i b = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(&B[i]));

                // Store to temporary arrays for scalar division
                alignas(64) int64_t temp_a[8], temp_b[8], temp_c[8];
                _mm512_storeu_si512(reinterpret_cast<__m512i *>(temp_a), a);
                _mm512_storeu_si512(reinterpret_cast<__m512i *>(temp_b), b);

                // Perform scalar division
                for (int j = 0; j < 8; j++)
                {
                    // Check for division by zero
                    if (temp_b[j] != 0)
                        temp_c[j] = temp_a[j] / temp_b[j];
                    else
                        temp_c[j] = 0; // Or set to a defined value for division by zero
                }

                // Load results back to SIMD register and store
                __m512i c = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(temp_c));
                _mm512_storeu_si512(reinterpret_cast<__m512i *>(&C[i]), c);
            }
        }
    }
    // process data haven't finished yet
    for (; i < shape; i++)
    {
        C[i] = A[i] / B[i];
    }
}

template <typename data_type>
void simd_power(const data_type *A, const data_type *B, data_type *C, size_t shape)
{
    size_t i = 0;
    if constexpr (std::is_same_v<data_type, float>)
    {
        for (; i + 16 <= shape; i += 16)
        {
            __m512 a = _mm512_loadu_ps(&A[i]);
            __m512 b = _mm512_loadu_ps(&B[i]);
#if 0
            // Create masks for special cases
            __mmask16 zero_base_mask = _mm512_cmpeq_ps_mask(a, _mm512_setzero_ps());
            __mmask16 neg_base_mask = _mm512_cmplt_ps_mask(a, _mm512_setzero_ps());
#endif
            // Take absolute value of bases for log calculation
            __m512 abs_a = _mm512_abs_ps(a);

            // For pow(x,y) = exp(y * log(x))
            // Note: This uses temp arrays since direct SVML functions might not be available
            alignas(64) float temp_a[16], temp_b[16], temp_c[16];
            _mm512_storeu_ps(temp_a, abs_a);
            _mm512_storeu_ps(temp_b, b);

            for (int j = 0; j < 16; j++)
            {
                // Handle special cases
                if (temp_a[j] == 0.0f)
                {
                    // 0^anything = 0, except 0^0 = 1
                    temp_c[j] = (temp_b[j] == 0.0f) ? 1.0f : 0.0f;
                }
                else
                {
                    // Use pow(x,y) = exp(y * log(x))
                    temp_c[j] = std::exp(temp_b[j] * std::log(temp_a[j]));

                    // For negative bases, check if exponent is integer and odd/even
                    if (j < 16 && _mm512_mask_test_epi32_mask(1 << j, _mm512_castps_si512(a), _mm512_set1_epi32(0x80000000)))
                    {
                        float intpart;
                        if (std::modf(temp_b[j], &intpart) == 0.0f)
                        {
                            // Integer exponent
                            int exponent = static_cast<int>(temp_b[j]);
                            if (exponent % 2 != 0)
                            {
                                // Odd exponent, negate result
                                temp_c[j] = -temp_c[j];
                            }
                        }
                        else
                        {
                            // Non-integer exponent with negative base
                            // This would result in a complex number, but we don't support complex
                            // So we'll set to NaN
                            temp_c[j] = std::numeric_limits<float>::quiet_NaN();
                        }
                    }
                }
            }

            __m512 c = _mm512_loadu_ps(temp_c);
            _mm512_storeu_ps(&C[i], c);
        }
    }
    else if constexpr (std::is_same_v<data_type, double>)
    {
        for (; i + 8 <= shape; i += 8)
        {
            __m512d a = _mm512_loadu_pd(&A[i]);
            __m512d b = _mm512_loadu_pd(&B[i]);

#if 0
            // Create masks for special cases
            __mmask8 zero_base_mask = _mm512_cmpeq_pd_mask(a, _mm512_setzero_pd());
            __mmask8 neg_base_mask = _mm512_cmplt_pd_mask(a, _mm512_setzero_pd());
#endif
            // Take absolute value of bases for log calculation
            __m512d abs_a = _mm512_abs_pd(a);

            // For pow(x,y) = exp(y * log(x))
            alignas(64) double temp_a[8], temp_b[8], temp_c[8];
            _mm512_storeu_pd(temp_a, abs_a);
            _mm512_storeu_pd(temp_b, b);

            for (int j = 0; j < 8; j++)
            {
                // Handle special cases
                if (temp_a[j] == 0.0)
                {
                    // 0^anything = 0, except 0^0 = 1
                    temp_c[j] = (temp_b[j] == 0.0) ? 1.0 : 0.0;
                }
                else
                {
                    // Use pow(x,y) = exp(y * log(x))
                    temp_c[j] = std::exp(temp_b[j] * std::log(temp_a[j]));

                    // For negative bases, check if exponent is integer and odd/even
                    if (j < 8 && _mm512_mask_test_epi64_mask(1 << j, _mm512_castpd_si512(a), _mm512_set1_epi64(0x8000000000000000)))
                    {
                        double intpart;
                        if (std::modf(temp_b[j], &intpart) == 0.0)
                        {
                            // Integer exponent
                            long long exponent = static_cast<long long>(temp_b[j]);
                            if (exponent % 2 != 0)
                            {
                                // Odd exponent, negate result
                                temp_c[j] = -temp_c[j];
                            }
                        }
                        else
                        {
                            // Non-integer exponent with negative base
                            // This would result in a complex number
                            temp_c[j] = std::numeric_limits<double>::quiet_NaN();
                        }
                    }
                }
            }

            __m512d c = _mm512_loadu_pd(temp_c);
            _mm512_storeu_pd(&C[i], c);
        }
    }
    else if constexpr (std::is_integral_v<data_type>)
    {
        if constexpr (sizeof(data_type) == 4) // 32-bit integers
        {
            for (; i + 16 <= shape; i += 16)
            {
                __m512i a = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(&A[i]));
                __m512i b = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(&B[i]));

                alignas(64) int32_t temp_a[16], temp_b[16], temp_c[16];
                _mm512_storeu_si512(reinterpret_cast<__m512i *>(temp_a), a);
                _mm512_storeu_si512(reinterpret_cast<__m512i *>(temp_b), b);

                for (int j = 0; j < 16; j++)
                {
                    // Handle special cases for integer power
                    if (temp_a[j] == 0)
                    {
                        temp_c[j] = (temp_b[j] == 0) ? 1 : 0;
                    }
                    else if (temp_b[j] == 0)
                    {
                        temp_c[j] = 1; // x^0 = 1
                    }
                    else if (temp_b[j] < 0)
                    {
                        temp_c[j] = 0; // Integer division result of 0 for negative exponents (with integers)
                    }
                    else
                    {
                        // Use exponentiation by squaring for positive exponents
                        int32_t result = 1;
                        int32_t base = temp_a[j];
                        int32_t exp = temp_b[j];

                        while (exp > 0)
                        {
                            if (exp & 1)
                            {
                                result *= base;
                            }
                            base *= base;
                            exp >>= 1;
                        }
                        temp_c[j] = result;
                    }
                }

                __m512i c = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(temp_c));
                _mm512_storeu_si512(reinterpret_cast<__m512i *>(&C[i]), c);
            }
        }
        else if constexpr (sizeof(data_type) == 8) // 64-bit integers
        {
            for (; i + 8 <= shape; i += 8)
            {
                __m512i a = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(&A[i]));
                __m512i b = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(&B[i]));

                alignas(64) int64_t temp_a[8], temp_b[8], temp_c[8];
                _mm512_storeu_si512(reinterpret_cast<__m512i *>(temp_a), a);
                _mm512_storeu_si512(reinterpret_cast<__m512i *>(temp_b), b);

                for (int j = 0; j < 8; j++)
                {
                    // Handle special cases for integer power
                    if (temp_a[j] == 0)
                    {
                        temp_c[j] = (temp_b[j] == 0) ? 1 : 0;
                    }
                    else if (temp_b[j] == 0)
                    {
                        temp_c[j] = 1; // x^0 = 1
                    }
                    else if (temp_b[j] < 0)
                    {
                        temp_c[j] = 0; // Integer division result of 0 for negative exponents (with integers)
                    }
                    else
                    {
                        // Use exponentiation by squaring for positive exponents
                        int64_t result = 1;
                        int64_t base = temp_a[j];
                        int64_t exp = temp_b[j];

                        while (exp > 0)
                        {
                            if (exp & 1)
                            {
                                result *= base;
                            }
                            base *= base;
                            exp >>= 1;
                        }
                        temp_c[j] = result;
                    }
                }

                __m512i c = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(temp_c));
                _mm512_storeu_si512(reinterpret_cast<__m512i *>(&C[i]), c);
            }
        }
    }

    // Process remaining elements
    for (; i < shape; i++)
    {
        if constexpr (std::is_floating_point_v<data_type>)
        {
            C[i] = std::pow(A[i], B[i]);
        }
        else
        {
            // Integer power
            if (A[i] == 0)
            {
                C[i] = (B[i] == 0) ? 1 : 0;
            }
            else if (B[i] == 0)
            {
                C[i] = 1;
            }
            else if (B[i] < 0)
            {
                C[i] = 0;
            }
            else
            {
                data_type result = 1;
                data_type base = A[i];
                data_type exp = B[i];

                while (exp > 0)
                {
                    if (exp & 1)
                    {
                        result *= base;
                    }
                    base *= base;
                    exp >>= 1;
                }
                C[i] = result;
            }
        }
    }
}

//=====================================================================================//
// for boolean operations
template <typename data_type>
bool simd_eq(const data_type *a_ptr, const data_type *b_ptr, size_t size)
{
    size_t i = 0;
    if constexpr (std::is_same_v<data_type, float>)
    {
        constexpr size_t vec_size = 8; // 512 bits / 32 bits = 16 floats
        for (; i + vec_size <= size; i += vec_size)
        {
            __m512 a_vec = _mm512_loadu_ps(&a_ptr[i]);
            __m512 b_vec = _mm512_loadu_ps(&b_ptr[i]);
            // Compare 16 elements at once, returns 16-bit mask
            __mmask16 mask = _mm512_cmpeq_ps_mask(a_vec, b_vec);
            // If any elements differ (mask not all 1s), vectors are not equal
            if (mask != 0xFFFF)
            {
                return false;
            }
        }
    }
    else if constexpr (std::is_same_v<data_type, double>)
    {
        constexpr size_t vec_size = 8; // 512 bits / 64 bits = 8 doubles
        for (; i + vec_size <= size; i += vec_size)
        {
            __m512d a_vec = _mm512_loadu_pd(&a_ptr[i]);
            __m512d b_vec = _mm512_loadu_pd(&b_ptr[i]);
            // Compare 8 elements at once, returns 8-bit mask
            __mmask8 mask = _mm512_cmpeq_pd_mask(a_vec, b_vec);
            // If any elements differ (mask not all 1s), vectors are not equal
            if (mask != 0xFF)
            {
                return false;
            }
        }
    }
    else if constexpr (std::is_integral_v<data_type>)
    {
        if constexpr (sizeof(data_type) == 4)
        { // int32_t
            constexpr size_t vec_size = 16;

            for (; i + vec_size <= size; i += vec_size)
            {
                __m512i a_vec = _mm512_loadu_si512((__m512i *)&a_ptr[i]);
                __m512i b_vec = _mm512_loadu_si512((__m512i *)&b_ptr[i]);

                __mmask16 mask = _mm512_cmpeq_epi32_mask(a_vec, b_vec);
                if (mask != 0xFFFF)
                {
                    return false;
                }
            }
        }
        else if constexpr (sizeof(data_type) == 8)
        { // int64_t
            constexpr size_t vec_size = 8;

            for (; i + vec_size <= size; i += vec_size)
            {
                __m512i a_vec = _mm512_loadu_si512((__m512i *)&a_ptr[i]);
                __m512i b_vec = _mm512_loadu_si512((__m512i *)&b_ptr[i]);

                __mmask8 mask = _mm512_cmpeq_epi64_mask(a_vec, b_vec);
                if (mask != 0xFF)
                {
                    return false;
                }
            }
        }
    }
    for (; i < size; i++)
    {
        if (a_ptr[i] != b_ptr[i])
        {
            return false;
        }
    }
    return true;
}

template <typename data_type>
bool simd_larger(const data_type *a_ptr, const data_type *b_ptr, size_t size)
{
    size_t i = 0;

    if constexpr (std::is_same_v<data_type, float>)
    {
        constexpr size_t vec_size = 16; // 512 bits / 32 bits = 16 floats
        for (; i + vec_size <= size; i += vec_size)
        {
            __m512 a_vec = _mm512_loadu_ps(&a_ptr[i]);
            __m512 b_vec = _mm512_loadu_ps(&b_ptr[i]);

            // Compare if a > b for all 16 elements
            __mmask16 mask = _mm512_cmp_ps_mask(a_vec, b_vec, _CMP_GT_OQ);

            if (mask != 0xFFFF)
            {
                return false;
            }
        }
    }
    else if constexpr (std::is_same_v<data_type, double>)
    {
        constexpr size_t vec_size = 8; // 512 bits / 64 bits = 8 doubles
        for (; i + vec_size <= size; i += vec_size)
        {
            __m512d a_vec = _mm512_loadu_pd(&a_ptr[i]);
            __m512d b_vec = _mm512_loadu_pd(&b_ptr[i]);

            // Compare if a > b for all 8 elements
            __mmask8 mask = _mm512_cmp_pd_mask(a_vec, b_vec, _CMP_GT_OQ);

            if (mask != 0xFF)
            {
                return false;
            }
        }
    }
    else if constexpr (std::is_integral_v<data_type>)
    {
        if constexpr (sizeof(data_type) == 4) // int32_t
        {
            constexpr size_t vec_size = 16;

            for (; i + vec_size <= size; i += vec_size)
            {
                __m512i a_vec = _mm512_loadu_si512((__m512i *)&a_ptr[i]);
                __m512i b_vec = _mm512_loadu_si512((__m512i *)&b_ptr[i]);

                __mmask16 mask = _mm512_cmpgt_epi32_mask(a_vec, b_vec);
                if (mask != 0xFFFF)
                {
                    return false;
                }
            }
        }
        else if constexpr (sizeof(data_type) == 8) // int64_t
        {
            constexpr size_t vec_size = 8;

            for (; i + vec_size <= size; i += vec_size)
            {
                __m512i a_vec = _mm512_loadu_si512((__m512i *)&a_ptr[i]);
                __m512i b_vec = _mm512_loadu_si512((__m512i *)&b_ptr[i]);

                __mmask8 mask = _mm512_cmpgt_epi64_mask(a_vec, b_vec);
                if (mask != 0xFF)
                {
                    return false;
                }
            }
        }
    }

    // Process remaining elements
    for (; i < size; i++)
    {
        if (!(a_ptr[i] > b_ptr[i]))
        {
            return false;
        }
    }

    // If we've reached here, all elements in a are greater than b
    return true;
}

template <typename data_type>
bool simd_smaller(const data_type *a_ptr, const data_type *b_ptr, size_t size)
{
    size_t i = 0;

    if constexpr (std::is_same_v<data_type, float>)
    {
        constexpr size_t vec_size = 16; // 512 bits / 32 bits = 16 floats
        for (; i + vec_size <= size; i += vec_size)
        {
            __m512 a_vec = _mm512_loadu_ps(&a_ptr[i]);
            __m512 b_vec = _mm512_loadu_ps(&b_ptr[i]);

            // Compare if a < b for all 16 elements, returns 16-bit mask
            __mmask16 mask = _mm512_cmplt_ps_mask(a_vec, b_vec);

            // If any element of a is not less than b (mask not all 1s), return false
            if (mask != 0xFFFF)
            {
                return false;
            }
        }
    }
    else if constexpr (std::is_same_v<data_type, double>)
    {
        constexpr size_t vec_size = 8; // 512 bits / 64 bits = 8 doubles
        for (; i + vec_size <= size; i += vec_size)
        {
            __m512d a_vec = _mm512_loadu_pd(&a_ptr[i]);
            __m512d b_vec = _mm512_loadu_pd(&b_ptr[i]);

            // Compare if a < b for all 8 elements, returns 8-bit mask
            __mmask8 mask = _mm512_cmplt_pd_mask(a_vec, b_vec);

            // If any element of a is not less than b (mask not all 1s), return false
            if (mask != 0xFF)
            {
                return false;
            }
        }
    }
    else if constexpr (std::is_integral_v<data_type>)
    {
        if constexpr (sizeof(data_type) == 4) // int32_t
        {
            constexpr size_t vec_size = 16;

            for (; i + vec_size <= size; i += vec_size)
            {
                __m512i a_vec = _mm512_loadu_si512((__m512i *)&a_ptr[i]);
                __m512i b_vec = _mm512_loadu_si512((__m512i *)&b_ptr[i]);

                __mmask16 mask = _mm512_cmplt_epi32_mask(a_vec, b_vec);
                if (mask != 0xFFFF)
                {
                    return false;
                }
            }
        }
        else if constexpr (sizeof(data_type) == 8) // int64_t
        {
            constexpr size_t vec_size = 8;

            for (; i + vec_size <= size; i += vec_size)
            {
                __m512i a_vec = _mm512_loadu_si512((__m512i *)&a_ptr[i]);
                __m512i b_vec = _mm512_loadu_si512((__m512i *)&b_ptr[i]);

                __mmask8 mask = _mm512_cmplt_epi64_mask(a_vec, b_vec);
                if (mask != 0xFF)
                {
                    return false;
                }
            }
        }
    }

    // Process remaining elements
    for (; i < size; i++)
    {
        if (!(a_ptr[i] < b_ptr[i]))
        {
            return false;
        }
    }

    // If we've reached here, all elements in a are less than b
    return true;
}
```
