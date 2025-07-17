#pragma once

// Explicit template instantiations
#include "element_wise_expli.tpp"
//==============================================================================================//
template <typename data_type>
void simd_elem_add(data_type *A, size_t shape, const data_type &scalor)
{
    size_t i = 0;
    if constexpr (std::is_same_v<data_type, float>)
    {
        __m512 scalar_vec = _mm512_set1_ps(scalor); // Broadcast scalar to all elements of the vector
#pragma unroll
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
#pragma unroll
        for (; i + 8 <= shape; i += 8)
        {
            __m512d a = _mm512_loadu_pd(&A[i]); // Load 8 doubles from memory
            a = _mm512_add_pd(a, scalar_vec);   // Perform element-wise addition
            _mm512_storeu_pd(&A[i], a);         // Store the result back to memory
        }
    }
    else if constexpr (std::is_integral_v<data_type>)
    {
        if constexpr (sizeof(data_type) == 1) // 8-bit integers
        {
            __m512i scalar_vec = _mm512_set1_epi8(static_cast<char>(scalor));
#pragma unroll
            for (; i + 64 <= shape; i += 64) // Process 64 elements at once
            {
                __m512i a = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(&A[i]));
                a = _mm512_add_epi8(a, scalar_vec); // Using AVX-512BW instruction
                _mm512_storeu_si512(reinterpret_cast<__m512i *>(&A[i]), a);
            }
        }
        else if constexpr (sizeof(data_type) == 2) // 16-bit integers
        {
            __m512i scalar_vec = _mm512_set1_epi16(static_cast<short>(scalor));
#pragma unroll
            for (; i + 32 <= shape; i += 32) // Process 32 elements at once
            {
                __m512i a = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(&A[i]));
                a = _mm512_add_epi16(a, scalar_vec); // Using AVX-512BW instruction
                _mm512_storeu_si512(reinterpret_cast<__m512i *>(&A[i]), a);
            }
        }
        else if constexpr (sizeof(data_type) == 4) // 32-bit integers
        {
            __m512i scalar_vec = _mm512_set1_epi32(static_cast<int>(scalor));
#pragma unroll
            for (; i + 16 <= shape; i += 16)
            {
                __m512i a = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(&A[i]));
                a = _mm512_add_epi32(a, scalar_vec);
                _mm512_storeu_si512(reinterpret_cast<__m512i *>(&A[i]), a);
            }
        }
        else if constexpr (sizeof(data_type) == 8) // 64-bit integers
        {
            __m512i scalar_vec = _mm512_set1_epi64(static_cast<long long>(scalor));
#pragma unroll
            for (; i + 8 <= shape; i += 8)
            {
                __m512i a = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(&A[i]));
                a = _mm512_add_epi64(a, scalar_vec);
                _mm512_storeu_si512(reinterpret_cast<__m512i *>(&A[i]), a);
            }
        }
    }

// Process remaining elements
#pragma unroll
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
#pragma unroll
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
#pragma unroll
        for (; i + 8 <= shape; i += 8)
        {
            __m512d a = _mm512_loadu_pd(&A[i]);
            a = _mm512_sub_pd(a, scalar_vec);
            _mm512_storeu_pd(&A[i], a);
        }
    }
    else if constexpr (std::is_integral_v<data_type>)
    {
        if constexpr (sizeof(data_type) == 1) // 8-bit integers
        {
            __m512i scalar_vec = _mm512_set1_epi8(static_cast<char>(scalor));
#pragma unroll
            for (; i + 64 <= shape; i += 64) // Process 64 elements at once
            {
                __m512i a = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(&A[i]));
                a = _mm512_sub_epi8(a, scalar_vec); // Using AVX-512BW instruction
                _mm512_storeu_si512(reinterpret_cast<__m512i *>(&A[i]), a);
            }
        }
        else if constexpr (sizeof(data_type) == 2) // 16-bit integers
        {
            __m512i scalar_vec = _mm512_set1_epi16(static_cast<short>(scalor));
#pragma unroll
            for (; i + 32 <= shape; i += 32) // Process 32 elements at once
            {
                __m512i a = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(&A[i]));
                a = (a, scalar_vec); // Using AVX-512BW instruction
                _mm512_storeu_si512(reinterpret_cast<__m512i *>(&A[i]), a);
            }
        }
        else if constexpr (sizeof(data_type) == 4) // 32-bit integers
        {
            __m512i scalar_vec = _mm512_set1_epi32(static_cast<int>(scalor));
#pragma unroll
            for (; i + 16 <= shape; i += 16)
            {
                __m512i a = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(&A[i]));
                a = _mm512_sub_epi32(a, scalar_vec);
                _mm512_storeu_si512(reinterpret_cast<__m512i *>(&A[i]), a);
            }
        }
        else if constexpr (sizeof(data_type) == 8) // 64-bit integers
        {
            __m512i scalar_vec = _mm512_set1_epi64(static_cast<long long>(scalor));
#pragma unroll
            for (; i + 8 <= shape; i += 8)
            {
                __m512i a = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(&A[i]));
                a = _mm512_sub_epi64(a, scalar_vec);
                _mm512_storeu_si512(reinterpret_cast<__m512i *>(&A[i]), a);
            }
        }
    }
// Process remaining elements
#pragma unroll
    for (; i < shape; i++)
    {
        A[i] -= scalor;
    }
}

template <typename data_type>
void simd_elem_sub(const data_type &scalor, data_type *A, size_t shape)
{
    size_t i = 0;
    if constexpr (std::is_same_v<data_type, float>)
    {
        __m512 scalar_vec = _mm512_set1_ps(scalor);
#pragma unroll
        for (; i + 16 <= shape; i += 16)
        {
            __m512 a = _mm512_loadu_ps(&A[i]);
            // Subtract array from scalar: scalar - A
            a = _mm512_sub_ps(scalar_vec, a);
            _mm512_storeu_ps(&A[i], a);
        }
    }
    else if constexpr (std::is_same_v<data_type, double>)
    {
        __m512d scalar_vec = _mm512_set1_pd(scalor);
#pragma unroll
        for (; i + 8 <= shape; i += 8)
        {
            __m512d a = _mm512_loadu_pd(&A[i]);
            // Subtract array from scalar: scalar - A
            a = _mm512_sub_pd(scalar_vec, a);
            _mm512_storeu_pd(&A[i], a);
        }
    }
    else if constexpr (std::is_integral_v<data_type>)
    {
        if constexpr (sizeof(data_type) == 1) // 8-bit integers
        {
            __m512i scalar_vec = _mm512_set1_epi8(static_cast<char>(scalor));
#pragma unroll
            for (; i + 64 <= shape; i += 64) // Process 64 elements at once
            {
                __m512i a = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(&A[i]));
                // Subtract array from scalar: scalar - A
                a = _mm512_sub_epi8(scalar_vec, a);
                _mm512_storeu_si512(reinterpret_cast<__m512i *>(&A[i]), a);
            }
        }
        else if constexpr (sizeof(data_type) == 2) // 16-bit integers
        {
            __m512i scalar_vec = _mm512_set1_epi16(static_cast<short>(scalor));
#pragma unroll
            for (; i + 32 <= shape; i += 32) // Process 32 elements at once
            {
                __m512i a = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(&A[i]));
                // Subtract array from scalar: scalar - A
                a = _mm512_sub_epi16(scalar_vec, a);
                _mm512_storeu_si512(reinterpret_cast<__m512i *>(&A[i]), a);
            }
        }
        else if constexpr (sizeof(data_type) == 4) // 32-bit integers
        {
            __m512i scalar_vec = _mm512_set1_epi32(static_cast<int>(scalor));
#pragma unroll
            for (; i + 16 <= shape; i += 16)
            {
                __m512i a = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(&A[i]));
                // Subtract array from scalar: scalar - A
                a = _mm512_sub_epi32(scalar_vec, a);
                _mm512_storeu_si512(reinterpret_cast<__m512i *>(&A[i]), a);
            }
        }
        else if constexpr (sizeof(data_type) == 8) // 64-bit integers
        {
            __m512i scalar_vec = _mm512_set1_epi64(static_cast<long long>(scalor));
#pragma unroll
            for (; i + 8 <= shape; i += 8)
            {
                __m512i a = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(&A[i]));
                // Subtract array from scalar: scalar - A
                a = _mm512_sub_epi64(scalar_vec, a);
                _mm512_storeu_si512(reinterpret_cast<__m512i *>(&A[i]), a);
            }
        }
    }

// Process remaining elements with OpenMP parallelization
#pragma omp parallel for
    for (size_t j = i; j < shape; j++)
    {
        A[j] = scalor - A[j];
    }
}

template <typename data_type>
void simd_elem_mul(data_type *A, size_t shape, const data_type &scalor)
{
    size_t i = 0;
    if constexpr (std::is_same_v<data_type, float>)
    {
        __m512 scalar_vec = _mm512_set1_ps(scalor);
#pragma unroll
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
#pragma unroll
        for (; i + 8 <= shape; i += 8)
        {
            __m512d a = _mm512_loadu_pd(&A[i]);
            a = _mm512_mul_pd(a, scalar_vec);
            _mm512_storeu_pd(&A[i], a);
        }
    }
    else if constexpr (std::is_integral_v<data_type>)
    {
        if constexpr (sizeof(data_type) == 2) // 16-bit integers
        {
            __m512i scalar_vec = _mm512_set1_epi16(static_cast<short>(scalor));
#pragma unroll
            for (; i + 32 <= shape; i += 32) // Process 32 elements at once
            {
                __m512i a = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(&A[i]));
                a = _mm512_mullo_epi16(a, scalar_vec); // Using AVX-512BW instruction
                _mm512_storeu_si512(reinterpret_cast<__m512i *>(&A[i]), a);
            }
        }
        else if constexpr (sizeof(data_type) == 4) // 32-bit integers
        {
            __m512i scalar_vec = _mm512_set1_epi32(static_cast<int>(scalor));
#pragma unroll
            for (; i + 16 <= shape; i += 16)
            {
                __m512i a = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(&A[i]));
                a = _mm512_mullo_epi32(a, scalar_vec);
                _mm512_storeu_si512(reinterpret_cast<__m512i *>(&A[i]), a);
            }
        }
        else if constexpr (sizeof(data_type) == 8) // 64-bit integers
        {
            __m512i scalar_vec = _mm512_set1_epi64(static_cast<long long>(scalor));
#pragma unroll
            for (; i + 8 <= shape; i += 8)
            {
                __m512i a = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(&A[i]));
                a = _mm512_mullo_epi64(a, scalar_vec);
                _mm512_storeu_si512(reinterpret_cast<__m512i *>(&A[i]), a);
            }
        }
    }
    // Process remaining elements
#pragma unroll
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
void simd_elem_div(const data_type &scalor, data_type *A, size_t shape)
{
    size_t i = 0;
    if constexpr (std::is_same_v<data_type, float>)
    {
        __m512 scalar_vec = _mm512_set1_ps(scalor);
        for (; i + 16 <= shape; i += 16)
        {
            __m512 a = _mm512_loadu_ps(&A[i]);

            // Handle division by zero cases
            alignas(64) float temp_in[16], temp_out[16];
            _mm512_storeu_ps(temp_in, a);

#pragma omp parallel for simd
            for (int j = 0; j < 16; ++j)
            {
                if (temp_in[j] == 0.0f)
                {
                    temp_out[j] = std::numeric_limits<float>::infinity();
                }
                else
                {
                    temp_out[j] = scalor / temp_in[j];
                }
            }

            __m512 result = _mm512_loadu_ps(temp_out);
            _mm512_storeu_ps(&A[i], result);
        }
    }
    else if constexpr (std::is_same_v<data_type, double>)
    {
        __m512d scalar_vec = _mm512_set1_pd(scalor);
        for (; i + 8 <= shape; i += 8)
        {
            __m512d a = _mm512_loadu_pd(&A[i]);

            // Handle division by zero cases
            alignas(64) double temp_in[8], temp_out[8];
            _mm512_storeu_pd(temp_in, a);

#pragma omp parallel for simd
            for (int j = 0; j < 8; ++j)
            {
                if (temp_in[j] == 0.0)
                {
                    temp_out[j] = std::numeric_limits<double>::infinity();
                }
                else
                {
                    temp_out[j] = scalor / temp_in[j];
                }
            }

            __m512d result = _mm512_loadu_pd(temp_out);
            _mm512_storeu_pd(&A[i], result);
        }
    }
    else if constexpr (std::is_integral_v<data_type>)
    {
        if constexpr (sizeof(data_type) == 1) // 8-bit integers
        {
            for (; i + 64 <= shape; i += 64)
            {
                __m512i a = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(&A[i]));
                alignas(64) signed char temp_in[64];
                alignas(64) data_type temp_out[64];
                _mm512_storeu_si512(reinterpret_cast<__m512i *>(temp_in), a);

#pragma omp parallel for simd
                for (int j = 0; j < 64; ++j)
                {
                    // Handle division by zero and set to max value for integers
                    if (temp_in[j] == 0)
                    {
                        temp_out[j] = std::numeric_limits<data_type>::max();
                    }
                    else
                    {
                        temp_out[j] = scalor / temp_in[j];
                    }
                }

                __m512i result = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(temp_out));
                _mm512_storeu_si512(reinterpret_cast<__m512i *>(&A[i]), result);
            }
        }
        else if constexpr (sizeof(data_type) == 2) // 16-bit integers
        {
            for (; i + 32 <= shape; i += 32)
            {
                __m512i a = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(&A[i]));
                alignas(64) short temp_in[32];
                alignas(64) data_type temp_out[32];
                _mm512_storeu_si512(reinterpret_cast<__m512i *>(temp_in), a);

#pragma omp parallel for simd
                for (int j = 0; j < 32; ++j)
                {
                    if (temp_in[j] == 0)
                    {
                        temp_out[j] = std::numeric_limits<data_type>::max();
                    }
                    else
                    {
                        temp_out[j] = scalor / temp_in[j];
                    }
                }

                __m512i result = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(temp_out));
                _mm512_storeu_si512(reinterpret_cast<__m512i *>(&A[i]), result);
            }
        }
        else if constexpr (sizeof(data_type) == 4) // 32-bit integers
        {
            for (; i + 16 <= shape; i += 16)
            {
                __m512i a = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(&A[i]));
                alignas(64) int temp_in[16];
                alignas(64) data_type temp_out[16];
                _mm512_storeu_si512(reinterpret_cast<__m512i *>(temp_in), a);

#pragma omp parallel for simd
                for (int j = 0; j < 16; ++j)
                {
                    if (temp_in[j] == 0)
                    {
                        temp_out[j] = std::numeric_limits<data_type>::max();
                    }
                    else
                    {
                        temp_out[j] = scalor / temp_in[j];
                    }
                }

                __m512i result = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(temp_out));
                _mm512_storeu_si512(reinterpret_cast<__m512i *>(&A[i]), result);
            }
        }
        else if constexpr (sizeof(data_type) == 8) // 64-bit integers
        {
            for (; i + 8 <= shape; i += 8)
            {
                __m512i a = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(&A[i]));
                alignas(64) long long temp_in[8];
                alignas(64) data_type temp_out[8];
                _mm512_storeu_si512(reinterpret_cast<__m512i *>(temp_in), a);

#pragma omp parallel for simd
                for (int j = 0; j < 8; ++j)
                {
                    if (temp_in[j] == 0)
                    {
                        temp_out[j] = std::numeric_limits<data_type>::max();
                    }
                    else
                    {
                        temp_out[j] = scalor / temp_in[j];
                    }
                }

                __m512i result = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(temp_out));
                _mm512_storeu_si512(reinterpret_cast<__m512i *>(&A[i]), result);
            }
        }
    }

// Process remaining elements with OpenMP parallelization
#pragma omp parallel for
    for (size_t j = i; j < shape; ++j)
    {
        if (A[j] == 0)
        {
            if constexpr (std::is_floating_point_v<data_type>)
            {
                A[j] = std::numeric_limits<data_type>::infinity();
            }
            else
            {
                A[j] = std::numeric_limits<data_type>::max();
            }
        }
        else
        {
            A[j] = scalor / A[j];
        }
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
void simd_elem_power(const data_type &scalor, data_type *A, size_t shape)
{
    size_t i = 0;
    if constexpr (std::is_same_v<data_type, float>)
    {
        for (; i + 16 <= shape; i += 16)
        {
            __m512 a = _mm512_loadu_ps(&A[i]);

            // Use temporary arrays for calculation
            alignas(64) float temp_in[16], temp_out[16];
            _mm512_storeu_ps(temp_in, a);

#pragma omp parallel for simd
            for (int j = 0; j < 16; j++)
            {
                // Handle special cases
                if (scalor == 0.0f)
                {
                    // 0^x = 0 for x > 0, 0^0 = 1, 0^x = inf for x < 0
                    if (temp_in[j] > 0.0f)
                        temp_out[j] = 0.0f;
                    else if (temp_in[j] == 0.0f)
                        temp_out[j] = 1.0f;
                    else
                        temp_out[j] = std::numeric_limits<float>::infinity();
                }
                else if (scalor == 1.0f)
                {
                    // 1^x = 1 for any x
                    temp_out[j] = 1.0f;
                }
                else if (scalor < 0.0f)
                {
                    // For negative bases, check if exponent is integer and odd/even
                    if (std::floor(temp_in[j]) == temp_in[j])
                    {
                        // Integer exponent
                        if (static_cast<int>(temp_in[j]) % 2 == 0)
                        {
                            // Even exponent: (-a)^n = a^n
                            temp_out[j] = std::pow(std::abs(scalor), temp_in[j]);
                        }
                        else
                        {
                            // Odd exponent: (-a)^n = -(a^n)
                            temp_out[j] = -std::pow(std::abs(scalor), temp_in[j]);
                        }
                    }
                    else
                    {
                        // Negative base with non-integer exponent is complex
                        temp_out[j] = std::numeric_limits<float>::quiet_NaN();
                    }
                }
                else
                {
                    // Normal case: positive base
                    temp_out[j] = std::pow(scalor, temp_in[j]);
                }
            }

            __m512 result = _mm512_loadu_ps(temp_out);
            _mm512_storeu_ps(&A[i], result);
        }
    }
    else if constexpr (std::is_same_v<data_type, double>)
    {
        for (; i + 8 <= shape; i += 8)
        {
            __m512d a = _mm512_loadu_pd(&A[i]);

            // Use temporary arrays for calculation
            alignas(64) double temp_in[8], temp_out[8];
            _mm512_storeu_pd(temp_in, a);

#pragma omp parallel for simd
            for (int j = 0; j < 8; j++)
            {
                // Handle special cases
                if (scalor == 0.0)
                {
                    // 0^x = 0 for x > 0, 0^0 = 1, 0^x = inf for x < 0
                    if (temp_in[j] > 0.0)
                        temp_out[j] = 0.0;
                    else if (temp_in[j] == 0.0)
                        temp_out[j] = 1.0;
                    else
                        temp_out[j] = std::numeric_limits<double>::infinity();
                }
                else if (scalor == 1.0)
                {
                    // 1^x = 1 for any x
                    temp_out[j] = 1.0;
                }
                else if (scalor < 0.0)
                {
                    // For negative bases, check if exponent is integer and odd/even
                    if (std::floor(temp_in[j]) == temp_in[j])
                    {
                        // Integer exponent
                        if (static_cast<int>(temp_in[j]) % 2 == 0)
                        {
                            // Even exponent: (-a)^n = a^n
                            temp_out[j] = std::pow(std::abs(scalor), temp_in[j]);
                        }
                        else
                        {
                            // Odd exponent: (-a)^n = -(a^n)
                            temp_out[j] = -std::pow(std::abs(scalor), temp_in[j]);
                        }
                    }
                    else
                    {
                        // Negative base with non-integer exponent is complex
                        temp_out[j] = std::numeric_limits<double>::quiet_NaN();
                    }
                }
                else
                {
                    // Normal case: positive base
                    temp_out[j] = std::pow(scalor, temp_in[j]);
                }
            }

            __m512d result = _mm512_loadu_pd(temp_out);
            _mm512_storeu_pd(&A[i], result);
        }
    }
    else if constexpr (std::is_integral_v<data_type>)
    {
        // For integer types, use a more optimized approach

        if constexpr (sizeof(data_type) == 1) // 8-bit integers
        {
            for (; i + 64 <= shape; i += 64)
            {
                __m512i a = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(&A[i]));
                alignas(64) signed char temp_in[64];
                alignas(64) data_type temp_out[64];
                _mm512_storeu_si512(reinterpret_cast<__m512i *>(temp_in), a);

#pragma omp parallel for simd
                for (int j = 0; j < 64; ++j)
                {
                    // Special cases for integer exponentiation
                    if (scalor == 0)
                    {
                        temp_out[j] = (temp_in[j] == 0) ? 1 : 0;
                    }
                    else if (scalor == 1)
                    {
                        temp_out[j] = 1;
                    }
                    else if (temp_in[j] == 0)
                    {
                        temp_out[j] = 1; // any^0 = 1
                    }
                    else if (temp_in[j] < 0)
                    {
                        // Handle negative exponents for integers (usually 0)
                        temp_out[j] = 0;
                    }
                    else if (temp_in[j] > 10) // Prevent overflow for small integer types
                    {
                        temp_out[j] = (scalor < 0 && temp_in[j] % 2) ? std::numeric_limits<data_type>::min() : std::numeric_limits<data_type>::max();
                    }
                    else
                    {
                        // Compute scalor^temp_in[j] using exponentiation by squaring
                        data_type result = 1;
                        data_type base = scalor;
                        data_type exp = temp_in[j];

                        while (exp > 0)
                        {
                            if (exp % 2 == 1)
                            {
                                result *= base;
                            }
                            base *= base;
                            exp /= 2;
                        }

                        temp_out[j] = result;
                    }
                }

                __m512i result = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(temp_out));
                _mm512_storeu_si512(reinterpret_cast<__m512i *>(&A[i]), result);
            }
        }
        else if constexpr (sizeof(data_type) == 2) // 16-bit integers
        {
            for (; i + 32 <= shape; i += 32)
            {
                __m512i a = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(&A[i]));
                alignas(64) short temp_in[32];
                alignas(64) data_type temp_out[32];
                _mm512_storeu_si512(reinterpret_cast<__m512i *>(temp_in), a);

#pragma omp parallel for simd
                for (int j = 0; j < 32; ++j)
                {
                    // Special cases for integer exponentiation
                    if (scalor == 0)
                    {
                        temp_out[j] = (temp_in[j] == 0) ? 1 : 0;
                    }
                    else if (scalor == 1)
                    {
                        temp_out[j] = 1;
                    }
                    else if (temp_in[j] == 0)
                    {
                        temp_out[j] = 1; // any^0 = 1
                    }
                    else if (temp_in[j] < 0)
                    {
                        // Handle negative exponents for integers (usually 0)
                        temp_out[j] = 0;
                    }
                    else if (temp_in[j] > 15) // Prevent overflow for 16-bit
                    {
                        temp_out[j] = (scalor < 0 && temp_in[j] % 2) ? std::numeric_limits<data_type>::min() : std::numeric_limits<data_type>::max();
                    }
                    else
                    {
                        // Compute scalor^temp_in[j] using exponentiation by squaring
                        data_type result = 1;
                        data_type base = scalor;
                        data_type exp = temp_in[j];

                        while (exp > 0)
                        {
                            if (exp % 2 == 1)
                            {
                                result *= base;
                            }
                            base *= base;
                            exp /= 2;
                        }

                        temp_out[j] = result;
                    }
                }

                __m512i result = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(temp_out));
                _mm512_storeu_si512(reinterpret_cast<__m512i *>(&A[i]), result);
            }
        }
        else if constexpr (sizeof(data_type) == 4) // 32-bit integers
        {
            for (; i + 16 <= shape; i += 16)
            {
                __m512i a = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(&A[i]));
                alignas(64) int temp_in[16];
                alignas(64) data_type temp_out[16];
                _mm512_storeu_si512(reinterpret_cast<__m512i *>(temp_in), a);

#pragma omp parallel for simd
                for (int j = 0; j < 16; ++j)
                {
                    // Special cases for integer exponentiation
                    if (scalor == 0)
                    {
                        temp_out[j] = (temp_in[j] == 0) ? 1 : 0;
                    }
                    else if (scalor == 1)
                    {
                        temp_out[j] = 1;
                    }
                    else if (temp_in[j] == 0)
                    {
                        temp_out[j] = 1; // any^0 = 1
                    }
                    else if (temp_in[j] < 0)
                    {
                        // Handle negative exponents for integers (usually 0)
                        temp_out[j] = 0;
                    }
                    else if (temp_in[j] > 30) // Prevent overflow for 32-bit
                    {
                        temp_out[j] = (scalor < 0 && temp_in[j] % 2) ? std::numeric_limits<data_type>::min() : std::numeric_limits<data_type>::max();
                    }
                    else
                    {
                        // Compute scalor^temp_in[j] using exponentiation by squaring
                        data_type result = 1;
                        data_type base = scalor;
                        data_type exp = temp_in[j];

                        while (exp > 0)
                        {
                            if (exp % 2 == 1)
                            {
                                result *= base;
                            }
                            base *= base;
                            exp /= 2;
                        }

                        temp_out[j] = result;
                    }
                }

                __m512i result = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(temp_out));
                _mm512_storeu_si512(reinterpret_cast<__m512i *>(&A[i]), result);
            }
        }
        else if constexpr (sizeof(data_type) == 8) // 64-bit integers
        {
            for (; i + 8 <= shape; i += 8)
            {
                __m512i a = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(&A[i]));
                alignas(64) long long temp_in[8];
                alignas(64) data_type temp_out[8];
                _mm512_storeu_si512(reinterpret_cast<__m512i *>(temp_in), a);

#pragma omp parallel for simd
                for (int j = 0; j < 8; ++j)
                {
                    // Special cases for integer exponentiation
                    if (scalor == 0)
                    {
                        temp_out[j] = (temp_in[j] == 0) ? 1 : 0;
                    }
                    else if (scalor == 1)
                    {
                        temp_out[j] = 1;
                    }
                    else if (temp_in[j] == 0)
                    {
                        temp_out[j] = 1; // any^0 = 1
                    }
                    else if (temp_in[j] < 0)
                    {
                        // Handle negative exponents for integers (usually 0)
                        temp_out[j] = 0;
                    }
                    else if (temp_in[j] > 62) // Prevent overflow for 64-bit
                    {
                        temp_out[j] = (scalor < 0 && temp_in[j] % 2) ? std::numeric_limits<data_type>::min() : std::numeric_limits<data_type>::max();
                    }
                    else
                    {
                        // Compute scalor^temp_in[j] using exponentiation by squaring
                        data_type result = 1;
                        data_type base = scalor;
                        data_type exp = temp_in[j];

                        while (exp > 0)
                        {
                            if (exp % 2 == 1)
                            {
                                result *= base;
                            }
                            base *= base;
                            exp /= 2;
                        }

                        temp_out[j] = result;
                    }
                }

                __m512i result = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(temp_out));
                _mm512_storeu_si512(reinterpret_cast<__m512i *>(&A[i]), result);
            }
        }
    }

// Process remaining elements with OpenMP parallelization
#pragma omp parallel for
    for (size_t j = i; j < shape; ++j)
    {
        if constexpr (std::is_floating_point_v<data_type>)
        {
            if (scalor == 0.0)
            {
                // 0^x = 0 for x > 0, 0^0 = 1, 0^x = inf for x < 0
                if (A[j] > 0.0)
                    A[j] = 0.0;
                else if (A[j] == 0.0)
                    A[j] = 1.0;
                else
                    A[j] = std::numeric_limits<data_type>::infinity();
            }
            else if (scalor == 1.0)
            {
                // 1^x = 1 for any x
                A[j] = 1.0;
            }
            else if (scalor < 0.0)
            {
                // For negative bases, check if exponent is integer and odd/even
                if (std::floor(A[j]) == A[j])
                {
                    // Integer exponent
                    if (static_cast<int>(A[j]) % 2 == 0)
                    {
                        // Even exponent: (-a)^n = a^n
                        A[j] = std::pow(std::abs(scalor), A[j]);
                    }
                    else
                    {
                        // Odd exponent: (-a)^n = -(a^n)
                        A[j] = -std::pow(std::abs(scalor), A[j]);
                    }
                }
                else
                {
                    // Negative base with non-integer exponent is complex
                    A[j] = std::numeric_limits<data_type>::quiet_NaN();
                }
            }
            else
            {
                // Normal case: positive base
                A[j] = std::pow(scalor, A[j]);
            }
        }
        else if constexpr (std::is_integral_v<data_type>)
        {
            if (scalor == 0)
            {
                A[j] = (A[j] == 0) ? 1 : 0;
            }
            else if (scalor == 1)
            {
                A[j] = 1;
            }
            else if (A[j] == 0)
            {
                A[j] = 1; // any^0 = 1
            }
            else if (A[j] < 0)
            {
                // Handle negative exponents for integers (usually 0)
                A[j] = 0;
            }
            else if (A[j] > 20) // Prevent overflow
            {
                A[j] = (scalor < 0 && A[j] % 2) ? std::numeric_limits<data_type>::min() : std::numeric_limits<data_type>::max();
            }
            else
            {
                data_type result = 1;
                data_type base = scalor;
                data_type exp = A[j];

                while (exp > 0)
                {
                    if (exp % 2 == 1)
                    {
                        result *= base;
                    }
                    base *= base;
                    exp /= 2;
                }

                A[j] = result;
            }
        }
    }
}

template <typename data_type>
void simd_elem_mod(data_type *A, size_t shape, const data_type &scalor)
{
    size_t i = 0;

    if constexpr (std::is_same_v<data_type, float>)
    {
        for (; i + 16 <= shape; i += 16)
        {
            __m512 a = _mm512_loadu_ps(&A[i]); // Tải 16 phần tử float
            alignas(64) float temp[16];
            _mm512_storeu_ps(temp, a); // Lưu tạm thời để xử lý fmodf

#pragma unroll
            for (int j = 0; j < 16; ++j)
            {
                temp[j] = fmodf(temp[j], scalor); // Sử dụng fmodf thay cho std::fmod
            }

            __m512 result = _mm512_loadu_ps(temp);
            _mm512_storeu_ps(&A[i], result); // Lưu kết quả trở lại
        }
    }
    else if constexpr (std::is_same_v<data_type, double>)
    {
        for (; i + 8 <= shape; i += 8)
        {
            __m512d a = _mm512_loadu_pd(&A[i]); // Tải 8 phần tử double
            alignas(64) double temp[8];
            _mm512_storeu_pd(temp, a); // Lưu tạm thời để xử lý fmod

#pragma unroll
            for (int j = 0; j < 8; ++j)
            {
                temp[j] = std::fmod(temp[j], scalor);
            }

            __m512d result = _mm512_loadu_pd(temp);
            _mm512_storeu_pd(&A[i], result); // Lưu kết quả trở lại
        }
    }
    else if constexpr (std::is_integral_v<data_type>)
    {
        // Replace line 477 and surrounding code with:
        if constexpr (sizeof(data_type) == 4) // 32-bit integers
        {
            __m512i scalar_vec = _mm512_set1_epi32(static_cast<int>(scalor));
            for (; i + 16 <= shape; i += 16)
            {
                __m512i a = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(&A[i]));
                alignas(64) int temp[16];
                _mm512_storeu_si512(reinterpret_cast<__m512i *>(temp), a);

#pragma unroll
                for (int j = 0; j < 16; j++)
                {
                    temp[j] %= static_cast<int>(scalor);
                }

                a = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(temp));
                _mm512_storeu_si512(reinterpret_cast<__m512i *>(&A[i]), a);
            }
        }
        else if constexpr (sizeof(data_type) == 8) // 64-bit integers
        {
            __m512i scalar_vec = _mm512_set1_epi64(static_cast<long long>(scalor));
            for (; i + 8 <= shape; i += 8)
            {
                __m512i a = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(&A[i]));
                alignas(64) data_type temp[8];
                _mm512_storeu_si512(reinterpret_cast<__m512i *>(temp), a);

#pragma unroll
                for (size_t j = 0; j < 8; ++j)
                {
                    temp[j] %= scalor;
                }

                a = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(temp));
                _mm512_storeu_si512(reinterpret_cast<__m512i *>(&A[i]), a);
            }
        }
    }

    // Xử lý các phần tử còn lại
    for (; i < shape; ++i)
    {
        if constexpr (std::is_same_v<data_type, float>)
            A[i] = fmodf(A[i], scalor); // Sử dụng fmodf thay cho std::fmod
        else if constexpr (std::is_same_v<data_type, double>)
            A[i] = std::fmod(A[i], scalor);
        else if constexpr (std::is_integral_v<data_type>)
            A[i] %= scalor;
    }
}

template <typename data_type>
void simd_elem_mod(const data_type &scalor, data_type *A, size_t shape)
{
    size_t i = 0;

    if constexpr (std::is_same_v<data_type, float>)
    {
        for (; i + 16 <= shape; i += 16)
        {
            __m512 a = _mm512_loadu_ps(&A[i]);

            // Use temporary arrays for calculation
            alignas(64) float temp_in[16], temp_out[16];
            _mm512_storeu_ps(temp_in, a);

#pragma omp parallel for simd
            for (int j = 0; j < 16; ++j)
            {
                // Handle division by zero
                if (temp_in[j] == 0.0f)
                {
                    temp_out[j] = std::numeric_limits<float>::quiet_NaN();
                }
                else
                {
                    // scalor % A[i] = fmod(scalor, A[i])
                    temp_out[j] = fmodf(scalor, temp_in[j]);
                }
            }

            __m512 result = _mm512_loadu_ps(temp_out);
            _mm512_storeu_ps(&A[i], result);
        }
    }
    else if constexpr (std::is_same_v<data_type, double>)
    {
        for (; i + 8 <= shape; i += 8)
        {
            __m512d a = _mm512_loadu_pd(&A[i]);

            // Use temporary arrays for calculation
            alignas(64) double temp_in[8], temp_out[8];
            _mm512_storeu_pd(temp_in, a);

#pragma omp parallel for simd
            for (int j = 0; j < 8; ++j)
            {
                // Handle division by zero
                if (temp_in[j] == 0.0)
                {
                    temp_out[j] = std::numeric_limits<double>::quiet_NaN();
                }
                else
                {
                    // scalor % A[i] = fmod(scalor, A[i])
                    temp_out[j] = std::fmod(scalor, temp_in[j]);
                }
            }

            __m512d result = _mm512_loadu_pd(temp_out);
            _mm512_storeu_pd(&A[i], result);
        }
    }
    else if constexpr (std::is_integral_v<data_type>)
    {
        if constexpr (sizeof(data_type) == 1) // 8-bit integers
        {
            for (; i + 64 <= shape; i += 64)
            {
                __m512i a = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(&A[i]));
                alignas(64) signed char temp_in[64];
                alignas(64) data_type temp_out[64];
                _mm512_storeu_si512(reinterpret_cast<__m512i *>(temp_in), a);

#pragma omp parallel for simd
                for (int j = 0; j < 64; ++j)
                {
                    // Handle modulo by zero
                    if (temp_in[j] == 0)
                    {
                        temp_out[j] = 0; // Division by zero in modulo is undefined
                    }
                    else
                    {
                        // scalor % A[i]
                        temp_out[j] = scalor % temp_in[j];
                    }
                }

                __m512i result = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(temp_out));
                _mm512_storeu_si512(reinterpret_cast<__m512i *>(&A[i]), result);
            }
        }
        else if constexpr (sizeof(data_type) == 2) // 16-bit integers
        {
            for (; i + 32 <= shape; i += 32)
            {
                __m512i a = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(&A[i]));
                alignas(64) short temp_in[32];
                alignas(64) data_type temp_out[32];
                _mm512_storeu_si512(reinterpret_cast<__m512i *>(temp_in), a);

#pragma omp parallel for simd
                for (int j = 0; j < 32; ++j)
                {
                    if (temp_in[j] == 0)
                    {
                        temp_out[j] = 0;
                    }
                    else
                    {
                        temp_out[j] = scalor % temp_in[j];
                    }
                }

                __m512i result = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(temp_out));
                _mm512_storeu_si512(reinterpret_cast<__m512i *>(&A[i]), result);
            }
        }
        else if constexpr (sizeof(data_type) == 4) // 32-bit integers
        {
            for (; i + 16 <= shape; i += 16)
            {
                __m512i a = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(&A[i]));
                alignas(64) int temp_in[16];
                alignas(64) data_type temp_out[16];
                _mm512_storeu_si512(reinterpret_cast<__m512i *>(temp_in), a);

#pragma omp parallel for simd
                for (int j = 0; j < 16; ++j)
                {
                    if (temp_in[j] == 0)
                    {
                        temp_out[j] = 0;
                    }
                    else
                    {
                        temp_out[j] = scalor % temp_in[j];
                    }
                }

                __m512i result = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(temp_out));
                _mm512_storeu_si512(reinterpret_cast<__m512i *>(&A[i]), result);
            }
        }
        else if constexpr (sizeof(data_type) == 8) // 64-bit integers
        {
            for (; i + 8 <= shape; i += 8)
            {
                __m512i a = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(&A[i]));
                alignas(64) long long temp_in[8];
                alignas(64) data_type temp_out[8];
                _mm512_storeu_si512(reinterpret_cast<__m512i *>(temp_in), a);

#pragma omp parallel for simd
                for (int j = 0; j < 8; ++j)
                {
                    if (temp_in[j] == 0)
                    {
                        temp_out[j] = 0;
                    }
                    else
                    {
                        temp_out[j] = scalor % temp_in[j];
                    }
                }

                __m512i result = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(temp_out));
                _mm512_storeu_si512(reinterpret_cast<__m512i *>(&A[i]), result);
            }
        }
    }

// Process remaining elements with OpenMP parallelization
#pragma omp parallel for
    for (size_t j = i; j < shape; ++j)
    {
        if (A[j] == 0)
        {
            if constexpr (std::is_floating_point_v<data_type>)
            {
                A[j] = std::numeric_limits<data_type>::quiet_NaN();
            }
            else
            {
                A[j] = 0;
            }
        }
        else
        {
            if constexpr (std::is_same_v<data_type, float>)
            {
                A[j] = fmodf(scalor, A[j]);
            }
            else if constexpr (std::is_same_v<data_type, double>)
            {
                A[j] = std::fmod(scalor, A[j]);
            }
            else if constexpr (std::is_integral_v<data_type>)
            {
                A[j] = scalor % A[j];
            }
        }
    }
}

template <typename data_type>
bool simd_elem_eq(const data_type *A, size_t shape, const data_type &scalor)
{
    size_t i = 0;
    if constexpr (std::is_same_v<data_type, float>)
    {
        __m512 scalar_vec = _mm512_set1_ps(scalor);
#pragma unroll
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
#pragma unroll
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
        if constexpr (sizeof(data_type) == 1) // 8-bit integers
        {
            __m512i scalar_vec = _mm512_set1_epi8(static_cast<char>(scalor));
#pragma unroll
            for (; i + 64 <= shape; i += 64) // Process 64 elements at once
            {
                __m512i a = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(&A[i]));
                __mmask64 mask = _mm512_cmpeq_epi8_mask(a, scalar_vec);
                if (mask != 0xFFFFFFFFFFFFFFFFULL) // All 64 bits should be set
                    return false;
            }
        }
        else if constexpr (sizeof(data_type) == 2) // 16-bit integers
        {
            __m512i scalar_vec = _mm512_set1_epi16(static_cast<short>(scalor));
#pragma unroll
            for (; i + 32 <= shape; i += 32) // Process 32 elements at once
            {
                __m512i a = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(&A[i]));
                __mmask32 mask = _mm512_cmpeq_epi16_mask(a, scalar_vec);
                if (mask != 0xFFFFFFFF) // All 32 bits should be set
                    return false;
            }
        }
        else if constexpr (sizeof(data_type) == 4) // 32-bit integers
        {
            __m512i scalar_vec = _mm512_set1_epi32(static_cast<int>(scalor));
#pragma unroll
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
            __m512i scalar_vec = _mm512_set1_epi64(static_cast<long long>(scalor));
#pragma unroll
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
#pragma unroll
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
#pragma unroll
        for (; i + 16 <= shape; i += 16)
        {
            __m512 a = _mm512_loadu_ps(&A[i]);
            __mmask16 mask = _mm512_cmp_ps_mask(a, scalar_vec, _CMP_GT_OQ);
            if (mask != 0xFFFF) // Check if ALL elements are larger than scalar
                return false;
        }
    }
    else if constexpr (std::is_same_v<data_type, double>)
    {
        __m512d scalar_vec = _mm512_set1_pd(scalor);
#pragma unroll
        for (; i + 8 <= shape; i += 8)
        {
            __m512d a = _mm512_loadu_pd(&A[i]);
            __mmask8 mask = _mm512_cmp_pd_mask(a, scalar_vec, _CMP_GT_OQ);
            if (mask != 0xFF) // Check if ALL elements are larger than scalar
                return false;
        }
    }
    else if constexpr (std::is_integral_v<data_type>)
    {
        if constexpr (sizeof(data_type) == 1) // 8-bit integers
        {
            __m512i scalar_vec = _mm512_set1_epi8(static_cast<char>(scalor));
#pragma unroll
            for (; i + 64 <= shape; i += 64) // Process 64 elements at once
            {
                __m512i a = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(&A[i]));
                __mmask64 mask = _mm512_cmpgt_epi8_mask(a, scalar_vec);
                if (mask != 0xFFFFFFFFFFFFFFFFULL) // All 64 bits should be set
                    return false;
            }
        }
        else if constexpr (sizeof(data_type) == 2) // 16-bit integers
        {
            __m512i scalar_vec = _mm512_set1_epi16(static_cast<short>(scalor));
#pragma unroll
            for (; i + 32 <= shape; i += 32) // Process 32 elements at once
            {
                __m512i a = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(&A[i]));
                __mmask32 mask = _mm512_cmpgt_epi16_mask(a, scalar_vec);
                if (mask != 0xFFFFFFFF) // All 32 bits should be set
                    return false;
            }
        }
        else if constexpr (sizeof(data_type) == 4) // 32-bit integers
        {
            __m512i scalar_vec = _mm512_set1_epi32(static_cast<int>(scalor));
#pragma unroll
            for (; i + 16 <= shape; i += 16)
            {
                __m512i a = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(&A[i]));
                __mmask16 mask = _mm512_cmpgt_epi32_mask(a, scalar_vec);
                if (mask != 0xFFFF) // All 16 bits should be set
                    return false;
            }
        }
        else if constexpr (sizeof(data_type) == 8) // 64-bit integers
        {
            __m512i scalar_vec = _mm512_set1_epi64(static_cast<long long>(scalor));
#pragma unroll
            for (; i + 8 <= shape; i += 8)
            {
                __m512i a = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(&A[i]));
                __mmask8 mask = _mm512_cmpgt_epi64_mask(a, scalar_vec);
                if (mask != 0xFF) // All 8 bits should be set
                    return false;
            }
        }
    }
    // Process remaining elements
#pragma unroll
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
#pragma unroll
        for (; i + 16 <= shape; i += 16)
        {
            __m512 a = _mm512_loadu_ps(&A[i]);
            __mmask16 mask = _mm512_cmp_ps_mask(a, scalar_vec, _CMP_LT_OQ);
            if (mask != 0xFFFF) // Check if ALL elements are smaller than scalar
                return false;
        }
    }
    else if constexpr (std::is_same_v<data_type, double>)
    {
        __m512d scalar_vec = _mm512_set1_pd(scalor);
#pragma unroll
        for (; i + 8 <= shape; i += 8)
        {
            __m512d a = _mm512_loadu_pd(&A[i]);
            __mmask8 mask = _mm512_cmp_pd_mask(a, scalar_vec, _CMP_LT_OQ);
            if (mask != 0xFF) // Check if ALL elements are smaller than scalar
                return false;
        }
    }
    else if constexpr (std::is_integral_v<data_type>)
    {
        if constexpr (sizeof(data_type) == 1) // 8-bit integers
        {
            __m512i scalar_vec = _mm512_set1_epi8(static_cast<char>(scalor));
#pragma unroll
            for (; i + 64 <= shape; i += 64) // Process 64 elements at once
            {
                __m512i a = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(&A[i]));
                __mmask64 mask = _mm512_cmplt_epi8_mask(a, scalar_vec);
                if (mask != 0xFFFFFFFFFFFFFFFFULL) // All 64 bits should be set
                    return false;
            }
        }
        else if constexpr (sizeof(data_type) == 2) // 16-bit integers
        {
            __m512i scalar_vec = _mm512_set1_epi16(static_cast<short>(scalor));
#pragma unroll
            for (; i + 32 <= shape; i += 32) // Process 32 elements at once
            {
                __m512i a = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(&A[i]));
                __mmask32 mask = _mm512_cmplt_epi16_mask(a, scalar_vec);
                if (mask != 0xFFFFFFFF) // All 32 bits should be set
                    return false;
            }
        }
        else if constexpr (sizeof(data_type) == 4) // 32-bit integers
        {
            __m512i scalar_vec = _mm512_set1_epi32(static_cast<int>(scalor));
#pragma unroll
            for (; i + 16 <= shape; i += 16)
            {
                __m512i a = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(&A[i]));
                __mmask16 mask = _mm512_cmplt_epi32_mask(a, scalar_vec);
                if (mask != 0xFFFF) // All 16 bits should be set
                    return false;
            }
        }
        else if constexpr (sizeof(data_type) == 8) // 64-bit integers
        {
            __m512i scalar_vec = _mm512_set1_epi64(static_cast<long long>(scalor));
#pragma unroll
            for (; i + 8 <= shape; i += 8)
            {
                __m512i a = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(&A[i]));
                __mmask8 mask = _mm512_cmplt_epi64_mask(a, scalar_vec);
                if (mask != 0xFF) // All 8 bits should be set
                    return false;
            }
        }
    }
    // Process remaining elements
#pragma unroll
    for (; i < shape; i++)
    {
        if (A[i] >= scalor) // Check if element is NOT smaller than scalar
            return false;
    }
    return true;
}

template <typename data_type>
void simd_elem_log(data_type *A, size_t shape, const data_type &scalor)
{
    size_t i = 0;
    if constexpr (std::is_same_v<data_type, float>)
    {
        // Natural log of the scalar (for changing base)
        float log_base = std::log(scalor);

        for (; i + 16 <= shape; i += 16)
        {
            __m512 a = _mm512_loadu_ps(&A[i]);

            // AVX-512 doesn't have direct log instructions, use temp arrays
            alignas(64) float temp_in[16], temp_out[16];
            _mm512_storeu_ps(temp_in, a);

#pragma omp parallel for simd
            for (int j = 0; j < 16; ++j)
            {
                // Handle special cases
                if (temp_in[j] <= 0.0f)
                {
                    temp_out[j] = std::numeric_limits<float>::quiet_NaN();
                }
                else if (scalor <= 0.0f || scalor == 1.0f)
                {
                    temp_out[j] = std::numeric_limits<float>::quiet_NaN();
                }
                else
                {
                    // log_base(x) = log(x) / log(base)
                    temp_out[j] = std::log(temp_in[j]) / log_base;
                }
            }

            __m512 result = _mm512_loadu_ps(temp_out);
            _mm512_storeu_ps(&A[i], result);
        }
    }
    else if constexpr (std::is_same_v<data_type, double>)
    {
        // Natural log of the scalar (for changing base)
        double log_base = std::log(scalor);

        for (; i + 8 <= shape; i += 8)
        {
            __m512d a = _mm512_loadu_pd(&A[i]);

            alignas(64) double temp_in[8], temp_out[8];
            _mm512_storeu_pd(temp_in, a);

#pragma omp parallel for simd
            for (int j = 0; j < 8; ++j)
            {
                // Handle special cases
                if (temp_in[j] <= 0.0)
                {
                    temp_out[j] = std::numeric_limits<double>::quiet_NaN();
                }
                else if (scalor <= 0.0 || scalor == 1.0)
                {
                    temp_out[j] = std::numeric_limits<double>::quiet_NaN();
                }
                else
                {
                    // log_base(x) = log(x) / log(base)
                    temp_out[j] = std::log(temp_in[j]) / log_base;
                }
            }

            __m512d result = _mm512_loadu_pd(temp_out);
            _mm512_storeu_pd(&A[i], result);
        }
    }
    else if constexpr (std::is_integral_v<data_type>)
    {
        // For integers, convert to double for calculation
        double log_base = std::log(static_cast<double>(scalor));

        if constexpr (sizeof(data_type) == 1) // 8-bit integers
        {
            for (; i + 64 <= shape; i += 64)
            {
                __m512i a = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(&A[i]));
                alignas(64) signed char temp_in[64];
                alignas(64) data_type temp_out[64];
                _mm512_storeu_si512(reinterpret_cast<__m512i *>(temp_in), a);

#pragma omp parallel for simd
                for (int j = 0; j < 64; ++j)
                {
                    // Handle special cases for integers
                    if (temp_in[j] <= 0 || scalor <= 0 || scalor == 1)
                    {
                        temp_out[j] = 0; // Or another appropriate error value
                    }
                    else
                    {
                        // Integer logarithm (floor of the actual log value)
                        temp_out[j] = static_cast<data_type>(
                            std::floor(std::log(static_cast<double>(temp_in[j])) / log_base));
                    }
                }

                __m512i result = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(temp_out));
                _mm512_storeu_si512(reinterpret_cast<__m512i *>(&A[i]), result);
            }
        }
        else if constexpr (sizeof(data_type) == 2) // 16-bit integers
        {
            for (; i + 32 <= shape; i += 32)
            {
                __m512i a = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(&A[i]));
                alignas(64) short temp_in[32];
                alignas(64) data_type temp_out[32];
                _mm512_storeu_si512(reinterpret_cast<__m512i *>(temp_in), a);

#pragma omp parallel for simd
                for (int j = 0; j < 32; ++j)
                {
                    // Handle special cases for integers
                    if (temp_in[j] <= 0 || scalor <= 0 || scalor == 1)
                    {
                        temp_out[j] = 0; // Or another appropriate error value
                    }
                    else
                    {
                        // Integer logarithm (floor of the actual log value)
                        temp_out[j] = static_cast<data_type>(
                            std::floor(std::log(static_cast<double>(temp_in[j])) / log_base));
                    }
                }

                __m512i result = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(temp_out));
                _mm512_storeu_si512(reinterpret_cast<__m512i *>(&A[i]), result);
            }
        }
        else if constexpr (sizeof(data_type) == 4) // 32-bit integers
        {
            for (; i + 16 <= shape; i += 16)
            {
                __m512i a = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(&A[i]));
                alignas(64) int temp_in[16];
                alignas(64) data_type temp_out[16];
                _mm512_storeu_si512(reinterpret_cast<__m512i *>(temp_in), a);

#pragma omp parallel for simd
                for (int j = 0; j < 16; ++j)
                {
                    // Handle special cases for integers
                    if (temp_in[j] <= 0 || scalor <= 0 || scalor == 1)
                    {
                        temp_out[j] = 0; // Or another appropriate error value
                    }
                    else
                    {
                        // Integer logarithm (floor of the actual log value)
                        temp_out[j] = static_cast<data_type>(
                            std::floor(std::log(static_cast<double>(temp_in[j])) / log_base));
                    }
                }

                __m512i result = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(temp_out));
                _mm512_storeu_si512(reinterpret_cast<__m512i *>(&A[i]), result);
            }
        }
        else if constexpr (sizeof(data_type) == 8) // 64-bit integers
        {
            for (; i + 8 <= shape; i += 8)
            {
                __m512i a = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(&A[i]));
                alignas(64) long long temp_in[8];
                alignas(64) data_type temp_out[8];
                _mm512_storeu_si512(reinterpret_cast<__m512i *>(temp_in), a);

#pragma omp parallel for simd
                for (int j = 0; j < 8; ++j)
                {
                    if (temp_in[j] <= 0 || scalor <= 0 || scalor == 1)
                    {
                        temp_out[j] = 0;
                    }
                    else
                    {
                        temp_out[j] = static_cast<data_type>(
                            std::floor(std::log(static_cast<double>(temp_in[j])) / log_base));
                    }
                }

                __m512i result = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(temp_out));
                _mm512_storeu_si512(reinterpret_cast<__m512i *>(&A[i]), result);
            }
        }
    }

// Process remaining elements
// Process remaining elements with proper OpenMP canonical form
#pragma omp parallel for
    for (size_t j = i; j < shape; ++j)
    {
        if constexpr (std::is_floating_point_v<data_type>)
        {
            if (A[j] <= 0 || scalor <= 0 || scalor == 1)
            {
                A[j] = std::numeric_limits<data_type>::quiet_NaN();
            }
            else
            {
                A[j] = std::log(A[j]) / std::log(scalor);
            }
        }
        else if constexpr (std::is_integral_v<data_type>)
        {
            if (A[j] <= 0 || scalor <= 0 || scalor == 1)
            {
                A[j] = 0;
            }
            else
            {
                A[j] = static_cast<data_type>(
                    std::floor(std::log(static_cast<double>(A[j])) / std::log(static_cast<double>(scalor))));
            }
        }
    }
}

template <typename data_type>
void simd_elem_sqrt(const data_type &scalor, data_type *A, size_t shape)
{
    const data_type convert_scalor = 1 / (data_type)scalor;
    simd_elem_power(A, shape, convert_scalor);
}

template <typename data_type>
void simd_elem_xor(const data_type &scalor, data_type *A, size_t shape)
{
    // XOR is typically only meaningful for integral types
    static_assert(std::is_integral_v<data_type>, "XOR operation is only supported for integral types.");

    size_t i = 0;

    if constexpr (std::is_integral_v<data_type>)
    {
        // Determine vector size based on data_type size
        constexpr int type_size = sizeof(data_type);
        constexpr int VEC_SIZE = 64 / type_size; // Number of elements per 512-bit vector

        // Load the scalar into a full AVX-512 register
        __m512i scalar_vec;
        if constexpr (type_size == 1)
        { // 8-bit
            scalar_vec = _mm512_set1_epi8(static_cast<char>(scalor));
        }
        else if constexpr (type_size == 2)
        { // 16-bit
            scalar_vec = _mm512_set1_epi16(static_cast<short>(scalor));
        }
        else if constexpr (type_size == 4)
        { // 32-bit
            scalar_vec = _mm512_set1_epi32(static_cast<int>(scalor));
        }
        else if constexpr (type_size == 8)
        { // 64-bit
            scalar_vec = _mm512_set1_epi64(static_cast<long long>(scalor));
        }

        // Process elements in chunks using AVX-512
        // No #pragma unroll here as the loop bound is data-dependent
        for (; i + VEC_SIZE <= shape; i += VEC_SIZE)
        {
            // Load data from the array into an AVX-512 register
            __m512i a_vec = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(&A[i]));

            // Perform the XOR operation with the scalar vector
            a_vec = _mm512_xor_si512(scalar_vec, a_vec);

            // Store the result back to the array
            _mm512_storeu_si512(reinterpret_cast<__m512i *>(&A[i]), a_vec);
        }
    }
    // else: If somehow called with non-integral, do nothing in SIMD part

    // Process remaining elements using a scalar loop
    // #pragma unroll // Unrolling might be less effective here due to small remaining count
    for (; i < shape; ++i)
    {
        if constexpr (std::is_integral_v<data_type>)
        {
            A[i] = scalor ^ A[i];
        }
        // else: If somehow called with non-integral, do nothing in scalar part
    }
}