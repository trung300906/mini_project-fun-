#pragma once

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

// Generic IN-PLACE SIMD type casting with AVX-512
template <typename SrcType, typename DstType>
void simd_cast(SrcType *data, size_t shape)
{
    // No operation needed if types are identical
    if constexpr (std::is_same_v<SrcType, DstType>)
    {
        return;
    }

    // Determine loop direction based on type sizes
    constexpr bool is_expanding = sizeof(DstType) > sizeof(SrcType);
    constexpr bool is_shrinking = sizeof(DstType) < sizeof(SrcType);
    constexpr bool is_same_size = sizeof(DstType) == sizeof(SrcType);

    DstType *dst_data = reinterpret_cast<DstType *>(data); // Logical destination pointer

    // ==============================================================
    // == Expanding Conversions (Process Backwards) sizeof(Dst) > sizeof(Src) ==
    // ==============================================================
    if constexpr (is_expanding)
    {
        // --- Float -> Double (Expanding) ---
        if constexpr (std::is_same_v<SrcType, float> && std::is_same_v<DstType, double>)
        {
            const size_t step = 8; // Process 8 elements (256 bits of float -> 512 bits of double)
#pragma omp parallel for
            for (size_t j = shape; j >= step; j -= step) // Loop backwards
            {
                size_t current_idx = j - step;
                // Load 8 floats (offset based on SrcType)
                __m256 src_vec = _mm256_loadu_ps(reinterpret_cast<const float *>(&data[current_idx]));
                // Convert to 8 doubles
                __m512d dst_vec = _mm512_cvtps_pd(src_vec);
                // Store 8 doubles (offset based on DstType)
                _mm512_storeu_pd(reinterpret_cast<double *>(&dst_data[current_idx]), dst_vec);
            }
// Handle remaining elements at the beginning (scalar)
#pragma omp parallel for // Can parallelize the scalar part too, though overhead might be high
            for (size_t k = 0; k < shape % step; ++k)
            {
                dst_data[k] = static_cast<DstType>(data[k]);
            }
        }
        // --- int32 -> double (Expanding) ---
        else if constexpr (std::is_integral_v<SrcType> && sizeof(SrcType) == 4 && std::is_same_v<DstType, double>)
        {
            const size_t step = 8; // Process 8 elements (256 bits of int32 -> 512 bits of double)
#pragma omp parallel for
            for (size_t j = shape; j >= step; j -= step) // Loop backwards
            {
                size_t current_idx = j - step;
                // Load 8 int32s (offset based on SrcType)
                __m256i src_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(&data[current_idx]));
                // Convert to 8 doubles
                __m512d dst_vec = _mm512_cvtepi32_pd(src_vec);
                // Store 8 doubles (offset based on DstType)
                _mm512_storeu_pd(reinterpret_cast<double *>(&dst_data[current_idx]), dst_vec);
            }
            // Handle remaining elements at the beginning (scalar)
#pragma omp parallel for
            for (size_t k = 0; k < shape % step; ++k)
            {
                dst_data[k] = static_cast<DstType>(data[k]);
            }
        }
        // --- Other Expanding Conversions (Fallback to Backward Scalar) ---
        else
        {
// Process backwards to avoid overwriting unprocessed data
#pragma omp parallel for
            for (size_t j = shape; j > 0; --j)
            {
                // Use placement new or careful casting if constructors/destructors matter
                // For POD types, simple assignment after casting the pointer is okay conceptually,
                // but direct assignment modifies memory based on SrcType size.
                // We need to read SrcType and write DstType logically at the same location.
                // This is tricky and often requires a temporary buffer or careful pointer arithmetic.
                // A simpler (but potentially slower) scalar approach:
                dst_data[j - 1] = static_cast<DstType>(data[j - 1]); // Read SrcType, convert, write DstType
                                                                     // This relies on the compiler/memory model handling
                                                                     // the read before the (larger) write completes overwriting.
                                                                     // It's generally safer for POD types.
            }
            // A more robust scalar approach might use a temporary variable:
            /*
            if (shape > 0) { // Avoid underflow with size_t for shape=0
                #pragma omp parallel
                {
                    // Temporary variable per thread might be needed if SrcType/DstType are complex
                    #pragma omp for schedule(static) // Use static schedule for predictable backward processing
                    for (long long j = static_cast<long long>(shape - 1); j >= 0; --j) {
                        // Read before potential overwrite
                        SrcType temp_val = data[j];
                        // Now write the converted value. The write might overlap with
                        // data[j-1] or others, but we've already read data[j].
                        dst_data[j] = static_cast<DstType>(temp_val);
                    }
                }
            }
            */
        }
    }
    // ==============================================================
    // == Shrinking Conversions (Process Forwards) sizeof(Dst) < sizeof(Src) ==
    // ==============================================================
    else if constexpr (is_shrinking)
    {
        // --- Double -> Float (Shrinking) ---
        if constexpr (std::is_same_v<SrcType, double> && std::is_same_v<DstType, float>)
        {
            const size_t step = 16; // Process 16 elements (2x512 bits of double -> 512 bits of float)
            size_t vec_end = shape - (shape % step);
#pragma omp parallel for
            for (size_t j = 0; j < vec_end; j += step)
            {
                // Load 16 doubles (offset based on SrcType)
                __m512d src_vec1 = _mm512_loadu_pd(reinterpret_cast<const double *>(&data[j]));
                __m512d src_vec2 = _mm512_loadu_pd(reinterpret_cast<const double *>(&data[j + 8]));
                // Convert to 16 floats
                __m256 dst_vec_lo = _mm512_cvtpd_ps(src_vec1);
                __m256 dst_vec_hi = _mm512_cvtpd_ps(src_vec2);
                // Combine into one 512-bit float vector
                __m512 dst_vec = _mm512_insertf32x8(_mm512_castps256_ps512(dst_vec_lo), dst_vec_hi, 1);
                // Store 16 floats (offset based on DstType)
                _mm512_storeu_ps(reinterpret_cast<float *>(&dst_data[j]), dst_vec);
            }
            // Handle remaining elements at the end (scalar)
#pragma omp parallel for
            for (size_t k = vec_end; k < shape; ++k)
            {
                dst_data[k] = static_cast<DstType>(data[k]);
            }
        }
        // --- Double -> int32 (Shrinking) ---
        else if constexpr (std::is_same_v<SrcType, double> && std::is_integral_v<DstType> && sizeof(DstType) == 4)
        {
            const size_t step = 8; // Process 8 elements (512 bits double -> 256 bits int32)
            size_t vec_end = shape - (shape % step);
#pragma omp parallel for
            for (size_t j = 0; j < vec_end; j += step)
            {
                // Load 8 doubles (offset based on SrcType)
                __m512d src_vec = _mm512_loadu_pd(reinterpret_cast<const double *>(&data[j]));
                // Convert to 8 int32s (rounding towards zero)
                // Use _mm512_cvt_roundpd_epi32 for specific rounding modes if needed
                __m256i dst_vec = _mm512_cvtpd_epi32(src_vec);
                // Store 8 int32s (offset based on DstType)
                _mm256_storeu_si256(reinterpret_cast<__m256i *>(&dst_data[j]), dst_vec);
            }
// Handle remaining elements at the end (scalar, with rounding)
#pragma omp parallel for
            for (size_t k = vec_end; k < shape; ++k)
            {
                dst_data[k] = static_cast<DstType>(std::round(data[k])); // Use std::round for consistency
            }
        }
        // --- Other Shrinking Conversions (Fallback to Forward Scalar) ---
        else
        {
// Process forward is safe for shrinking
#pragma omp parallel for
            for (size_t j = 0; j < shape; ++j)
            {
                // Round floats/doubles when converting to integer
                if constexpr (std::is_floating_point_v<SrcType> && std::is_integral_v<DstType>)
                {
                    dst_data[j] = static_cast<DstType>(std::round(data[j]));
                }
                else
                {
                    dst_data[j] = static_cast<DstType>(data[j]);
                }
            }
        }
    }

    // ==============================================================
    // == Same Size Conversions (Process Forwards) sizeof(Dst) == sizeof(Src) ==
    // ==============================================================
    else // implies is_same_size
    {
        // --- Float -> int32 (Same Size) ---
        if constexpr (std::is_same_v<SrcType, float> && std::is_integral_v<DstType> && sizeof(DstType) == 4)
        {
            const size_t step = 16; // Process 16 elements (512 bits float -> 512 bits int32)
            size_t vec_end = shape - (shape % step);
#pragma omp parallel for
            for (size_t j = 0; j < vec_end; j += step)
            {
                // Load 16 floats
                __m512 src_vec = _mm512_loadu_ps(reinterpret_cast<const float *>(&data[j]));
                // Convert to 16 int32s (rounding towards zero)
                // Use _mm512_cvt_roundps_epi32 for specific rounding modes if needed
                __m512i dst_vec = _mm512_cvtps_epi32(src_vec);
                // Store 16 int32s
                _mm512_storeu_si512(reinterpret_cast<__m512i *>(&dst_data[j]), dst_vec);
            }
// Handle remaining elements at the end (scalar, with rounding)
#pragma omp parallel for
            for (size_t k = vec_end; k < shape; ++k)
            {
                dst_data[k] = static_cast<DstType>(std::round(data[k])); // Use std::round for consistency
            }
        }
        // --- int32 -> Float (Same Size) ---
        else if constexpr (std::is_integral_v<SrcType> && sizeof(SrcType) == 4 && std::is_same_v<DstType, float>)
        {
            const size_t step = 16; // Process 16 elements (512 bits int32 -> 512 bits float)
            size_t vec_end = shape - (shape % step);
#pragma omp parallel for
            for (size_t j = 0; j < vec_end; j += step)
            {
                // Load 16 int32s
                __m512i src_vec = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(&data[j]));
                // Convert to 16 floats
                __m512 dst_vec = _mm512_cvtepi32_ps(src_vec);
                // Store 16 floats
                _mm512_storeu_ps(reinterpret_cast<float *>(&dst_data[j]), dst_vec);
            }
// Handle remaining elements at the end (scalar)
#pragma omp parallel for
            for (size_t k = vec_end; k < shape; ++k)
            {
                dst_data[k] = static_cast<DstType>(data[k]);
            }
        }
        // --- Integer -> Integer (Same Size) ---
        // Includes cases like int32->uint32, int64->uint64 etc.
        // Effectively a reinterpret_cast of the block, SIMD copy is fast.
        else if constexpr (std::is_integral_v<SrcType> && std::is_integral_v<DstType>)
        {
            // Treat as raw data copy using SIMD load/store
            const size_t step = 64 / sizeof(SrcType); // Number of elements in 512 bits
            size_t vec_end = shape - (shape % step);
#pragma omp parallel for
            for (size_t j = 0; j < vec_end; j += step)
            {
                __m512i vec = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(&data[j]));
                _mm512_storeu_si512(reinterpret_cast<__m512i *>(&dst_data[j]), vec);
            }
            // Handle remaining elements at the end (scalar)
#pragma omp parallel for // Can parallelize scalar copy too
            for (size_t k = vec_end; k < shape; ++k)
            {
                dst_data[k] = static_cast<DstType>(data[k]); // Simple scalar cast
            }
        }
        // --- Other Same Size Conversions (Fallback to Forward Scalar) ---
        // E.g., custom types if they happened to have the same size
        else
        {
            // Process forward is safe for same-size
#pragma omp parallel for
            for (size_t j = 0; j < shape; ++j)
            {
                dst_data[j] = static_cast<DstType>(data[j]);
            }
        }
    }
}

template <typename data_type>
void simd_sin(data_type *A, size_t shape)
{
    size_t i = 0;

    if constexpr (std::is_same_v<data_type, float>)
    {
        for (; i + 16 <= shape; i += 16)
        {
            __m512 a = _mm512_loadu_ps(&A[i]);

            // No direct sine AVX-512 instruction, use temporary arrays
            alignas(64) float temp_in[16], temp_out[16];
            _mm512_storeu_ps(temp_in, a);

#pragma omp parallel for simd
            for (int j = 0; j < 16; ++j)
            {
                temp_out[j] = std::sin(temp_in[j]);
            }

            // Load results back to SIMD register and store
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
                temp_out[j] = std::sin(temp_in[j]);
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
                    // Convert to float, apply sin, round to nearest integer
                    temp_out[j] = static_cast<data_type>(
                        std::round(std::sin(static_cast<float>(temp_in[j]))));
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
                    // Convert to float, apply sin, round to nearest integer
                    temp_out[j] = static_cast<data_type>(
                        std::round(std::sin(static_cast<float>(temp_in[j]))));
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
                    // Convert to double for better precision with larger integers
                    temp_out[j] = static_cast<data_type>(
                        std::round(std::sin(static_cast<double>(temp_in[j]))));
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
                    // Convert to double for better precision with larger integers
                    temp_out[j] = static_cast<data_type>(
                        std::round(std::sin(static_cast<double>(temp_in[j]))));
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
            A[j] = std::sin(A[j]);
        }
        else if constexpr (std::is_integral_v<data_type>)
        {
            // For integers, we convert to floating point, apply sin, then round back
            if constexpr (sizeof(data_type) <= 4)
                A[j] = static_cast<data_type>(std::round(std::sin(static_cast<float>(A[j]))));
            else
                A[j] = static_cast<data_type>(std::round(std::sin(static_cast<double>(A[j]))));
        }
    }
}

template <typename data_type>
void simd_cos(data_type *A, size_t shape)
{
    size_t i = 0;

    if constexpr (std::is_same_v<data_type, float>)
    {
        for (; i + 16 <= shape; i += 16)
        {
            __m512 a = _mm512_loadu_ps(&A[i]);

            // No direct cosine AVX-512 instruction, use temporary arrays
            alignas(64) float temp_in[16], temp_out[16];
            _mm512_storeu_ps(temp_in, a);

#pragma omp parallel for simd
            for (int j = 0; j < 16; ++j)
            {
                temp_out[j] = std::cos(temp_in[j]);
            }

            // Load results back to SIMD register and store
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
                temp_out[j] = std::cos(temp_in[j]);
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
                    // Convert to float, apply cos, round to nearest integer
                    temp_out[j] = static_cast<data_type>(
                        std::round(std::cos(static_cast<float>(temp_in[j]))));
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
                    // Convert to float, apply cos, round to nearest integer
                    temp_out[j] = static_cast<data_type>(
                        std::round(std::cos(static_cast<float>(temp_in[j]))));
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
                    // Convert to double for better precision with larger integers
                    temp_out[j] = static_cast<data_type>(
                        std::round(std::cos(static_cast<double>(temp_in[j]))));
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
                    // Convert to double for better precision with larger integers
                    temp_out[j] = static_cast<data_type>(
                        std::round(std::cos(static_cast<double>(temp_in[j]))));
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
            A[j] = std::cos(A[j]);
        }
        else if constexpr (std::is_integral_v<data_type>)
        {
            // For integers, we convert to floating point, apply cos, then round back
            if constexpr (sizeof(data_type) <= 4)
                A[j] = static_cast<data_type>(std::round(std::cos(static_cast<float>(A[j]))));
            else
                A[j] = static_cast<data_type>(std::round(std::cos(static_cast<double>(A[j]))));
        }
    }
}
template <typename data_type>
void simd_tan(data_type *A, size_t shape)
{
    size_t i = 0;

    if constexpr (std::is_same_v<data_type, float>)
    {
        for (; i + 16 <= shape; i += 16)
        {
            __m512 a = _mm512_loadu_ps(&A[i]);

            // No direct tangent AVX-512 instruction, use temporary arrays
            alignas(64) float temp_in[16], temp_out[16];
            _mm512_storeu_ps(temp_in, a);

#pragma omp parallel for simd
            for (int j = 0; j < 16; ++j)
            {
                temp_out[j] = std::tan(temp_in[j]);
            }

            // Load results back to SIMD register and store
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
                temp_out[j] = std::tan(temp_in[j]);
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
                    // Convert to float, apply tan, round to nearest integer
                    temp_out[j] = static_cast<data_type>(
                        std::round(std::tan(static_cast<float>(temp_in[j]))));
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
                    // Convert to float, apply tan, round to nearest integer
                    temp_out[j] = static_cast<data_type>(
                        std::round(std::tan(static_cast<float>(temp_in[j]))));
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
                    // Convert to double for better precision with larger integers
                    temp_out[j] = static_cast<data_type>(
                        std::round(std::tan(static_cast<double>(temp_in[j]))));
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
                    // Convert to double for better precision with larger integers
                    temp_out[j] = static_cast<data_type>(
                        std::round(std::tan(static_cast<double>(temp_in[j]))));
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
            A[j] = std::tan(A[j]);
        }
        else if constexpr (std::is_integral_v<data_type>)
        {
            // For integers, we convert to floating point, apply tan, then round back
            if constexpr (sizeof(data_type) <= 4)
                A[j] = static_cast<data_type>(std::round(std::tan(static_cast<float>(A[j]))));
            else
                A[j] = static_cast<data_type>(std::round(std::tan(static_cast<double>(A[j]))));
        }
    }
}

template <typename data_type>
void simd_arcsin(data_type *A, size_t shape)
{
    size_t i = 0;

    if constexpr (std::is_same_v<data_type, float>)
    {
        for (; i + 16 <= shape; i += 16)
        {
            __m512 a = _mm512_loadu_ps(&A[i]);

            // No direct arcsin AVX-512 instruction, use temporary arrays
            alignas(64) float temp_in[16], temp_out[16];
            _mm512_storeu_ps(temp_in, a);

// Apply std::asin element-wise using OpenMP SIMD
// Note: std::asin domain is [-1, 1]. Input outside this range results in NaN.
#pragma omp parallel for simd
            for (int j = 0; j < 16; ++j)
            {
                temp_out[j] = std::asin(temp_in[j]);
            }

            // Load results back to SIMD register and store
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

// Apply std::asin element-wise using OpenMP SIMD
// Note: std::asin domain is [-1, 1]. Input outside this range results in NaN.
#pragma omp parallel for simd
            for (int j = 0; j < 8; ++j)
            {
                temp_out[j] = std::asin(temp_in[j]);
            }

            __m512d result = _mm512_loadu_pd(temp_out);
            _mm512_storeu_pd(&A[i], result);
        }
    }
    else if constexpr (std::is_integral_v<data_type>)
    {
        // Arcsin is typically not meaningful for integers directly,
        // as the domain is [-1, 1]. Most integers fall outside this.
        // We'll implement it by casting to float/double, applying asin,
        // and casting back, but the results might be limited (often 0 or NaN/error).
        // Consider if this operation is truly needed for integer types.

        if constexpr (sizeof(data_type) == 1) // 8-bit integers
        {
            for (; i + 64 <= shape; i += 64)
            {
                __m512i a = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(&A[i]));
                alignas(64) signed char temp_in[64]; // Use signed char for potential -1
                alignas(64) data_type temp_out[64];
                _mm512_storeu_si512(reinterpret_cast<__m512i *>(temp_in), a);

#pragma omp parallel for simd
                for (int j = 0; j < 64; ++j)
                {
                    // Cast to float, apply asin, round, cast back
                    float val_f = static_cast<float>(temp_in[j]);
                    temp_out[j] = static_cast<data_type>(std::round(std::asin(val_f)));
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
                    float val_f = static_cast<float>(temp_in[j]);
                    temp_out[j] = static_cast<data_type>(std::round(std::asin(val_f)));
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
                    // Use double for potentially better intermediate precision
                    double val_d = static_cast<double>(temp_in[j]);
                    temp_out[j] = static_cast<data_type>(std::round(std::asin(val_d)));
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
                    double val_d = static_cast<double>(temp_in[j]);
                    temp_out[j] = static_cast<data_type>(std::round(std::asin(val_d)));
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
            A[j] = std::asin(A[j]);
        }
        else if constexpr (std::is_integral_v<data_type>)
        {
            // Cast, apply asin, round back
            if constexpr (sizeof(data_type) <= 4)
            {
                float val_f = static_cast<float>(A[j]);
                A[j] = static_cast<data_type>(std::round(std::asin(val_f)));
            }
            else
            {
                double val_d = static_cast<double>(A[j]);
                A[j] = static_cast<data_type>(std::round(std::asin(val_d)));
            }
        }
    }
}

template <typename data_type>
void simd_arccos(data_type *A, size_t shape)
{
    size_t i = 0;

    if constexpr (std::is_same_v<data_type, float>)
    {
        for (; i + 16 <= shape; i += 16)
        {
            __m512 a = _mm512_loadu_ps(&A[i]);

            // No direct arccos AVX-512 instruction, use temporary arrays
            alignas(64) float temp_in[16], temp_out[16];
            _mm512_storeu_ps(temp_in, a);

// Apply std::acos element-wise using OpenMP SIMD
// Note: std::acos domain is [-1, 1]. Input outside this range results in NaN.
#pragma omp parallel for simd
            for (int j = 0; j < 16; ++j)
            {
                temp_out[j] = std::acos(temp_in[j]);
            }

            // Load results back to SIMD register and store
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

// Apply std::acos element-wise using OpenMP SIMD
// Note: std::acos domain is [-1, 1]. Input outside this range results in NaN.
#pragma omp parallel for simd
            for (int j = 0; j < 8; ++j)
            {
                temp_out[j] = std::acos(temp_in[j]);
            }

            __m512d result = _mm512_loadu_pd(temp_out);
            _mm512_storeu_pd(&A[i], result);
        }
    }
    else if constexpr (std::is_integral_v<data_type>)
    {
        // Arccos is typically not meaningful for integers directly,
        // as the domain is [-1, 1]. Most integers fall outside this.
        // We'll implement it by casting to float/double, applying acos,
        // and casting back, but the results might be limited (often 0 or NaN/error).
        // Consider if this operation is truly needed for integer types.

        if constexpr (sizeof(data_type) == 1) // 8-bit integers
        {
            for (; i + 64 <= shape; i += 64)
            {
                __m512i a = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(&A[i]));
                alignas(64) signed char temp_in[64]; // Use signed char for potential -1
                alignas(64) data_type temp_out[64];
                _mm512_storeu_si512(reinterpret_cast<__m512i *>(temp_in), a);

#pragma omp parallel for simd
                for (int j = 0; j < 64; ++j)
                {
                    // Cast to float, apply acos, round, cast back
                    float val_f = static_cast<float>(temp_in[j]);
                    temp_out[j] = static_cast<data_type>(std::round(std::acos(val_f)));
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
                    float val_f = static_cast<float>(temp_in[j]);
                    temp_out[j] = static_cast<data_type>(std::round(std::acos(val_f)));
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
                    // Use double for potentially better intermediate precision
                    double val_d = static_cast<double>(temp_in[j]);
                    temp_out[j] = static_cast<data_type>(std::round(std::acos(val_d)));
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
                    double val_d = static_cast<double>(temp_in[j]);
                    temp_out[j] = static_cast<data_type>(std::round(std::acos(val_d)));
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
            A[j] = std::acos(A[j]);
        }
        else if constexpr (std::is_integral_v<data_type>)
        {
            // Cast, apply acos, round back
            if constexpr (sizeof(data_type) <= 4)
            {
                float val_f = static_cast<float>(A[j]);
                A[j] = static_cast<data_type>(std::round(std::acos(val_f)));
            }
            else
            {
                double val_d = static_cast<double>(A[j]);
                A[j] = static_cast<data_type>(std::round(std::acos(val_d)));
            }
        }
    }
}

// ...existing code...

template <typename data_type>
void simd_arctan(data_type *A, size_t shape)
{
    size_t i = 0;

    if constexpr (std::is_same_v<data_type, float>)
    {
        for (; i + 16 <= shape; i += 16)
        {
            __m512 a = _mm512_loadu_ps(&A[i]);

            // No direct arctan AVX-512 instruction, use temporary arrays
            alignas(64) float temp_in[16], temp_out[16];
            _mm512_storeu_ps(temp_in, a);

// Apply std::atan element-wise using OpenMP SIMD
#pragma omp parallel for simd
            for (int j = 0; j < 16; ++j)
            {
                temp_out[j] = std::atan(temp_in[j]);
            }

            // Load results back to SIMD register and store
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

// Apply std::atan element-wise using OpenMP SIMD
#pragma omp parallel for simd
            for (int j = 0; j < 8; ++j)
            {
                temp_out[j] = std::atan(temp_in[j]);
            }

            __m512d result = _mm512_loadu_pd(temp_out);
            _mm512_storeu_pd(&A[i], result);
        }
    }
    else if constexpr (std::is_integral_v<data_type>)
    {
        // Arctan for integers: cast to float/double, apply atan, round back.
        if constexpr (sizeof(data_type) == 1) // 8-bit integers
        {
            for (; i + 64 <= shape; i += 64)
            {
                __m512i a = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(&A[i]));
                alignas(64) signed char temp_in[64]; // Use signed char for range
                alignas(64) data_type temp_out[64];
                _mm512_storeu_si512(reinterpret_cast<__m512i *>(temp_in), a);

#pragma omp parallel for simd
                for (int j = 0; j < 64; ++j)
                {
                    // Cast to float, apply atan, round, cast back
                    float val_f = static_cast<float>(temp_in[j]);
                    temp_out[j] = static_cast<data_type>(std::round(std::atan(val_f)));
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
                    float val_f = static_cast<float>(temp_in[j]);
                    temp_out[j] = static_cast<data_type>(std::round(std::atan(val_f)));
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
                    // Use double for potentially better intermediate precision
                    double val_d = static_cast<double>(temp_in[j]);
                    temp_out[j] = static_cast<data_type>(std::round(std::atan(val_d)));
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
                    double val_d = static_cast<double>(temp_in[j]);
                    temp_out[j] = static_cast<data_type>(std::round(std::atan(val_d)));
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
            A[j] = std::atan(A[j]);
        }
        else if constexpr (std::is_integral_v<data_type>)
        {
            // Cast, apply atan, round back
            if constexpr (sizeof(data_type) <= 4)
            {
                float val_f = static_cast<float>(A[j]);
                A[j] = static_cast<data_type>(std::round(std::atan(val_f)));
            }
            else
            {
                double val_d = static_cast<double>(A[j]);
                A[j] = static_cast<data_type>(std::round(std::atan(val_d)));
            }
        }
    }
}

// ...existing code...

template <typename data_type>
void simd_cotan(data_type *A, size_t shape)
{
    size_t i = 0;

    if constexpr (std::is_floating_point_v<data_type>)
    {
        // Define constants for 1.0f and 1.0
        const data_type one = static_cast<data_type>(1.0);

        if constexpr (std::is_same_v<data_type, float>)
        {
            const __m512 ones = _mm512_set1_ps(one);
            for (; i + 16 <= shape; i += 16)
            {
                __m512 a = _mm512_loadu_ps(&A[i]);

                // Calculate tangent first using temporary arrays
                alignas(64) float temp_in[16], temp_tan[16];
                _mm512_storeu_ps(temp_in, a);

#pragma omp parallel for simd
                for (int j = 0; j < 16; ++j)
                {
                    temp_tan[j] = std::tan(temp_in[j]);
                }

                // Load tangent results and calculate cotangent (1 / tan)
                __m512 tan_vec = _mm512_loadu_ps(temp_tan);
                __m512 result = _mm512_div_ps(ones, tan_vec); // cot(x) = 1 / tan(x)
                _mm512_storeu_ps(&A[i], result);
            }
        }
        else if constexpr (std::is_same_v<data_type, double>)
        {
            const __m512d ones = _mm512_set1_pd(one);
            for (; i + 8 <= shape; i += 8)
            {
                __m512d a = _mm512_loadu_pd(&A[i]);

                // Calculate tangent first using temporary arrays
                alignas(64) double temp_in[8], temp_tan[8];
                _mm512_storeu_pd(temp_in, a);

#pragma omp parallel for simd
                for (int j = 0; j < 8; ++j)
                {
                    temp_tan[j] = std::tan(temp_in[j]);
                }

                // Load tangent results and calculate cotangent (1 / tan)
                __m512d tan_vec = _mm512_loadu_pd(temp_tan);
                __m512d result = _mm512_div_pd(ones, tan_vec); // cot(x) = 1 / tan(x)
                _mm512_storeu_pd(&A[i], result);
            }
        }
    }
    else if constexpr (std::is_integral_v<data_type>)
    {
        // Cotangent for integers: cast to float/double, apply 1/tan, round back.
        // Note: Result might be inaccurate or lead to large values/infinity near multiples of pi.
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
                    float val_f = static_cast<float>(temp_in[j]);
                    float tan_val = std::tan(val_f);
                    // Avoid division by zero, result in 0 for integer cotangent in this case?
                    // Or let it potentially become large/INF and round? Rounding INF is problematic.
                    // Let's round the result, acknowledging potential issues.
                    temp_out[j] = (tan_val == 0.0f) ? static_cast<data_type>(0) // Or some large value?
                                                    : static_cast<data_type>(std::round(1.0f / tan_val));
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
                    float val_f = static_cast<float>(temp_in[j]);
                    float tan_val = std::tan(val_f);
                    temp_out[j] = (tan_val == 0.0f) ? static_cast<data_type>(0)
                                                    : static_cast<data_type>(std::round(1.0f / tan_val));
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
                    double val_d = static_cast<double>(temp_in[j]);
                    double tan_val = std::tan(val_d);
                    temp_out[j] = (tan_val == 0.0) ? static_cast<data_type>(0)
                                                   : static_cast<data_type>(std::round(1.0 / tan_val));
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
                    double val_d = static_cast<double>(temp_in[j]);
                    double tan_val = std::tan(val_d);
                    temp_out[j] = (tan_val == 0.0) ? static_cast<data_type>(0)
                                                   : static_cast<data_type>(std::round(1.0 / tan_val));
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
            data_type tan_val = std::tan(A[j]);
            // Handle potential division by zero for floating point (results in INF)
            A[j] = static_cast<data_type>(1.0) / tan_val;
        }
        else if constexpr (std::is_integral_v<data_type>)
        {
            // Cast, apply 1/tan, round back
            if constexpr (sizeof(data_type) <= 4)
            {
                float val_f = static_cast<float>(A[j]);
                float tan_val = std::tan(val_f);
                A[j] = (tan_val == 0.0f) ? static_cast<data_type>(0)
                                         : static_cast<data_type>(std::round(1.0f / tan_val));
            }
            else
            {
                double val_d = static_cast<double>(A[j]);
                double tan_val = std::tan(val_d);
                A[j] = (tan_val == 0.0) ? static_cast<data_type>(0)
                                        : static_cast<data_type>(std::round(1.0 / tan_val));
            }
        }
    }
}

// ...existing code...

template <typename data_type>
void simd_sinh(data_type *A, size_t shape)
{
    size_t i = 0;

    if constexpr (std::is_same_v<data_type, float>)
    {
        for (; i + 16 <= shape; i += 16)
        {
            __m512 a = _mm512_loadu_ps(&A[i]);

            // No direct sinh AVX-512 instruction, use temporary arrays
            alignas(64) float temp_in[16], temp_out[16];
            _mm512_storeu_ps(temp_in, a);

// Apply std::sinh element-wise using OpenMP SIMD
#pragma omp parallel for simd
            for (int j = 0; j < 16; ++j)
            {
                temp_out[j] = std::sinh(temp_in[j]);
            }

            // Load results back to SIMD register and store
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

// Apply std::sinh element-wise using OpenMP SIMD
#pragma omp parallel for simd
            for (int j = 0; j < 8; ++j)
            {
                temp_out[j] = std::sinh(temp_in[j]);
            }

            __m512d result = _mm512_loadu_pd(temp_out);
            _mm512_storeu_pd(&A[i], result);
        }
    }
    else if constexpr (std::is_integral_v<data_type>)
    {
        // Sinh for integers: cast to float/double, apply sinh, round back.
        if constexpr (sizeof(data_type) == 1) // 8-bit integers
        {
            for (; i + 64 <= shape; i += 64)
            {
                __m512i a = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(&A[i]));
                alignas(64) signed char temp_in[64]; // Use signed char for range
                alignas(64) data_type temp_out[64];
                _mm512_storeu_si512(reinterpret_cast<__m512i *>(temp_in), a);

#pragma omp parallel for simd
                for (int j = 0; j < 64; ++j)
                {
                    // Cast to float, apply sinh, round, cast back
                    float val_f = static_cast<float>(temp_in[j]);
                    temp_out[j] = static_cast<data_type>(std::round(std::sinh(val_f)));
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
                    float val_f = static_cast<float>(temp_in[j]);
                    temp_out[j] = static_cast<data_type>(std::round(std::sinh(val_f)));
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
                    // Use double for potentially better intermediate precision
                    double val_d = static_cast<double>(temp_in[j]);
                    temp_out[j] = static_cast<data_type>(std::round(std::sinh(val_d)));
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
                    double val_d = static_cast<double>(temp_in[j]);
                    temp_out[j] = static_cast<data_type>(std::round(std::sinh(val_d)));
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
            A[j] = std::sinh(A[j]);
        }
        else if constexpr (std::is_integral_v<data_type>)
        {
            // Cast, apply sinh, round back
            if constexpr (sizeof(data_type) <= 4)
            {
                float val_f = static_cast<float>(A[j]);
                A[j] = static_cast<data_type>(std::round(std::sinh(val_f)));
            }
            else
            {
                double val_d = static_cast<double>(A[j]);
                A[j] = static_cast<data_type>(std::round(std::sinh(val_d)));
            }
        }
    }
}
template <typename data_type>
void simd_cosh(data_type *A, size_t shape)
{
    size_t i = 0;

    if constexpr (std::is_same_v<data_type, float>)
    {
        for (; i + 16 <= shape; i += 16)
        {
            __m512 a = _mm512_loadu_ps(&A[i]);

            // No direct cosh AVX-512 instruction, use temporary arrays
            alignas(64) float temp_in[16], temp_out[16];
            _mm512_storeu_ps(temp_in, a);

// Apply std::cosh element-wise using OpenMP SIMD
#pragma omp parallel for simd
            for (int j = 0; j < 16; ++j)
            {
                temp_out[j] = std::cosh(temp_in[j]);
            }

            // Load results back to SIMD register and store
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

// Apply std::cosh element-wise using OpenMP SIMD
#pragma omp parallel for simd
            for (int j = 0; j < 8; ++j)
            {
                temp_out[j] = std::cosh(temp_in[j]);
            }

            __m512d result = _mm512_loadu_pd(temp_out);
            _mm512_storeu_pd(&A[i], result);
        }
    }
    else if constexpr (std::is_integral_v<data_type>)
    {
        // Cosh for integers: cast to float/double, apply cosh, round back.
        if constexpr (sizeof(data_type) == 1) // 8-bit integers
        {
            for (; i + 64 <= shape; i += 64)
            {
                __m512i a = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(&A[i]));
                alignas(64) signed char temp_in[64]; // Use signed char for range
                alignas(64) data_type temp_out[64];
                _mm512_storeu_si512(reinterpret_cast<__m512i *>(temp_in), a);

#pragma omp parallel for simd
                for (int j = 0; j < 64; ++j)
                {
                    // Cast to float, apply cosh, round, cast back
                    float val_f = static_cast<float>(temp_in[j]);
                    temp_out[j] = static_cast<data_type>(std::round(std::cosh(val_f)));
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
                    float val_f = static_cast<float>(temp_in[j]);
                    temp_out[j] = static_cast<data_type>(std::round(std::cosh(val_f)));
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
                    // Use double for potentially better intermediate precision
                    double val_d = static_cast<double>(temp_in[j]);
                    temp_out[j] = static_cast<data_type>(std::round(std::cosh(val_d)));
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
                    double val_d = static_cast<double>(temp_in[j]);
                    temp_out[j] = static_cast<data_type>(std::round(std::cosh(val_d)));
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
            A[j] = std::cosh(A[j]);
        }
        else if constexpr (std::is_integral_v<data_type>)
        {
            // Cast, apply cosh, round back
            if constexpr (sizeof(data_type) <= 4)
            {
                float val_f = static_cast<float>(A[j]);
                A[j] = static_cast<data_type>(std::round(std::cosh(val_f)));
            }
            else
            {
                double val_d = static_cast<double>(A[j]);
                A[j] = static_cast<data_type>(std::round(std::cosh(val_d)));
            }
        }
    }
}

// ...existing code...

template <typename data_type>
void simd_tanh(data_type *A, size_t shape)
{
    size_t i = 0;

    if constexpr (std::is_same_v<data_type, float>)
    {
        for (; i + 16 <= shape; i += 16)
        {
            __m512 a = _mm512_loadu_ps(&A[i]);

            // No direct tanh AVX-512 instruction, use temporary arrays
            alignas(64) float temp_in[16], temp_out[16];
            _mm512_storeu_ps(temp_in, a);

// Apply std::tanh element-wise using OpenMP SIMD
#pragma omp parallel for simd
            for (int j = 0; j < 16; ++j)
            {
                temp_out[j] = std::tanh(temp_in[j]);
            }

            // Load results back to SIMD register and store
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

// Apply std::tanh element-wise using OpenMP SIMD
#pragma omp parallel for simd
            for (int j = 0; j < 8; ++j)
            {
                temp_out[j] = std::tanh(temp_in[j]);
            }

            __m512d result = _mm512_loadu_pd(temp_out);
            _mm512_storeu_pd(&A[i], result);
        }
    }
    else if constexpr (std::is_integral_v<data_type>)
    {
        // Tanh for integers: cast to float/double, apply tanh, round back.
        if constexpr (sizeof(data_type) == 1) // 8-bit integers
        {
            for (; i + 64 <= shape; i += 64)
            {
                __m512i a = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(&A[i]));
                alignas(64) signed char temp_in[64]; // Use signed char for range
                alignas(64) data_type temp_out[64];
                _mm512_storeu_si512(reinterpret_cast<__m512i *>(temp_in), a);

#pragma omp parallel for simd
                for (int j = 0; j < 64; ++j)
                {
                    // Cast to float, apply tanh, round, cast back
                    float val_f = static_cast<float>(temp_in[j]);
                    temp_out[j] = static_cast<data_type>(std::round(std::tanh(val_f)));
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
                    float val_f = static_cast<float>(temp_in[j]);
                    temp_out[j] = static_cast<data_type>(std::round(std::tanh(val_f)));
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
                    // Use double for potentially better intermediate precision
                    double val_d = static_cast<double>(temp_in[j]);
                    temp_out[j] = static_cast<data_type>(std::round(std::tanh(val_d)));
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
                    double val_d = static_cast<double>(temp_in[j]);
                    temp_out[j] = static_cast<data_type>(std::round(std::tanh(val_d)));
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
            A[j] = std::tanh(A[j]);
        }
        else if constexpr (std::is_integral_v<data_type>)
        {
            // Cast, apply tanh, round back
            if constexpr (sizeof(data_type) <= 4)
            {
                float val_f = static_cast<float>(A[j]);
                A[j] = static_cast<data_type>(std::round(std::tanh(val_f)));
            }
            else
            {
                double val_d = static_cast<double>(A[j]);
                A[j] = static_cast<data_type>(std::round(std::tanh(val_d)));
            }
        }
    }
}

template <typename data_type>
void simd_arcsinh(data_type *A, size_t shape)
{
    size_t i = 0;
    if constexpr (std::is_floating_point_v<data_type>)
    {
        for (; i + 16 <= shape; i += 16)
        {
            __m512 a = _m512_loadu_ps(&A[i]);
            alignas(64) float temp_in[16], temp_out[16];
            _mm512_storeu_ps(temp_in, a);

#pragma omp parallel for simd
            for (int j = 0; j < 16; j++)
            {
                temp_out[j] = std::asinh(temp_in[j]);
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
            alignas(64) double temp_in[8], temp_out[8];
            _mm512_storeu_pd(temp_in, a);
#pragma omp parallel for simd
            for (int j = 0; j < 8; j++)
            {
                temp_out[j] = std::asinh(temp_in[j]);
            }
            __m512d result = _mm512_loadu_pd(temp_out);
            _mm512_storeu_pd(&A[i], result);
        }
    }
    else if constexpr (std::is_integral_v<data_type>)
    {
        // Arcsinh for integers: cast to float/double, apply asinh, round back.
        if constexpr (sizeof(data_type) == 1) // 8-bit integers
        {
            for (; i + 64 <= shape; i += 64)
            {
                __m512i a = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(&A[i]));
                alignas(64) signed char temp_in[64]; // Use signed char for range
                alignas(64) data_type temp_out[64];
                _mm512_storeu_si512(reinterpret_cast<__m512i *>(temp_in), a);

#pragma omp parallel for simd
                for (int j = 0; j < 64; ++j)
                {
                    // Cast to float, apply asinh, round, cast back
                    float val_f = static_cast<float>(temp_in[j]);
                    temp_out[j] = static_cast<data_type>(std::round(std::asinh(val_f)));
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
                    float val_f = static_cast<float>(temp_in[j]);
                    temp_out[j] = static_cast<data_type>(std::round(std::asinh(val_f)));
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
                    // Use double for potentially better intermediate precision
                    double val_d = static_cast<double>(temp_in[j]);
                    temp_out[j] = static_cast<data_type>(std::round(std::asinh(val_d)));
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
                    double val_d = static_cast<double>(temp_in[j]);
                    temp_out[j] = static_cast<data_type>(std::round(std::asinh(val_d)));
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
            A[j] = std::asinh(A[j]);
        }
        else if constexpr (std::is_integral_v<data_type>)
        {
            // Cast, apply asinh, round back
            if constexpr (sizeof(data_type) <= 4)
            {
                float val_f = static_cast<float>(A[j]);
                A[j] = static_cast<data_type>(std::round(std::asinh(val_f)));
            }
            else
            {
                double val_d = static_cast<double>(A[j]);
                A[j] = static_cast<data_type>(std::round(std::asinh(val_d)));
            }
        }
    }
}

// ...existing code...

template <typename data_type>
void simd_arccosh(data_type *A, size_t shape) // arccosh function implementation
{
    size_t i = 0;

    if constexpr (std::is_same_v<data_type, float>)
    {
        for (; i + 16 <= shape; i += 16)
        {
            __m512 a = _mm512_loadu_ps(&A[i]);

            // No direct arccosh AVX-512 instruction, use temporary arrays
            alignas(64) float temp_in[16], temp_out[16];
            _mm512_storeu_ps(temp_in, a);

// Apply std::acosh element-wise using OpenMP SIMD
// Note: std::acosh domain is [1, +inf). Input < 1 results in NaN.
#pragma omp parallel for simd
            for (int j = 0; j < 16; ++j)
            {
                temp_out[j] = std::acosh(temp_in[j]);
            }

            // Load results back to SIMD register and store
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

// Apply std::acosh element-wise using OpenMP SIMD
// Note: std::acosh domain is [1, +inf). Input < 1 results in NaN.
#pragma omp parallel for simd
            for (int j = 0; j < 8; ++j)
            {
                temp_out[j] = std::acosh(temp_in[j]);
            }

            __m512d result = _mm512_loadu_pd(temp_out);
            _mm512_storeu_pd(&A[i], result);
        }
    }
    else if constexpr (std::is_integral_v<data_type>)
    {
        // Arccosh is typically not meaningful for integers < 1.
        // We'll implement it by casting to float/double, applying acosh,
        // and casting back, but results for inputs < 1 will be NaN/error,
        // and results for 0 will likely be NaN/error after cast.
        // Consider if this operation is truly needed for integer types.

        if constexpr (sizeof(data_type) == 1) // 8-bit integers
        {
            for (; i + 64 <= shape; i += 64)
            {
                __m512i a = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(&A[i]));
                alignas(64) unsigned char temp_in[64]; // Use unsigned char as domain is >= 1
                alignas(64) data_type temp_out[64];
                _mm512_storeu_si512(reinterpret_cast<__m512i *>(temp_in), a);

#pragma omp parallel for simd
                for (int j = 0; j < 64; ++j)
                {
                    // Cast to float, apply acosh, round, cast back
                    float val_f = static_cast<float>(temp_in[j]);
                    temp_out[j] = static_cast<data_type>(std::round(std::acosh(val_f)));
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
                alignas(64) unsigned short temp_in[32]; // Domain >= 1
                alignas(64) data_type temp_out[32];
                _mm512_storeu_si512(reinterpret_cast<__m512i *>(temp_in), a);

#pragma omp parallel for simd
                for (int j = 0; j < 32; ++j)
                {
                    float val_f = static_cast<float>(temp_in[j]);
                    temp_out[j] = static_cast<data_type>(std::round(std::acosh(val_f)));
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
                alignas(64) unsigned int temp_in[16]; // Domain >= 1
                alignas(64) data_type temp_out[16];
                _mm512_storeu_si512(reinterpret_cast<__m512i *>(temp_in), a);

#pragma omp parallel for simd
                for (int j = 0; j < 16; ++j)
                {
                    // Use double for potentially better intermediate precision
                    double val_d = static_cast<double>(temp_in[j]);
                    temp_out[j] = static_cast<data_type>(std::round(std::acosh(val_d)));
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
                alignas(64) unsigned long long temp_in[8]; // Domain >= 1
                alignas(64) data_type temp_out[8];
                _mm512_storeu_si512(reinterpret_cast<__m512i *>(temp_in), a);

#pragma omp parallel for simd
                for (int j = 0; j < 8; ++j)
                {
                    double val_d = static_cast<double>(temp_in[j]);
                    temp_out[j] = static_cast<data_type>(std::round(std::acosh(val_d)));
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
            A[j] = std::acosh(A[j]);
        }
        else if constexpr (std::is_integral_v<data_type>)
        {
            // Cast, apply acosh, round back
            if constexpr (sizeof(data_type) <= 4)
            {
                // Use unsigned intermediate type if original is unsigned
                using IntermediateInt = std::conditional_t<std::is_signed_v<data_type>, int, unsigned int>;
                float val_f = static_cast<float>(static_cast<IntermediateInt>(A[j]));
                A[j] = static_cast<data_type>(std::round(std::acosh(val_f)));
            }
            else
            {
                using IntermediateInt = std::conditional_t<std::is_signed_v<data_type>, long long, unsigned long long>;
                double val_d = static_cast<double>(static_cast<IntermediateInt>(A[j]));
                A[j] = static_cast<data_type>(std::round(std::acosh(val_d)));
            }
        }
    }
}

// ...existing code...

template <typename data_type>
void simd_arctanh(data_type *A, size_t shape) // arctanh function implementation
{
    size_t i = 0;

    if constexpr (std::is_same_v<data_type, float>)
    {
        for (; i + 16 <= shape; i += 16)
        {
            __m512 a = _mm512_loadu_ps(&A[i]);

            // No direct arctanh AVX-512 instruction, use temporary arrays
            alignas(64) float temp_in[16], temp_out[16];
            _mm512_storeu_ps(temp_in, a);

// Apply std::atanh element-wise using OpenMP SIMD
// Note: std::atanh domain is (-1, 1). Input outside this range results in NaN/Inf.
#pragma omp parallel for simd
            for (int j = 0; j < 16; ++j)
            {
                temp_out[j] = std::atanh(temp_in[j]);
            }

            // Load results back to SIMD register and store
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

// Apply std::atanh element-wise using OpenMP SIMD
// Note: std::atanh domain is (-1, 1). Input outside this range results in NaN/Inf.
#pragma omp parallel for simd
            for (int j = 0; j < 8; ++j)
            {
                temp_out[j] = std::atanh(temp_in[j]);
            }

            __m512d result = _mm512_loadu_pd(temp_out);
            _mm512_storeu_pd(&A[i], result);
        }
    }
    else if constexpr (std::is_integral_v<data_type>)
    {
        // Arctanh is typically not meaningful for integers, as the domain is (-1, 1).
        // Only 0 is valid. Other inputs will result in NaN/Inf.
        // We implement by casting, applying, rounding, but results are very limited.

        if constexpr (sizeof(data_type) == 1) // 8-bit integers
        {
            for (; i + 64 <= shape; i += 64)
            {
                __m512i a = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(&A[i]));
                alignas(64) signed char temp_in[64]; // Use signed char for potential -1 (though domain is (-1,1))
                alignas(64) data_type temp_out[64];
                _mm512_storeu_si512(reinterpret_cast<__m512i *>(temp_in), a);

#pragma omp parallel for simd
                for (int j = 0; j < 64; ++j)
                {
                    // Cast to float, apply atanh, round, cast back
                    float val_f = static_cast<float>(temp_in[j]);
                    temp_out[j] = static_cast<data_type>(std::round(std::atanh(val_f))); // Will be 0 for input 0, NaN/Inf otherwise
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
                    float val_f = static_cast<float>(temp_in[j]);
                    temp_out[j] = static_cast<data_type>(std::round(std::atanh(val_f)));
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
                    // Use double for potentially better intermediate precision
                    double val_d = static_cast<double>(temp_in[j]);
                    temp_out[j] = static_cast<data_type>(std::round(std::atanh(val_d)));
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
                    double val_d = static_cast<double>(temp_in[j]);
                    temp_out[j] = static_cast<data_type>(std::round(std::atanh(val_d)));
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
            A[j] = std::atanh(A[j]);
        }
        else if constexpr (std::is_integral_v<data_type>)
        {
            // Cast, apply atanh, round back
            if constexpr (sizeof(data_type) <= 4)
            {
                float val_f = static_cast<float>(A[j]);
                A[j] = static_cast<data_type>(std::round(std::atanh(val_f)));
            }
            else
            {
                double val_d = static_cast<double>(A[j]);
                A[j] = static_cast<data_type>(std::round(std::atanh(val_d)));
            }
        }
    }
}

// ...existing code...

template <typename data_type>
void simd_sec(data_type *A, size_t shape) // sec function implementation
{
    size_t i = 0;

    if constexpr (std::is_floating_point_v<data_type>)
    {
        // Define constants for 1.0f and 1.0
        const data_type one = static_cast<data_type>(1.0);

        if constexpr (std::is_same_v<data_type, float>)
        {
            const __m512 ones = _mm512_set1_ps(one);
            for (; i + 16 <= shape; i += 16)
            {
                __m512 a = _mm512_loadu_ps(&A[i]);

                // Calculate cosine first using temporary arrays
                alignas(64) float temp_in[16], temp_cos[16];
                _mm512_storeu_ps(temp_in, a);

#pragma omp parallel for simd
                for (int j = 0; j < 16; ++j)
                {
                    temp_cos[j] = std::cos(temp_in[j]);
                }

                // Load cosine results and calculate secant (1 / cos)
                __m512 cos_vec = _mm512_loadu_ps(temp_cos);
                __m512 result = _mm512_div_ps(ones, cos_vec); // sec(x) = 1 / cos(x)
                _mm512_storeu_ps(&A[i], result);
            }
        }
        else if constexpr (std::is_same_v<data_type, double>)
        {
            const __m512d ones = _mm512_set1_pd(one);
            for (; i + 8 <= shape; i += 8)
            {
                __m512d a = _mm512_loadu_pd(&A[i]);

                // Calculate cosine first using temporary arrays
                alignas(64) double temp_in[8], temp_cos[8];
                _mm512_storeu_pd(temp_in, a);

#pragma omp parallel for simd
                for (int j = 0; j < 8; ++j)
                {
                    temp_cos[j] = std::cos(temp_in[j]);
                }

                // Load cosine results and calculate secant (1 / cos)
                __m512d cos_vec = _mm512_loadu_pd(temp_cos);
                __m512d result = _mm512_div_pd(ones, cos_vec); // sec(x) = 1 / cos(x)
                _mm512_storeu_pd(&A[i], result);
            }
        }
    }
    else if constexpr (std::is_integral_v<data_type>)
    {
        // Secant for integers: cast to float/double, apply 1/cos, round back.
        // Note: Result might be inaccurate or lead to large values/infinity near pi/2 + k*pi.
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
                    float val_f = static_cast<float>(temp_in[j]);
                    float cos_val = std::cos(val_f);
                    // Avoid division by zero, result in 0 for integer secant in this case?
                    temp_out[j] = (cos_val == 0.0f) ? static_cast<data_type>(0)
                                                    : static_cast<data_type>(std::round(1.0f / cos_val));
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
                    float val_f = static_cast<float>(temp_in[j]);
                    float cos_val = std::cos(val_f);
                    temp_out[j] = (cos_val == 0.0f) ? static_cast<data_type>(0)
                                                    : static_cast<data_type>(std::round(1.0f / cos_val));
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
                    double val_d = static_cast<double>(temp_in[j]);
                    double cos_val = std::cos(val_d);
                    temp_out[j] = (cos_val == 0.0) ? static_cast<data_type>(0)
                                                   : static_cast<data_type>(std::round(1.0 / cos_val));
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
                    double val_d = static_cast<double>(temp_in[j]);
                    double cos_val = std::cos(val_d);
                    temp_out[j] = (cos_val == 0.0) ? static_cast<data_type>(0)
                                                   : static_cast<data_type>(std::round(1.0 / cos_val));
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
            data_type cos_val = std::cos(A[j]);
            // Handle potential division by zero for floating point (results in INF)
            A[j] = static_cast<data_type>(1.0) / cos_val;
        }
        else if constexpr (std::is_integral_v<data_type>)
        {
            // Cast, apply 1/cos, round back
            if constexpr (sizeof(data_type) <= 4)
            {
                float val_f = static_cast<float>(A[j]);
                float cos_val = std::cos(val_f);
                A[j] = (cos_val == 0.0f) ? static_cast<data_type>(0)
                                         : static_cast<data_type>(std::round(1.0f / cos_val));
            }
            else
            {
                double val_d = static_cast<double>(A[j]);
                double cos_val = std::cos(val_d);
                A[j] = (cos_val == 0.0) ? static_cast<data_type>(0)
                                        : static_cast<data_type>(std::round(1.0 / cos_val));
            }
        }
    }
}

// ...existing code...

template <typename data_type>
void simd_cosec(data_type *A, size_t shape) // cosec function implementation
{
    size_t i = 0;

    if constexpr (std::is_floating_point_v<data_type>)
    {
        // Define constants for 1.0f and 1.0
        const data_type one = static_cast<data_type>(1.0);

        if constexpr (std::is_same_v<data_type, float>)
        {
            const __m512 ones = _mm512_set1_ps(one);
            for (; i + 16 <= shape; i += 16)
            {
                __m512 a = _mm512_loadu_ps(&A[i]);

                // Calculate sine first using temporary arrays
                alignas(64) float temp_in[16], temp_sin[16];
                _mm512_storeu_ps(temp_in, a);

#pragma omp parallel for simd
                for (int j = 0; j < 16; ++j)
                {
                    temp_sin[j] = std::sin(temp_in[j]);
                }

                // Load sine results and calculate cosecant (1 / sin)
                __m512 sin_vec = _mm512_loadu_ps(temp_sin);
                __m512 result = _mm512_div_ps(ones, sin_vec); // cosec(x) = 1 / sin(x)
                _mm512_storeu_ps(&A[i], result);
            }
        }
        else if constexpr (std::is_same_v<data_type, double>)
        {
            const __m512d ones = _mm512_set1_pd(one);
            for (; i + 8 <= shape; i += 8)
            {
                __m512d a = _mm512_loadu_pd(&A[i]);

                // Calculate sine first using temporary arrays
                alignas(64) double temp_in[8], temp_sin[8];
                _mm512_storeu_pd(temp_in, a);

#pragma omp parallel for simd
                for (int j = 0; j < 8; ++j)
                {
                    temp_sin[j] = std::sin(temp_in[j]);
                }

                // Load sine results and calculate cosecant (1 / sin)
                __m512d sin_vec = _mm512_loadu_pd(temp_sin);
                __m512d result = _mm512_div_pd(ones, sin_vec); // cosec(x) = 1 / sin(x)
                _mm512_storeu_pd(&A[i], result);
            }
        }
    }
    else if constexpr (std::is_integral_v<data_type>)
    {
        // Cosecant for integers: cast to float/double, apply 1/sin, round back.
        // Note: Result might be inaccurate or lead to large values/infinity near k*pi.
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
                    float val_f = static_cast<float>(temp_in[j]);
                    float sin_val = std::sin(val_f);
                    // Avoid division by zero
                    temp_out[j] = (sin_val == 0.0f) ? static_cast<data_type>(0)
                                                    : static_cast<data_type>(std::round(1.0f / sin_val));
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
                    float val_f = static_cast<float>(temp_in[j]);
                    float sin_val = std::sin(val_f);
                    temp_out[j] = (sin_val == 0.0f) ? static_cast<data_type>(0)
                                                    : static_cast<data_type>(std::round(1.0f / sin_val));
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
                    double val_d = static_cast<double>(temp_in[j]);
                    double sin_val = std::sin(val_d);
                    temp_out[j] = (sin_val == 0.0) ? static_cast<data_type>(0)
                                                   : static_cast<data_type>(std::round(1.0 / sin_val));
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
                    double val_d = static_cast<double>(temp_in[j]);
                    double sin_val = std::sin(val_d);
                    temp_out[j] = (sin_val == 0.0) ? static_cast<data_type>(0)
                                                   : static_cast<data_type>(std::round(1.0 / sin_val));
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
            data_type sin_val = std::sin(A[j]);
            // Handle potential division by zero for floating point (results in INF)
            A[j] = static_cast<data_type>(1.0) / sin_val;
        }
        else if constexpr (std::is_integral_v<data_type>)
        {
            // Cast, apply 1/sin, round back
            if constexpr (sizeof(data_type) <= 4)
            {
                float val_f = static_cast<float>(A[j]);
                float sin_val = std::sin(val_f);
                A[j] = (sin_val == 0.0f) ? static_cast<data_type>(0)
                                         : static_cast<data_type>(std::round(1.0f / sin_val));
            }
            else
            {
                double val_d = static_cast<double>(A[j]);
                double sin_val = std::sin(val_d);
                A[j] = (sin_val == 0.0) ? static_cast<data_type>(0)
                                        : static_cast<data_type>(std::round(1.0 / sin_val));
            }
        }
    }
}

// ...existing code...

template <typename data_type>
void simd_arcsec(data_type *A, size_t shape) // arcsec function implementation
{
    size_t i = 0;

    if constexpr (std::is_floating_point_v<data_type>)
    {
        // Define constants for 1.0f and 1.0
        const data_type one = static_cast<data_type>(1.0);

        if constexpr (std::is_same_v<data_type, float>)
        {
            const __m512 ones = _mm512_set1_ps(one);
            for (; i + 16 <= shape; i += 16)
            {
                __m512 a = _mm512_loadu_ps(&A[i]);

                // Calculate 1/x
                __m512 inv_a = _mm512_div_ps(ones, a);

                // Calculate arccos(1/x) using temporary arrays
                alignas(64) float temp_in[16], temp_out[16];
                _mm512_storeu_ps(temp_in, inv_a);

// Apply std::acos element-wise using OpenMP SIMD
// Note: Domain for arcsec(x) is |x| >= 1, so 1/x is in [-1, 1] (excluding 0).
// std::acos domain is [-1, 1]. Input outside this range results in NaN.
#pragma omp parallel for simd
                for (int j = 0; j < 16; ++j)
                {
                    temp_out[j] = std::acos(temp_in[j]);
                }

                // Load results back to SIMD register and store
                __m512 result = _mm512_loadu_ps(temp_out);
                _mm512_storeu_ps(&A[i], result);
            }
        }
        else if constexpr (std::is_same_v<data_type, double>)
        {
            const __m512d ones = _mm512_set1_pd(one);
            for (; i + 8 <= shape; i += 8)
            {
                __m512d a = _mm512_loadu_pd(&A[i]);

                // Calculate 1/x
                __m512d inv_a = _mm512_div_pd(ones, a);

                // Calculate arccos(1/x) using temporary arrays
                alignas(64) double temp_in[8], temp_out[8];
                _mm512_storeu_pd(temp_in, inv_a);

// Apply std::acos element-wise using OpenMP SIMD
#pragma omp parallel for simd
                for (int j = 0; j < 8; ++j)
                {
                    temp_out[j] = std::acos(temp_in[j]);
                }

                // Load results back to SIMD register and store
                __m512d result = _mm512_loadu_pd(temp_out);
                _mm512_storeu_pd(&A[i], result);
            }
        }
    }
    else if constexpr (std::is_integral_v<data_type>)
    {
        // Arcsec for integers: cast to float/double, apply acos(1/x), round back.
        // Domain |x| >= 1. Integers 0, -1, 1 are edge cases.
        // x=0 -> division by zero (INF), acos(INF) -> NaN.
        // |x|<1 -> 1/|x| > 1, acos(>1) -> NaN.
        // Only meaningful for |x| >= 1.

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
                    float val_f = static_cast<float>(temp_in[j]);
                    float inv_val = (val_f == 0.0f) ? std::numeric_limits<float>::quiet_NaN() // Avoid division by zero explicitly
                                                    : 1.0f / val_f;
                    temp_out[j] = static_cast<data_type>(std::round(std::acos(inv_val)));
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
                    float val_f = static_cast<float>(temp_in[j]);
                    float inv_val = (val_f == 0.0f) ? std::numeric_limits<float>::quiet_NaN()
                                                    : 1.0f / val_f;
                    temp_out[j] = static_cast<data_type>(std::round(std::acos(inv_val)));
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
                    double val_d = static_cast<double>(temp_in[j]);
                    double inv_val = (val_d == 0.0) ? std::numeric_limits<double>::quiet_NaN()
                                                    : 1.0 / val_d;
                    temp_out[j] = static_cast<data_type>(std::round(std::acos(inv_val)));
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
                    double val_d = static_cast<double>(temp_in[j]);
                    double inv_val = (val_d == 0.0) ? std::numeric_limits<double>::quiet_NaN()
                                                    : 1.0 / val_d;
                    temp_out[j] = static_cast<data_type>(std::round(std::acos(inv_val)));
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
            // Handle potential division by zero for floating point (results in INF -> acos(INF) -> NaN)
            data_type inv_val = static_cast<data_type>(1.0) / A[j];
            A[j] = std::acos(inv_val);
        }
        else if constexpr (std::is_integral_v<data_type>)
        {
            // Cast, apply acos(1/x), round back
            if constexpr (sizeof(data_type) <= 4)
            {
                float val_f = static_cast<float>(A[j]);
                float inv_val = (val_f == 0.0f) ? std::numeric_limits<float>::quiet_NaN()
                                                : 1.0f / val_f;
                A[j] = static_cast<data_type>(std::round(std::acos(inv_val)));
            }
            else
            {
                double val_d = static_cast<double>(A[j]);
                double inv_val = (val_d == 0.0) ? std::numeric_limits<double>::quiet_NaN()
                                                : 1.0 / val_d;
                A[j] = static_cast<data_type>(std::round(std::acos(inv_val)));
            }
        }
    }
}

// ...existing code...

template <typename data_type>
void simd_arccosec(data_type *A, size_t shape) // arccosec function implementation
{
    size_t i = 0;

    if constexpr (std::is_floating_point_v<data_type>)
    {
        // Define constants for 1.0f and 1.0
        const data_type one = static_cast<data_type>(1.0);

        if constexpr (std::is_same_v<data_type, float>)
        {
            const __m512 ones = _mm512_set1_ps(one);
            for (; i + 16 <= shape; i += 16)
            {
                __m512 a = _mm512_loadu_ps(&A[i]);

                // Calculate 1/x
                __m512 inv_a = _mm512_div_ps(ones, a);

                // Calculate arcsin(1/x) using temporary arrays
                alignas(64) float temp_in[16], temp_out[16];
                _mm512_storeu_ps(temp_in, inv_a);

// Apply std::asin element-wise using OpenMP SIMD
// Note: Domain for arccsc(x) is |x| >= 1, so 1/x is in [-1, 1] (excluding 0).
// std::asin domain is [-1, 1]. Input outside this range results in NaN.
#pragma omp parallel for simd
                for (int j = 0; j < 16; ++j)
                {
                    temp_out[j] = std::asin(temp_in[j]);
                }

                // Load results back to SIMD register and store
                __m512 result = _mm512_loadu_ps(temp_out);
                _mm512_storeu_ps(&A[i], result);
            }
        }
        else if constexpr (std::is_same_v<data_type, double>)
        {
            const __m512d ones = _mm512_set1_pd(one);
            for (; i + 8 <= shape; i += 8)
            {
                __m512d a = _mm512_loadu_pd(&A[i]);

                // Calculate 1/x
                __m512d inv_a = _mm512_div_pd(ones, a);

                // Calculate arcsin(1/x) using temporary arrays
                alignas(64) double temp_in[8], temp_out[8];
                _mm512_storeu_pd(temp_in, inv_a);

// Apply std::asin element-wise using OpenMP SIMD
#pragma omp parallel for simd
                for (int j = 0; j < 8; ++j)
                {
                    temp_out[j] = std::asin(temp_in[j]);
                }

                // Load results back to SIMD register and store
                __m512d result = _mm512_loadu_pd(temp_out);
                _mm512_storeu_pd(&A[i], result);
            }
        }
    }
    else if constexpr (std::is_integral_v<data_type>)
    {
        // Arccosec for integers: cast to float/double, apply asin(1/x), round back.
        // Domain |x| >= 1. Integers 0, -1, 1 are edge cases.
        // x=0 -> division by zero (INF), asin(INF) -> NaN.
        // |x|<1 -> 1/|x| > 1, asin(>1) -> NaN.
        // Only meaningful for |x| >= 1.

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
                    float val_f = static_cast<float>(temp_in[j]);
                    float inv_val = (val_f == 0.0f) ? std::numeric_limits<float>::quiet_NaN() // Avoid division by zero explicitly
                                                    : 1.0f / val_f;
                    temp_out[j] = static_cast<data_type>(std::round(std::asin(inv_val)));
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
                    float val_f = static_cast<float>(temp_in[j]);
                    float inv_val = (val_f == 0.0f) ? std::numeric_limits<float>::quiet_NaN()
                                                    : 1.0f / val_f;
                    temp_out[j] = static_cast<data_type>(std::round(std::asin(inv_val)));
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
                    double val_d = static_cast<double>(temp_in[j]);
                    double inv_val = (val_d == 0.0) ? std::numeric_limits<double>::quiet_NaN()
                                                    : 1.0 / val_d;
                    temp_out[j] = static_cast<data_type>(std::round(std::asin(inv_val)));
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
                    double val_d = static_cast<double>(temp_in[j]);
                    double inv_val = (val_d == 0.0) ? std::numeric_limits<double>::quiet_NaN()
                                                    : 1.0 / val_d;
                    temp_out[j] = static_cast<data_type>(std::round(std::asin(inv_val)));
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
            // Handle potential division by zero for floating point (results in INF -> asin(INF) -> NaN)
            data_type inv_val = static_cast<data_type>(1.0) / A[j];
            A[j] = std::asin(inv_val);
        }
        else if constexpr (std::is_integral_v<data_type>)
        {
            // Cast, apply asin(1/x), round back
            if constexpr (sizeof(data_type) <= 4)
            {
                float val_f = static_cast<float>(A[j]);
                float inv_val = (val_f == 0.0f) ? std::numeric_limits<float>::quiet_NaN()
                                                : 1.0f / val_f;
                A[j] = static_cast<data_type>(std::round(std::asin(inv_val)));
            }
            else
            {
                double val_d = static_cast<double>(A[j]);
                double inv_val = (val_d == 0.0) ? std::numeric_limits<double>::quiet_NaN()
                                                : 1.0 / val_d;
                A[j] = static_cast<data_type>(std::round(std::asin(inv_val)));
            }
        }
    }
}

// ... potentially more functions ...