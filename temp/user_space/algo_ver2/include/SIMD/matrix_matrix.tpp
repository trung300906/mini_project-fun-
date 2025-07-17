#pragma once

// explicit header
#include "matrix_matrix_expli.tpp"
//==========================================================================//
template <typename data_type>
void simd_add(const data_type *A, const data_type *B, data_type *C, size_t shape)
{
    size_t i = 0;

    // Sử dụng constexpr if để chuyên biệt hóa tại thời điểm biên dịch
    if constexpr (std::is_same_v<data_type, float>)
    {
        // AVX-512 xử lý 16 float (16 * 32-bit = 512 bits) cùng lúc
#pragma unroll
        for (; i + 16 <= shape; i += 16)
        {
            // Sử dụng load/store không căn chỉnh (an toàn hơn nếu không chắc chắn về alignment)
            __m512 a = _mm512_loadu_ps(&A[i]);
            __m512 b = _mm512_loadu_ps(&B[i]);
            __m512 c = _mm512_add_ps(a, b); // Lệnh cộng cho float
            _mm512_storeu_ps(&C[i], c);
        }
    }
    else if constexpr (std::is_same_v<data_type, double>)
    {
        // AVX-512 xử lý 8 double (8 * 64-bit = 512 bits) cùng lúc
#pragma unroll
        for (; i + 8 <= shape; i += 8)
        {
            __m512d a = _mm512_loadu_pd(&A[i]);
            __m512d b = _mm512_loadu_pd(&B[i]);
            __m512d c = _mm512_add_pd(a, b); // Lệnh cộng cho double
            _mm512_storeu_pd(&C[i], c);
        }
    }
    else if constexpr (std::is_integral_v<data_type>) // Xử lý các kiểu số nguyên
    {
        constexpr size_t type_size = sizeof(data_type);

        // Kiểm tra xem kích thước kiểu có được hỗ trợ bởi lệnh SIMD cụ thể không
        if constexpr (type_size == 1 || type_size == 2 || type_size == 4 || type_size == 8)
        {
            // Tính số phần tử vừa trong một vector 512-bit
            constexpr size_t elements_per_vector = 512 / (type_size * 8); // 512 bits / (size_in_bits)
#pragma unroll
            for (; i + elements_per_vector <= shape; i += elements_per_vector)
            {
                // Load 512 bits dữ liệu (unaligned)
                // reinterpret_cast là cần thiết khi làm việc với __m512i và các kiểu dữ liệu cơ bản
                const __m512i *ptr_a = reinterpret_cast<const __m512i *>(&A[i]);
                const __m512i *ptr_b = reinterpret_cast<const __m512i *>(&B[i]);
                __m512i *ptr_c = reinterpret_cast<__m512i *>(&C[i]);

                __m512i a = _mm512_loadu_si512(ptr_a); // Load không căn chỉnh
                __m512i b = _mm512_loadu_si512(ptr_b); // Load không căn chỉnh
                __m512i c;

                // *** Sửa lỗi quan trọng: Chọn đúng lệnh cộng dựa trên kích thước kiểu ***
                if constexpr (type_size == 1)
                {                              // 8-bit (char, int8_t, uint8_t)
                    c = _mm512_add_epi8(a, b); // Yêu cầu cờ AVX512BW
                }
                else if constexpr (type_size == 2)
                {                               // 16-bit (short, int16_t, uint16_t)
                    c = _mm512_add_epi16(a, b); // Yêu cầu cờ AVX512BW
                }
                else if constexpr (type_size == 4)
                {                               // 32-bit (int, int32_t, uint32_t)
                    c = _mm512_add_epi32(a, b); // Yêu cầu cờ AVX512F
                }
                else if constexpr (type_size == 8)
                {                               // 64-bit (long long, int64_t, uint64_t)
                    c = _mm512_add_epi64(a, b); // Yêu cầu cờ AVX512DQ
                }

                // Store 512 bits kết quả (unaligned)
                _mm512_storeu_si512(ptr_c, c);
            }
        }
        // else: Nếu kích thước kiểu không phải là 1, 2, 4, hoặc 8 bytes,
        //       (ví dụ __int128), mã sẽ không vào nhánh SIMD này
        //       và sẽ được xử lý bởi vòng lặp scalar cuối cùng.
    }
    // else if constexpr (/* Các kiểu dữ liệu khác như số phức có thể thêm ở đây */) { ... }

    // Vòng lặp cuối cùng: Xử lý các phần tử còn lại không đủ một vector SIMD
    // Vòng lặp này không thay đổi so với bản gốc và là cần thiết.
    for (; i < shape; ++i)
    {
        C[i] = A[i] + B[i];
    }
}

template <typename data_type>
void simd_sub(const data_type *A, const data_type *B, data_type *C, size_t shape)
{
    size_t i = 0;

    // Sử dụng constexpr if để chuyên biệt hóa tại thời điểm biên dịch
    if constexpr (std::is_same_v<data_type, float>)
    {
        // AVX-512 xử lý 16 float cùng lúc
#pragma unroll
        for (; i + 16 <= shape; i += 16)
        {
            // Load/Store không căn chỉnh
            __m512 a = _mm512_loadu_ps(&A[i]);
            __m512 b = _mm512_loadu_ps(&B[i]);
            __m512 c = _mm512_sub_ps(a, b); // << Lệnh trừ cho float
            _mm512_storeu_ps(&C[i], c);
        }
    }
    else if constexpr (std::is_same_v<data_type, double>)
    {
        // AVX-512 xử lý 8 double cùng lúc
#pragma unroll
        for (; i + 8 <= shape; i += 8)
        {
            // Load/Store không căn chỉnh
            __m512d a = _mm512_loadu_pd(&A[i]);
            __m512d b = _mm512_loadu_pd(&B[i]);
            __m512d c = _mm512_sub_pd(a, b); // << Lệnh trừ cho double
            _mm512_storeu_pd(&C[i], c);
        }
    }
    else if constexpr (std::is_integral_v<data_type>) // Xử lý các kiểu số nguyên
    {
        constexpr size_t type_size = sizeof(data_type);

        // Kiểm tra kích thước kiểu có phù hợp với lệnh SIMD không
        if constexpr (type_size == 1 || type_size == 2 || type_size == 4 || type_size == 8)
        {
            // Tính số phần tử vừa trong một vector 512-bit
            constexpr size_t elements_per_vector = 512 / (type_size * 8);
#pragma unroll
            for (; i + elements_per_vector <= shape; i += elements_per_vector)
            {
                // Load 512 bits dữ liệu (unaligned)
                const __m512i *ptr_a = reinterpret_cast<const __m512i *>(&A[i]);
                const __m512i *ptr_b = reinterpret_cast<const __m512i *>(&B[i]);
                __m512i *ptr_c = reinterpret_cast<__m512i *>(&C[i]);

                __m512i a = _mm512_loadu_si512(ptr_a);
                __m512i b = _mm512_loadu_si512(ptr_b);
                __m512i c;

                // *** Sửa lỗi: Chọn đúng lệnh TRỪ dựa trên kích thước kiểu ***
                if constexpr (type_size == 1)
                {                              // 8-bit
                    c = _mm512_sub_epi8(a, b); // Yêu cầu AVX512BW
                }
                else if constexpr (type_size == 2)
                {                               // 16-bit
                    c = _mm512_sub_epi16(a, b); // Yêu cầu AVX512BW
                }
                else if constexpr (type_size == 4)
                {                               // 32-bit
                    c = _mm512_sub_epi32(a, b); // Yêu cầu AVX512F
                }
                else if constexpr (type_size == 8)
                {                               // 64-bit
                    c = _mm512_sub_epi64(a, b); // Yêu cầu AVX512DQ
                }

                // Store 512 bits kết quả (unaligned)
                _mm512_storeu_si512(ptr_c, c);
            }
        }
        // else: Các kiểu số nguyên không hỗ trợ sẽ rơi vào vòng lặp scalar
    }
    // else if constexpr (/* other types */) { ... }

    // Vòng lặp cuối: Xử lý các phần tử còn lại bằng phép trừ thông thường
    for (; i < shape; ++i)
    {
        C[i] = A[i] - B[i]; // Phép trừ scalar
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
        constexpr size_t type_size = sizeof(data_type);

        // Kiểm tra kích thước kiểu có phù hợp với lệnh SIMD không
        if constexpr (type_size == 1 || type_size == 2 || type_size == 4 || type_size == 8)
        {
            // Tính số phần tử vừa trong một vector 512-bit
            constexpr size_t elements_per_vector = 512 / (type_size * 8);
#pragma unroll
            for (; i + elements_per_vector <= shape; i += elements_per_vector)
            {
                // Load 512 bits dữ liệu (unaligned)
                const __m512i *ptr_a = reinterpret_cast<const __m512i *>(&A[i]);
                const __m512i *ptr_b = reinterpret_cast<const __m512i *>(&B[i]);
                __m512i *ptr_c = reinterpret_cast<__m512i *>(&C[i]);

                __m512i a = _mm512_loadu_si512(ptr_a);
                __m512i b = _mm512_loadu_si512(ptr_b);
                __m512i c;

                // *** Sửa lỗi: Chọn đúng lệnh TRỪ dựa trên kích thước kiểu ***
                if constexpr (type_size == 2)
                {                                 // 16-bit
                    c = _mm512_mullo_epi16(a, b); // Yêu cầu AVX512BW
                }
                else if constexpr (type_size == 4)
                {                                 // 32-bit
                    c = _mm512_mullo_epi32(a, b); // Yêu cầu AVX512F
                }
                else if constexpr (type_size == 8)
                {                                 // 64-bit
                    c = _mm512_mullo_epi64(a, b); // Yêu cầu AVX512DQ
                }

                // Store 512 bits kết quả (unaligned)
                _mm512_storeu_si512(ptr_c, c);
            }
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
#pragma unroll
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
#pragma unroll
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
#pragma unroll
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
#pragma unroll
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

#pragma unroll
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

template <typename data_type>
void simd_mod(const data_type *A, const data_type *B, data_type *C, size_t shape)
{
    size_t i = 0;

    if constexpr (std::is_same_v<data_type, float>)
    {
#pragma unroll
        for (; i + 16 <= shape; i += 16)
        {
            __m512 a = _mm512_loadu_ps(&A[i]);
            __m512 b = _mm512_loadu_ps(&B[i]);

            // No direct intrinsic for float modulo, use temporary arrays
            alignas(64) float temp_a[16], temp_b[16], temp_c[16];
            _mm512_storeu_ps(temp_a, a);
            _mm512_storeu_ps(temp_b, b);
            for (int j = 0; j < 16; j++)
            {
                // Check for division by zero
                if (temp_b[j] == 0.0f)
                {
                    temp_c[j] = std::numeric_limits<float>::quiet_NaN();
                }
                else
                {
                    temp_c[j] = fmodf(temp_a[j], temp_b[j]);
                }
            }

            __m512 c = _mm512_loadu_ps(temp_c);
            _mm512_storeu_ps(&C[i], c);
        }
    }
    else if constexpr (std::is_same_v<data_type, double>)
    {
#pragma unroll
        for (; i + 8 <= shape; i += 8)
        {
            __m512d a = _mm512_loadu_pd(&A[i]);
            __m512d b = _mm512_loadu_pd(&B[i]);

            // No direct intrinsic for double modulo, use temporary arrays
            alignas(64) double temp_a[8], temp_b[8], temp_c[8];
            _mm512_storeu_pd(temp_a, a);
            _mm512_storeu_pd(temp_b, b);
            for (int j = 0; j < 8; j++)
            {
                // Check for division by zero
                if (temp_b[j] == 0.0)
                {
                    temp_c[j] = std::numeric_limits<double>::quiet_NaN();
                }
                else
                {
                    temp_c[j] = std::fmod(temp_a[j], temp_b[j]);
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
#pragma unroll
            for (; i + 16 <= shape; i += 16)
            {
                __m512i a = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(&A[i]));
                __m512i b = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(&B[i]));

                // No direct intrinsic for integer modulo, use temporary arrays
                alignas(64) int32_t temp_a[16], temp_b[16], temp_c[16];
                _mm512_storeu_si512(reinterpret_cast<__m512i *>(temp_a), a);
                _mm512_storeu_si512(reinterpret_cast<__m512i *>(temp_b), b);
                for (int j = 0; j < 16; j++)
                {
                    // Check for division by zero
                    if (temp_b[j] == 0)
                    {
                        temp_c[j] = 0; // Define behavior for division by zero
                    }
                    else
                    {
                        temp_c[j] = temp_a[j] % temp_b[j];
                    }
                }

                __m512i c = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(temp_c));
                _mm512_storeu_si512(reinterpret_cast<__m512i *>(&C[i]), c);
            }
        }
        else if constexpr (sizeof(data_type) == 8) // 64-bit integers
        {
#pragma unroll
            for (; i + 8 <= shape; i += 8)
            {
                __m512i a = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(&A[i]));
                __m512i b = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(&B[i]));

                // No direct intrinsic for integer modulo, use temporary arrays
                alignas(64) int64_t temp_a[8], temp_b[8], temp_c[8];
                _mm512_storeu_si512(reinterpret_cast<__m512i *>(temp_a), a);
                _mm512_storeu_si512(reinterpret_cast<__m512i *>(temp_b), b);
                for (int j = 0; j < 8; j++)
                {
                    // Check for division by zero
                    if (temp_b[j] == 0)
                    {
                        temp_c[j] = 0; // Define behavior for division by zero
                    }
                    else
                    {
                        temp_c[j] = temp_a[j] % temp_b[j];
                    }
                }

                __m512i c = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(temp_c));
                _mm512_storeu_si512(reinterpret_cast<__m512i *>(&C[i]), c);
            }
        }
    }
#pragma unroll
    // Process remaining elements
    for (; i < shape; i++)
    {
        if constexpr (std::is_same_v<data_type, float>)
        {
            C[i] = (B[i] == 0.0f) ? std::numeric_limits<float>::quiet_NaN() : fmodf(A[i], B[i]);
        }
        else if constexpr (std::is_same_v<data_type, double>)
        {
            C[i] = (B[i] == 0.0) ? std::numeric_limits<double>::quiet_NaN() : std::fmod(A[i], B[i]);
        }
        else if constexpr (std::is_integral_v<data_type>)
        {
            C[i] = (B[i] == 0) ? 0 : A[i] % B[i]; // Define behavior for division by zero
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
        constexpr size_t vec_size = 16; // 512 bits / 32 bits = 16 floats
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
        if constexpr (sizeof(data_type) == 1) // 8-bit integers
        {
            constexpr size_t vec_size = 64; // 512 bits / 8 bits = 64 bytes

            for (; i + vec_size <= size; i += vec_size)
            {
                __m512i a_vec = _mm512_loadu_si512((__m512i *)&a_ptr[i]);
                __m512i b_vec = _mm512_loadu_si512((__m512i *)&b_ptr[i]);

                __mmask64 mask = _mm512_cmpeq_epi8_mask(a_vec, b_vec);
                if (mask != 0xFFFFFFFFFFFFFFFFULL) // All 64 bits should be set
                {
                    return false;
                }
            }
        }
        else if constexpr (sizeof(data_type) == 2) // 16-bit integers
        {
            constexpr size_t vec_size = 32; // 512 bits / 16 bits = 32 shorts

            for (; i + vec_size <= size; i += vec_size)
            {
                __m512i a_vec = _mm512_loadu_si512((__m512i *)&a_ptr[i]);
                __m512i b_vec = _mm512_loadu_si512((__m512i *)&b_ptr[i]);

                __mmask32 mask = _mm512_cmpeq_epi16_mask(a_vec, b_vec);
                if (mask != 0xFFFFFFFF) // All 32 bits should be set
                {
                    return false;
                }
            }
        }
        else if constexpr (sizeof(data_type) == 4) // 32-bit integers
        {
            constexpr size_t vec_size = 16; // 512 bits / 32 bits = 16 ints

            for (; i + vec_size <= size; i += vec_size)
            {
                __m512i a_vec = _mm512_loadu_si512((__m512i *)&a_ptr[i]);
                __m512i b_vec = _mm512_loadu_si512((__m512i *)&b_ptr[i]);

                __mmask16 mask = _mm512_cmpeq_epi32_mask(a_vec, b_vec);
                if (mask != 0xFFFF) // All 16 bits should be set
                {
                    return false;
                }
            }
        }
        else if constexpr (sizeof(data_type) == 8) // 64-bit integers
        {
            constexpr size_t vec_size = 8; // 512 bits / 64 bits = 8 longs

            for (; i + vec_size <= size; i += vec_size)
            {
                __m512i a_vec = _mm512_loadu_si512((__m512i *)&a_ptr[i]);
                __m512i b_vec = _mm512_loadu_si512((__m512i *)&b_ptr[i]);

                __mmask8 mask = _mm512_cmpeq_epi64_mask(a_vec, b_vec);
                if (mask != 0xFF) // All 8 bits should be set
                {
                    return false;
                }
            }
        }
    }

    // Process remaining elements
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
        if constexpr (sizeof(data_type) == 1) // 8-bit integers
        {
            constexpr size_t vec_size = 64; // 512 bits / 8 bits = 64 bytes

            for (; i + vec_size <= size; i += vec_size)
            {
                __m512i a_vec = _mm512_loadu_si512((__m512i *)&a_ptr[i]);
                __m512i b_vec = _mm512_loadu_si512((__m512i *)&b_ptr[i]);

                __mmask64 mask = _mm512_cmpgt_epi8_mask(a_vec, b_vec);
                if (mask != 0xFFFFFFFFFFFFFFFFULL) // All 64 bits should be set
                {
                    return false;
                }
            }
        }
        else if constexpr (sizeof(data_type) == 2) // 16-bit integers
        {
            constexpr size_t vec_size = 32; // 512 bits / 16 bits = 32 shorts

            for (; i + vec_size <= size; i += vec_size)
            {
                __m512i a_vec = _mm512_loadu_si512((__m512i *)&a_ptr[i]);
                __m512i b_vec = _mm512_loadu_si512((__m512i *)&b_ptr[i]);

                __mmask32 mask = _mm512_cmpgt_epi16_mask(a_vec, b_vec);
                if (mask != 0xFFFFFFFF) // All 32 bits should be set
                {
                    return false;
                }
            }
        }
        else if constexpr (sizeof(data_type) == 4) // 32-bit integers
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
        else if constexpr (sizeof(data_type) == 8) // 64-bit integers
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
        if constexpr (sizeof(data_type) == 1) // 8-bit integers
        {
            constexpr size_t vec_size = 64; // 512 bits / 8 bits = 64 bytes

            for (; i + vec_size <= size; i += vec_size)
            {
                __m512i a_vec = _mm512_loadu_si512((__m512i *)&a_ptr[i]);
                __m512i b_vec = _mm512_loadu_si512((__m512i *)&b_ptr[i]);

                __mmask64 mask = _mm512_cmplt_epi8_mask(a_vec, b_vec);
                if (mask != 0xFFFFFFFFFFFFFFFFULL) // All 64 bits should be set
                {
                    return false;
                }
            }
        }
        else if constexpr (sizeof(data_type) == 2) // 16-bit integers
        {
            constexpr size_t vec_size = 32; // 512 bits / 16 bits = 32 shorts

            for (; i + vec_size <= size; i += vec_size)
            {
                __m512i a_vec = _mm512_loadu_si512((__m512i *)&a_ptr[i]);
                __m512i b_vec = _mm512_loadu_si512((__m512i *)&b_ptr[i]);

                __mmask32 mask = _mm512_cmplt_epi16_mask(a_vec, b_vec);
                if (mask != 0xFFFFFFFF) // All 32 bits should be set
                {
                    return false;
                }
            }
        }
        else if constexpr (sizeof(data_type) == 4) // int32_t
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

template <typename data_type>
void simd_log(const data_type *A, const data_type *B, data_type *C, size_t shape)
{
    size_t i = 0;

    if constexpr (std::is_same_v<data_type, float>)
    {
#pragma unroll
        for (; i + 16 <= shape; i += 16)
        {
            __m512 a = _mm512_loadu_ps(&A[i]);
            __m512 b = _mm512_loadu_ps(&B[i]);

            // AVX-512 doesn't have direct log instructions, use temp arrays
            alignas(64) float temp_a[16], temp_b[16], temp_c[16];
            _mm512_storeu_ps(temp_a, a);
            _mm512_storeu_ps(temp_b, b);

#pragma omp parallel for simd
            for (int j = 0; j < 16; ++j)
            {
                // Handle special cases
                if (temp_a[j] <= 0.0f)
                {
                    temp_c[j] = std::numeric_limits<float>::quiet_NaN();
                }
                else if (temp_b[j] <= 0.0f || temp_b[j] == 1.0f)
                {
                    temp_c[j] = std::numeric_limits<float>::quiet_NaN();
                }
                else
                {
                    // log_base(x) = log(x) / log(base)
                    temp_c[j] = std::log(temp_a[j]) / std::log(temp_b[j]);
                }
            }

            __m512 c = _mm512_loadu_ps(temp_c);
            _mm512_storeu_ps(&C[i], c);
        }
    }
    else if constexpr (std::is_same_v<data_type, double>)
    {
#pragma unroll
        for (; i + 8 <= shape; i += 8)
        {
            __m512d a = _mm512_loadu_pd(&A[i]);
            __m512d b = _mm512_loadu_pd(&B[i]);

            alignas(64) double temp_a[8], temp_b[8], temp_c[8];
            _mm512_storeu_pd(temp_a, a);
            _mm512_storeu_pd(temp_b, b);

#pragma omp parallel for simd
            for (int j = 0; j < 8; ++j)
            {
                // Handle special cases
                if (temp_a[j] <= 0.0)
                {
                    temp_c[j] = std::numeric_limits<double>::quiet_NaN();
                }
                else if (temp_b[j] <= 0.0 || temp_b[j] == 1.0)
                {
                    temp_c[j] = std::numeric_limits<double>::quiet_NaN();
                }
                else
                {
                    // log_base(x) = log(x) / log(base)
                    temp_c[j] = std::log(temp_a[j]) / std::log(temp_b[j]);
                }
            }

            __m512d c = _mm512_loadu_pd(temp_c);
            _mm512_storeu_pd(&C[i], c);
        }
    }
    else if constexpr (std::is_integral_v<data_type>)
    {
        if constexpr (sizeof(data_type) == 1) // 8-bit integers
        {
#pragma unroll
            for (; i + 64 <= shape; i += 64)
            {
                __m512i a = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(&A[i]));
                __m512i b = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(&B[i]));

                alignas(64) signed char temp_a[64], temp_b[64];
                alignas(64) data_type temp_c[64];
                _mm512_storeu_si512(reinterpret_cast<__m512i *>(temp_a), a);
                _mm512_storeu_si512(reinterpret_cast<__m512i *>(temp_b), b);

#pragma omp parallel for simd
                for (int j = 0; j < 64; ++j)
                {
                    // Handle special cases for integers
                    if (temp_a[j] <= 0 || temp_b[j] <= 0 || temp_b[j] == 1)
                    {
                        temp_c[j] = 0; // Or another appropriate error value
                    }
                    else
                    {
                        // Integer logarithm (floor of the actual log value)
                        temp_c[j] = static_cast<data_type>(
                            std::floor(std::log(static_cast<double>(temp_a[j])) /
                                       std::log(static_cast<double>(temp_b[j]))));
                    }
                }

                __m512i c = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(temp_c));
                _mm512_storeu_si512(reinterpret_cast<__m512i *>(&C[i]), c);
            }
        }
        else if constexpr (sizeof(data_type) == 2) // 16-bit integers
        {
#pragma unroll
            for (; i + 32 <= shape; i += 32)
            {
                __m512i a = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(&A[i]));
                __m512i b = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(&B[i]));

                alignas(64) short temp_a[32], temp_b[32];
                alignas(64) data_type temp_c[32];
                _mm512_storeu_si512(reinterpret_cast<__m512i *>(temp_a), a);
                _mm512_storeu_si512(reinterpret_cast<__m512i *>(temp_b), b);

#pragma omp parallel for simd
                for (int j = 0; j < 32; ++j)
                {
                    if (temp_a[j] <= 0 || temp_b[j] <= 0 || temp_b[j] == 1)
                    {
                        temp_c[j] = 0;
                    }
                    else
                    {
                        temp_c[j] = static_cast<data_type>(
                            std::floor(std::log(static_cast<double>(temp_a[j])) /
                                       std::log(static_cast<double>(temp_b[j]))));
                    }
                }

                __m512i c = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(temp_c));
                _mm512_storeu_si512(reinterpret_cast<__m512i *>(&C[i]), c);
            }
        }
        else if constexpr (sizeof(data_type) == 4) // 32-bit integers
        {
#pragma unroll
            for (; i + 16 <= shape; i += 16)
            {
                __m512i a = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(&A[i]));
                __m512i b = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(&B[i]));

                alignas(64) int temp_a[16], temp_b[16];
                alignas(64) data_type temp_c[16];
                _mm512_storeu_si512(reinterpret_cast<__m512i *>(temp_a), a);
                _mm512_storeu_si512(reinterpret_cast<__m512i *>(temp_b), b);

#pragma omp parallel for simd
                for (int j = 0; j < 16; ++j)
                {
                    if (temp_a[j] <= 0 || temp_b[j] <= 0 || temp_b[j] == 1)
                    {
                        temp_c[j] = 0;
                    }
                    else
                    {
                        temp_c[j] = static_cast<data_type>(
                            std::floor(std::log(static_cast<double>(temp_a[j])) /
                                       std::log(static_cast<double>(temp_b[j]))));
                    }
                }

                __m512i c = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(temp_c));
                _mm512_storeu_si512(reinterpret_cast<__m512i *>(&C[i]), c);
            }
        }
        else if constexpr (sizeof(data_type) == 8) // 64-bit integers
        {
#pragma unroll
            for (; i + 8 <= shape; i += 8)
            {
                __m512i a = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(&A[i]));
                __m512i b = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(&B[i]));

                alignas(64) long long temp_a[8], temp_b[8];
                alignas(64) data_type temp_c[8];
                _mm512_storeu_si512(reinterpret_cast<__m512i *>(temp_a), a);
                _mm512_storeu_si512(reinterpret_cast<__m512i *>(temp_b), b);

#pragma omp parallel for simd
                for (int j = 0; j < 8; ++j)
                {
                    if (temp_a[j] <= 0 || temp_b[j] <= 0 || temp_b[j] == 1)
                    {
                        temp_c[j] = 0;
                    }
                    else
                    {
                        temp_c[j] = static_cast<data_type>(
                            std::floor(std::log(static_cast<double>(temp_a[j])) /
                                       std::log(static_cast<double>(temp_b[j]))));
                    }
                }

                __m512i c = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(temp_c));
                _mm512_storeu_si512(reinterpret_cast<__m512i *>(&C[i]), c);
            }
        }
    }

// Process remaining elements
#pragma omp parallel for
    for (size_t j = i; j < shape; ++j)
    {
        if constexpr (std::is_floating_point_v<data_type>)
        {
            if (A[j] <= 0 || B[j] <= 0 || B[j] == 1)
            {
                C[j] = std::numeric_limits<data_type>::quiet_NaN();
            }
            else
            {
                C[j] = std::log(A[j]) / std::log(B[j]);
            }
        }
        else if constexpr (std::is_integral_v<data_type>)
        {
            if (A[j] <= 0 || B[j] <= 0 || B[j] == 1)
            {
                C[j] = 0;
            }
            else
            {
                C[j] = static_cast<data_type>(
                    std::floor(std::log(static_cast<double>(A[j])) /
                               std::log(static_cast<double>(B[j]))));
            }
        }
    }
}

// ... existing code ...

template <typename data_type>
void simd_sqrt(const data_type *A, const data_type *B, data_type *C, size_t shape)
{
    size_t i = 0;

    // --- SIMD Processing ---

    if constexpr (std::is_floating_point_v<data_type>)
    {
        if constexpr (std::is_same_v<data_type, float>)
        {
            // Constants for edge cases
            const __m512 zero_ps = _mm512_setzero_ps();
            const __m512 one_ps = _mm512_set1_ps(1.0f);
            const __m512 nan_ps = _mm512_set1_ps(std::numeric_limits<float>::quiet_NaN());
            // Use std::numeric_limits directly where needed for Inf to potentially avoid compiler warnings
            const float infinity_f = std::numeric_limits<float>::infinity();
            const __m512 inf_ps = _mm512_set1_ps(infinity_f);

            // Temporary arrays for std::pow calculation
            alignas(64) float temp_b[16], temp_exp[16], temp_c[16];

            for (; i + 16 <= shape; i += 16)
            {
                const __m512 a_ps = _mm512_loadu_ps(&A[i]);
                const __m512 b_ps = _mm512_loadu_ps(&B[i]);

                // --- Masks (Keep these as they are useful) ---
                const __mmask16 a_zero_mask = _mm512_cmp_ps_mask(a_ps, zero_ps, _CMP_EQ_OQ);
                const __mmask16 b_pos_mask = _mm512_cmp_ps_mask(b_ps, zero_ps, _CMP_GT_OQ);
                const __mmask16 b_zero_mask = _mm512_cmp_ps_mask(b_ps, zero_ps, _CMP_EQ_OQ);
                const __mmask16 b_neg_mask = _mm512_cmp_ps_mask(b_ps, zero_ps, _CMP_LT_OQ);

                // --- Calculate Exponent (masked) ---
                // exponent = 1.0f / a (only where A != 0)
                const __m512 exponent_ps = _mm512_maskz_div_ps(~a_zero_mask, one_ps, a_ps);

                // --- Calculate pow(B, exponent) using std::pow via temporary arrays ---
                _mm512_storeu_ps(temp_b, b_ps);
                _mm512_storeu_ps(temp_exp, exponent_ps); // Store potentially invalid exponents too

#pragma omp parallel for simd
                for (int j = 0; j < 16; ++j)
                {
                    // Let std::pow handle B=0 cases internally for now.
                    // Use double intermediate for precision in pow.
                    // This calculation happens even for masked-out elements, but will be overwritten.
                    temp_c[j] = static_cast<float>(std::pow(static_cast<double>(temp_b[j]), static_cast<double>(temp_exp[j])));
                }

                // Load the results computed by std::pow
                __m512 result_ps = _mm512_loadu_ps(temp_c);

                // --- Apply Masked Overwrites for Edge Cases ---

                // Case 1: A != 0, B == 0
                __mmask16 b_zero_case_mask = (~a_zero_mask) & b_zero_mask;
                if (b_zero_case_mask != 0) // Check if this case exists
                {
                    // Need exponent sign to determine 0 or Inf
                    __mmask16 exp_pos_mask = _mm512_cmp_ps_mask(exponent_ps, zero_ps, _CMP_GT_OQ);
                    __mmask16 exp_neg_mask = _mm512_cmp_ps_mask(exponent_ps, zero_ps, _CMP_LT_OQ);
                    // Apply 0 where B==0 and Exp>0
                    result_ps = _mm512_mask_mov_ps(result_ps, b_zero_case_mask & exp_pos_mask, zero_ps);
                    // Apply Inf where B==0 and Exp<0
                    result_ps = _mm512_mask_mov_ps(result_ps, b_zero_case_mask & exp_neg_mask, inf_ps);
                    // Note: std::pow(0,0) should yield 1. std::pow(0, NaN) yields NaN.
                    // The scalar loop result for these specific sub-cases (exp=0 or exp=NaN when B=0)
                    // might be more accurate than this mask overwrite. Consider if pow(0,0)=1 is critical.
                    // If pow(0,0)=1 is needed, we might need a specific mask for exp==0 & b==0.
                    __mmask16 exp_zero_mask = _mm512_cmp_ps_mask(exponent_ps, zero_ps, _CMP_EQ_OQ);
                    result_ps = _mm512_mask_mov_ps(result_ps, b_zero_case_mask & exp_zero_mask, one_ps); // pow(0,0) = 1
                }

                // Case 2: A != 0, B < 0 -> Result is NaN (std::pow might do this, mask ensures it)
                result_ps = _mm512_mask_mov_ps(result_ps, (~a_zero_mask) & b_neg_mask, nan_ps);

                // Case 3: A == 0 -> Result is NaN (Operation undefined)
                result_ps = _mm512_mask_mov_ps(result_ps, a_zero_mask, nan_ps);

                // Store the final combined result
                _mm512_storeu_ps(&C[i], result_ps);
            }
        }
        else // Must be double
        {
            // Constants for edge cases
            const __m512d zero_pd = _mm512_setzero_pd();
            const __m512d one_pd = _mm512_set1_pd(1.0);
            const __m512d nan_pd = _mm512_set1_pd(std::numeric_limits<double>::quiet_NaN());
            const double infinity_d = std::numeric_limits<double>::infinity();
            const __m512d inf_pd = _mm512_set1_pd(infinity_d);

            // Temporary arrays for std::pow calculation
            alignas(64) double temp_b[8], temp_exp[8], temp_c[8];

            for (; i + 8 <= shape; i += 8)
            {
                const __m512d a_pd = _mm512_loadu_pd(&A[i]);
                const __m512d b_pd = _mm512_loadu_pd(&B[i]);

                // --- Masks ---
                const __mmask8 a_zero_mask = _mm512_cmp_pd_mask(a_pd, zero_pd, _CMP_EQ_OQ);
                const __mmask8 b_pos_mask = _mm512_cmp_pd_mask(b_pd, zero_pd, _CMP_GT_OQ);
                const __mmask8 b_zero_mask = _mm512_cmp_pd_mask(b_pd, zero_pd, _CMP_EQ_OQ);
                const __mmask8 b_neg_mask = _mm512_cmp_pd_mask(b_pd, zero_pd, _CMP_LT_OQ);

                // --- Calculate Exponent (masked) ---
                const __m512d exponent_pd = _mm512_maskz_div_pd(~a_zero_mask, one_pd, a_pd);

                // --- Calculate pow(B, exponent) using std::pow via temporary arrays ---
                _mm512_storeu_pd(temp_b, b_pd);
                _mm512_storeu_pd(temp_exp, exponent_pd);

#pragma omp parallel for simd
                for (int j = 0; j < 8; ++j)
                {
                    temp_c[j] = std::pow(temp_b[j], temp_exp[j]);
                }

                // Load the results computed by std::pow
                __m512d result_pd = _mm512_loadu_pd(temp_c);

                // --- Apply Masked Overwrites for Edge Cases ---

                // Case 1: A != 0, B == 0
                __mmask8 b_zero_case_mask = (~a_zero_mask) & b_zero_mask;
                if (b_zero_case_mask != 0)
                {
                    __mmask8 exp_pos_mask = _mm512_cmp_pd_mask(exponent_pd, zero_pd, _CMP_GT_OQ);
                    __mmask8 exp_neg_mask = _mm512_cmp_pd_mask(exponent_pd, zero_pd, _CMP_LT_OQ);
                    result_pd = _mm512_mask_mov_pd(result_pd, b_zero_case_mask & exp_pos_mask, zero_pd);
                    result_pd = _mm512_mask_mov_pd(result_pd, b_zero_case_mask & exp_neg_mask, inf_pd);
                    // Handle pow(0,0) = 1
                    __mmask8 exp_zero_mask = _mm512_cmp_pd_mask(exponent_pd, zero_pd, _CMP_EQ_OQ);
                    result_pd = _mm512_mask_mov_pd(result_pd, b_zero_case_mask & exp_zero_mask, one_pd);
                }

                // Case 2: A != 0, B < 0 -> NaN
                result_pd = _mm512_mask_mov_pd(result_pd, (~a_zero_mask) & b_neg_mask, nan_pd);

                // Case 3: A == 0 -> NaN
                result_pd = _mm512_mask_mov_pd(result_pd, a_zero_mask, nan_pd);

                _mm512_storeu_pd(&C[i], result_pd);
            }
        }
    }
    // Integer Types (Keep the previous temporary array + scalar loop approach)
    else if constexpr (std::is_integral_v<data_type>)
    {
        constexpr int type_size = sizeof(data_type);
        constexpr int VEC_SIZE = 64 / type_size;
        alignas(64) data_type temp_a[VEC_SIZE];
        alignas(64) data_type temp_b[VEC_SIZE];
        alignas(64) data_type temp_c[VEC_SIZE];

        for (; i + VEC_SIZE <= shape; i += VEC_SIZE)
        {
            __m512i a_vec = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(&A[i]));
            __m512i b_vec = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(&B[i]));
            _mm512_storeu_si512(reinterpret_cast<__m512i *>(temp_a), a_vec);
            _mm512_storeu_si512(reinterpret_cast<__m512i *>(temp_b), b_vec);

#pragma omp parallel for simd
            for (int j = 0; j < VEC_SIZE; ++j)
            {
                if (temp_a[j] == 0) // Invalid exponent denominator
                {
                    temp_c[j] = 0; // Or some error indicator for integers
                }
                else if (temp_b[j] < 0) // Negative base with fractional exponent (1/A) is problematic
                {
                    // For integer root (1/A), result is often undefined or complex unless A is odd and B is negative.
                    // Simplest is to return 0 for integer types.
                    temp_c[j] = 0;
                }
                else if (temp_b[j] == 0)
                {
                    // 0^(1/A) is 0 for A > 0 (since A!=0 here).
                    temp_c[j] = 0;
                }
                else
                {
                    double exponent = 1.0 / static_cast<double>(temp_a[j]);
                    double result_double = std::pow(static_cast<double>(temp_b[j]), exponent);
                    temp_c[j] = static_cast<data_type>(std::round(result_double)); // Round
                }
            }
            __m512i result_vec = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(temp_c));
            _mm512_storeu_si512(reinterpret_cast<__m512i *>(&C[i]), result_vec);
        }
    }

// --- Scalar Fallback Loop (Keep using std::pow for consistency) ---
#pragma omp parallel for
    for (size_t j = i; j < shape; ++j)
    {
        if constexpr (std::is_floating_point_v<data_type>)
        {
            if (A[j] == 0.0)
            {
                C[j] = std::numeric_limits<data_type>::quiet_NaN();
            }
            else
            {
                // Use double for intermediate exponent calculation
                double exponent = 1.0 / static_cast<double>(A[j]);
                C[j] = static_cast<data_type>(std::pow(static_cast<double>(B[j]), exponent));
            }
        }
        else // Integer types
        {
            if (A[j] == 0)
            {
                C[j] = 0;
            }
            else if (B[j] < 0)
            {
                C[j] = 0; // Simplified error handling for integers
            }
            else if (B[j] == 0)
            {
                C[j] = 0; // 0^(1/A) = 0 for A!=0
            }
            else
            {
                double exponent = 1.0 / static_cast<double>(A[j]);
                double result_double = std::pow(static_cast<double>(B[j]), exponent);
                C[j] = static_cast<data_type>(std::round(result_double)); // Round
            }
        }
    }
}

template <typename data_type>
void simd_xor(const data_type *A, const data_type *B, data_type *C, size_t shape)
{
    // XOR is typically only meaningful for integral types.
    static_assert(std::is_integral_v<data_type>, "XOR operation is only supported for integral types.");

    size_t i = 0;

    // --- SIMD Part ---
    if constexpr (std::is_integral_v<data_type>)
    {
        // Determine vector size based on data_type size
        constexpr int type_size = sizeof(data_type);
        constexpr int VEC_SIZE = 64 / type_size; // Number of elements per 512-bit vector

        // Process elements in chunks using AVX-512
        for (; i + VEC_SIZE <= shape; i += VEC_SIZE)
        {
            // Load data from both arrays into AVX-512 registers
            __m512i a_vec = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(&A[i]));
            __m512i b_vec = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(&B[i]));

            // Perform the XOR operation
            __m512i result_vec = _mm512_xor_si512(a_vec, b_vec);

            // Store the result back to the result array C
            _mm512_storeu_si512(reinterpret_cast<__m512i *>(&C[i]), result_vec);
        }
    }
    // No SIMD implementation provided for float/double as XOR is not standard.
    // The static_assert above prevents instantiation for these types anyway.
    else if constexpr (std::is_same_v<data_type, float>)
    {
        // Empty branch - static_assert prevents this path
    }
    else if constexpr (std::is_same_v<data_type, double>)
    {
        // Empty branch - static_assert prevents this path
    }

    // --- Scalar Part ---
    // Process remaining elements using a scalar loop
    for (; i < shape; ++i)
    {
        // This part will only be reached if the static_assert is removed/bypassed.
        // Standard C++ does not define operator^ for floating-point types.
        if constexpr (std::is_integral_v<data_type>)
        {
            C[i] = A[i] ^ B[i];
        }
        // else: No operation for float/double
    }
}