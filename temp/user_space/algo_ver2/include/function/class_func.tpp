#pragma once
namespace numpy
{
    template <typename data_type>
    void ndarray<data_type>::print() const
    {
        // Hàm đệ quy để in mảng n chiều
        assert(!data.empty());
        std::function<void(const std::vector<size_t> &, std::vector<size_t> &, size_t, size_t)> recursive;
        recursive = [&](const std::vector<size_t> &index, std::vector<size_t> &path, size_t level = 0, size_t indent = 0)
        {
            if (level == index.size())
            {
                std::cout << std::string(indent, ' ') << "[";
                std::cout << (*this)(path);
                std::cout << "]\n";
                return;
            }
            std::cout << std::string(indent, ' ') << "[\n";
            for (size_t i = 0; i < index[level]; i++)
            {
                path[level] = i;
                recursive(index, path, level + 1, indent + 2);
            }
            std::cout << std::string(indent, ' ') << "]\n";
        };
        // Create a separate path vector to track current indices during recursion
        std::vector<size_t> path(shape.size(), 0);
        recursive(shape, path, 0, 0);
    }
    template <typename data_type>
    size_t ndarray<data_type>::size() const
    {
        return data.size();
    }

    // get funct
    template <typename data_type>
    std::vector<size_t> ndarray<data_type>::get_shape() const
    {
        return shape;
    }
    template <typename data_type>
    const std::vector<size_t> &ndarray<data_type>::get_strides() const
    {
        return strides;
    }
    template <typename data_type>
    const std::vector<data_type> &ndarray<data_type>::get_data() const
    {
        return data;
    }

    // set funct
    template <typename data_type>
    void ndarray<data_type>::set_shape(const std::vector<size_t> &shape_)
    {
        shape = shape_;
        strides.resize(shape.size());
        size_t total = 1;
        for (int i = shape.size() - 1; i >= 0; --i)
        {
            strides[i] = total;
            total *= shape[i];
        }
        data.resize(total, 0);
    }
    template <typename data_type>
    void ndarray<data_type>::set_data(const std::vector<data_type> &data_)
    {
        data = data_;
    }
    template <typename data_type>
    void ndarray<data_type>::set_strides(const std::vector<size_t> &strides_)
    {
        strides = strides_;
    }

    template <typename data_type>
    size_t ndarray<data_type>::ndim() const
    {
        return shape.size();
    }

    template <typename data_type>
    data_type ndarray<data_type>::sum()
    {
        data_type answer;
        sum_avx512(data.data(), data.size(), answer);
        return answer;
    }

    template <typename data_type>
    data_type ndarray<data_type>::sum(std::vector<size_t> &begin, std::vector<size_t> &end)
    {
        assert(begin.size() == shape.size() && end.size() == shape.size() && begin.size() == end.size());
        size_t idx_begin = Index(begin);
        size_t idx_end = Index(end);
        data_type answer;
        sum_avx512(data.data() + idx_begin, idx_end - idx_begin, answer);
        return answer;
    }

    template <typename data_type>
    std::tuple<data_type, int, typename std::vector<data_type>::iterator> ndarray<data_type>::max()
    {
        auto min = std::max(data.begin(), data.end());
        return std::make_tuple(*min, std::distance(data.begin(), min), min);
    }

    template <typename data_type>
    std::tuple<data_type, int, typename std::vector<data_type>::iterator> ndarray<data_type>::max(std::vector<size_t> &begin, std::vector<size_t> &end)
    {
        assert(begin.size() == shape.size() && end.size() == shape.size() && begin.size() == end.size());
        size_t idx_begin = Index(begin);
        size_t idx_end = Index(end);
        auto min = std::max(data.begin() + idx_begin, data.begin() + idx_end);
        return std::make_tuple(*min, std::distance(data.begin(), min), min);
    }

    template <typename data_type>
    std::tuple<data_type, int, typename std::vector<data_type>::iterator> ndarray<data_type>::min()
    {
        auto min = std::min(data.begin(), data.end());
        return std::make_tuple(*min, std::distance(data.begin(), min), min);
    }

    template <typename data_type>
    std::tuple<data_type, int, typename std::vector<data_type>::iterator> ndarray<data_type>::min(std::vector<size_t> &begin, std::vector<size_t> &end)
    {
        assert(begin.size() == shape.size() && end.size() == shape.size() && begin.size() == end.size());
        size_t idx_begin = Index(begin);
        size_t idx_end = Index(end);
        auto min = std::min(data.begin() + idx_begin, data.begin() + idx_end);
        return std::make_tuple(*min, std::distance(data.begin(), min), min);
    }

    template <typename data_type>
    ndarray<data_type> ndarray<data_type>::reshape(const std::vector<size_t> &shape_)
    {
        std::vector<size_t> strides_;
        strides_.resize(shape_.size());
        size_t total = 1;
        for (int i = shape_.size() - 1; i >= 0; --i)
        {
            strides_[i] = total;
            total *= shape_[i];
        }
        assert(total == data.size());
        this->shape = shape_;
        this->strides = strides_;
        return *this;
    }

    template <typename data_type>
    std::vector<data_type> ndarray<data_type>::flatten(char order)
    {
        if (order == 'C')
        {
            // C-order is already correct - just return data
            return data;
        }
        else if (order == 'F')
        {
            // Create a new vector to hold the F-order result
            std::vector<data_type> result(data.size());

            // For a simple 2D case:
            if (shape.size() == 2)
            {
                size_t rows = shape[0];
                size_t cols = shape[1];
                size_t index = 0;

                // F-order: column-major traversal
                for (size_t j = 0; j < cols; ++j)
                {
                    for (size_t i = 0; i < rows; ++i)
                    {
                        // Access elements in column-major order
                        size_t data_index = i * cols + j;
                        result[index++] = data[data_index];
                    }
                }
                return result;
            }
            // For the general n-dimensional case:
            else
            {
                // Initialize indices for each dimension
                std::vector<size_t> indices(shape.size(), 0);
                size_t result_index = 0;

                // Recursive function for F-order traversal - start from the LAST dimension
                std::function<void(int)> traverse_f;
                traverse_f = [&](int dim)
                {
                    if (dim < 0)
                    {
                        // We've assigned values to all dimensions
                        // Calculate the offset in the original data array
                        size_t offset = 0;
                        for (size_t i = 0; i < shape.size(); ++i)
                        {
                            offset += indices[i] * strides[i];
                        }
                        result[result_index++] = data[offset];
                        return;
                    }

                    for (indices[dim] = 0; indices[dim] < shape[dim]; ++indices[dim])
                    {
                        traverse_f(dim - 1);
                    }
                };

                // Start traversal from the last dimension
                traverse_f(shape.size() - 1);
                return result;
            }
        }
        else
        {
            throw std::invalid_argument("Order must be 'C' or 'F'");
        }
    }

    template <typename data_type>
    ndarray<data_type> ndarray<data_type>::transpose()
    {
        // Handle empty array case
        if (shape.empty() || data.empty())
        {
            return ndarray<data_type>();
        }

        size_t n_dims = shape.size();

        // Special case for 1D arrays - just return a copy
        if (n_dims == 1)
        {
            ndarray<data_type> result(*this);
            return result;
        }

        // Special fast path for 2D arrays (common case)
        if (n_dims == 2)
        {
            size_t rows = shape[0];
            size_t cols = shape[1];

            // Create result with reversed dimensions
            ndarray<data_type> result;
            result.shape = {cols, rows};
            result.strides = {1, cols}; // Correct strides for row-major layout
            result.data.resize(data.size());

// Optimized transpose for 2D case
#pragma omp parallel for collapse(2) if (rows * cols > 10000) // Only parallelize large matrices
            for (size_t i = 0; i < rows; ++i)
            {
                for (size_t j = 0; j < cols; ++j)
                {
                    result.data[j * rows + i] = data[i * cols + j];
                }
            }
            return result;
        }

        // General n-dimensional case
        std::vector<size_t> new_shape = shape;
        std::reverse(new_shape.begin(), new_shape.end());

        // Calculate new strides more efficiently
        std::vector<size_t> new_strides(n_dims);
        size_t stride = 1;
        for (int i = n_dims - 1; i >= 0; --i)
        {
            new_strides[i] = stride;
            stride *= new_shape[i];
        }

        // Create and prepare result array
        ndarray<data_type> result;
        result.shape = new_shape;
        result.strides = new_strides;
        result.data.resize(data.size());

        // Performance optimization: pre-compute total elements
        size_t total_elements = data.size();

        // Use a vector to track current indices - preallocate to avoid reallocation
        std::vector<size_t> current_indices(n_dims, 0);

        // Optimize main loop
        for (size_t i = 0; i < total_elements; ++i)
        {
            // Calculate original offset
            size_t original_offset = 0;
            for (size_t d = 0; d < n_dims; ++d)
            {
                original_offset += current_indices[d] * strides[d];
            }

            // Calculate transposed offset
            size_t transposed_offset = 0;
            for (size_t d = 0; d < n_dims; ++d)
            {
                transposed_offset += current_indices[n_dims - 1 - d] * result.strides[d];
            }

            // Copy data - remove bounds checking in the inner loop for performance
            result.data[transposed_offset] = data[original_offset];

            // Update indices for next element
            for (int d = n_dims - 1; d >= 0; --d)
            {
                current_indices[d]++;
                if (current_indices[d] < shape[d])
                    break;
                current_indices[d] = 0;
            }
        }
        return result;
    }

    template <typename data_type>
    ndarray<data_type> ndarray<data_type>::dot(const ndarray<data_type> &nd, const std::vector<size_t> &axis)
    {
        // Handle empty arrays
        if (shape.empty() || nd.shape.empty() || data.empty() || nd.data.empty())
        {
            return ndarray<data_type>();
        }

        //=============================================================================
        // CASE 1: 1D vector dot product (inner product) - Highly optimized
        //=============================================================================
        if (shape.size() == 1 && nd.shape.size() == 1)
        {
            if (shape[0] != nd.shape[0])
            {
                throw std::invalid_argument("Vectors must have the same length for dot product");
            }

            size_t len = shape[0];
            data_type result = 0;

#ifdef __AVX512F__
            if constexpr (std::is_same_v<data_type, float>)
            {
                const size_t simd_width = 16; // 16 floats in 512 bits
                size_t vec_end = (len / simd_width) * simd_width;
                __m512 sum_vec = _mm512_setzero_ps();

                // Process 16 elements at a time
                for (size_t i = 0; i < vec_end; i += simd_width)
                {
                    // Prefetch next cache lines
                    _mm_prefetch((const char *)&data[i + simd_width], _MM_HINT_T0);
                    _mm_prefetch((const char *)&nd.data[i + simd_width], _MM_HINT_T0);

                    __m512 vec1 = _mm512_loadu_ps(&data[i]);
                    __m512 vec2 = _mm512_loadu_ps(&nd.data[i]);

                    // Use FMA for better performance
                    sum_vec = _mm512_fmadd_ps(vec1, vec2, sum_vec);
                }

                // Horizontal sum of all elements in the AVX-512 register
                result = _mm512_reduce_add_ps(sum_vec);

                // Process remaining elements
                for (size_t i = vec_end; i < len; ++i)
                {
                    result += data[i] * nd.data[i];
                }
            }
            else if constexpr (std::is_same_v<data_type, double>)
            {
                const size_t simd_width = 8; // 8 doubles in 512 bits
                size_t vec_end = (len / simd_width) * simd_width;
                __m512d sum_vec = _mm512_setzero_pd();

                // Process 8 elements at a time
                for (size_t i = 0; i < vec_end; i += simd_width)
                {
                    // Prefetch next cache lines
                    _mm_prefetch((const char *)&data[i + simd_width], _MM_HINT_T0);
                    _mm_prefetch((const char *)&nd.data[i + simd_width], _MM_HINT_T0);

                    __m512d vec1 = _mm512_loadu_pd(&data[i]);
                    __m512d vec2 = _mm512_loadu_pd(&nd.data[i]);

                    // Use FMA for better performance
                    sum_vec = _mm512_fmadd_pd(vec1, vec2, sum_vec);
                }

                // Horizontal sum of all elements in the AVX-512 register
                result = _mm512_reduce_add_pd(sum_vec);

                // Process remaining elements
                for (size_t i = vec_end; i < len; ++i)
                {
                    result += data[i] * nd.data[i];
                }
            }
            else if constexpr (std::is_same_v<data_type, int32_t>)
            {
                const size_t simd_width = 16; // 16 int32s in 512 bits
                size_t vec_end = (len / simd_width) * simd_width;
                __m512i sum_vec = _mm512_setzero_si512();

                // Process 16 elements at a time
                for (size_t i = 0; i < vec_end; i += simd_width)
                {
                    __m512i vec1 = _mm512_loadu_si512((__m512i *)&data[i]);
                    __m512i vec2 = _mm512_loadu_si512((__m512i *)&nd.data[i]);

                    // Multiply and add using AVX-512
                    __m512i prod = _mm512_mullo_epi32(vec1, vec2);
                    sum_vec = _mm512_add_epi32(sum_vec, prod);
                }

                // Horizontal sum of all elements
                result = _mm512_reduce_add_epi32(sum_vec);

                // Process remaining elements
                for (size_t i = vec_end; i < len; ++i)
                {
                    result += data[i] * nd.data[i];
                }
            }
            else
#endif // __AVX512F__
            {
// Fallback scalar implementation for non-SIMD types or if AVX-512 not available
#pragma omp parallel for reduction(+ : result) if (len > 100000)
                for (size_t i = 0; i < len; ++i)
                {
                    result += data[i] * nd.data[i];
                }
            }

            // Return a scalar as a 1-element ndarray
            ndarray<data_type> result_array;
            result_array.data = {result};
            result_array.shape = {1};
            result_array.strides = {1};
            return result_array;
        }

        //=============================================================================
        // CASE 2: 2D matrix multiplication - Cache-optimized with SIMD
        //=============================================================================
        if (shape.size() == 2 && nd.shape.size() == 2)
        {
            size_t m = shape[0];
            size_t k = shape[1];
            size_t n = nd.shape[1];

            if (k != nd.shape[0])
            {
                throw std::invalid_argument("Matrix dimensions incompatible for multiplication");
            }

            // Create result array
            ndarray<data_type> result;
            result.shape = {m, n};
            result.strides = {n, 1};
            result.data.resize(m * n, 0);

            // Matrix transposition for B to improve memory access patterns
            std::vector<data_type> B_trans(k * n);

// Transpose B for better cache locality
#pragma omp parallel for collapse(2) if (k * n > 10000)
            for (size_t i = 0; i < k; i++)
            {
                for (size_t j = 0; j < n; j++)
                {
                    B_trans[j * k + i] = nd.data[i * n + j];
                }
            }

#ifdef __AVX512F__
            if constexpr (std::is_same_v<data_type, float>)
            {
                // Block sizes chosen for L1 and L2 cache optimization
                const int BLOCK_M = 64;
                const int BLOCK_N = 64;
                const int BLOCK_K = 64;

// Outer blocking for cache efficiency
#pragma omp parallel for collapse(2) schedule(dynamic)
                for (size_t i0 = 0; i0 < m; i0 += BLOCK_M)
                {
                    for (size_t j0 = 0; j0 < n; j0 += BLOCK_N)
                    {
                        // Actual block sizes (handling edge cases)
                        size_t i_end = std::min(i0 + BLOCK_M, m);
                        size_t j_end = std::min(j0 + BLOCK_N, n);

                        // Process one block
                        for (size_t k0 = 0; k0 < k; k0 += BLOCK_K)
                        {
                            size_t k_end = std::min(k0 + BLOCK_K, k);

                            // Compute the block multiplication
                            for (size_t i = i0; i < i_end; i++)
                            {
                                for (size_t j = j0; j < j_end; j++)
                                {
                                    // Use register blocking for innermost loop
                                    __m512 sum_vec = _mm512_setzero_ps();
                                    size_t k_vec_end = k0 + ((k_end - k0) / 16) * 16;

                                    // SIMD vectorized part
                                    for (size_t kk = k0; kk < k_vec_end; kk += 16)
                                    {
                                        // Prefetch next iterations
                                        _mm_prefetch((const char *)&data[i * k + kk + 16], _MM_HINT_T0);
                                        _mm_prefetch((const char *)&B_trans[j * k + kk + 16], _MM_HINT_T0);

                                        __m512 a_vec = _mm512_loadu_ps(&data[i * k + kk]);
                                        __m512 b_vec = _mm512_loadu_ps(&B_trans[j * k + kk]);
                                        sum_vec = _mm512_fmadd_ps(a_vec, b_vec, sum_vec);
                                    }

                                    // Reduce the sum vector
                                    float sum = _mm512_reduce_add_ps(sum_vec);

                                    // Handle remaining elements
                                    for (size_t kk = k_vec_end; kk < k_end; kk++)
                                    {
                                        sum += data[i * k + kk] * B_trans[j * k + kk];
                                    }

                                    result.data[i * n + j] += sum;
                                }
                            }
                        }
                    }
                }
                return result;
            }
            else if constexpr (std::is_same_v<data_type, double>)
            {
                // Similar blocking strategy for double precision
                const int BLOCK_M = 32;
                const int BLOCK_N = 32;
                const int BLOCK_K = 32;

// Outer blocking for cache efficiency
#pragma omp parallel for collapse(2) schedule(dynamic)
                for (size_t i0 = 0; i0 < m; i0 += BLOCK_M)
                {
                    for (size_t j0 = 0; j0 < n; j0 += BLOCK_N)
                    {
                        // Actual block sizes (handling edge cases)
                        size_t i_end = std::min(i0 + BLOCK_M, m);
                        size_t j_end = std::min(j0 + BLOCK_N, n);

                        // Process one block
                        for (size_t k0 = 0; k0 < k; k0 += BLOCK_K)
                        {
                            size_t k_end = std::min(k0 + BLOCK_K, k);

                            // Compute the block multiplication
                            for (size_t i = i0; i < i_end; i++)
                            {
                                for (size_t j = j0; j < j_end; j++)
                                {
                                    // Use register blocking for innermost loop
                                    __m512d sum_vec = _mm512_setzero_pd();
                                    size_t k_vec_end = k0 + ((k_end - k0) / 8) * 8;

                                    // SIMD vectorized part
                                    for (size_t kk = k0; kk < k_vec_end; kk += 8)
                                    {
                                        // Prefetch next iterations
                                        _mm_prefetch((const char *)&data[i * k + kk + 8], _MM_HINT_T0);
                                        _mm_prefetch((const char *)&B_trans[j * k + kk + 8], _MM_HINT_T0);

                                        __m512d a_vec = _mm512_loadu_pd(&data[i * k + kk]);
                                        __m512d b_vec = _mm512_loadu_pd(&B_trans[j * k + kk]);
                                        sum_vec = _mm512_fmadd_pd(a_vec, b_vec, sum_vec);
                                    }

                                    // Reduce the sum vector
                                    double sum = _mm512_reduce_add_pd(sum_vec);

                                    // Handle remaining elements
                                    for (size_t kk = k_vec_end; kk < k_end; kk++)
                                    {
                                        sum += data[i * k + kk] * B_trans[j * k + kk];
                                    }

                                    result.data[i * n + j] += sum;
                                }
                            }
                        }
                    }
                }
                return result;
            }
            else if constexpr (std::is_same_v<data_type, int32_t>)
            {
                // Integer-specific optimizations
                const int BLOCK_M = 64;
                const int BLOCK_N = 64;
                const int BLOCK_K = 64;

#pragma omp parallel for collapse(2)
                for (size_t i0 = 0; i0 < m; i0 += BLOCK_M)
                {
                    for (size_t j0 = 0; j0 < n; j0 += BLOCK_N)
                    {
                        size_t i_end = std::min(i0 + BLOCK_M, m);
                        size_t j_end = std::min(j0 + BLOCK_N, n);

                        for (size_t k0 = 0; k0 < k; k0 += BLOCK_K)
                        {
                            size_t k_end = std::min(k0 + BLOCK_K, k);

                            for (size_t i = i0; i < i_end; i++)
                            {
                                for (size_t j = j0; j < j_end; j++)
                                {
                                    __m512i sum_vec = _mm512_setzero_si512();
                                    size_t k_vec_end = k0 + ((k_end - k0) / 16) * 16;

                                    for (size_t kk = k0; kk < k_vec_end; kk += 16)
                                    {
                                        __m512i a_vec = _mm512_loadu_si512((__m512i *)&data[i * k + kk]);
                                        __m512i b_vec = _mm512_loadu_si512((__m512i *)&B_trans[j * k + kk]);

                                        // For int32 we use multiplication and addition
                                        __m512i prod = _mm512_mullo_epi32(a_vec, b_vec);
                                        sum_vec = _mm512_add_epi32(sum_vec, prod);
                                    }

                                    int sum = _mm512_reduce_add_epi32(sum_vec);

                                    for (size_t kk = k_vec_end; kk < k_end; kk++)
                                    {
                                        sum += data[i * k + kk] * B_trans[j * k + kk];
                                    }

                                    result.data[i * n + j] += sum;
                                }
                            }
                        }
                    }
                }
                return result;
            }
            else
#endif // __AVX512F__
            {
                // Standard tiled matrix multiplication algorithm for non-SIMD types
                const int BLOCK_SIZE = 64; // Block size for cache optimization

#pragma omp parallel for collapse(2)
                for (size_t i0 = 0; i0 < m; i0 += BLOCK_SIZE)
                {
                    for (size_t j0 = 0; j0 < n; j0 += BLOCK_SIZE)
                    {
                        size_t i_end = std::min(i0 + BLOCK_SIZE, m);
                        size_t j_end = std::min(j0 + BLOCK_SIZE, n);

                        for (size_t k0 = 0; k0 < k; k0 += BLOCK_SIZE)
                        {
                            size_t k_end = std::min(k0 + BLOCK_SIZE, k);

                            for (size_t i = i0; i < i_end; i++)
                            {
                                for (size_t j = j0; j < j_end; j++)
                                {
                                    data_type sum = 0;

                                    for (size_t kk = k0; kk < k_end; kk++)
                                    {
                                        sum += data[i * k + kk] * B_trans[j * k + kk];
                                    }

                                    result.data[i * n + j] += sum;
                                }
                            }
                        }
                    }
                }
                return result;
            }
        }

        //=============================================================================
        // CASE 3: N-dimensional tensor contraction
        //=============================================================================
        // Default behavior: contract over last axis of A and second-to-last axis of B
        std::vector<size_t> contract_axes;
        if (axis.empty())
        {
            if (shape.size() < 1 || nd.shape.size() < 2)
            {
                throw std::invalid_argument("Arrays must have enough dimensions for default contraction");
            }
            contract_axes = {shape.size() - 1, nd.shape.size() - 2};
        }
        else
        {
            contract_axes = axis;
            if (contract_axes.size() != 2 ||
                contract_axes[0] >= shape.size() ||
                contract_axes[1] >= nd.shape.size())
            {
                throw std::invalid_argument("Invalid contraction axes");
            }
        }

        // Check if the contraction dimensions match
        if (shape[contract_axes[0]] != nd.shape[contract_axes[1]])
        {
            throw std::invalid_argument("Contraction dimensions don't match");
        }

        // Construct the output shape
        std::vector<size_t> result_shape;
        std::vector<size_t> a_dims_to_keep;
        std::vector<size_t> b_dims_to_keep;

        for (size_t i = 0; i < shape.size(); ++i)
        {
            if (i != contract_axes[0])
            {
                result_shape.push_back(shape[i]);
                a_dims_to_keep.push_back(i);
            }
        }

        for (size_t i = 0; i < nd.shape.size(); ++i)
        {
            if (i != contract_axes[1])
            {
                result_shape.push_back(nd.shape[i]);
                b_dims_to_keep.push_back(i);
            }
        }

        // Prepare result array
        ndarray<data_type> result;
        result.shape = result_shape;

        // Calculate strides for result
        result.strides.resize(result_shape.size());
        size_t stride = 1;
        for (int i = result_shape.size() - 1; i >= 0; --i)
        {
            result.strides[i] = stride;
            stride *= result_shape[i];
        }

        // Initialize result data
        size_t total_size = 1;
        for (size_t s : result_shape)
        {
            total_size *= s;
        }
        result.data.resize(total_size, 0);

        // N-dimensional tensor contraction is inherently difficult to vectorize
        // due to complex indexing. We will optimize by using a thread-parallel approach
        // and cache-friendly memory access patterns.

        // Create pre-computed index maps for faster access
        std::vector<size_t> a_strides_for_contract(shape[contract_axes[0]]);
        std::vector<size_t> b_strides_for_contract(shape[contract_axes[0]]);

        for (size_t k = 0; k < shape[contract_axes[0]]; ++k)
        {
            a_strides_for_contract[k] = k * strides[contract_axes[0]];
            b_strides_for_contract[k] = k * nd.strides[contract_axes[1]];
        }

        // Helper function for flat index to multi-dim indices conversion
        auto flat_to_indices = [](size_t flat_idx, const std::vector<size_t> &shape)
        {
            std::vector<size_t> indices(shape.size());
            for (int i = shape.size() - 1; i >= 0; --i)
            {
                indices[i] = flat_idx % shape[i];
                flat_idx /= shape[i];
            }
            return indices;
        };

// Parallelize over result elements
#pragma omp parallel for
        for (size_t i = 0; i < total_size; ++i)
        {
            // Convert flat index to multi-dimensional indices for result
            std::vector<size_t> result_indices = flat_to_indices(i, result_shape);

            // Prepare for calculating A and B indices
            std::vector<size_t> a_indices(shape.size());
            std::vector<size_t> b_indices(nd.shape.size());

            // Fill in the non-contracted indices
            size_t a_idx_pos = 0;
            for (size_t dim : a_dims_to_keep)
            {
                a_indices[dim] = result_indices[a_idx_pos++];
            }

            size_t b_idx_pos = a_idx_pos;
            for (size_t dim : b_dims_to_keep)
            {
                b_indices[dim] = result_indices[b_idx_pos++];
            }

            // Calculate base offsets for A and B (without contracted dimension)
            size_t a_base_offset = 0;
            for (size_t dim = 0; dim < shape.size(); ++dim)
            {
                if (dim != contract_axes[0])
                {
                    a_base_offset += a_indices[dim] * strides[dim];
                }
            }

            size_t b_base_offset = 0;
            for (size_t dim = 0; dim < nd.shape.size(); ++dim)
            {
                if (dim != contract_axes[1])
                {
                    b_base_offset += b_indices[dim] * nd.strides[dim];
                }
            }

            // Perform the contraction (dot product along contracted dims)
            data_type sum = 0;

            // Try to use SIMD for the contracted dimension if possible
#ifdef __AVX512F__
            if constexpr (std::is_same_v<data_type, float>)
            {
                const size_t simd_width = 16;
                size_t k_end = (shape[contract_axes[0]] / simd_width) * simd_width;
                __m512 sum_vec = _mm512_setzero_ps();

                // Prepare arrays of offsets for vectorized access
                alignas(64) float a_values[16];
                alignas(64) float b_values[16];

                for (size_t k = 0; k < k_end; k += simd_width)
                {
                    // Gather elements from A and B into contiguous arrays
                    for (size_t j = 0; j < simd_width; ++j)
                    {
                        a_indices[contract_axes[0]] = k + j;
                        b_indices[contract_axes[1]] = k + j;

                        size_t a_offset = a_base_offset + a_strides_for_contract[k + j];
                        size_t b_offset = b_base_offset + b_strides_for_contract[k + j];

                        a_values[j] = data[a_offset];
                        b_values[j] = nd.data[b_offset];
                    }

                    // Load gathered data and multiply
                    __m512 a_vec = _mm512_load_ps(a_values);
                    __m512 b_vec = _mm512_load_ps(b_values);
                    sum_vec = _mm512_fmadd_ps(a_vec, b_vec, sum_vec);
                }

                // Horizontal sum
                sum += _mm512_reduce_add_ps(sum_vec);

                // Handle remaining elements
                for (size_t k = k_end; k < shape[contract_axes[0]]; ++k)
                {
                    a_indices[contract_axes[0]] = k;
                    b_indices[contract_axes[1]] = k;

                    size_t a_offset = a_base_offset + a_strides_for_contract[k];
                    size_t b_offset = b_base_offset + b_strides_for_contract[k];

                    sum += data[a_offset] * nd.data[b_offset];
                }
            }
            else
#endif // __AVX512F__
            {
                // Scalar implementation for other types
                for (size_t k = 0; k < shape[contract_axes[0]]; ++k)
                {
                    a_indices[contract_axes[0]] = k;
                    b_indices[contract_axes[1]] = k;

                    size_t a_offset = a_base_offset + a_strides_for_contract[k];
                    size_t b_offset = b_base_offset + b_strides_for_contract[k];

                    sum += data[a_offset] * nd.data[b_offset];
                }
            }

            result.data[i] = sum;
        }

        return result;
    }
}