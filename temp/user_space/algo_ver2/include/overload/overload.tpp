#pragma once
// operator function
namespace numpy
{
    template <typename data_type>
    data_type &ndarray<data_type>::operator()(const std::vector<size_t> &indices)
    {
        return data[Index(indices)];
    }
    template <typename data_type>
    const data_type &ndarray<data_type>::operator()(const std::vector<size_t> &indices) const
    {
        return data[Index(indices)];
    }
    //================================================================================================//
    //================================================================================================//
    // operator calculate
    template <typename data_type>
    ndarray<data_type> ndarray<data_type>::operator=(const ndarray<data_type> &nd)
    {
        shape = nd.shape;
        strides = nd.strides;
        data = nd.data;
        return *this;
    }
    // operator add
    template <typename data_type>
    ndarray<data_type> ndarray<data_type>::operator+(const ndarray<data_type> &nd)
    {
        assert(nd.data.size() == data.size());
        assert(nd.shape == shape);
        assert(nd.strides == strides);
        ndarray<data_type> answer({shape}); // make index and shape like raw data
        simd_add(data.data(), nd.data.data(), answer.data.data(), data.size());
        return answer;
    }
    template <typename data_type>
    ndarray<data_type> ndarray<data_type>::operator+(const data_type &scalor)
    {
        assert(!shape.empty());
        assert(!strides.empty());
        ndarray<data_type> answer(*this);
        simd_elem_add(answer.data.data(), answer.data.size(), scalor);
        return answer;
    }

    // operator sub
    template <typename data_type>
    ndarray<data_type> ndarray<data_type>::operator-(const ndarray<data_type> &nd)
    {
        assert(nd.data.size() == data.size());
        assert(nd.shape == shape);
        assert(nd.strides == strides);
        ndarray<data_type> answer({shape});
        simd_sub(data.data(), nd.data.data(), answer.data.data(), data.size());
        return answer;
    }
    template <typename data_type>
    ndarray<data_type> ndarray<data_type>::operator-(const data_type &scalor)
    {
        assert(!shape.empty());
        assert(!strides.empty());
        ndarray<data_type> answer(*this);
        simd_elem_sub(answer.data.data(), answer.data.size(), scalor);
        return answer;
    }

    // operator mul
    // missing operator* for matrix_matrix
    template <typename data_type>
    ndarray<data_type> ndarray<data_type>::operator*(const ndarray<data_type> &nd)
    {
        assert(nd.data.size() == data.size());
        assert(nd.shape == shape);
        assert(nd.strides == strides);
        ndarray<data_type> answer({shape});
        simd_mul(data.data(), nd.data.data(), answer.data.data(), data.size());
        return answer;
    }
    template <typename data_type>
    ndarray<data_type> ndarray<data_type>::operator*(const data_type &scalor)
    {
        assert(!shape.empty());
        assert(!strides.empty());
        ndarray<data_type> answer(*this);
        simd_elem_mul(answer.data.data(), answer.data.size(), scalor);
        return answer;
    }

    // operator div
    // missing operator/ for matrix_matrix
    template <typename data_type>
    ndarray<data_type> ndarray<data_type>::operator/(const ndarray<data_type> &nd)
    {
        assert(nd.data.size() == data.size());
        assert(nd.shape == shape);
        assert(nd.strides == strides);
        ndarray<data_type> answer({shape});
        simd_div(data.data(), nd.data.data(), answer.data.data(), data.size());
        return answer;
    }
    template <typename data_type>
    ndarray<data_type> ndarray<data_type>::operator/(const data_type &scalor)
    {
        assert(!shape.empty());
        assert(!strides.empty());
        assert(scalor != 0);
        ndarray<data_type> answer(*this);
        simd_elem_div(answer.data.data(), answer.data.size(), scalor);
        return answer;
    }

    // operator power
    template <typename data_type>
    ndarray<data_type> ndarray<data_type>::operator^(const ndarray<data_type> &nd)
    {
        assert(nd.data.size() == data.size());
        assert(nd.shape == shape);
        assert(nd.strides == strides);
        ndarray<data_type> answer({shape});
        simd_power(data.data(), nd.data.data(), answer.data.data(), data.size());
        return answer;
    }
    template <typename data_type>
    ndarray<data_type> ndarray<data_type>::operator^(const data_type &scalor)
    {
        assert(!shape.empty());
        assert(!strides.empty());
        ndarray<data_type> answer(*this);
        simd_elem_power(answer.data.data(), answer.data.size(), scalor);
        return answer;
    }

    // operator mod
    template <typename data_type>
    ndarray<data_type> ndarray<data_type>::operator%(const ndarray<data_type> &nd)
    {
        assert(nd.data.size() == data.size());
        assert(nd.shape == shape);
        assert(nd.strides == strides);
        ndarray<data_type> answer({shape});
        simd_mod(data.data(), nd.data.data(), answer.data.data(), data.size());
        return answer;
    }

    template <typename data_type>
    ndarray<data_type> ndarray<data_type>::operator%(const data_type &scalor)
    {
        assert(!shape.empty());
        assert(!strides.empty());
        ndarray<data_type> answer(*this);
        simd_elem_mod(answer.data.data(), answer.data.size(), scalor);
        return answer;
    }
    //================================================================================================//
    // boolean operations

    template <typename data_type>
    bool ndarray<data_type>::operator==(const ndarray<data_type> &nd)
    {
        assert(nd.data.size() == data.size());
        assert(nd.shape == shape);
        assert(nd.strides == strides);
        return simd_eq(data.data(), nd.data.data(), data.size());
    }
    template <typename data_type>
    bool ndarray<data_type>::operator==(const data_type &scalor)
    {
        assert(!shape.empty());
        assert(!strides.empty());
        return simd_elem_eq(data.data(), data.size(), scalor);
    }

    template <typename data_type>
    bool ndarray<data_type>::operator!=(const ndarray<data_type> &nd)
    {
        assert(nd.data.size() == data.size());
        assert(nd.shape == shape);
        assert(nd.strides == strides);
        return !simd_eq(data.data(), nd.data.data(), data.size());
    }
    template <typename data_type>
    bool ndarray<data_type>::operator!=(const data_type &scalor)
    {
        assert(!shape.empty());
        assert(!strides.empty());
        return !simd_elem_eq(data.data(), data.size(), scalor);
    }

    template <typename data_type>
    bool ndarray<data_type>::operator<(const ndarray<data_type> &nd)
    {
        assert(nd.data.size() == data.size());
        assert(nd.shape == shape);
        assert(nd.strides == strides);
        return simd_smaller(data.data(), nd.data.data(), data.size());
    }
    template <typename data_type>
    bool ndarray<data_type>::operator<(const data_type &scalor)
    {
        assert(!shape.empty());
        assert(!strides.empty());
        return simd_elem_smaller(data.data(), data.size(), scalor);
    }
    template <typename data_type>
    bool ndarray<data_type>::operator<=(const ndarray<data_type> &nd)
    {
        assert(nd.data.size() == data.size());
        assert(nd.shape == shape);
        assert(nd.strides == strides);
        return !simd_larger(data.data(), nd.data.data(), data.size());
    }
    template <typename data_type>
    bool ndarray<data_type>::operator<=(const data_type &scalor)
    {
        assert(!shape.empty());
        assert(!strides.empty());
        return !simd_elem_larger(data.data(), data.size(), scalor);
    }

    template <typename data_type>
    bool ndarray<data_type>::operator>(const ndarray<data_type> &nd)
    {
        assert(nd.data.size() == data.size());
        assert(nd.shape == shape);
        assert(nd.strides == strides);
        return simd_larger(data.data(), nd.data.data(), data.size());
    }
    template <typename data_type>
    bool ndarray<data_type>::operator>(const data_type &scalor)
    {
        assert(!shape.empty());
        assert(!strides.empty());
        return simd_elem_larger(data.data(), data.size(), scalor);
    }
    template <typename data_type>
    bool ndarray<data_type>::operator>=(const ndarray<data_type> &nd)
    {
        assert(nd.data.size() == data.size());
        assert(nd.shape == shape);
        assert(nd.strides == strides);
        return !simd_smaller(data.data(), nd.data.data(), data.size());
    }
    template <typename data_type>
    bool ndarray<data_type>::operator>=(const data_type &scalor)
    {
        assert(!shape.empty());
        assert(!strides.empty());
        return !simd_elem_smaller(data.data(), data.size(), scalor);
    }
    template <typename data_type>
    ndarray<data_type> ndarray<data_type>::operator+=(const ndarray<data_type> &nd)
    {
        assert(nd.data.size() == data.size());
        assert(nd.shape == shape);
        assert(nd.strides == strides);
        simd_add(data.data(), nd.data.data(), data.data(), data.size());
        return *this;
    }
    template <typename data_type>
    ndarray<data_type> ndarray<data_type>::operator+=(const data_type &scalor)
    {
        assert(!shape.empty());
        assert(!strides.empty());
        simd_elem_add(data.data(), data.size(), scalor);
        return *this;
    }
    template <typename data_type>
    ndarray<data_type> ndarray<data_type>::operator-=(const ndarray<data_type> &nd)
    {
        assert(nd.data.size() == data.size());
        assert(nd.shape == shape);
        assert(nd.strides == strides);
        simd_sub(data.data(), nd.data.data(), data.data(), data.size());
        return *this;
    }
    template <typename data_type>
    ndarray<data_type> ndarray<data_type>::operator-=(const data_type &scalor)
    {
        assert(!shape.empty());
        assert(!strides.empty());
        simd_elem_sub(data.data(), data.size(), scalor);
        return *this;
    }
    template <typename data_type>
    ndarray<data_type> ndarray<data_type>::operator*=(const ndarray<data_type> &nd)
    {
        assert(nd.data.size() == data.size());
        assert(nd.shape == shape);
        assert(nd.strides == strides);
        simd_mul(data.data(), nd.data.data(), data.data(), data.size());
        return *this;
    }
    template <typename data_type>
    ndarray<data_type> ndarray<data_type>::operator*=(const data_type &scalor)
    {
        assert(!shape.empty());
        assert(!strides.empty());
        simd_elem_mul(data.data(), data.size(), scalor);
        return *this;
    }
    template <typename data_type>
    ndarray<data_type> ndarray<data_type>::operator/=(const ndarray<data_type> &nd)
    {
        assert(nd.data.size() == data.size());
        assert(nd.shape == shape);
        assert(nd.strides == strides);
        simd_div(data.data(), nd.data.data(), data.data(), data.size());
        return *this;
    }
    template <typename data_type>
    ndarray<data_type> ndarray<data_type>::operator/=(const data_type &scalor)
    {
        assert(!shape.empty());
        assert(!strides.empty());
        assert(scalor != 0);
        simd_elem_div(data.data(), data.size(), scalor);
        return *this;
    }
    template <typename data_type>
    ndarray<data_type> ndarray<data_type>::operator^=(const ndarray<data_type> &nd)
    {
        assert(nd.data.size() == data.size());
        assert(nd.shape == shape);
        assert(nd.strides == strides);
        simd_power(data.data(), nd.data.data(), data.data(), data.size());
        return *this;
    }
    template <typename data_type>
    ndarray<data_type> ndarray<data_type>::operator^=(const data_type &scalor)
    {
        assert(!shape.empty());
        assert(!strides.empty());
        simd_elem_power(data.data(), data.size(), scalor);
        return *this;
    }
    template <typename data_type>
    ndarray<data_type> ndarray<data_type>::operator%=(const ndarray<data_type> &nd)
    {
        assert(nd.data.size() == data.size());
        assert(nd.shape == shape);
        assert(nd.strides == strides);
        simd_mod(data.data(), nd.data.data(), data.data(), data.size());
        return *this;
    }
    template <typename data_type>
    ndarray<data_type> ndarray<data_type>::operator%=(const data_type &scalor)
    {
        assert(!shape.empty());
        assert(!strides.empty());
        simd_elem_mod(data.data(), data.size(), scalor);
        return *this;
    }
    template <typename data_type>
    ndarray<data_type> &ndarray<data_type>::operator++()
    {
        assert(!shape.empty());
        assert(!strides.empty());
        simd_elem_add(data.data(), data.size(), 1);
        return *this;
    }
    template <typename data_type>
    ndarray<data_type> ndarray<data_type>::operator++(int)
    {
        assert(!shape.empty());
        assert(!strides.empty());
        ndarray<data_type> answer(*this);
        simd_elem_add(answer.data.data(), answer.data.size(), 1);
        return answer;
    }
    template <typename data_type>
    ndarray<data_type> &ndarray<data_type>::operator--()
    {
        assert(!shape.empty());
        assert(!strides.empty());
        simd_elem_sub(data.data(), data.size(), 1);
        return *this;
    }
    template <typename data_type>
    ndarray<data_type> ndarray<data_type>::operator--(int)
    {
        assert(!shape.empty());
        assert(!strides.empty());
        ndarray<data_type> answer(*this);
        simd_elem_sub(answer.data.data(), answer.data.size(), 1);
        return answer;
    }
}
