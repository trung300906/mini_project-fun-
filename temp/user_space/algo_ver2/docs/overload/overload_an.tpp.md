```cpp
#pragma once
template <typename data_type>
std::ostream &operator<<(std::ostream &out, const std::vector<data_type> &shape)
{
    out << "(";
    for (size_t i = 0; i < shape.size(); i++)
    {
        out << shape[i];
        if (i != shape.size() - 1)
        {
            out << ", ";
        }
    }

    out << ")";
    return out;
}
// for vector class boolean in C++
template <typename data_type>
bool operator==(const std::vector<data_type> &a, const std::vector<data_type> &b)
{
    assert(a.size() == b.size()); // conditional check except
    // compare function parameter a and b
    return simd_eq(a.data(), b.data(), a.size());
}

template <typename data_type>
bool operator!=(const std::vector<data_type> &a, const std::vector<data_type> &b)
{
    return !(a == b);
}
template <typename data_type>
bool operator<(const std::vector<data_type> &a, const std::vector<data_type> &b)
{
    assert(a.size() == b.size()); // conditional check except
    // compare function parameter a and b
    return simd_smaller(a.data(), b.data(), a.size());
}

template <typename data_type>
bool operator>(const std::vector<data_type> &a, const std::vector<data_type> &b)
{
    assert(a.size() == b.size()); // conditional check except
    // compare function parameter a and b
    return simd_larger(a.data(), b.data(), a.size());
}

#if 1
// Vector comparison operators remain the same...
namespace numpy
{
    template <typename data_type>
    ndarray<data_type> operator+(const data_type &scalor, const ndarray<data_type> &nd)
    {
        // Create a copy of the input array
        ndarray<data_type> result = nd;

        // Apply scalar operation to the copy
        simd_elem_add(result.data.data(), result.data.size(), scalor);

        return result;
    }

    template <typename data_type>
    ndarray<data_type> operator-(const data_type &scalor, const ndarray<data_type> &nd)
    {
        ndarray<data_type> result = nd;
        simd_elem_sub(scalor, result.data.data(), result.data.size());
        return result;
    }

    template <typename data_type>
    ndarray<data_type> operator*(const data_type &scalor, const ndarray<data_type> &nd)
    {
        ndarray<data_type> result = nd;
        simd_elem_mul(result.data.data(), result.data.size(), scalor);
        return result;
    }

    template <typename data_type>
    ndarray<data_type> operator/(const data_type &scalor, const ndarray<data_type> &nd)
    {
        ndarray<data_type> result = nd;
        simd_elem_div(scalor, result.data.data(), result.data.size());
        return result;
    }

    template <typename data_type>
    ndarray<data_type> operator^(const data_type &scalor, const ndarray<data_type> &nd)
    {
        ndarray<data_type> result = nd;
        simd_elem_power(scalor, result.data.data(), result.data.size());
        return result;
    }

    template <typename data_type>
    ndarray<data_type> operator%(const data_type &scalor, const ndarray<data_type> &nd)
    {
        ndarray<data_type> result = nd;
        simd_elem_mod(scalor, result.data.data(), result.data.size());
        return result;
    }

    // Comparison operators
    template <typename data_type>
    bool operator==(const data_type &scalor, const ndarray<data_type> &nd)
    {
        return simd_elem_eq(nd.data.data(), nd.data.size(), scalor);
    }

    template <typename data_type>
    bool operator!=(const data_type &scalor, const ndarray<data_type> &nd)
    {
        return !simd_elem_eq(nd.data.data(), nd.data.size(), scalor);
    }

    template <typename data_type>
    bool operator<(const data_type &scalor, const ndarray<data_type> &nd)
    {
        return simd_elem_smaller(scalor, nd.data.data(), nd.data.size());
    }

    template <typename data_type>
    bool operator<=(const data_type &scalor, const ndarray<data_type> &nd)
    {
        return !simd_elem_larger(scalor, nd.data.data(), nd.data.size());
    }

    template <typename data_type>
    bool operator>(const data_type &scalor, const ndarray<data_type> &nd)
    {
        return simd_elem_larger(scalor, nd.data.data(), nd.data.size());
    }

    template <typename data_type>
    bool operator>=(const data_type &scalor, const ndarray<data_type> &nd)
    {
        return !simd_elem_smaller(scalor, nd.data.data(), nd.data.size());
    }
}
#endif

```

include into [[overload/index]]
