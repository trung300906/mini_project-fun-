#pragma once
template <typename data_type>
size_t size(ndarray<data_type> &nd, size_t axis)
{
    if (axis >= nd.get_shape().size())
    {
        throw std::out_of_range("Axis out of range");
    }
    return nd.get_shape()[axis];
}

// e^matrix
template <typename data_type>
ndarray<data_type> exp(ndarray<data_type> &nd)
{
    assert(!nd.get_shape().empty());
    assert(!nd.get_strides().empty());
    assert(nd.get_data().empty());
    const size_t size = nd.size();
    if constexpr (std::is_same_v<data_type, float>)
    {
        const float exp = 2.718281828459045235360287471352662497757247093699959574966;
        simd_elem_power(exp, nd.get_data().data(), size);
        return nd;
    }
    else if constexpr (std::is_same_v<data_type, double>)
    {
        const double exp = 2.718281828459045235360287471352662497757247093699959574966;
        simd_elem_power(exp, nd.get_data().data(), size);
        return nd;
    }
    else if constexpr (std::is_integral_v<data_type>)
    {
        if constexpr (sizeof(data_type) == 1)
        {
            const float exp = 2.718281828459045235360287471352662497757247093699959574966;
            simd_cast<int, float>(nd.get_data().data(), size);
            simd_elem_power(exp, nd.get_data().data(), size);
            return nd;
        }
        else if constexpr (sizeof(data_type) == 2)
        {
            const double exp = 2.718281828459045235360287471352662497757247093699959574966;
            simd_cast<int, double>(nd.get_data().data(), size);
            simd_elem_power(exp, nd.get_data().data(), size);
            return nd;
        }
        else if constexpr (sizeof(data_type) == 4)
        {
            const float exp = 2.718281828459045235360287471352662497757247093699959574966;
            simd_cast<int, float>(nd.get_data().data(), size);
            simd_elem_power(exp, nd.get_data().data(), size);
            return nd;
        }
        else if constexpr (sizeof(data_type) == 8)
        {
            const double exp = 2.718281828459045235360287471352662497757247093699959574966;
            simd_cast<int, double>(nd.get_data().data(), size);
            simd_elem_power(exp, nd.get_data().data(), size);
            return nd;
        }
    }
    return nd;
}
#if 1
// for all log function
//(element_wise)
template <typename data_type>
ndarray<data_type> log(const data_type &scalor, ndarray<data_type> &nd){
    assert(!nd.get_shape().empty());
    assert(!nd.get_strides().empty());
    assert(nd.get_data().empty());
    const size_t size = nd.size();
    if constexpr (std::is_same_v<data_type, float>)
    {
        simd_elem_log(scalor, nd.get_data().data(), size);
        return nd;
    }
    else if constexpr (std::is_same_v<data_type, double>)
    {
        simd_elem_log(scalor, nd.get_data().data(), size);
        return nd;
    }
    else if constexpr (std::is_integral_v<data_type>)
    {
        if constexpr (sizeof(data_type) == 1)
        {
            simd_cast<int, float>(nd.get_data().data(), size);
            simd_elem_log(scalor, nd.get_data().data(), size);
            return nd;
        }
        else if constexpr (sizeof(data_type) == 2)
        {   
            simd_cast<int, double>(nd.get_data().data(), size);
            simd_elem_log(scalor, nd.get_data().data(), size);
            return nd;
        }
        else if constexpr (sizeof(data_type) == 4)
        {
            simd_cast<int, float>(nd.get_data().data(), size);
            simd_elem_log(scalor, nd.get_data().data(), size);
            return nd;
        }
        else if constexpr (sizeof(data_type) == 8)
        {
            simd_cast<int, double>(nd.get_data().data(), size);
            simd_elem_log(scalor, nd.get_data().data(), size);
            return nd;
        }
    }
    return nd;
}

template <typename data_type>
ndarray<data_type> log10(ndarray<data_type> &nd){
    assert(!nd.get_shape().empty());
    assert(!nd.get_strides().empty());
    assert(nd.get_data().empty());
    const size_t size = nd.size();
    if constexpr (std::is_same_v<data_type, float>)
    {
        simd_elem_log(10.0, nd.get_data().data(), size);
        return nd;
    }
    else if constexpr (std::is_same_v<data_type, double>)
    {
        simd_elem_log(10.0, nd.get_data().data(), size);
        return nd;
    }
    else if constexpr (std::is_integral_v<data_type>)
    {
        if constexpr (sizeof(data_type) == 1)
        {
            simd_cast<int, float>(nd.get_data().data(), size);
            simd_elem_log(10.0, nd.get_data().data(), size);
            return nd;
        }
        else if constexpr (sizeof(data_type) == 2)
        {   
            simd_cast<int, double>(nd.get_data().data(), size);
            simd_elem_log(10.0, nd.get_data().data(), size);
            return nd;
        }
        else if constexpr (sizeof(data_type) == 4)
        {
            simd_cast<int, float>(nd.get_data().data(), size);
            simd_elem_log(10.0, nd.get_data().data(), size);
            return nd;
        }
        else if constexpr (sizeof(data_type) == 8)
        {
            simd_cast<int, double>(nd.get_data().data(), size);
            simd_elem_log(10.0, nd.get_data().data(), size);
            return nd;
        }
    }
    return nd;
}

template <typename data_type>
ndarray<data_type> log2(ndarray<data_type> &nd){
    assert(!nd.get_shape().empty());
    assert(!nd.get_strides().empty());
    assert(nd.get_data().empty());
    const size_t size = nd.size();
    if constexpr (std::is_same_v<data_type, float>)
    {
        simd_elem_log(2.0, nd.get_data().data(), size);
        return nd;
    }
    else if constexpr (std::is_same_v<data_type, double>)
    {
        simd_elem_log(2.0, nd.get_data().data(), size);
        return nd;
    }
    else if constexpr (std::is_integral_v<data_type>)
    {
        if constexpr (sizeof(data_type) == 1)
        {
            simd_cast<int, float>(nd.get_data().data(), size);
            simd_elem_log(2.0, nd.get_data().data(), size);
            return nd;
        }
        else if constexpr (sizeof(data_type) == 2)
        {   
            simd_cast<int, double>(nd.get_data().data(), size);
            simd_elem_log(2.0, nd.get_data().data(), size);
            return nd;
        }
        else if constexpr (sizeof(data_type) == 4)
        {
            simd_cast<int, float>(nd.get_data().data(), size);
            simd_elem_log(2.0, nd.get_data().data(), size);
            return nd;
        }
        else if constexpr (sizeof(data_type) == 8)
        {
            simd_cast<int, double>(nd.get_data().data(), size);
            simd_elem_log(2.0, nd.get_data().data(), size);
            return nd;
        }
    }
    return nd;
}

template <typename data_type>
ndarray<data_type> ln(ndarray<data_type> &nd){
    assert(!nd.get_shape().empty());
    assert(!nd.get_strides().empty());
    assert(nd.get_data().empty());
    const size_t size = nd.size();
    const exp = 2.718281828459045235360287471352662497757247093699959574966;
    if constexpr (std::is_same_v<data_type, float>)
    {
        simd_elem_log(scalor, nd.get_data().data(), size);
        return nd;
    }
    else if constexpr (std::is_same_v<data_type, double>)
    {
        simd_elem_log(scalor, nd.get_data().data(), size);
        return nd;
    }
    else if constexpr (std::is_integral_v<data_type>)
    {
        if constexpr (sizeof(data_type) == 1)
        {
            simd_cast<int, float>(nd.get_data().data(), size);
            simd_elem_log(scalor, nd.get_data().data(), size);
            return nd;
        }
        else if constexpr (sizeof(data_type) == 2)
        {   
            simd_cast<int, double>(nd.get_data().data(), size);
            simd_elem_log(scalor, nd.get_data().data(), size);
            return nd;
        }
        else if constexpr (sizeof(data_type) == 4)
        {
            simd_cast<int, float>(nd.get_data().data(), size);
            simd_elem_log(scalor, nd.get_data().data(), size);
            return nd;
        }
        else if constexpr (sizeof(data_type) == 8)
        {
            simd_cast<int, double>(nd.get_data().data(), size);
            simd_elem_log(scalor, nd.get_data().data(), size);
            return nd;
        }
    }
    return nd;
}

// matrix log matrix
template <typename data_type>
ndarray<data_type> log(const ndarray<data_type> &first, const ndarray<data_type> &second){
    assert(!first.get_shape().empty());
    assert(!first.get_strides().empty());
    assert(!first.get_data().empty());
    assert(!second.get_shape().empty());
    assert(!second.get_strides().empty());
    assert(!second.get_data().empty());
    ndarray<data_type> answer({first.get_shape()});
    if constexpr (is_integral_v<data_type)
    {
        if constexpr (sizeof(data_type) == 1)
        {
            simd_cast<int, float>(first.get_data().data(), first.size());
            simd_cast<int, float>(second.get_data().data(), second.size());
            simd_log(first.get_data().data(), second.get_data().data(), answer.get_data().data(), first.size());
            return answer;
        }
        else if constexpr (sizeof(data_type) == 2)
        {
            simd_cast<int, double>(first.get_data().data(), first.size());
            simd_cast<int, double>(second.get_data().data(), second.size());
            simd_log(first.get_data().data(), second.get_data().data(), answer.get_data().data(), first.size());
            return answer;
        } 
        else if constexpr (sizeof(data_type) == 4)
        {
            simd_cast<int, float>(first.get_data().data(), first.size());
            simd_cast<int, float>(second.get_data().data(), second.size());
            simd_log(first.get_data().data(), second.get_data().data(), answer.get_data().data(), first.size());
            return answer;
        }
        else if constexpr (sizeof(data_type) == 8)
        {
            simd_cast<int, double>(first.get_data().data(), first.size());
            simd_cast<int, double>(second.get_data().data(), second.size());
            simd_log(first.get_data().data(), second.get_data().data(), answer.get_data().data(), first.size());
            return answer;
        }
    } else if constexpr (std::is_same_v<data_type, float>)
    {
        simd_log(first.get_data().data(), second.get_data().data(), answer.get_data().data(), first.size());
        return answer;
    }
    else if constexpr (std::is_same_v<data_type, double>)
    {
        simd_log(first.get_data().data(), second.get_data().data(), answer.get_data().data(), first.size());
        return answer;
    }
    return answer;
}

// for sin
template <typename data_type>
ndarray<data_type> sin(ndarray<data_type> &nd){
    assert(!nd.get_shape().empty());
    assert(!nd.get_strides().empty());
    assert(nd.get_data().empty());
    if constexpr(std::is_same_v<data_type, float>){
        simd_sin(nd.get_data().data(), nd.size());
        return nd;
    } else if constexpr(std::is_same_v<data_type, double>){
        simd_sin(nd.get_data().data(), nd.size());
        return nd;
    } else if constexpr(std::is_integral_v<data_type>){
        if constexpr (sizeof(data_type) == 1) // 8 bit
        {
            simd_cast<int, float>(nd.get_data().data(), nd.size());
            simd_sin(nd.get_data().data(), nd.size());
            return nd;
        }
        else if constexpr (sizeof(data_type) == 2) // 16
        {
            simd_cast<int, double>(nd.get_data().data(), nd.size());
            simd_sin(nd.get_data().data(), nd.size());
            return nd;
        }
        else if constexpr (sizeof(data_type) == 4) // 32
        {
            simd_cast<int, float>(nd.get_data().data(), nd.size());
            simd_sin(nd.get_data().data(), nd.size());
            return nd;
        }
        else if constexpr (sizeof(data_type) == 8) // 64
        {
            simd_cast<int, double>(nd.get_data().data(), nd.size());
            simd_sin(nd.get_data().data(), nd.size());
            return nd;
        }
    }
    return nd;
}

// for cos
template <typename data_type>
ndarray<data_type> cos(ndarray<data_type> &nd){
    assert(!nd.get_shape().empty());
    assert(!nd.get_strides().empty());
    assert(nd.get_data().empty());
    if constexpr(std::is_same_v<data_type, float>){
        simd_cos(nd.get_data().data(), nd.size());
        return nd;
    } else if constexpr(std::is_same_v<data_type, double>){
        simd_cos(nd.get_data().data(), nd.size());
        return nd;
    } else if constexpr(std::is_integral_v<data_type>){
        if constexpr (sizeof(data_type) == 1) // 8 bit
        {
            simd_cast<int, float>(nd.get_data().data(), nd.size());
            simd_cos(nd.get_data().data(), nd.size());
            return nd;
        }
        else if constexpr (sizeof(data_type) == 2) // 16
        {
            simd_cast<int, double>(nd.get_data().data(), nd.size());
            simd_cos(nd.get_data().data(), nd.size());
            return nd;
        }
        else if constexpr (sizeof(data_type) == 4) // 32
        {
            simd_cast<int, float>(nd.get_data().data(), nd.size());
            simd_cos(nd.get_data().data(), nd.size());
            return nd;
        }
        else if constexpr (sizeof(data_type) == 8) // 64
        {
            simd_cast<int, double>(nd.get_data().data(), nd.size());
            simd_cos(nd.get_data().data(), nd.size());
            return nd;
        }
    }
    return nd;
}

// for tan
template <typename data_type>
ndarray<data_type> tan(ndarray<data_type> &nd){
    assert(!nd.get_shape().empty());
    assert(!nd.get_strides().empty());
    assert(nd.get_data().empty());
    if constexpr(std::is_same_v<data_type, float>){
        simd_tan(nd.get_data().data(), nd.size());
        return nd;
    } else if constexpr(std::is_same_v<data_type, double>){
        simd_tan(nd.get_data().data(), nd.size());
        return nd;
    } else if constexpr(std::is_integral_v<data_type>){
        if constexpr (sizeof(data_type) == 1) // 8 bit
        {
            simd_cast<int, float>(nd.get_data().data(), nd.size());
            simd_tan(nd.get_data().data(), nd.size());
            return nd;
        }
        else if constexpr (sizeof(data_type) == 2) // 16
        {
            simd_cast<int, double>(nd.get_data().data(), nd.size());
            simd_tan(nd.get_data().data(), nd.size());
            return nd;
        }
        else if constexpr (sizeof(data_type) == 4) // 32
        {
            simd_cast<int, float>(nd.get_data().data(), nd.size());
            simd_tan(nd.get_data().data(), nd.size());
            return nd;
        }
        else if constexpr (sizeof(data_type) == 8) // 64
        {
            simd_cast<int, double>(nd.get_data().data(), nd.size());
            simd_tan(nd.get_data().data(), nd.size());
            return nd;
        }
    }
    return nd;
}

// for cotan
template <typename data_type>
ndarray<data_type> cotan(ndarray<data_type> &nd){
    assert(!nd.get_shape().empty());
    assert(!nd.get_strides().empty());
    assert(nd.get_data().empty());
    if constexpr(std::is_same_v<data_type, float>){
        simd_cotan(nd.get_data().data(), nd.size());
        return nd;
    } else if constexpr(std::is_same_v<data_type, double>){
        simd_cotan(nd.get_data().data(), nd.size());
        return nd;
    } else if constexpr(std::is_integral_v<data_type>){
        if constexpr (sizeof(data_type) == 1) // 8 bit
        {
            simd_cast<int, float>(nd.get_data().data(), nd.size());
            simd_cotan(nd.get_data().data(), nd.size());
            return nd;
        }
        else if constexpr (sizeof(data_type) == 2) // 16
        {
            simd_cast<int, double>(nd.get_data().data(), nd.size());
            simd_cotan(nd.get_data().data(), nd.size());
            return nd;
        }
        else if constexpr (sizeof(data_type) == 4) // 32
        {
            simd_cast<int, float>(nd.get_data().data(), nd.size());
            simd_cotan(nd.get_data().data(), nd.size());
            return nd;
        }
        else if constexpr (sizeof(data_type) == 8) // 64
        {
            simd_cast<int, double>(nd.get_data().data(), nd.size());
            simd_cotan(nd.get_data().data(), nd.size());
            return nd;
        }
    }
    return nd;
}

// for sinh
template <typename data_type>
ndarray<data_type> sinh(ndarray<data_type> &nd){
    assert(!nd.get_shape().empty());
    assert(!nd.get_strides().empty());
    assert(nd.get_data().empty());
    if constexpr(std::is_same_v<data_type, float>){
        simd_sinh(nd.get_data().data(), nd.size());
        return nd;
    } else if constexpr(std::is_same_v<data_type, double>){
        simd_sinh(nd.get_data().data(), nd.size());
        return nd;
    } else if constexpr(std::is_integral_v<data_type>){
        if constexpr (sizeof(data_type) == 1) // 8 bit
        {
            simd_cast<int, float>(nd.get_data().data(), nd.size());
            simd_sinh(nd.get_data().data(), nd.size());
            return nd;
        }
        else if constexpr (sizeof(data_type) == 2) // 16
        {
            simd_cast<int, double>(nd.get_data().data(), nd.size());
            simd_sinh(nd.get_data().data(), nd.size());
            return nd;
        }
        else if constexpr (sizeof(data_type) == 4) // 32
        {
            simd_cast<int, float>(nd.get_data().data(), nd.size());
            simd_sinh(nd.get_data().data(), nd.size());
            return nd;
        }
        else if constexpr (sizeof(data_type) == 8) // 64
        {
            simd_cast<int, double>(nd.get_data().data(), nd.size());
            simd_sinh(nd.get_data().data(), nd.size());
            return nd;
        }
    }
    return nd;
}

// for cosh
template <typename data_type>
ndarray<data_type> cosh(ndarray<data_type> &nd){
    assert(!nd.get_shape().empty());
    assert(!nd.get_strides().empty());
    assert(nd.get_data().empty());
    if constexpr(std::is_same_v<data_type, float>){
        simd_cosh(nd.get_data().data(), nd.size());
        return nd;
    } else if constexpr(std::is_same_v<data_type, double>){
        simd_cosh(nd.get_data().data(), nd.size());
        return nd;
    } else if constexpr(std::is_integral_v<data_type>){
        if constexpr (sizeof(data_type) == 1) // 8 bit
        {
            simd_cast<int, float>(nd.get_data().data(), nd.size());
            simd_cosh(nd.get_data().data(), nd.size());
            return nd;
        }
        else if constexpr (sizeof(data_type) == 2) // 16
        {
            simd_cast<int, double>(nd.get_data().data(), nd.size());
            simd_cosh(nd.get_data().data(), nd.size());
            return nd;
        }
        else if constexpr (sizeof(data_type) == 4) // 32
        {
            simd_cast<int, float>(nd.get_data().data(), nd.size());
            simd_cosh(nd.get_data().data(), nd.size());
            return nd;
        }
        else if constexpr (sizeof(data_type) == 8) // 64
        {
            simd_cast<int, double>(nd.get_data().data(), nd.size());
            simd_cosh(nd.get_data().data(), nd.size());
            return nd;
        }
    }
    return nd;
}

// for tanh
template <typename data_type>
ndarray<data_type> tanh(ndarray<data_type> &nd) {
    assert(!nd.get_shape().empty());
    assert(!nd.get_strides().empty());
    assert(nd.get_data().empty());
    if constexpr(std::is_same_v<data_type, float>){
        simd_tanh(nd.get_data().data(), nd.size());
        return nd;
    } else if constexpr(std::is_same_v<data_type, double>){
        simd_tanh(nd.get_data().data(), nd.size());
        return nd;
    } else if constexpr(std::is_integral_v<data_type>){
        if constexpr (sizeof(data_type) == 1) // 8 bit
        {
            simd_cast<int, float>(nd.get_data().data(), nd.size());
            simd_tanh(nd.get_data().data(), nd.size());
            return nd;
        }
        else if constexpr (sizeof(data_type) == 2) // 16
        {
            simd_cast<int, double>(nd.get_data().data(), nd.size());
            simd_tanh(nd.get_data().data(), nd.size());
            return nd;
        }
        else if constexpr (sizeof(data_type) == 4) // 32
        {
            simd_cast<int, float>(nd.get_data().data(), nd.size());
            simd_tanh(nd.get_data().data(), nd.size());
            return nd;
        }
        else if constexpr (sizeof(data_type) == 8) // 64
        {
            simd_cast<int, double>(nd.get_data().data(), nd.size());
            simd_tanh(nd.get_data().data(), nd.size());
            return nd;
        }
    }
    return nd;
}

// for arcsin
template <typename data_type>
ndarray<data_type> arcsin(ndarray<data_type> &nd){
    assert(!nd.get_shape().empty());
    assert(!nd.get_strides().empty());
    assert(nd.get_data().empty());
    if constexpr(std::is_same_v<data_type, float>){
        simd_arcsin(nd.get_data().data(), nd.size());
        return nd;
    } else if constexpr(std::is_same_v<data_type, double>){
        simd_arcsin(nd.get_data().data(), nd.size());
        return nd;
    } else if constexpr(std::is_integral_v<data_type>){
        if constexpr (sizeof(data_type) == 1) // 8 bit
        {
            simd_cast<int, float>(nd.get_data().data(), nd.size());
            simd_arcsin(nd.get_data().data(), nd.size());
            return nd;
        }
        else if constexpr (sizeof(data_type) == 2) // 16
        {
            simd_cast<int, double>(nd.get_data().data(), nd.size());
            simd_arcsin(nd.get_data().data(), nd.size());
            return nd;
        }
        else if constexpr (sizeof(data_type) == 4) // 32
        {
            simd_cast<int, float>(nd.get_data().data(), nd.size());
            simd_arcsin(nd.get_data().data(), nd.size());
            return nd;
        }
        else if constexpr (sizeof(data_type) == 8) // 64
        {
            simd_cast<int, double>(nd.get_data().data(), nd.size());
            simd_arcsin(nd.get_data().data(), nd.size());
            return nd;
        }
    }
    return nd;
}

// for arccos
template <typename data_type>
ndarray<data_type> arccos(ndarray<data_type> &nd){
    assert(!nd.get_shape().empty());
    assert(!nd.get_strides().empty());
    assert(nd.get_data().empty());
    if constexpr(std::is_same_v<data_type, float>){
        simd_arccos(nd.get_data().data(), nd.size());
        return nd;
    } else if constexpr(std::is_same_v<data_type, double>){
        simd_arccos(nd.get_data().data(), nd.size());
        return nd;
    } else if constexpr(std::is_integral_v<data_type>){
        if constexpr (sizeof(data_type) == 1) // 8 bit
        {
            simd_cast<int, float>(nd.get_data().data(), nd.size());
            simd_arccos(nd.get_data().data(), nd.size());
            return nd;
        }
        else if constexpr (sizeof(data_type) == 2) // 16
        {
            simd_cast<int, double>(nd.get_data().data(), nd.size());
            simd_arccos(nd.get_data().data(), nd.size());
            return nd;
        }
        else if constexpr (sizeof(data_type) == 4) // 32
        {
            simd_cast<int, float>(nd.get_data().data(), nd.size());
            simd_arccos(nd.get_data().data(), nd.size());
            return nd;
        }
        else if constexpr (sizeof(data_type) == 8) // 64
        {
            simd_cast<int, double>(nd.get_data().data(), nd.size());
            simd_arccos(nd.get_data().data(), nd.size());
            return nd;
        }
    }
    return nd;
}

// for arctan
template <typename data_type>
ndarray<data_type> arctan(ndarray<data_type> &nd){
    assert(!nd.get_shape().empty());
    assert(!nd.get_strides().empty());
    assert(nd.get_data().empty());
    if constexpr(std::is_same_v<data_type, float>){
        simd_arctan(nd.get_data().data(), nd.size());
        return nd;
    } else if constexpr(std::is_same_v<data_type, double>){
        simd_arctan(nd.get_data().data(), nd.size());
        return nd;
    } else if constexpr(std::is_integral_v<data_type>){
        if constexpr (sizeof(data_type) == 1) // 8 bit
        {
            simd_cast<int, float>(nd.get_data().data(), nd.size());
            simd_arctan(nd.get_data().data(), nd.size());
            return nd;
        }
        else if constexpr (sizeof(data_type) == 2) // 16
        {
            simd_cast<int, double>(nd.get_data().data(), nd.size());
            simd_arctan(nd.get_data().data(), nd.size());
            return nd;
        }
        else if constexpr (sizeof(data_type) == 4) // 32
        {
            simd_cast<int, float>(nd.get_data().data(), nd.size());
            simd_arctan(nd.get_data().data(), nd.size());
            return nd;
        }
        else if constexpr (sizeof(data_type) == 8) // 64
        {
            simd_cast<int, double>(nd.get_data().data(), nd.size());
            simd_arctan(nd.get_data().data(), nd.size());
            return nd;
        }
    }
    return nd;
}

// for arcsinh
template <typename data_type>
ndarray<data_type> arcsinh(ndarray<data_type> &nd){
    assert(!nd.get_shape().empty());
    assert(!nd.get_strides().empty());
    assert(nd.get_data().empty());
    if constexpr(std::is_same_v<data_type, float>){
        simd_arcsinh(nd.get_data().data(), nd.size());
        return nd;
    } else if constexpr(std::is_same_v<data_type, double>){
        simd_arcsinh(nd.get_data().data(), nd.size());
        return nd;
    } else if constexpr(std::is_integral_v<data_type>){
        if constexpr (sizeof(data_type) == 1) // 8 bit
        {
            simd_cast<int, float>(nd.get_data().data(), nd.size());
            simd_arcsinh(nd.get_data().data(), nd.size());
            return nd;
        }
        else if constexpr (sizeof(data_type) == 2) // 16
        {
            simd_cast<int, double>(nd.get_data().data(), nd.size());
            simd_arcsinh(nd.get_data().data(), nd.size());
            return nd;
        }
        else if constexpr (sizeof(data_type) == 4) // 32
        {
            simd_cast<int, float>(nd.get_data().data(), nd.size());
            simd_arcsinh(nd.get_data().data(), nd.size());
            return nd;
        }
        else if constexpr (sizeof(data_type) == 8) // 64
        {
            simd_cast<int, double>(nd.get_data().data(), nd.size());
            simd_arcsinh(nd.get_data().data(), nd.size());
            return nd;
        }
    }
    return nd;
}

// for arccosh
template <typename data_type>
ndarray<data_type> arccosh(ndarray<data_type> &nd){
    assert(!nd.get_shape().empty());
    assert(!nd.get_strides().empty());
    assert(nd.get_data().empty());
    if constexpr(std::is_same_v<data_type, float>){
        simd_arccosh(nd.get_data().data(), nd.size());
        return nd;
    } else if constexpr(std::is_same_v<data_type, double>){
        simd_arccosh(nd.get_data().data(), nd.size());
        return nd;
    } else if constexpr(std::is_integral_v<data_type>){
        if constexpr (sizeof(data_type) == 1) // 8 bit
        {
            simd_cast<int, float>(nd.get_data().data(), nd.size());
            simd_arccosh(nd.get_data().data(), nd.size());
            return nd;
        }
        else if constexpr (sizeof(data_type) == 2) // 16
        {
            simd_cast<int, double>(nd.get_data().data(), nd.size());
            simd_arccosh(nd.get_data().data(), nd.size());
            return nd;
        }
        else if constexpr (sizeof(data_type) == 4) // 32
        {
            simd_cast<int, float>(nd.get_data().data(), nd.size());
            simd_arccosh(nd.get_data().data(), nd.size());
            return nd;
        }
        else if constexpr (sizeof(data_type) == 8) // 64
        {
            simd_cast<int, double>(nd.get_data().data(), nd.size());
            simd_arccosh(nd.get_data().data(), nd.size());
            return nd;
        }
    }
    return nd;
}

// for arctanh
template <typename data_type>
ndarray<data_type> arctanh(ndarray<data_type> &nd){
    assert(!nd.get_shape().empty());
    assert(!nd.get_strides().empty());
    assert(nd.get_data().empty());
    if constexpr(std::is_same_v<data_type, float>){
        simd_arctanh(nd.get_data().data(), nd.size());
        return nd;
    } else if constexpr(std::is_same_v<data_type, double>){
        simd_arctanh(nd.get_data().data(), nd.size());
        return nd;
    } else if constexpr(std::is_integral_v<data_type>){
        if constexpr (sizeof(data_type) == 1) // 8 bit
        {
            simd_cast<int, float>(nd.get_data().data(), nd.size());
            simd_arctanh(nd.get_data().data(), nd.size());
            return nd;
        }
        else if constexpr (sizeof(data_type) == 2) // 16
        {
            simd_cast<int, double>(nd.get_data().data(), nd.size());
            simd_arctanh(nd.get_data().data(), nd.size());
            return nd;
        }
        else if constexpr (sizeof(data_type) == 4) // 32
        {
            simd_cast<int, float>(nd.get_data().data(), nd.size());
            simd_arctanh(nd.get_data().data(), nd.size());
            return nd;
        }
        else if constexpr (sizeof(data_type) == 8) // 64
        {
            simd_cast<int, double>(nd.get_data().data(), nd.size());
            simd_arctanh(nd.get_data().data(), nd.size());
            return nd;
        }
    }
    return nd;
}

// for sec
template <typename data_type>
ndarray<data_type> sec(ndarray<data_type> &nd){
    assert(!nd.get_shape().empty());
    assert(!nd.get_strides().empty());
    assert(nd.get_data().empty());
    if constexpr(std::is_same_v<data_type, float>){
        simd_sec(nd.get_data().data(), nd.size());
        return nd;
    } else if constexpr(std::is_same_v<data_type, double>){
        simd_sec(nd.get_data().data(), nd.size());
        return nd;
    } else if constexpr(std::is_integral_v<data_type>){
        if constexpr (sizeof(data_type) == 1) // 8 bit
        {
            simd_cast<int, float>(nd.get_data().data(), nd.size());
            simd_sec(nd.get_data().data(), nd.size());
            return nd;
        }
        else if constexpr (sizeof(data_type) == 2) // 16
        {
            simd_cast<int, double>(nd.get_data().data(), nd.size());
            simd_sec(nd.get_data().data(), nd.size());
            return nd;
        }
        else if constexpr (sizeof(data_type) == 4) // 32
        {
            simd_cast<int, float>(nd.get_data().data(), nd.size());
            simd_sec(nd.get_data().data(), nd.size());
            return nd;
        }
        else if constexpr (sizeof(data_type) == 8) // 64
        {
            simd_cast<int, double>(nd.get_data().data(), nd.size());
            simd_sec(nd.get_data().data(), nd.size());
            return nd;
        }
    }
    return nd;
}

// for cosec
template <typename data_type>
ndarray<data_type> cosec(ndarray<data_type> &nd){
    assert(!nd.get_shape().empty());
    assert(!nd.get_strides().empty());
    assert(nd.get_data().empty());
    if constexpr(std::is_same_v<data_type, float>){
        simd_cosec(nd.get_data().data(), nd.size());
        return nd;
    } else if constexpr(std::is_same_v<data_type, double>){
        simd_cosec(nd.get_data().data(), nd.size());
        return nd;
    } else if constexpr(std::is_integral_v<data_type>){
        if constexpr (sizeof(data_type) == 1) // 8 bit
        {
            simd_cast<int, float>(nd.get_data().data(), nd.size());
            simd_cosec(nd.get_data().data(), nd.size());
            return nd;
        }
        else if constexpr (sizeof(data_type) == 2) // 16
        {
            simd_cast<int, double>(nd.get_data().data(), nd.size());
            simd_cosec(nd.get_data().data(), nd.size());
            return nd;
        }
        else if constexpr (sizeof(data_type) == 4) // 32
        {
            simd_cast<int, float>(nd.get_data().data(), nd.size());
            simd_cosec(nd.get_data().data(), nd.size());
            return nd;
        }
        else if constexpr (sizeof(data_type) == 8) // 64
        {
            simd_cast<int, double>(nd.get_data().data(), nd.size());
            simd_cosec(nd.get_data().data(), nd.size());
            return nd;
        }
    }
    return nd;
}

// for arcsec
template <typename data_type>
ndarray<data_type> arcsec(ndarray<data_type> &nd){
    assert(!nd.get_shape().empty());
    assert(!nd.get_strides().empty());
    assert(nd.get_data().empty());
    if constexpr(std::is_same_v<data_type, float>){
        simd_arcsec(nd.get_data().data(), nd.size());
        return nd;
    } else if constexpr(std::is_same_v<data_type, double>){
        simd_arcsec(nd.get_data().data(), nd.size());
        return nd;
    } else if constexpr(std::is_integral_v<data_type>){
        if constexpr (sizeof(data_type) == 1) // 8 bit
        {
            simd_cast<int, float>(nd.get_data().data(), nd.size());
            simd_arcsec(nd.get_data().data(), nd.size());
            return nd;
        }
        else if constexpr (sizeof(data_type) == 2) // 16
        {
            simd_cast<int, double>(nd.get_data().data(), nd.size());
            simd_arcsec(nd.get_data().data(), nd.size());
            return nd;
        }
        else if constexpr (sizeof(data_type) == 4) // 32
        {
            simd_cast<int, float>(nd.get_data().data(), nd.size());
            simd_arcsec(nd.get_data().data(), nd.size());
            return nd;
        }
        else if constexpr (sizeof(data_type) == 8) // 64
        {
            simd_cast<int, double>(nd.get_data().data(), nd.size());
            simd_arcsec(nd.get_data().data(), nd.size());
            return nd;
        }
    }
    return nd;
}

// for arccosec
template <typename data_type>
ndarray<data_type> arccosec(ndarray<data_type> &nd){
    assert(!nd.get_shape().empty());
    assert(!nd.get_strides().empty());
    assert(nd.get_data().empty());
    if constexpr(std::is_same_v<data_type, float>){
        simd_arccosec(nd.get_data().data(), nd.size());
        return nd;
    } else if constexpr(std::is_same_v<data_type, double>){
        simd_arccosec(nd.get_data().data(), nd.size());
        return nd;
    } else if constexpr(std::is_integral_v<data_type>){
        if constexpr (sizeof(data_type) == 1) // 8 bit
        {
            simd_cast<int, float>(nd.get_data().data(), nd.size());
            simd_arccosec(nd.get_data().data(), nd.size());
            return nd;
        }
        else if constexpr (sizeof(data_type) == 2) // 16
        {
            simd_cast<int, double>(nd.get_data().data(), nd.size());
            simd_arccosec(nd.get_data().data(), nd.size());
            return nd;
        }
        else if constexpr (sizeof(data_type) == 4) // 32
        {
            simd_cast<int, float>(nd.get_data().data(), nd.size());
            simd_arccosec(nd.get_data().data(), nd.size());
            return nd;
        }
        else if constexpr (sizeof(data_type) == 8) // 64
        {
            simd_cast<int, double>(nd.get_data().data(), nd.size());
            simd_arccosec(nd.get_data().data(), nd.size());
            return nd;
        }
    }
    return nd;
}

// for arccotan
template <typename data_type>
ndarray<data_type> arccotan(ndarray<data_type> &nd);

// for sqrt
template <typename data_type>
ndarray<data_type> sqrt(ndarray<data_type> &nd){
    assert(!nd.get_shape().empty());
    assert(!nd.get_strides().empty());
    assert(nd.get_data().empty());
    const size_t size = nd.size();
    const data_type scalor = 2.0;
    if constexpr (std::is_same_v<data_type, float>)
    {
        simd_elem_sqrt(scalor ,nd.get_data().data(), size);
        return nd;
    }
    else if constexpr (std::is_same_v<data_type, double>)
    {
        simd_elem_sqrt(scalor ,nd.get_data().data(), size);
        return nd;
    }
    else if constexpr (std::is_integral_v<data_type>)
    {
        if constexpr (sizeof(data_type) == 1)
        {
            simd_cast<int, float>(nd.get_data().data(), size);
            simd_elem_sqrt(scalor ,nd.get_data().data(), size);
            return nd;
        }
        else if constexpr (sizeof(data_type) == 2)
        {   
            simd_cast<int, double>(nd.get_data().data(), size);
            simd_elem_sqrt(scalor , nd.get_data().data(), size);
            return nd;
        }
        else if constexpr (sizeof(data_type) == 4)
        {
            simd_cast<int, float>(nd.get_data().data(), size);
            simd_elem_sqrt(scalor , nd.get_data().data(), size);
            return nd;
        }
        else if constexpr (sizeof(data_type) == 8)
        {
            simd_cast<int, double>(nd.get_data().data(), size);
            simd_elem_sqrt(scalor, nd.get_data().data(), size);
            return nd;
        }
    }
    return nd;
}

template <typename data_type>
ndarray<data_type> sqrt(const data_type x, ndarray<data_type> &nd){
    assert(!nd.get_shape().empty());
    assert(!nd.get_strides().empty());
    assert(nd.get_data().empty());
    const size_t size = nd.size();
    const data_type scalor = x;
    if constexpr (std::is_same_v<data_type, float>)
    {
        simd_elem_sqrt(scalor ,nd.get_data().data(), size);
        return nd;
    }
    else if constexpr (std::is_same_v<data_type, double>)
    {
        simd_elem_sqrt(scalor ,nd.get_data().data(), size);
        return nd;
    }
    else if constexpr (std::is_integral_v<data_type>)
    {
        if constexpr (sizeof(data_type) == 1)
        {
            simd_cast<int, float>(nd.get_data().data(), size);
            simd_elem_sqrt(scalor ,nd.get_data().data(), size);
            return nd;
        }
        else if constexpr (sizeof(data_type) == 2)
        {   
            simd_cast<int, double>(nd.get_data().data(), size);
            simd_elem_sqrt(scalor , nd.get_data().data(), size);
            return nd;
        }
        else if constexpr (sizeof(data_type) == 4)
        {
            simd_cast<int, float>(nd.get_data().data(), size);
            simd_elem_sqrt(scalor , nd.get_data().data(), size);
            return nd;
        }
        else if constexpr (sizeof(data_type) == 8)
        {
            simd_cast<int, double>(nd.get_data().data(), size);
            simd_elem_sqrt(scalor, nd.get_data().data(), size);
            return nd;
        }
    }
    return nd;
}

template <typename data_type>
ndarray<data_type> sqrt(ndarray<data_type> &first, ndarray<data_type> &second){
    assert(!first.get_shape().empty());
    assert(!first.get_strides().empty());
    assert(!first.get_data().empty());
    assert(!second.get_shape().empty());
    assert(!second.get_strides().empty());
    assert(!second.get_data().empty());
    ndarray<data_type> answer({first.get_shape()});
    if constexpr (is_integral_v<data_type>)
    {
        if constexpr (sizeof(data_type) == 1)
        {
            simd_cast<int, float>(first.get_data().data(), first.size());
            simd_cast<int, float>(second.get_data().data(), second.size());
            simd_sqrt(first.get_data().data(), second.get_data().data(), answer.get_data().data(), first.size());
            return answer;
        }
        else if constexpr (sizeof(data_type) == 2)
        {
            simd_cast<int, double>(first.get_data().data(), first.size());
            simd_cast<int, double>(second.get_data().data(), second.size());
            simd_sqrt(first.get_data().data(), second.get_data().data(), answer.get_data().data(), first.size());
            return answer;
        } 
        else if constexpr (sizeof(data_type) == 4)
        {
            simd_cast<int, float>(first.get_data().data(), first.size());
            simd_cast<int, float>(second.get_data().data(), second.size());
            simd_sqrt(first.get_data().data(), second.get_data().data(), answer.get_data().data(), first.size());
            return answer;
        }
        else if constexpr (sizeof(data_type) == 8)
        {
            simd_cast<int, double>(first.get_data().data(), first.size());
            simd_cast<int, double>(second.get_data().data(), second.size());
            simd_sqrt(first.get_data().data(), second.get_data().data(), answer.get_data().data(), first.size());
            return answer;
        }
    } else if constexpr (std::is_same_v<data_type, float>)
    {
        simd_sqrt(first.get_data().data(), second.get_data().data(), answer.get_data().data(), first.size());
        return answer;
    }
    else if constexpr (std::is_same_v<data_type, double>)
    {
        simd_sqrt(first.get_data().data(), second.get_data().data(), answer.get_data().data(), first.size());
        return answer;
    }
}

// for load file
template <typename data_type>
ndarray<data_type> loadtxt(const std::string &filename)
{
    std::ifstream file(filename);
    if (!file.is_open())
    {
        throw std::runtime_error("Could not open file: " + filename);
    }

    std::vector<data_type> data;
    std::vector<size_t> shape;
    std::string line;
    size_t num_columns = 0;

    while (std::getline(file, line))
    {
        std::istringstream line_stream(line);
        data_type value;
        size_t current_columns = 0;

        while (line_stream >> value)
        {
            data.push_back(value);
            ++current_columns;
        }

        if (num_columns == 0)
        {
            num_columns = current_columns;
        }
        else if (current_columns != num_columns)
        {
            throw std::runtime_error("Inconsistent number of columns in file: " + filename);
        }
    }

    file.close();

    if (num_columns == 0 || data.empty())
    {
        throw std::runtime_error("File is empty or invalid: " + filename);
    }

    shape.push_back(data.size() / num_columns);
    shape.push_back(num_columns);

    return ndarray<data_type>(shape, data);
}

// for save file
template <typename data_type>
void savetxt(const std::string &filename, const ndarray<data_type> &nd)
{
    std::ofstream file(filename);
    if (!file.is_open())
    {
        throw std::runtime_error("Could not open file for writing: " + filename);
    }

    const auto& shape = nd.get_shape();
    const auto& data = nd.get_data();

    // Set precision for floating-point types
    if constexpr (std::is_floating_point_v<data_type>)
    {
        file << std::fixed << std::setprecision(18); // Adjust precision as needed
    }

    if (shape.size() == 1) // Handle 1D array
    {
        for (size_t i = 0; i < shape[0]; ++i)
        {
            file << data[i] << (i == shape[0] - 1 ? "" : " ");
        }
        file << std::endl;
    }
    else if (shape.size() == 2) // Handle 2D array
    {
        size_t rows = shape[0];
        size_t cols = shape[1];
        for (size_t i = 0; i < rows; ++i)
        {
            for (size_t j = 0; j < cols; ++j)
            {
                // Assuming row-major storage (C-style)
                file << data[i * cols + j] << (j == cols - 1 ? "" : " ");
            }
            file << std::endl;
        }
    }
    else
    {
        // For higher dimensions, flatten or represent differently?
        // Current implementation throws error for > 2D.
        file.close(); // Close file before throwing
        throw std::runtime_error("savetxt currently only supports 1D and 2D arrays.");
    }

    file.close();
}

// for xor function (will using in simd technology, just for int type)
template <typename data_type>
ndarray<data_type> _xor(const ndarray<data_type> &first, const ndarray<data_type> &second){
    static_assert(std::is_integral_v<data_type>, "XOR operation is only supported for integral types.");
    assert(!first.get_shape().empty());
    assert(!first.get_strides().empty());
    assert(!first.get_data().empty());
    assert(!second.get_shape().empty());
    assert(!second.get_strides().empty());
    assert(!second.get_data().empty());
    assert(first.size() == second.size());
    ndarray<data_type> answer({first.get_shape()});
    if constexpr (std::is_same_v<data_type, int>)
    {
        simd_xor(first.get_data().data(), second.get_data().data(), answer.get_data().data(), first.size());
        return answer;
    }
    else if constexpr (std::is_same_v<data_type, long>)
    {
        simd_xor(first.get_data().data(), second.get_data().data(), answer.get_data().data(), first.size());
        return answer;
    }
    else if constexpr (std::is_same_v<data_type, short>)
    {
        simd_xor(first.get_data().data(), second.get_data().data(), answer.get_data().data(), first.size());
        return answer;
    }
    return answer;
}

template <typename data_type>
ndarray<data_type> _xor(const data_type &scalor, const ndarray<data_type> &nd){
    static_assert(std::is_integral_v<data_type>, "XOR operation is only supported for integral types.");
    assert(!nd.get_shape().empty());
    assert(!nd.get_strides().empty());
    assert(!nd.get_data().empty());
    const size_t size = nd.size();
    if constexpr (std::is_same_v<data_type, int>)
    {
        simd_xor(scalor, nd.get_data().data(), size);
        return nd;
    }
    else if constexpr (std::is_same_v<data_type, long>)
    {
        simd_xor(scalor, nd.get_data().data(), size);
        return nd;
    }
    else if constexpr (std::is_same_v<data_type, short>)
    {
        simd_xor(scalor, nd.get_data().data(), size);
        return nd;
    }
    return nd;
}
#endif