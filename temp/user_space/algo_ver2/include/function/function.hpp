#pragma once
#ifndef FUNCTION_HPP
#define FUNCTION_HPP

namespace numpy
{
    template <typename data_type>
    size_t size(ndarray<data_type> &nd, size_t axis = 0);

    // e^matrix
    template <typename data_type>
    ndarray<data_type> exp(ndarray<data_type> &nd);

    // for all log function
    template <typename data_type>
    ndarray<data_type> log(const data_type &scalor, ndarray<data_type> &nd);

    template <typename data_type>
    ndarray<data_type> log10(ndarray<data_type> &nd);

    template <typename data_type>
    ndarray<data_type> log2(ndarray<data_type> &nd);

    template <typename data_type>
    ndarray<data_type> ln(ndarray<data_type> &nd);

    // matrix log matrix(element_wise)
    template <typename data_type>
    ndarray<data_type> log(const ndarray<data_type> &first, const ndarray<data_type> &second);

    // for sin
    template <typename data_type>
    ndarray<data_type> sin(ndarray<data_type> &nd);

    // for cos
    template <typename data_type>
    ndarray<data_type> cos(ndarray<data_type> &nd);

    // for tan
    template <typename data_type>
    ndarray<data_type> tan(ndarray<data_type> &nd);

    // for cotan
    template <typename data_type>
    ndarray<data_type> cotan(ndarray<data_type> &nd);

    // for sinh
    template <typename data_type>
    ndarray<data_type> sinh(ndarray<data_type> &nd);

    // for cosh
    template <typename data_type>
    ndarray<data_type> cosh(ndarray<data_type> &nd);

    // for tanh
    template <typename data_type>
    ndarray<data_type> tanh(ndarray<data_type> &nd);

    // for arcsin
    template <typename data_type>
    ndarray<data_type> arcsin(ndarray<data_type> &nd);

    // for arccos
    template <typename data_type>
    ndarray<data_type> arccos(ndarray<data_type> &nd);

    // for arctan
    template <typename data_type>
    ndarray<data_type> arctan(ndarray<data_type> &nd);

    // for arcsinh
    template <typename data_type>
    ndarray<data_type> arcsinh(ndarray<data_type> &nd);

    // for arccosh
    template <typename data_type>
    ndarray<data_type> arccosh(ndarray<data_type> &nd);

    // for arctanh
    template <typename data_type>
    ndarray<data_type> arctanh(ndarray<data_type> &nd);

    // for sec
    template <typename data_type>
    ndarray<data_type> sec(ndarray<data_type> &nd);

    // for cosec
    template <typename data_type>
    ndarray<data_type> cosec(ndarray<data_type> &nd);

    // for arcsec
    template <typename data_type>
    ndarray<data_type> arcsec(ndarray<data_type> &nd);

    // for arccosec
    template <typename data_type>
    ndarray<data_type> arccosec(ndarray<data_type> &nd);

    // for arccotan
    template <typename data_type>
    ndarray<data_type> arccotan(ndarray<data_type> &nd);

    // for sqrt
    template <typename data_type>
    ndarray<data_type> sqrt(ndarray<data_type> &nd);

    template <typename data_type>
    ndarray<data_type> sqrt(const float x, ndarray<data_type> &nd);

    template <typename data_type>
    ndarray<data_type> sqrt(ndarray<data_type> &first, ndarray<data_type> &second);

    // for load file
    template <typename data_type>
    ndarray<data_type> loadtxt(const std::string &filename); // not implent

    // for save file
    template <typename data_type>
    void savetxt(const std::string &filename, const ndarray<data_type> &nd); // not implent

    // for xor function (will using in simd technology)
    template <typename data_type>
    ndarray<data_type> _xor(const ndarray<data_type> &first, const ndarray<data_type> &second); // not implent

    template <typename data_type>
    ndarray<data_type> _xor(const data_type &scalor, const ndarray<data_type> &nd); // not implent
#include "function.tpp"
}
#endif