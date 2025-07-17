#pragma once
#ifndef NUMPY_HPP
#define NUMPY_HPP
#include "SIMD/simd_index.hpp"
/*
    all things will store in 1D array, and because for that, it will be easy and  effecient for memory
    about index:
    just think from eazy way, if you have and 3D array, it's like a cube, and each layer is 2D array, and each row is 1D array
    and strides will store stride in 1D array raw, to access the next of layer, for example, if you have 3x4x5 array, the formula will be:
    shape = [3,4,5]
    strides[2] = 1 (alwasys 1 because this is the last layer, mean that no layer next to it to strides more, and every access in this layer will be like an 1D array)
    strides[1] = strides[2]*shape[2-1] = 1 * 5 = 5, mean that wanna acess layer, you must jump 5 element in 1D array
    strides[0] = strides[1]*shape[1-1] = 5 * 4 mean that wanna access next layer, you must jump 5*4 = 20 element in 1D array
    so, if you want to get value at (i, j, k), you just need to calculate index = i * stride[0] + j * stride[1] + k * stride[2]
    and if you want to get value at (i, j), you just need to calculate index = i * stride[0] + j * stride[1]
    and if you want to get value at (i), you just need to calculate index = i * stride[0]
*/
namespace numpy
{
#include "Commainitializer.tpp"
    template <typename data_type>
    class ndarray
    {
    protected:
        std::vector<data_type> data; // 1D array data
        std::vector<size_t> shape;   // store shape of array(ex 3x4x5 mean 3D array with 3 layer, each layer has 4 row, and each row has 5 element)
        std::vector<size_t> strides; // store and calculate strides for each dimension
        friend class CommaInitializer<data_type>;

    public:
        ndarray() : data({}), shape({}), strides({}) {};
        ndarray(const std::vector<size_t> &shape_) : shape(shape_)
        {
            strides.resize(shape.size());
            size_t total = 1;
            for (int i = shape.size() - 1; i >= 0; --i)
            {
                strides[i] = total;
                total *= shape[i];
            }
            data.resize(total, 0);
        }

        // setter and getter
        // get
        std::vector<size_t> get_shape() const; // already defined in class_func.tpp
        const std::vector<size_t> &get_strides() const;
        const std::vector<data_type> &get_data() const;
        // set
        void set_shape(const std::vector<size_t> &shape_);
        void set_strides(const std::vector<size_t> &strides_);
        void set_data(const std::vector<data_type> &data_);
        // this function will outcome the index of things we need to access
        size_t Index(const std::vector<size_t> &indices) const
        {
            assert(indices.size() == shape.size());
            size_t idx = 0;
            for (size_t i = 0; i < indices.size(); i++)
            {
                assert(indices[i] <= shape[i]);
                idx += indices[i] * strides[i];
            }
            return idx;
        }
        // operator function
        data_type &operator()(const std::vector<size_t> &indices);
        const data_type &operator()(const std::vector<size_t> &indices) const;

        //================================================================================================//
        // ostream operator
        friend std::ostream &operator<<(std::ostream &out, const ndarray<data_type> &nd)
        {
            // Hàm đệ quy để in mảng n chiều
            assert(!nd.data.empty());
            std::function<void(const std::vector<size_t> &, std::vector<size_t> &, size_t, size_t)> recursive;
            recursive = [&](const std::vector<size_t> &index, std::vector<size_t> &path, size_t level = 0, size_t indent = 0)
            {
                if (level == index.size())
                {
                    out << std::string(indent, ' ') << "[";
                    out << nd(path);
                    out << "]\n";
                    return;
                }
                out << std::string(indent, ' ') << "[\n";
                for (size_t i = 0; i < index[level]; i++)
                {
                    path[level] = i;
                    recursive(index, path, level + 1, indent + 2);
                }
                out << std::string(indent, ' ') << "]\n";
            };
            std::vector<size_t> path(nd.shape.size(), 0);
            recursive(nd.shape, path, 0, 0);
            return out;
        }
        friend std::istream &operator>>(std::istream &input, ndarray<data_type> &nd)
        {
            assert(!nd.data.empty());
            std::function<void(std::vector<size_t> &, std::vector<size_t> &, size_t)> recursive;
            recursive = [&](std::vector<size_t> &index, std::vector<size_t> &path, size_t level = 0)
            {
                if (level == index.size())
                {
                    std::cout << "[";
                    for (auto &elem : path)
                        std::cout << elem + 1 << " ";
                    std::cout << "]: ";
                    input >> nd(path);
                    std::cout << "\n";
                    return;
                }
                for (size_t i = 0; i < index[level]; i++)
                {
                    path[level] = i;
                    recursive(index, path, level + 1);
                }
            };
            std::vector<size_t> path(nd.shape.size(), 0);
            recursive(nd.shape, path, 0);
            return input;
        }

        //================================================================================================//
        // operator calculate
        ndarray<data_type> operator=(const ndarray<data_type> &nd);
        // operator add
        ndarray<data_type> operator+(const ndarray<data_type> &nd);
        ndarray<data_type> operator+(const data_type &scalor);
        friend ndarray<data_type> operator+(const data_type &scalor, const ndarray<data_type> &nd);
        // operator sub
        ndarray<data_type> operator-(const ndarray<data_type> &nd);
        ndarray<data_type> operator-(const data_type &scalor);
        friend ndarray<data_type> operator-(const data_type &scalor, const ndarray<data_type> &nd);
        // operator mul
        ndarray<data_type> operator*(const ndarray<data_type> &nd);
        ndarray<data_type> operator*(const data_type &scalor);
        friend ndarray<data_type> operator*(const data_type &scalor, const ndarray<data_type> &nd);
        // operator div
        ndarray<data_type> operator/(const ndarray<data_type> &nd);
        ndarray<data_type> operator/(const data_type &scalor);
        friend ndarray<data_type> operator/(const data_type &scalor, const ndarray<data_type> &nd);
        // operator power
        ndarray<data_type> operator^(const ndarray<data_type> &nd);
        ndarray<data_type> operator^(const data_type &scalor);
        friend ndarray<data_type> operator^(const data_type &scalor, const ndarray<data_type> &nd);
        // operator mod
        ndarray<data_type> operator%(const ndarray<data_type> &nd);
        ndarray<data_type> operator%(const data_type &scalor);
        friend ndarray<data_type> operator%(const data_type &scalor, const ndarray<data_type> &nd);
        //================================================================================================//
        // boolean operations

        bool operator==(const ndarray<data_type> &nd);
        bool operator==(const data_type &scalor);
        friend bool operator==(const data_type &scalor, const ndarray<data_type> &nd);
        bool operator!=(const ndarray<data_type> &nd);
        bool operator!=(const data_type &scalor);
        friend bool operator!=(const data_type &scalor, const ndarray<data_type> &nd);
        bool operator<(const ndarray<data_type> &nd);
        bool operator<(const data_type &scalor);
        friend bool operator<(const data_type &scalor, const ndarray<data_type> &nd);
        bool operator<=(const ndarray<data_type> &nd);
        bool operator<=(const data_type &scalor);
        friend bool operator<=(const data_type &scalor, const ndarray<data_type> &nd);
        bool operator>(const ndarray<data_type> &nd);
        bool operator>(const data_type &scalor);
        friend bool operator>(const data_type &scalor, const ndarray<data_type> &nd);
        bool operator>=(const ndarray<data_type> &nd);
        bool operator>=(const data_type &scalor);
        friend bool operator>=(const data_type &scalor, const ndarray<data_type> &nd);
        //================================================================================================//
        // ++ -- *= /= ^= %=

        ndarray<data_type> operator+=(const ndarray<data_type> &nd);
        ndarray<data_type> operator+=(const data_type &scalor);

        ndarray<data_type> operator-=(const ndarray<data_type> &nd);
        ndarray<data_type> operator-=(const data_type &scalor);

        ndarray<data_type> operator*=(const ndarray<data_type> &nd);
        ndarray<data_type> operator*=(const data_type &scalor);

        ndarray<data_type> operator/=(const ndarray<data_type> &nd);
        ndarray<data_type> operator/=(const data_type &scalor);

        ndarray<data_type> operator^=(const ndarray<data_type> &nd);
        ndarray<data_type> operator^=(const data_type &scalor);

        ndarray<data_type> operator%=(const ndarray<data_type> &nd);
        ndarray<data_type> operator%=(const data_type &scalor);

        ndarray<data_type> &operator++();
        ndarray<data_type> operator++(int);
        ndarray<data_type> &operator--();
        ndarray<data_type> operator--(int);
        //================================================================================================//
        // basic function
        void print() const;
        size_t size() const; // get number of matrix
        size_t ndim() const;
        data_type sum();
        data_type sum(std::vector<size_t> &begin, std::vector<size_t> &end);
        /*
        std::tuple<data_type, int, std::vector<data_type>::iterator>
        parameters describe:
        # data_type: the max or min value
        # int: the index of max or min value
        # std::vector<data_type>::iterator: the iterator of max or min value
        */
        std::tuple<data_type, int, typename std::vector<data_type>::iterator> max();
        std::tuple<data_type, int, typename std::vector<data_type>::iterator> max(std::vector<size_t> &begin, std::vector<size_t> &end);
        std::tuple<data_type, int, typename std::vector<data_type>::iterator> min();
        std::tuple<data_type, int, typename std::vector<data_type>::iterator> min(std::vector<size_t> &begin, std::vector<size_t> &end);
        ndarray<data_type> reshape(const std::vector<size_t> &shape_);
        std::vector<data_type> flatten(char order = 'C');
        ndarray<data_type> transpose();
        ndarray<data_type> dot(const ndarray<data_type> &nd, const std::vector<size_t> &axis);
    };
}
#include "overload/overload.tpp"    // for function overload
#include "function/class_func.tpp"  // for class's function
#include "overload/overload_an.tpp" // for another overload
#include "function/function.hpp"    // for basic function

#endif