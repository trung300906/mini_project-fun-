```cpp

#pragma once
template <typename data_type>
class ndarray; // Khai báo trước

// Lớp trung gian giúp gán giá trị tuần tự
template <typename data_type>
class CommaInitializer
{
private:
    ndarray<data_type> &mat;
    int index;

public:
    CommaInitializer(ndarray<data_type> &m, data_type firstValue) {}
    CommaInitializer &operator,(data_type value) {}
};
```
include into [[numpy.hpp]]