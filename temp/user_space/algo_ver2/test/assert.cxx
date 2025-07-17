#include <cassert>
#include <iostream>

int main()
{
    int x = -5;
    assert(x > 0); // Điều kiện `false`, chương trình bị dừng và báo lỗi

    std::cout << "Dòng này sẽ không được in ra!" << std::endl;
    return 0;
}
