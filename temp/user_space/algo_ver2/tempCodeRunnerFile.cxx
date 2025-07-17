#include <iostream>
#include <asm/fpu/api.h>

using namespace std;

int main()
{
    int a = 0;
    int b = 0;
    int c = 0;
    int d = 0;
    int e = 0;
    int f = 0;
    int g = 0;
    int h = 0;

    cout << "Hello World!" << "\n";

    asm(
        "movl $1, %eax;"
        "movl $2, %ebx;"
        "addl %ebx, %eax;"
        : "=a"(a), "=b"(b)
        :
        : "%eax", "%ebx");

    cout << "Result: " << a << endl;

    return 0;
}