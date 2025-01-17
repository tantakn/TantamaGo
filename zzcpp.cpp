#ifndef myMacro_hpp_INCLUDED
#include "myMacro.hpp"
#define myMacro_hpp_INCLUDED
#endif

int main()
{
    double a = 123.456789;
    double b = 12.3456789;
    double c = 1.23456789;

    cout << setprecision(3) << a << ", " << b << ", " << c << endl;
// 123, 12.3, 1.23
    cout << fixed << setprecision(3) << a << ", " << b << ", " << c << endl;
// 123.457, 12.346, 1.235
    cout << fixed << setprecision(2) << a << ", " << b << ", " << c << endl;
// 123.46, 12.35, 1.23
cout << scientific << a << ", " << b << ", " << c << endl;


    double positive = 123.456;
    double negative = -123.456;

    std::cout << std::fixed << std::setprecision(2) << std::showpoint;

    std::cout << std::setw(10) << positive << std::endl;
    std::cout << std::setw(10) << negative << std::endl;
//     123.46
//    -123.46
    std::cout << std::setw(4) << positive << std::endl;
    std::cout << std::setw(4) << negative << std::endl;
// 123.46
// -123.46

    std::cout << std::fixed << std::setprecision(2) << std::showpoint << std::showpos;

    std::cout << std::setw(4) << positive << std::endl;
    std::cout << std::setw(4) << negative << std::endl;
// +123.46
// -123.46

    return 0;
}