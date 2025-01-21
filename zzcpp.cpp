#include <iomanip>
#include <iostream>
#include <cmath>

int main()
{
    double num = 3.1415926535;
    int decimal = static_cast<int>((num - std::floor(num)) * 1000 + 0.5);
    std::cout << decimal << std::endl;
}