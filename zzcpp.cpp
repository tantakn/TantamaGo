#ifndef myMacro_hpp_INCLUDED
#include "myMacro.hpp"
#define myMacro_hpp_INCLUDED
#endif

int main()
{
    set<tuple<int, int>> s;

    s.insert(make_tuple(1, 2));
    s.insert(make_tuple(2, 3));
    s.insert(make_tuple(3, 4));

    // for (auto x : s)
    // {
    //     print(x);
    // }

    for (auto &x : s)
    {
        print(x);
        auto [a, b] = x;
        s.erase(x);
        s.insert(make_tuple(a+10, b));
    }
}