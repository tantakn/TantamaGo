#include <bits/stdc++.h>
using namespace std;
#include "myMacro.hpp"

struct test {
    test* parent;
    vector<test*> childrens;
    int value;

    test* AddChild()
    {
        test* child = new test(this);
        childrens.push_back(child);
        return child;
    }

    test(test* parent)
        : parent(parent), value(1234)
    {
        cout << "created" << endl;
    }

    ~test()
    {
        for (test* child : childrens) {
            child->value = -1;
            delete child;
            child = nullptr;
        }
        cout << "delete this" << endl;
    }
};




int main()
{
//     set<tuple<int, pair<int, string>>> tmp;
//     tmp.insert({1, {1, "a"}});
//     tmp.insert({1, {2, "a"}});
//     tmp.insert({4, {1, "a"}});
//     tmp.insert({4, {1, "b"}});

//     // print(*begin(t));
//     // print(*rbegin(t));

//     for (auto&& x : tmp) {
//         print(x);
//         x = {0, {0, "0"}};
//     }
//     print(tmp);
//     return 0;



    // vint v = {1, 2, 3, 4, 5};
    // for (int& i : v) {
    //     cout << i << endl;
    //     ++i;
    // }
    // print(v);

    float f;
    double d;
    long double ld;
    char c;
    int i;
    ll li;

    cout << sizeof(f) << endl;
    // 4
    cout << sizeof(d) << endl;
    // 8
    cout << sizeof(ld) << endl;
    // 16
    cout << sizeof(c) << endl;
    // 1
    cout << sizeof(i) << endl;
    // 4
    cout << sizeof(li) << endl;
    // 8
}
