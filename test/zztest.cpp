#include <bits/stdc++.h>
using namespace std;

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
    test* root = new test(nullptr);
    test* child1 = root->AddChild();

    cout << root->value << endl;
    // 1234
    cout << child1->value << endl;
    // 1234

    delete root;

    // delete してもアクセスはできる
    cout << root->value << endl;
    // 1234
    cout << child1->value << endl;
    // -1

    return 0;
}