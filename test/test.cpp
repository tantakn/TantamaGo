#include "myMacro.hpp"

int cnt = 0;
struct test {
    test *parent;
    map<int, test *> childrens;

    vint id;

    bool isRoot = false;

    test() : parent(nullptr) {
        id.push_back(cnt++);
    }

    test(test *inputparent) : parent(inputparent) {
        assert (parent != nullptr);
        parent->childrens[rand()] = this;
        id = parent->id;
        id.push_back(cnt++);
    };

    test* MakeChild() {
        return new test(this);
    }

    // ~test() {
    //     for (auto x : childrens) {
    //         // print("delete", x.second->id);
    //         // print(cnt--);
    //         delete x.second;
    //     }
    // }
};

// int Delete(test *p) {
//     for (auto x : p->childrens) {
//         Delete(x.second);
//     }
//     delete p;
//     return 0;
// }

bool bfs(test *p) {
    // print(++cnt);
    if (!(rand() % 200)) {
        return false;
    }
    bfs(p->MakeChild());
    if (!p->isRoot) {
        delete p;
    }
    return true;
}

int main()
{
    test *root = new test();

    root->isRoot = true;

    test *tmp = root;
    rep (i, 40) {
        print(i);
        bfs(tmp);
        // for (auto x : root->childrens) {
        //     delete x.second;
        //     // Delete(x.second);
        // }
    }

    delete root;

    print(cnt);

    return 0;
}