#include "./myMacro.hpp"

int main() {
    int c = 1;
    cout << char(c + 'A') << endl;

    string s = "abc";
    s += char(c + 'A');
    cout << s << endl;
}