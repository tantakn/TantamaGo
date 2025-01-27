#include "./myMacro.hpp"

// int main()
// {
//     vector<string> v;
//     string s;
//     string ss = "asdf zxcv";
//     // stringstream ss{ss}; 
//     while (getline(ss, s, ' ')) {
//         v.push_back(s);
//     }

//     print(v);
// }

#include <iostream>
#include <sstream>
#include <vector>
using namespace std;

int main(void){
    vector<string> v;                  // 分割した文字列を格納するvector
    string str, s;                      

    // str = "A B C D E"     ;             // 分割対象の文字列str

    getline(cin, str);

    stringstream ss{str};              // 入出力可能なsstreamに変換

    print(str);

    while ( getline(ss, s, ' ') ){     // スペース（' '）で区切って，格納
        v.push_back(s);
    }
    
    for (const string& s : v){         // vの中身を出力
        cout << s << endl;
    }
}
