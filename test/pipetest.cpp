#include "myMacro.hpp"

#define bufsize 1024

int main(void) {

    char data[bufsize] = {};

    FILE *fp = popen("python3 -u ./pipetest.py", "w");
    fputs("{\"hoge\": \"fuga\"}", fp);
    fgets(data, bufsize , fp);
    std::cout << "cpp: " << data << std::endl;
    pclose(fp);

    return(0);
}



// #include "myMacro.hpp"

// #include "json.hpp"

// using json = nlohmann::json;

// #define bufsize 1024

// int main(void) {

//     char data[bufsize] = {};

//     FILE *fp = popen("python3 ./pipetest.py", "w");
//     fputs("[[3,3,3,3,3,3,3,3,3,3,3],[3,0,1,0,1,1,1,0,1,0,3],[3,2,1,1,1,1,0,1,0,1,3],[3,1,1,0,1,1,1,1,1,1,3],[3,1,0,1,0,1,0,1,1,0,3],[3,1,1,1,1,1,1,1,1,1,3],[3,1,1,1,1,1,1,1,0,1,3],[3,1,0,1,1,0,1,1,1,1,3],[3,1,1,0,1,1,1,1,1,0,3],[3,0,1,1,1,1,0,1,1,1,3],[3,3,3,3,3,3,3,3,3,3,3],[2,6,7]]", fp);
//     FILE *fpp = popen("python3 ./pipetest.py", "r");
//     fgets(data, bufsize , fpp);
//     std::cout << data << std::endl;
//     pclose(fp);

//     cout << "cpp " << data << endl;

//     json j = json::parse(data);

//     cout << j.dump() << endl;

//     json a = j[0];

//     // json p = j[policy];

//     // json v = j[value];

//     // vector<vector<double>> policy = j["policy"].get<vector<vector<double>>>();
//     // rep(i, policy.size()) {
//     //     rep(j, policy[i].size()) {
//     //         cout << policy[i][j] << " ";
//     //     }
//     //     cout << endl;
//     // }
//     // vector<double> value = j["value"].get<vector<double>>();
//     // rep(i, value.size()) {
//     //     cout << value[i] << " ";
//     // }


//     return(0);
// }
