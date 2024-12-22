#include "myMacro.hpp"

#define bufsize 32

int main(void) {

    char data[bufsize] = {};

    FILE *fp = popen("python3 ./pipetest.py", "w");
    fputs("{\"hoge\": \"fuga\"}", fp);
    fgets(data, bufsize , fp);
    std::cout << data << std::endl;
    pclose(fp);

    return(0);
}
