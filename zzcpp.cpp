// #include <stdio.h>
// #include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <arpa/inet.h>
// #include<string>
// #include<iostream>

#include "myMacro.hpp"

int main() {
    int sockfd;
    int client_sockfd;
    sockaddr_in addr;

    socklen_t len = sizeof( sockaddr_in );
    sockaddr_in from_addr;

    char buf[1024];

    // 受信バッファ初期化
    memset( buf, 0, sizeof( buf ) );

    // ソケット生成
    sockfd = socket( AF_INET, SOCK_STREAM, 0 );

    int port=8000;
    // std::cout << "ポート番号を入力" << std::endl;
    // std::cin >> port;

    // 待ち受け用IP・ポート番号設定
    addr.sin_family = AF_INET;
    addr.sin_port = htons( port );
    addr.sin_addr.s_addr = INADDR_ANY;

    // バインド
    bind( sockfd, (struct sockaddr *)&addr, sizeof( addr ));

    // 受信
    listen( sockfd, SOMAXCONN );

    // コネクト
    client_sockfd = accept( sockfd, (struct sockaddr *)&from_addr, &len );


    // 受信
    int rsize;
    while(true) {
        rsize = recv( client_sockfd, buf, sizeof( buf ), 0 );

        if ( rsize == 0 ) {
            break;
        }
        else {
            std::cout<<"recvData  "<< buf << std::endl;

            sleep(1);

            string a = "Linux Server" + string(buf);

            std::cout<<"sendData  "<< a <<std::endl;

            write( client_sockfd, a.c_str(), a.length() );

            // Clear the buffer after sending the data
            memset(buf, 0, sizeof(buf));
        }
    }

    // ソケットクローズ
    close( client_sockfd );
    close( sockfd );

    return 0;
}