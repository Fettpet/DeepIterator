#include <iostream>
#include <vector>
#include <iostream>
#include <typeinfo>

#include "deepForeach.hpp"

int main(int argc, char **argv) {

    std::vector< std::vector<int > > data = { {0, 1, 2}, {3, 4,5,6}, {7, 8, 9}};

    for(auto vec: data)
    {
        for(auto elem: vec)
        {
            std::cout << elem << ",";
        }
        std::cout << std::endl;
    }

    std::vector<int> con = {1,2,3};

    deepForeach(con.begin(), con.end(), [](int x){std::cout << x << std::endl;});

    /*container con(data);

    foreach(con.begin(), con.end(), [](auto & x){x=x*2;});


    for(auto vec: data)
    {
        for(auto elem: vec)
        {
            std::cout << elem << ",";
        }
        std::cout << std::endl;
    }*/
}
