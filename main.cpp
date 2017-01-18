#include <iostream>
#include <vector>
#include <iostream>
#include <typeinfo>
#include <algorithm>
#include "deepForeach.hpp"
#include "isContainer.hpp"


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
    
    std::cout << "Is Container: " << std::boolalpha << isContainer<std::vector<int>>::value << std::endl;
    std::cout << "Is Container: " << std::boolalpha << isContainer<int>::value << std::endl;
    std::remove_reference<decltype(*(data.begin()))>::type::iterator iter;
std::cout << "Is Container: " << std::boolalpha << isContainer<decltype(*(data.begin())) >::value << std::endl;
    std::vector<std::vector< std::vector<int > > > datatest = { { {0, 1, 2}, {3, 4,5,6}, {7, 8, 9}}, { {0, 1, 2}, {3, 4,5,6}, {7, 8, 9}}};
    deepForeach(datatest,
                [](int& x){x*=2;});

    std::cout << con[2] << std::endl;
    /*container con(data);

    foreach(con.begin(), con.end(), [](auto & x){x=x*2;});
    */

    for(auto vec: data)
    {
        for(auto elem: vec)
        {
            std::cout << elem << ",";
        }
        std::cout << std::endl;
    }
}
