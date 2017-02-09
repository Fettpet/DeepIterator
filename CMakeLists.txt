cmake_minimum_required(VERSION 2.6)
project(itertest)
set (CMAKE_CXX_STANDARD 11)


include_directories("./")
add_executable(itertest main.cpp)

install(TARGETS itertest RUNTIME DESTINATION bin)
