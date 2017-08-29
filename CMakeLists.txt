cmake_minimum_required(VERSION 3.4)
project(itertest)

#set( CUDA_HOST_COMPILER "gcc-4.8")
set (CMAKE_BUILD_TYPE Debug)
set(CMAKE_VERBOSE_MAKEFILE OFF)
set (USE_OPEMMP ON)
set (USE_CUDA ON)
set (CMAKE_CXX_STANDARD 11)
set (MAKE_TEST ON)

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -O0")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wall -O3")


SET(CUDA_SEPARABLE_COMPILATION ON)

include_directories("./")

#set(BOOST_ROOT "/home/s6303743/boost/")
find_package(Boost COMPONENTS unit_test_framework REQUIRED)
if(Boost_FOUND)
    message(STATUS "Boost Include Dirs ${Boost_INCLUDE_DIRS}")
    message(STATUS "Boost LIBARAY Dirs  ${Boost_LIBRARIES}")
    include_directories(${Boost_INCLUDE_DIRS})
endif()

if(USE_OPEMMP)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -fPIC")  
endif()

if(USE_CUDA)
    find_package(CUDA QUIET REQUIRED)
    list(APPEND CUDA_NVCC_FLAGS "-arch=sm_35;-std=c++11;-O2;-DVERBOSEM;--expt-relaxed-constexpr")
    cuda_add_executable(itertest main.cpp)
else()
    add_executable(itertest main.cpp)
    install(TARGETS itertest RUNTIME DESTINATION bin)
endif()


################################################################################
#                           Make Test
################################################################################
if(MAKE_TEST)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -fopenmp")
        set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/Tests)
        file( GLOB APP_SOURCES Tests/*.cpp )
        foreach( testsourcefile ${APP_SOURCES} )
            message("${testsourcefile}")
            # I used a simple string replace, to cut off .cpp.
            string( REPLACE ".cpp" "" testname ${testsourcefile} )
            string(FIND ${testname} "/" position REVERSE)
            MATH(EXPR position "${position}+1")
            string(LENGTH ${testname} length)
            string(SUBSTRING ${testname} ${position} ${length} filename)
            cuda_add_executable( "${filename}" "${testsourcefile}" "Tests/Cuda/cuda.cu")
            # Make sure YourLib is linked to each app
            target_link_libraries( ${filename} ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY} )
        endforeach( testsourcefile ${APP_SOURCES} )
endif()
