cmake_minimum_required(VERSION 3.4)
add_library(DeepIterator INTERFACE)

set( CMAKE_BUILD_TYPE Debug )
set( CMAKE_VERBOSE_MAKEFILE OFF )
set( USE_CUDA ON )
set( CMAKE_CXX_STANDARD 11 )
set( MAKE_TEST ON )

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -Wextra -O0")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wall -Wextra -O3")


SET(CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(DeepIterator INTERFACE include/)




if(USE_CUDA)
    find_package(CUDA QUIET REQUIRED)
    list(APPEND CUDA_NVCC_FLAGS "-std=c++11;-O2;-DVERBOSEM;--expt-relaxed-constexpr")
endif()


################################################################################
#                           Make Test
################################################################################
if(MAKE_TEST)
        include_directories("./")
        include_directories("./include/")
        find_package(Boost COMPONENTS unit_test_framework REQUIRED)
        if(Boost_FOUND)
            message(STATUS "Boost Include Dirs ${Boost_INCLUDE_DIRS}")
            message(STATUS "Boost LIBARAY Dirs  ${Boost_LIBRARIES}")
            include_directories(${Boost_INCLUDE_DIRS})
        endif()

        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -fopenmp")
        set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/Tests)
        file( GLOB APP_SOURCES Test/*.cpp )
        foreach( testsourcefile ${APP_SOURCES} )
            # I used a simple string replace, to cut off .cpp.
            string( REPLACE ".cpp" "" testname ${testsourcefile} )
            string(FIND ${testname} "/" position REVERSE)
            MATH(EXPR position "${position}+1")
            string(LENGTH ${testname} length)
            string(SUBSTRING ${testname} ${position} ${length} filename)
            cuda_add_executable( "${filename}" "${testsourcefile}" "Test/Cuda/cuda.cu")
            # Make sure YourLib is linked to each app
            target_link_libraries( ${filename} ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY} )
        endforeach( testsourcefile ${APP_SOURCES} )
endif()
