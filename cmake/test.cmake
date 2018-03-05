MESSAGE(STATUS "include test the shit")
################################################################################
#                           Make Test
################################################################################
if(DEEPITERATOR_BUILD_TESTS)
        include_directories("./")
        include_directories("./include/")
        find_package(Boost COMPONENTS unit_test_framework REQUIRED)
        if(Boost_FOUND)
            message(STATUS "Boost Include Dirs ${Boost_INCLUDE_DIRS}")
            message(STATUS "Boost LIBARAY Dirs  ${Boost_LIBRARIES}")
            include_directories(${Boost_INCLUDE_DIRS})
        endif()

        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
        set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/Tests)
    #DEEPITERATOR_TEST_BIDIRECTIONAL
    #DEEPITERATOR_TEST_CUDA
    #DEEPITERATOR_TEST_FORWARD
    #DEEPITERATOR_TEST_NDARRAY
    #DEEPITERATOR_TEST_NESTEDITERATOR
    #DEEPITERATOR_TEST_OWNCATEGORIE
    #DEEPITERATOR_TEST_NEWCONTAINER
    #DEEPITERATOR_TEST_RANDOMACCESS
    #DEEPITERATOR_TEST_RUNTIME
    #DEEPITERATOR_TEST_SLICENAVIGATOR
    #DEEPITERATOR_TEST_UNNESTETITERATOR
        if(DEEPITERATOR_TEST_BIDIRECTIONAL)
            add_executable( "Bidirectional" "${_DEEPITERATOR_TEST_DIR}/Bidirectional.cpp")
        endif()
        if(DEEPITERATOR_TEST_FORWARD)
            add_executable( "Forward" "${_DEEPITERATOR_TEST_DIR}/Forward.cpp")
        endif()        
        if(DEEPITERATOR_TEST_NDARRAY)
            add_executable( "NdArray" "${_DEEPITERATOR_TEST_DIR}/NdArray.cpp")
        endif()        
        if(DEEPITERATOR_TEST_NESTEDITERATOR)
            add_executable( "NestedIterator" "${_DEEPITERATOR_TEST_DIR}/NestedIterator.cpp")
        endif()        
        if(DEEPITERATOR_TEST_OWNCATEGORIE)
            add_executable( "OwnCategorie" "${_DEEPITERATOR_TEST_DIR}/OwnCategorie.cpp")
        endif()
        if(DEEPITERATOR_TEST_NEWCONTAINER)
            add_executable( "NewContainer" "${_DEEPITERATOR_TEST_DIR}/NewContainer.cpp")
        endif()
        if(DEEPITERATOR_TEST_RANDOMACCESS)
            add_executable( "RandomAccess" "${_DEEPITERATOR_TEST_DIR}/RandomAccess.cpp")
        endif()
        if(DEEPITERATOR_TEST_RUNTIME)
            add_executable( "Runtime" "${_DEEPITERATOR_TEST_DIR}/Runtime.cpp")
        endif()
        if(DEEPITERATOR_TEST_SLICENAVIGATOR)
            add_executable( "SliceNavigator" "${_DEEPITERATOR_TEST_DIR}/SliceNavigator.cpp")
        endif()
        if(DEEPITERATOR_TEST_UNNESTETITERATOR)
            add_executable( "UnnestedIterator" "${_DEEPITERATOR_TEST_DIR}/UnnestedIterator.cpp")
        endif()

        if(DEEPITERATOR_TEST_CUDA)
            SET(CUDA_SEPARABLE_COMPILATION ON)
            find_package(CUDA QUIET REQUIRED)
            list(APPEND CUDA_NVCC_FLAGS "-std=c++11;-O2;-DVERBOSEM;--expt-relaxed-constexpr")
            cuda_add_executable("Cuda" "${_DEEPITERATOR_TEST_DIR}/Cuda.cpp" "${_DEEPITERATOR_TEST_DIR}/Cuda/cuda.cu")
        endif()
endif()
