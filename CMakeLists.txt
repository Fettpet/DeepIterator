cmake_minimum_required(VERSION 3.4)
project(itertest)
set (CMAKE_CXX_STANDARD 11)
set (MAKE_TEST ON)
set (CMAKE_BUILD_TYPE DEBUG)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
include_directories("./")
add_executable(itertest main.cpp)

install(TARGETS itertest RUNTIME DESTINATION bin)


################################################################################
#                           Make Test
################################################################################
if(MAKE_TEST)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/Tests)
    file( GLOB APP_SOURCES Tests/*.cpp )
    foreach( testsourcefile ${APP_SOURCES} )
        # I used a simple string replace, to cut off .cpp.
        string( REPLACE ".cpp" "" testname ${testsourcefile} )
        string(FIND ${testname} "/" position REVERSE)
        MATH(EXPR position "${position}+1")
        string(LENGTH ${testname} length)
        string(SUBSTRING ${testname} ${position} ${length} filename)
        add_executable( "${filename}" ${testsourcefile} )
        # Make sure YourLib is linked to each app
        target_link_libraries( ${filename} ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY} )
    endforeach( testsourcefile ${APP_SOURCES} )
endif()