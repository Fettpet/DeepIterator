#
# Copyright 2018 Sebastian Hahn
#
# This file is part of DeepIterator.
#
# DeepIterator is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# DeepIterator is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with DeepIterator.
# If not, see <http://www.gnu.org/licenses/>.
#

################################################################################

cmake_minimum_required(VERSION 3.7)

project(DeepIterator VERSION 0.1.0 LANGUAGES CXX)


################################################################################
# Variables
################################################################################
SET(_DEEPITERATOR_ROOT_DIR ${CMAKE_CURRENT_LIST_DIR})
SET(DeepIterator_INCLUDE_DIR "${_DEEPITERATOR_ROOT_DIR}/include")
SET(_DEEPITERATOR_TEST_DIR "${_DEEPITERATOR_ROOT_DIR}/Test")
UNSET(DEEPITERATOR_FILES_HEADER)
SET(_DEEPITERATOR_COMMON_FILE "${_DEEPITERATOR_ROOT_DIR}/cmake/common.cmake")
INCLUDE("${_DEEPITERATOR_COMMON_FILE}")

################################################################################
# Options and Variants 
################################################################################
function(deepiterator_option name description default)
    set(DeepIterator_${name} ${default} CACHE STRING "${description}")
    set_property(CACHE DeepIterator_${name} PROPERTY
        STRINGS "ON;TRUE;AUTO;OFF;FALSE"
    )
    if(DeepIterator_HAVE_${name})
        set(DeepIterator_HAVE_${name} TRUE)
    else()
        set(DeepIterator_HAVE_${name})
    endif()
    set(DeepIterator_CONFIG_OPTIONS ${DeepIterator_CONFIG_OPTIONS} ${name} PARENT_SCOPE)
endfunction()


# Set Release if it isn't set
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING
        "Choose the build type, e.g. Debug." FORCE)
endif()

# Options.
OPTION(DEEPITERATOR_DEBUG "Use debugging informations" ON)
OPTION(DEEPITERATOR_BUILD_TESTS "Build the test to verify the correctness of the deepiterator." ON)

OPTION(DEEPITERATOR_TEST_BIDIRECTIONAL "Build the Bidirectional Test" ON)
OPTION(DEEPITERATOR_TEST_CUDA "Build the Cuda Test" ON)
OPTION(DEEPITERATOR_TEST_FORWARD "Build the Forward Test" ON)
OPTION(DEEPITERATOR_TEST_NDARRAY "Build the NdArray Test" ON)
OPTION(DEEPITERATOR_TEST_NESTEDITERATOR "Build the Nested Test" ON)
OPTION(DEEPITERATOR_TEST_NEWCONTAINER "Build the new container Test" ON)
OPTION(DEEPITERATOR_TEST_OWNCATEGORIE "Build the own categorie Test" ON)
OPTION(DEEPITERATOR_TEST_RANDOMACCESS "Build the Random access Test" ON)
OPTION(DEEPITERATOR_TEST_RUNTIME "Build the runtime Test" ON)
OPTION(DEEPITERATOR_TEST_SLICENAVIGATOR "Build the slice navigator Test" ON)
OPTION(DEEPITERATOR_TEST_UNNESTETITERATOR "Build the unnested Test" ON)

mark_as_advanced(FORCE 
    DEEPITERATOR_TEST_BIDIRECTIONAL
    DEEPITERATOR_TEST_CUDA
    DEEPITERATOR_TEST_FORWARD
    DEEPITERATOR_TEST_NDARRAY
    DEEPITERATOR_TEST_NESTEDITERATOR
    DEEPITERATOR_TEST_OWNCATEGORIE
    DEEPITERATOR_TEST_NEWCONTAINER
    DEEPITERATOR_TEST_RANDOMACCESS
    DEEPITERATOR_TEST_RUNTIME
    DEEPITERATOR_TEST_SLICENAVIGATOR
    DEEPITERATOR_TEST_UNNESTETITERATOR
)

# sources
append_recursive_files_add_to_src_group(
    "${DeepIterator_INCLUDE_DIR}" 
    "${DeepIterator_INCLUDE_DIR}"
    "hpp" 
    DEEPITERATOR_FILES_HEADER
)

# install  #####################################################################
#
add_library(DeepIterator INTERFACE)

configure_file(
    ${_DEEPITERATOR_ROOT_DIR}/DeepIteratorConfig.cmake.in
    ${DeepIterator_BINARY_DIR}/DeepIteratorConfig.cmake
    @ONLY
)

include(CMakePackageConfigHelpers)
write_basic_package_version_file("DeepIteratorConfigVersion.cmake"
    VERSION ${DeepIterator_VERSION}
    COMPATIBILITY SameMajorVersion # actually: SameMinorVersion
)

set_property(TARGET DeepIterator APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>  
    $<INSTALL_INTERFACE:include/>
)

target_include_directories(DeepIterator INTERFACE  
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>  
    $<INSTALL_INTERFACE:include/>
)

install(TARGETS DeepIterator EXPORT DeepIteratorTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    PUBLIC_HEADER DESTINATION include
    INCLUDES DESTINATION include
)

install(
    DIRECTORY ${CMAKE_SOURCE_DIR}/include DESTINATION ${CMAKE_INSTALL_PREFIX}
)

#install(EXPORT DeepIteratorTargets DESTINATION lib/cmake/DeepIterator)

install(
    FILES
        ${DeepIterator_BINARY_DIR}/DeepIteratorConfig.cmake
        ${DeepIterator_BINARY_DIR}/DeepIteratorConfigVersion.cmake
    DESTINATION lib/cmake/DeepIterator
)

install(EXPORT DeepIteratorTargets
    FILE DeepIteratorTargets.cmake
    NAMESPACE DeepIterator::
    DESTINATION lib/cmake/DeepIterator
)

# TEST Things ######################################
include(cmake/test.cmake)
enable_testing()


if(DEEPITERATOR_TEST_BIDIRECTIONAL)
    add_test(
        NAME Bidirectional        
        COMMAND Tests/Bidirectional
    )
endif()
if(DEEPITERATOR_TEST_FORWARD)
    add_test(
        NAME Forward        
        COMMAND Tests/Forward
    )
endif()        
if(DEEPITERATOR_TEST_NDARRAY)
    add_test(
        NAME NdArray        
        COMMAND Tests/NdArray
    )
endif()        
if(DEEPITERATOR_TEST_NESTEDITERATOR)
    add_test(
        NAME NestedIterator        
        COMMAND Tests/NestedIterator
    )
endif()        
if(DEEPITERATOR_TEST_OWNCATEGORIE)
    add_test(
        NAME OwnCategorie        
        COMMAND Tests/OwnCategorie
    )
endif()
if(DEEPITERATOR_TEST_NEWCONTAINER)
    add_test(
        NAME NewContainer        
        COMMAND Tests/NewContainer
    )
endif()
if(DEEPITERATOR_TEST_RANDOMACCESS)
    add_test(
        NAME RandomAccess        
        COMMAND Tests/RandomAccess
    )
endif()
if(DEEPITERATOR_TEST_RUNTIME)
    add_test(
        NAME Runtime        
        COMMAND Tests/Runtime
    )
endif()
if(DEEPITERATOR_TEST_SLICENAVIGATOR)
    add_test(
        NAME SliceNavigator        
        COMMAND Tests/SliceNavigator
    )
endif()
if(DEEPITERATOR_TEST_UNNESTETITERATOR)
    add_test(
        NAME UnnestedIterator        
        COMMAND Tests/UnnestedIterator
    )
endif()
if(DEEPITERATOR_TEST_CUDA)
    add_test(
        NAME Cuda        
        COMMAND Tests/Cuda
    )
endif()

