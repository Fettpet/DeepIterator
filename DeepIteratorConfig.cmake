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
# Required cmake version.

CMAKE_MINIMUM_REQUIRED(VERSION 3.4.0)

################################################################################

UNSET(DeepIterator_FOUND)
UNSET(DeepIterator_VERSION)
UNSET(DeepIterator_COMPILE_OPTIONS)
UNSET(DeepIterator_COMPILE_DEFINITIONS)
UNSET(DeepIterator_DEFINITIONS)
UNSET(DeepIterator_INCLUDE_DIR)
UNSET(DeepIterator_INCLUDE_DIRS)
UNSET(DeepIterator_LIBRARY)
UNSET(DeepIterator_LIBRARIES)


#-------------------------------------------------------------------------------
# Common.

# Directory of this file
# Directory of this file.
SET(_DEEPITERATOR_ROOT_DIR ${CMAKE_CURRENT_LIST_DIR})
# Normalize the path (e.g. remove ../)
GET_FILENAME_COMPONENT(_DEEPITERATOR_ROOT_DIR "${_DEEPITERATOR_ROOT_DIR}" ABSOLUTE)

SET(_DEEPITERATOR_COMMON_FILE "${_DEEPITERATOR_ROOT_DIR}/cmake/common.cmake")
INCLUDE("${_DEEPITERATOR_COMMON_FILE}")

set(DeepIterator_FOUND True)
#-------------------------------------------------------------------------------
# Options.
OPTION(DEEPITERATOR_DEBUG "Use debugging informations" ON)
OPTION(DEEPITERATOR_BUILD_TESTS "Build the test to verify the correctness of the deepiterator." ON)

OPTION(DEEPITERATOR_TEST_BIDIRECTIONAL "Build the Bidirectional Test" ON)
OPTION(DEEPITERATOR_TEST_CUDA "Build the Bidirectional Test" ON)
OPTION(DEEPITERATOR_TEST_FORWARD "Build the Bidirectional Test" ON)
OPTION(DEEPITERATOR_TEST_NDARRAY "Build the Bidirectional Test" ON)
OPTION(DEEPITERATOR_TEST_NESTEDITERATOR "Build the Bidirectional Test" ON)
OPTION(DEEPITERATOR_TEST_NEWCONTAINER "Build the Bidirectional Test" ON)
OPTION(DEEPITERATOR_TEST_OWNCATEGORIE "Build the Bidirectional Test" ON)
OPTION(DEEPITERATOR_TEST_RANDOMACCESS "Build the Bidirectional Test" ON)
OPTION(DEEPITERATOR_TEST_RUNTIME "Build the Bidirectional Test" ON)
OPTION(DEEPITERATOR_TEST_SLICENAVIGATOR "Build the Bidirectional Test" ON)
OPTION(DEEPITERATOR_TEST_UNNESTETITERATOR "Build the Bidirectional Test" ON)

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

SET(_DEEPITERATOR_TEST_FILE "${_DEEPITERATOR_ROOT_DIR}/cmake/test.cmake")
SET(_DEEPITERATOR_TEST_DIR "${_DEEPITERATOR_ROOT_DIR}/Test")
MESSAGE(STATUS "Deep Iterator Test Dir: ${_DEEPITERATOR_TEST_DIR}")
INCLUDE("${_DEEPITERATOR_TEST_FILE}")
MESSAGE(STATUS "Test Cmake: ${_DEEPITERATOR_TEST_FILE}") 
#Add all the  include files in all recursive subdirectories and group them accordingly.
#
SET(_DEEPITERATOR_INCLUDE_DIRECTORY "${_DEEPITERATOR_ROOT_DIR}/include")
append_recursive_files_add_to_src_group(
    "${_DEEPITERATOR_INCLUDE_DIRECTORY}" 
    "${_DEEPITERATOR_INCLUDE_DIRECTORY}"
    "hpp" 
    _DEEPITERATOR_FILES_HEADER
)


#-------------------------------------------------------------------------------
# Debug output of common variables.
IF(${DEEPITERATOR_DEBUG} )
    MESSAGE(STATUS "_DEEPITERATOR_ROOT_DIR : ${_DEEPITERATOR_ROOT_DIR}")
#    MESSAGE(STATUS "_ALPAKA_COMMON_FILE : ${_ALPAKA_COMMON_FILE}")
#    MESSAGE(STATUS "_ALPAKA_ADD_EXECUTABLE_FILE : ${_ALPAKA_ADD_EXECUTABLE_FILE}")
#    MESSAGE(STATUS "_ALPAKA_ADD_LIBRARY_FILE : ${_ALPAKA_ADD_LIBRARY_FILE}")
    MESSAGE(STATUS "CMAKE_BUILD_TYPE : ${CMAKE_BUILD_TYPE}")
    MESSAGE(STATUS "DeepIterator include directory: ${_DEEPITERATOR_INCLUDE_DIRECTORY}")
    MESSAGE(STATUS "Files : ${_DEEPITERATOR_FILES_HEADER}")
ENDIF()

#-------------------------------------------------------------------------------
# Set return values.
SET(DeepIterator_INCLUDE_DIR ${_DEEPITERATOR_INCLUDE_DIRECTORY})
include_directories(${DeepIterator_INCLUDE_DIR})


###############################################################################
# FindPackage options

# Handles the REQUIRED, QUIET and version-related arguments for FIND_PACKAGE.
# NOTE: We do not check for alpaka_LIBRARIES and alpaka_DEFINITIONS because they can be empty.
INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(
    "DeepIterator"
    FOUND_VAR DeepIterator_FOUND
    REQUIRED_VARS DeepIterator_INCLUDE_DIR
    VERSION_VAR 0.1
)

