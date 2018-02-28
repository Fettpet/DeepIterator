#.rst:
# Findalpaka
# ----------
#
# Abstraction library for iterator
# https://github.com/Fettpet/DeepIterator
#
# Finding and Using alpaka
# ^^^^^^^^^^^^^^^^^^^^^
#
# .. code-block:: cmake
#
#   FIND_PACKAGE(DeepIterator
#     [version] [EXACT]     # Minimum or EXACT version, e.g. 1.0.0
#     [REQUIRED]            # Fail with an error if alpaka or a required
#                           # component is not found
#     [QUIET]               # Do not warn if this module was not found
#     [COMPONENTS <...>]    # Compiled in components: ignored
#   )
#   TARGET_LINK_LIBRARIES(<target> PUBLIC alpaka)
#
# To provide a hint to this module where to find the alpaka installation,
# set the ALPAKA_ROOT variable.
#
# This module requires Boost. Make sure to provide a valid install of it
# under the environment variable BOOST_ROOT.
#


################################################################################
# Copyright 2018 Sebastian Hahn
#
# Permission to use, copy, modify, and/or distribute this software for any
# purpose with or without fee is hereby granted, provided that the above
# copyright notice and this permission notice appear in all copies.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
# WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY
# SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER
# RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT,
# NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE
# USE OR PERFORMANCE OF THIS SOFTWARE.

FIND_PATH(
    _DeepIterator_ROOT_DIR
    NAMES "include/DeepIterator.hpp"
    HINTS "${DeepIterator_ROOT}" ENV ALPAKA_ROOT
    DOC "DeepIterator ROOT location")

IF(_DeepIterator_ROOT_DIR)
    INCLUDE("${_DeepIterator_ROOT_DIR}/DeepIteratorConfig.cmake")
ELSE()
    MESSAGE(FATAL_ERROR "DeepIterator could not be found!")
ENDIF()
