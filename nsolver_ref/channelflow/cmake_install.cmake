# Install script for directory: /home/ebenq/Dev/c/channelflow/channelflow

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/home/ebenq/.local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "debug")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "0")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set path to fallback-tool for dependency-resolution.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/nix/store/ii75mhh7sxl11167m1b86p0qrjsjyjmd-gcc-wrapper-14-20241116/bin/objdump")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libchflow.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libchflow.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libchflow.so"
         RPATH "/home/ebenq/.local/lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES "/home/ebenq/Dev/c/channelflow/channelflow/libchflow.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libchflow.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libchflow.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libchflow.so"
         OLD_RPATH "/home/ebenq/Dev/c/channelflow/nsolver:"
         NEW_RPATH "/home/ebenq/.local/lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/nix/store/ii75mhh7sxl11167m1b86p0qrjsjyjmd-gcc-wrapper-14-20241116/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libchflow.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  include("/home/ebenq/Dev/c/channelflow/channelflow/CMakeFiles/chflow.dir/install-cxx-module-bmi-debug.cmake" OPTIONAL)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/channelflow" TYPE FILE FILES
    "/home/ebenq/Dev/c/channelflow/channelflow/bandedtridiag.h"
    "/home/ebenq/Dev/c/channelflow/channelflow/basisfunc.h"
    "/home/ebenq/Dev/c/channelflow/channelflow/cfmpi.h"
    "/home/ebenq/Dev/c/channelflow/channelflow/chebyshev.h"
    "/home/ebenq/Dev/c/channelflow/channelflow/diffops.h"
    "/home/ebenq/Dev/c/channelflow/channelflow/dnsflags.h"
    "/home/ebenq/Dev/c/channelflow/channelflow/dnsalgo.h"
    "/home/ebenq/Dev/c/channelflow/channelflow/nse.h"
    "/home/ebenq/Dev/c/channelflow/channelflow/dns.h"
    "/home/ebenq/Dev/c/channelflow/channelflow/flowfield.h"
    "/home/ebenq/Dev/c/channelflow/channelflow/helmholtz.h"
    "/home/ebenq/Dev/c/channelflow/channelflow/periodicfunc.h"
    "/home/ebenq/Dev/c/channelflow/channelflow/poissonsolver.h"
    "/home/ebenq/Dev/c/channelflow/channelflow/realprofile.h"
    "/home/ebenq/Dev/c/channelflow/channelflow/realprofileng.h"
    "/home/ebenq/Dev/c/channelflow/channelflow/symmetry.h"
    "/home/ebenq/Dev/c/channelflow/channelflow/tausolver.h"
    "/home/ebenq/Dev/c/channelflow/channelflow/turbstats.h"
    "/home/ebenq/Dev/c/channelflow/channelflow/utilfuncs.h"
    "/home/ebenq/Dev/c/channelflow/channelflow/cfdsi.h"
    "/home/ebenq/Dev/c/channelflow/channelflow/laurettedsi.h"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/channelflow" TYPE FILE FILES "/home/ebenq/Dev/c/channelflow/channelflow/config.h")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
if(CMAKE_INSTALL_LOCAL_ONLY)
  file(WRITE "/home/ebenq/Dev/c/channelflow/channelflow/install_local_manifest.txt"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
endif()
