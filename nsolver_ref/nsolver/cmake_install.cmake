# Install script for directory: /home/ebenq/Dev/c/channelflow/nsolver

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
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libnsolver.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libnsolver.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libnsolver.so"
         RPATH "/home/ebenq/.local/lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES "/home/ebenq/Dev/c/channelflow/nsolver/libnsolver.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libnsolver.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libnsolver.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libnsolver.so"
         OLD_RPATH "::::::::::::::::::::::"
         NEW_RPATH "/home/ebenq/.local/lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/nix/store/ii75mhh7sxl11167m1b86p0qrjsjyjmd-gcc-wrapper-14-20241116/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libnsolver.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  include("/home/ebenq/Dev/c/channelflow/nsolver/CMakeFiles/nsolver.dir/install-cxx-module-bmi-debug.cmake" OPTIONAL)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/nsolver" TYPE FILE FILES
    "/home/ebenq/Dev/c/channelflow/nsolver/arnoldi.h"
    "/home/ebenq/Dev/c/channelflow/nsolver/lanczos.h"
    "/home/ebenq/Dev/c/channelflow/nsolver/bicgstab.h"
    "/home/ebenq/Dev/c/channelflow/nsolver/bicgstabl.h"
    "/home/ebenq/Dev/c/channelflow/nsolver/continuation.h"
    "/home/ebenq/Dev/c/channelflow/nsolver/dsi.h"
    "/home/ebenq/Dev/c/channelflow/nsolver/gmres.h"
    "/home/ebenq/Dev/c/channelflow/nsolver/fgmres.h"
    "/home/ebenq/Dev/c/channelflow/nsolver/newtonalgorithm.h"
    "/home/ebenq/Dev/c/channelflow/nsolver/newton.h"
    "/home/ebenq/Dev/c/channelflow/nsolver/nsolver.h"
    "/home/ebenq/Dev/c/channelflow/nsolver/eigenvals.h"
    "/home/ebenq/Dev/c/channelflow/nsolver/multiShootingDSI.h"
    )
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
if(CMAKE_INSTALL_LOCAL_ONLY)
  file(WRITE "/home/ebenq/Dev/c/channelflow/nsolver/install_local_manifest.txt"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
endif()
