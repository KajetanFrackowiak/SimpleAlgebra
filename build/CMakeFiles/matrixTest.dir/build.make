# CMAKE generated file: DO NOT EDIT!
# Generated by "MinGW Makefiles" Generator, CMake Version 3.28

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

SHELL = cmd.exe

# The CMake executable.
CMAKE_COMMAND = "C:\Program Files\cmake-3.28.3-windows-x86_64\bin\cmake.exe"

# The command to remove a file.
RM = "C:\Program Files\cmake-3.28.3-windows-x86_64\bin\cmake.exe" -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = C:\Users\kajte\Desktop\SimpleAlgebra

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = C:\Users\kajte\Desktop\SimpleAlgebra\build

# Include any dependencies generated for this target.
include CMakeFiles/matrixTest.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/matrixTest.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/matrixTest.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/matrixTest.dir/flags.make

CMakeFiles/matrixTest.dir/src/matrix.cpp.obj: CMakeFiles/matrixTest.dir/flags.make
CMakeFiles/matrixTest.dir/src/matrix.cpp.obj: CMakeFiles/matrixTest.dir/includes_CXX.rsp
CMakeFiles/matrixTest.dir/src/matrix.cpp.obj: C:/Users/kajte/Desktop/SimpleAlgebra/src/matrix.cpp
CMakeFiles/matrixTest.dir/src/matrix.cpp.obj: CMakeFiles/matrixTest.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=C:\Users\kajte\Desktop\SimpleAlgebra\build\CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/matrixTest.dir/src/matrix.cpp.obj"
	g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/matrixTest.dir/src/matrix.cpp.obj -MF CMakeFiles\matrixTest.dir\src\matrix.cpp.obj.d -o CMakeFiles\matrixTest.dir\src\matrix.cpp.obj -c C:\Users\kajte\Desktop\SimpleAlgebra\src\matrix.cpp

CMakeFiles/matrixTest.dir/src/matrix.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/matrixTest.dir/src/matrix.cpp.i"
	g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E C:\Users\kajte\Desktop\SimpleAlgebra\src\matrix.cpp > CMakeFiles\matrixTest.dir\src\matrix.cpp.i

CMakeFiles/matrixTest.dir/src/matrix.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/matrixTest.dir/src/matrix.cpp.s"
	g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S C:\Users\kajte\Desktop\SimpleAlgebra\src\matrix.cpp -o CMakeFiles\matrixTest.dir\src\matrix.cpp.s

CMakeFiles/matrixTest.dir/test/matrixTest.cpp.obj: CMakeFiles/matrixTest.dir/flags.make
CMakeFiles/matrixTest.dir/test/matrixTest.cpp.obj: CMakeFiles/matrixTest.dir/includes_CXX.rsp
CMakeFiles/matrixTest.dir/test/matrixTest.cpp.obj: C:/Users/kajte/Desktop/SimpleAlgebra/test/matrixTest.cpp
CMakeFiles/matrixTest.dir/test/matrixTest.cpp.obj: CMakeFiles/matrixTest.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=C:\Users\kajte\Desktop\SimpleAlgebra\build\CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/matrixTest.dir/test/matrixTest.cpp.obj"
	g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/matrixTest.dir/test/matrixTest.cpp.obj -MF CMakeFiles\matrixTest.dir\test\matrixTest.cpp.obj.d -o CMakeFiles\matrixTest.dir\test\matrixTest.cpp.obj -c C:\Users\kajte\Desktop\SimpleAlgebra\test\matrixTest.cpp

CMakeFiles/matrixTest.dir/test/matrixTest.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/matrixTest.dir/test/matrixTest.cpp.i"
	g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E C:\Users\kajte\Desktop\SimpleAlgebra\test\matrixTest.cpp > CMakeFiles\matrixTest.dir\test\matrixTest.cpp.i

CMakeFiles/matrixTest.dir/test/matrixTest.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/matrixTest.dir/test/matrixTest.cpp.s"
	g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S C:\Users\kajte\Desktop\SimpleAlgebra\test\matrixTest.cpp -o CMakeFiles\matrixTest.dir\test\matrixTest.cpp.s

# Object files for target matrixTest
matrixTest_OBJECTS = \
"CMakeFiles/matrixTest.dir/src/matrix.cpp.obj" \
"CMakeFiles/matrixTest.dir/test/matrixTest.cpp.obj"

# External object files for target matrixTest
matrixTest_EXTERNAL_OBJECTS =

matrixTest.exe: CMakeFiles/matrixTest.dir/src/matrix.cpp.obj
matrixTest.exe: CMakeFiles/matrixTest.dir/test/matrixTest.cpp.obj
matrixTest.exe: CMakeFiles/matrixTest.dir/build.make
matrixTest.exe: C:/Users/kajte/Desktop/SimpleAlgebra/googletest/build/lib/libgtest.a
matrixTest.exe: C:/Users/kajte/Desktop/SimpleAlgebra/googletest/build/lib/libgtest_main.a
matrixTest.exe: CMakeFiles/matrixTest.dir/linkLibs.rsp
matrixTest.exe: CMakeFiles/matrixTest.dir/objects1.rsp
matrixTest.exe: CMakeFiles/matrixTest.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=C:\Users\kajte\Desktop\SimpleAlgebra\build\CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable matrixTest.exe"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles\matrixTest.dir\link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/matrixTest.dir/build: matrixTest.exe
.PHONY : CMakeFiles/matrixTest.dir/build

CMakeFiles/matrixTest.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles\matrixTest.dir\cmake_clean.cmake
.PHONY : CMakeFiles/matrixTest.dir/clean

CMakeFiles/matrixTest.dir/depend:
	$(CMAKE_COMMAND) -E cmake_depends "MinGW Makefiles" C:\Users\kajte\Desktop\SimpleAlgebra C:\Users\kajte\Desktop\SimpleAlgebra C:\Users\kajte\Desktop\SimpleAlgebra\build C:\Users\kajte\Desktop\SimpleAlgebra\build C:\Users\kajte\Desktop\SimpleAlgebra\build\CMakeFiles\matrixTest.dir\DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/matrixTest.dir/depend

