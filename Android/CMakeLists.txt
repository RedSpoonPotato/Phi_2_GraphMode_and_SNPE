cmake_minimum_required(VERSION 3.22.1)

project("nativecr")

add_library(nativecr SHARED native-lib.cpp)

add_library(test2 SHARED test2.cpp)
target_include_directories(test2 PUBLIC test2.h)
target_include_directories(nativecr PUBLIC test2.h)
target_include_directories(nativecr PUBLIC test.h)

target_include_directories(nativecr PUBLIC test3.h)
target_include_directories(nativecr PUBLIC android_main.h)

add_library(libc++_shared SHARED IMPORTED)
add_library(libmain SHARED IMPORTED)
add_library(libSNPE SHARED IMPORTED)
#add_library(libUdoDecodePackageImplCpu SHARED IMPORTED)
#add_library(libUdoDecodePackageReg SHARED IMPORTED)
#add_library(libtest3 SHARED IMPORTED)

# change to "arm64-v8a" to "x86_64" to use x86_64 .so files, assuming all directories are valid
set (targetArch "arm64-v8a")

# ensure you are using same android platform version as project (android-24) when compiling these
set_target_properties(libc++_shared PROPERTIES IMPORTED_LOCATION C:/Users/kunal/AndroidStudioProjects/NativeCR/app/src/main/jniLibs/${targetArch}/libc++_shared.so)
set_target_properties(libmain PROPERTIES IMPORTED_LOCATION C:/Users/kunal/AndroidStudioProjects/NativeCR/app/src/main/jniLibs/${targetArch}/libmain.so)
set_target_properties(libSNPE PROPERTIES IMPORTED_LOCATION C:/Users/kunal/AndroidStudioProjects/NativeCR/app/src/main/jniLibs/${targetArch}/libSNPE.so)
#set_target_properties(libUdoDecodePackageImplCpu PROPERTIES IMPORTED_LOCATION C:/Users/kunal/AndroidStudioProjects/NativeCR/app/src/main/jniLibs/${targetArch}/libUdoDecodePackageImplCpu.so)
#set_target_properties(libUdoDecodePackageReg PROPERTIES IMPORTED_LOCATION C:/Users/kunal/AndroidStudioProjects/NativeCR/app/src/main/jniLibs/${targetArch}/libUdoDecodePackageReg.so)
#set_target_properties(libtest3 PROPERTIES IMPORTED_LOCATION C:/Users/kunal/AndroidStudioProjects/NativeCR/app/src/main/jniLibs/${targetArch}/libtest3.so)


#target_link_libraries(nativecr libc++_shared)
#target_link_libraries(nativecr libtest3)
target_link_libraries(nativecr libmain libSNPE libc++_shared)

find_library(log-lib log)

target_link_libraries(nativecr ${log-lib})
target_link_libraries(nativecr test2)
