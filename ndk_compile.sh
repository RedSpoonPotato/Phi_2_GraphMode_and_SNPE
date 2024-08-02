export ANDROID_NDK_ROOT="/home/kernal1/android-ndk-r19c/build/"
export PATH="/home/kernal1/android-ndk-r19c/toolchains/llvm/prebuilt/linux-x86_64/bin:/opt/qcom/aistack/snpe/2.20.0.240223/bin/x86_64-linux-clang:/opt/qcom/aistack/qnn/2.16.4.231110/bin/x86_64-linux-clang:/home/kernal1/my_env/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/usr/lib/wsl/lib:/snap/bin"
export PATH=$ANDROID_NDK_ROOT:$PATH
ndk-build NDK_TOOLCHAIN_VERSION=clang APP_STL=c++_shared
