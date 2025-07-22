@echo off
REM 进入构建目录
REM cd out\build\x64-debug
cd out\build\x64-release

REM 运行 ninja 编译
ninja

REM 编译成功后运行程序
if %ERRORLEVEL% neq 0 (
    echo Build failed.
    pause
    exit /b %ERRORLEVEL%
)
cd refocus_cmake
refocus_cmake.exe

pause
