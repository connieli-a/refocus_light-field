@echo off
REM 进入构建目录
cd out\build\x64-debug
REM cd out\build\x64-release

REM 运行 ninja 编译
ninja -v

REM 编译成功后运行程序
if %ERRORLEVEL% neq 0 (
    echo Build failed.
    pause
    exit /b %ERRORLEVEL%
)
cd ..\..\..
.\out\build\x64-debug\refocus_cmake\refocus_cmake.exe

pause
