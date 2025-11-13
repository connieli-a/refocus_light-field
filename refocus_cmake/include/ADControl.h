#include <windows.h>
#include <stdio.h>
#include "GPC3100/Fbiad.h"
#include <cmath>
#include <iostream>
#include <conio.h>

// -------------------------------------------------------
// コールバック関数
// -------------------------------------------------------
#ifdef _WIN64
void CALLBACK EventProc(PVOID dwUser)
{
	printf("Event callback fired!\n");
}
#else
void CALLBACK EventProc(DWORD dwUser){}
#endif

class ADControl{
    public:
        
        ADControl(HANDLE* hDeviceHandle);
	    ~ADControl();
    private:
        ADBOARDSPEC 	BoardSpec;
        ADSMPLREQ		AdSmplConfig;
        unsigned char	bSmplData[2];
        unsigned short	wSmplData[2];
        unsigned long	dwSmplData[2];
        int		nRet;
	    HANDLE	hDeviceHandle;		// デバイスハンドル

        int Initialization(HANDLE* hDeviceHandle);
        int GetData(HANDLE hDeviceHandle);
        int End();
};

