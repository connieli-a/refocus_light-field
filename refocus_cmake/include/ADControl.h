#include <windows.h>
#include <stdio.h>
#include "GPC3100/Fbiad.h"
#include <cmath>
#include <iostream>
#include <conio.h>

// -------------------------------------------------------
// コールバック関数
// -------------------------------------------------------


class ADControl{
    public:
        
        ADControl();
	    ~ADControl();
        bool IsReady() const { return initialized; }
        int GetData(float& voltage);
    private:
        ADBOARDSPEC 	BoardSpec;
        ADSMPLREQ		AdSmplConfig;
        unsigned char	bSmplData[2];
        unsigned short	wSmplData[2];
        unsigned long	dwSmplData[2];
	    HANDLE	hDeviceHandle;		// デバイスハンドル
        bool initialized = false;
    
        int Initialization();
        void End();
};

extern ADControl ad;