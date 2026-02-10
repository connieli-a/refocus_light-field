#include <windows.h>
#include <stdio.h>
#include <conio.h>
#include <math.h>
#include <malloc.h>
#include	<mmsystem.h>
#include  "GPC3300/Fbida.h"
#include  <iostream>

class DAControl{
    public:
        DAControl();
        ~DAControl();
        void outputLight(float value, int channel);
        void outputVoltage(float value, int channel);
       
    private:
        HANDLE		hDeviceHandle;		// Device handle
        WORD		Data[2];			// Output data storage area
        DASMPLCHREQ	DaSmplChReq[2];		// Output conditions setting structure
        int			nRet;
        DABOARDSPEC DaBoardSpec;
        float DA_MIN = -10;
        float DA_MAX = 10;
        void getData(float value, int channel);
        
};

extern DAControl da;