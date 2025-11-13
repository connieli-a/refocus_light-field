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
        void outputLight(float value);
    private:
        HANDLE		hDeviceHandle;		// Device handle
        WORD		Data[2];			// Output data storage area
        DASMPLCHREQ	DaSmplChReq[2];		// Output conditions setting structure
        int			nRet;
        DABOARDSPEC DaBoardSpec;

        void getData(float value);
        
};

extern DAControl da;