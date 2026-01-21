#include "include/DAControl.h"


// ----------------------------------------------------------------
//
//		Example of output one analog data output
//
// ----------------------------------------------------------------
DAControl::DAControl(){
    #ifdef _WIN64
	hDeviceHandle = DaOpen((LPCSTR)"FBIDA1");

    #else
        hDeviceHandle = DaOpen((LPCTSTR)"FBIDA1");
    #endif
    if( hDeviceHandle == NULL ){
		fprintf( stderr, "ERROR: Failed to open device!\n");
		return ;
	}
	nRet = DaGetDeviceInfo(hDeviceHandle, &DaBoardSpec);
	if (nRet != DA_ERROR_SUCCESS) {
		printf("DaGetDeviceInfo errr(%lx)", nRet);
		return;
	}
	//setupchannel
	DaSmplChReq[0].ulChNo = 1;
	DaSmplChReq[0].ulRange = DA_10V;
}
DAControl::~DAControl(){
    // Close the device.
	DaClose(hDeviceHandle);
}

void DAControl::outputVoltage(float value){
	if(!hDeviceHandle) return;
	if (value < DA_MIN) value = DA_MIN;
	if (value > DA_MAX) value = DA_MAX;
	getData(value);
}
void DAControl::outputLight(float value){
	if(!hDeviceHandle) return;
	if(!std::isnan(value)){
		getData(5);
	}
}
void DAControl::getData(float value){
    
	double vref = 10.0;
	unsigned long N = DaBoardSpec.ulResolution;
	// std::cout << N << std::endl;
	// double voltage = ((double)Data[0] / (pow(2, N) - 1)) * (2.0 * vref) - vref;
	unsigned long rawData = static_cast<unsigned long>(
		((value + vref) / (2 * vref)) * (pow(2 , N) - 1)
	);
	// Start the analog output.
	Data[0] = rawData;
	nRet = DaOutputDA(hDeviceHandle, 1, &DaSmplChReq[0], Data);
	if (nRet != DA_ERROR_SUCCESS) {
		printf("DaOutputDA errr(%lx)", nRet);
		return;
	}
}
