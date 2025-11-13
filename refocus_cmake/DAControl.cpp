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
}
DAControl::~DAControl(){
    // Close the device.
	DaClose(hDeviceHandle);
}

void DAControl::outputLight(float value){
	if(!hDeviceHandle) return;
	getData(value);
}
void DAControl::getData(float value){
    // Configure the number of output channels to 2 channels.
	DaSmplChReq[0].ulChNo = 1;
	DaSmplChReq[0].ulRange = DA_10V;
	
	//DaSmplChReq[1].ulChNo = 2;
	//DaSmplChReq[1].ulRange = DA_5V;

	double vref = 10.0;
	double outputvoltage = 0.0;
	if(!std::isnan(value)){
		outputvoltage = 5;
	}
	
	nRet = DaGetDeviceInfo(hDeviceHandle, &DaBoardSpec);
	if (nRet != DA_ERROR_SUCCESS) {
		printf("DaGetDeviceInfo errr(%lx)", nRet);
		DaClose(hDeviceHandle);
		return;
	}
	unsigned long N = DaBoardSpec.ulResolution;
	// std::cout << N << std::endl;
	// double voltage = ((double)Data[0] / (pow(2, N) - 1)) * (2.0 * vref) - vref;
	unsigned long rawData = static_cast<unsigned long>(
		((outputvoltage + vref) / (2 * vref)) * (pow(2 , N) - 1)
	);
	// Start the analog output.
	Data[0] = rawData;
	nRet = DaOutputDA(hDeviceHandle, 1, &DaSmplChReq[0], Data);
	if (nRet != DA_ERROR_SUCCESS) {
		printf("DaOutputDA errr(%lx)", nRet);

		DaClose(hDeviceHandle);
		return;
	}
}
// void main(void)
// {
	

// 	// Open a device.


	

	
// 	while (!_kbhit()) {
// 		Sleep(10000);
// 	}
	
// }
