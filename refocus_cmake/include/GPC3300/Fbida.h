// -----------------------------------------------------------------------
//
//		Header file
//
//		File name: FbiDa.h
//
//		Copyright 1999, 2010 Interface Corporation. All rights reserved.
// -----------------------------------------------------------------------

#include "GpcDa.h"

#ifdef __cplusplus
extern	"C" {
#endif

#if !defined(_FBIDALIB_)
#define FBIDAAPI
#else
#define FBIDAAPI __declspec(dllexport)
#endif

//-----------------------------------------------------------------------------------------------
//
//		Function Declaration
//
//-----------------------------------------------------------------------------------------------
#ifdef _WIN64
FBIDAAPI HANDLE WINAPI DaOpen(LPCSTR);
FBIDAAPI int WINAPI DaSetBoardConfig(HANDLE, ULONG, HANDLE, LPDACALLBACK, PVOID);
FBIDAAPI int WINAPI DaSetCountEvent(HANDLE, ULONG, HANDLE, LPDACALLBACK, PVOID);
#else
FBIDAAPI HANDLE WINAPI DaOpen(LPCTSTR);
FBIDAAPI int WINAPI DaSetBoardConfig(HANDLE, ULONG, HANDLE, LPDACALLBACK, DWORD);
FBIDAAPI int WINAPI DaSetCountEvent(HANDLE, ULONG, HANDLE, LPDACALLBACK, DWORD);
#endif
FBIDAAPI int WINAPI DaClose(HANDLE);
FBIDAAPI int WINAPI DaGetDeviceInfo(HANDLE, PDABOARDSPEC);
FBIDAAPI int WINAPI DaGetBoardConfig(HANDLE, ULONG *, ULONG *);
FBIDAAPI int WINAPI DaSetSamplingConfig(HANDLE, PDASMPLREQ);
FBIDAAPI int WINAPI DaGetSamplingConfig(HANDLE, PDASMPLREQ);
FBIDAAPI int WINAPI DaSetMode(HANDLE, PDAMODEREQ);
FBIDAAPI int WINAPI DaGetMode(HANDLE, PDAMODEREQ);
FBIDAAPI int WINAPI DaSetSamplingData(HANDLE, PVOID, ULONG);
FBIDAAPI int WINAPI DaClearSamplingData(HANDLE);
FBIDAAPI int WINAPI DaStartSampling(HANDLE, ULONG);
FBIDAAPI int WINAPI DaSyncSampling(HANDLE, ULONG);
FBIDAAPI int WINAPI DaStartFileSampling(HANDLE, LPCTSTR, ULONG, ULONG);
FBIDAAPI int WINAPI DaStopSampling(HANDLE);
FBIDAAPI int WINAPI DaGetStatus(HANDLE, ULONG *, ULONG *, ULONG *, ULONG *);
FBIDAAPI int WINAPI DaOutputDA(HANDLE, ULONG, PDASMPLCHREQ, LPVOID);
FBIDAAPI int WINAPI DaInputDI(HANDLE, DWORD *);
FBIDAAPI int WINAPI DaOutputDO(HANDLE, DWORD);
FBIDAAPI int WINAPI DaAdjustVR(HANDLE, ULONG, ULONG, ULONG, ULONG);
FBIDAAPI int WINAPI DaReadAdjustVR(HANDLE, ULONG);
FBIDAAPI int WINAPI DaReadAdjustVREx(HANDLE, ULONG, ULONG);

FBIDAAPI int WINAPI DaSetFifoConfig(HANDLE, PDAFIFOREQ);
FBIDAAPI int WINAPI DaGetFifoConfig(HANDLE, PDAFIFOREQ);
FBIDAAPI int WINAPI DaSetInterval(HANDLE, ULONG);
FBIDAAPI int WINAPI DaGetInterval(HANDLE, ULONG *);
FBIDAAPI int WINAPI DaSetFunction(HANDLE, ULONG, ULONG);
FBIDAAPI int WINAPI DaGetFunction(HANDLE, ULONG, ULONG *);

FBIDAAPI int WINAPI DaCalibration(HANDLE, ULONG, ULONG);

FBIDAAPI int WINAPI DaSetOutputMode(HANDLE, ULONG);
FBIDAAPI int WINAPI DaGetOutputMode(HANDLE, ULONG *);

FBIDAAPI int WINAPI DaSetCurrentDir(HANDLE hDeviceHandle, DWORD Data);
FBIDAAPI int WINAPI DaGetCurrentDir(HANDLE hDeviceHandle, DWORD *pData);
FBIDAAPI int WINAPI DaSetPowerSupply(HANDLE hDeviceHandle, DWORD ExOnOff);
FBIDAAPI int WINAPI DaGetPowerSupply(HANDLE hDeviceHandle, DWORD *ExOnOff);
FBIDAAPI int WINAPI DaSetExcessVoltage(HANDLE hDeviceHandle, DWORD ExOnOff);
FBIDAAPI int WINAPI DaGetRelayStatus(HANDLE hDeviceHandle, DWORD *Status);
FBIDAAPI int WINAPI DaGetOVStatus(HANDLE hDeviceHandle, DWORD *LowStatus, DWORD *HighStatus);

FBIDAAPI INT WINAPI DaCalibrationEx(HANDLE hDeviceHandle, ULONG Channel, ULONG Range, HANDLE hAdDeviceHandle);

int WINAPI DaDataConv(UINT, PVOID, UINT, PDASMPLREQ, UINT, PVOID, PUINT, PDASMPLREQ, UINT, UINT, LPDACONVPROC);
int WINAPI DaWriteFile(LPCTSTR, PVOID, ULONG, ULONG, ULONG);

#ifdef __cplusplus
}
#endif

//#endif
