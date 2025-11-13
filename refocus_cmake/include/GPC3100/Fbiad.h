// -----------------------------------------------------------------------
//
//    GPC/GPF-3100 Header file
//
//		File Name: FbiAd.h
//
//    Copyright 1999, 2010 Interface Corporation. All rights reserved.
// -----------------------------------------------------------------------

#include "GpcAd.h"

#ifdef __cplusplus
extern	"C" {
#endif

#if !defined(_FBIADLIB_)
#define FBIADAPI
#else
#define FBIADAPI __declspec(dllexport)
#endif

//-----------------------------------------------------------------------------------------------
//
//   GPC/GPF-3100 Function Declaration
//
//-----------------------------------------------------------------------------------------------
#ifdef _WIN64
FBIADAPI HANDLE WINAPI AdOpen(LPCSTR);
FBIADAPI int WINAPI AdSetBoardConfig(HANDLE, HANDLE, LPADCALLBACK, PVOID);
FBIADAPI int WINAPI AdLvSetSamplingConfig(HANDLE, ULONG, ULONG, float, ULONG, HANDLE, LPADCALLBACK, PVOID);
#else
FBIADAPI HANDLE WINAPI AdOpen(LPCTSTR);
FBIADAPI int WINAPI AdSetBoardConfig(HANDLE, HANDLE, LPADCALLBACK, DWORD);
FBIADAPI int WINAPI AdLvSetSamplingConfig(HANDLE, ULONG, ULONG, float, ULONG, HANDLE, LPADCALLBACK, DWORD);
#endif
FBIADAPI int WINAPI AdClose(HANDLE);
FBIADAPI int WINAPI AdGetDeviceInfo(HANDLE, PADBOARDSPEC);
FBIADAPI int WINAPI AdGetBoardConfig(HANDLE, ULONG *);
FBIADAPI int WINAPI AdSetSamplingConfig(HANDLE, PADSMPLREQ);
FBIADAPI int WINAPI AdGetSamplingConfig(HANDLE, PADSMPLREQ);
FBIADAPI int WINAPI AdGetSamplingData(HANDLE, PVOID, ULONG *);
FBIADAPI int WINAPI AdClearSamplingData(HANDLE);
FBIADAPI int WINAPI AdStartSampling(HANDLE, ULONG);
FBIADAPI int WINAPI AdStartFileSampling(HANDLE, LPCTSTR, ULONG);
FBIADAPI int WINAPI AdTriggerSampling(HANDLE, ULONG, ULONG, ULONG, ULONG, ULONG, ULONG);
FBIADAPI int WINAPI AdMemTriggerSampling(HANDLE, ULONG, PADSMPLCHREQ, ULONG, ULONG, ULONG, float, ULONG, ULONG);
FBIADAPI int WINAPI AdSyncSampling(HANDLE, ULONG);
FBIADAPI int WINAPI AdStopSampling(HANDLE);
FBIADAPI int WINAPI AdGetStatus(HANDLE, ULONG *, ULONG *, ULONG *);
FBIADAPI int WINAPI AdInputAD(HANDLE, ULONG, ULONG, PADSMPLCHREQ, LPVOID);
FBIADAPI int WINAPI AdInputDI(HANDLE, DWORD *);
FBIADAPI int WINAPI AdOutputDO(HANDLE, DWORD);
FBIADAPI int WINAPI AdAdjustVR(HANDLE, ULONG, ULONG, ULONG, ULONG, ULONG);
FBIADAPI int WINAPI AdReadAdjustVR(HANDLE, ULONG);
FBIADAPI int WINAPI AdReadAdjustVREx(HANDLE, ULONG, ULONG);

FBIADAPI int WINAPI AdBmSetSamplingConfig(HANDLE, PADBMSMPLREQ);
FBIADAPI int WINAPI AdBmGetSamplingConfig(HANDLE, PADBMSMPLREQ);
FBIADAPI int WINAPI AdBmGetSamplingData(HANDLE, PVOID, ULONG *);
FBIADAPI int WINAPI AdBmStartFileSampling(HANDLE, LPCTSTR, ULONG);

FBIADAPI int WINAPI AdLvGetSamplingConfig(HANDLE, ULONG, ULONG *, float *, ULONG *);
FBIADAPI int WINAPI AdLvCalibration(HANDLE, ULONG, ULONG);
FBIADAPI int WINAPI AdLvGetSamplingData(HANDLE, ULONG, PVOID, ULONG *);
FBIADAPI int WINAPI AdLvStartSampling(HANDLE, ULONG);
FBIADAPI int WINAPI AdLvStopSampling(HANDLE, ULONG);
FBIADAPI int WINAPI AdLvGetStatus(HANDLE, ULONG, ULONG *, ULONG *, ULONG *);
FBIADAPI int WINAPI AdMeasureTemperature(HANDLE, float *);
FBIADAPI INT WINAPI AdMeasureTemperatureEx(HANDLE, float *, ULONG);

FBIADAPI int WINAPI AdAllocateSamplingBuffer(HANDLE);
FBIADAPI int WINAPI AdReadSamplingBuffer(HANDLE, LONG, ULONG *, PVOID);

FBIADAPI int WINAPI AdDataConv(UINT, PVOID, UINT, PADSMPLREQ, UINT, PVOID, PUINT, PADSMPLREQ, UINT, UINT, LPCONVPROC);
FBIADAPI int WINAPI AdReadFile(LPCTSTR, PVOID, ULONG);

FBIADAPI int WINAPI AdSetRangeEvent(HANDLE, DWORD, DWORD);
FBIADAPI int WINAPI AdGetRangeEventStatus(HANDLE, ULONG *, ULONG *);
FBIADAPI int WINAPI AdResetRangeEvent(HANDLE);

FBIADAPI int WINAPI AdGetOverRangeChStatus(HANDLE, ULONG *);
FBIADAPI int WINAPI AdResetOverRangeCh(HANDLE);

FBIADAPI int WINAPI AdSetInterval(HANDLE, ULONG);
FBIADAPI int WINAPI AdGetInterval(HANDLE, ULONG *);
FBIADAPI int WINAPI AdSetFunction(HANDLE, ULONG, ULONG);
FBIADAPI int WINAPI AdGetFunction(HANDLE, ULONG, ULONG *);
FBIADAPI int WINAPI AdSetFilter(HANDLE, ULONG);
FBIADAPI int WINAPI AdGetFilter(HANDLE, ULONG*);

FBIADAPI int WINAPI AdFifoGetSamplingData(HANDLE, PVOID, PVOID, ULONG *);

FBIADAPI int WINAPI AdCommonGetPciDeviceInfo(HANDLE, DWORD *, DWORD *, DWORD *, DWORD *, DWORD *, DWORD *, DWORD *, DWORD *, DWORD *, DWORD *, DWORD *, DWORD *, DWORD *, DWORD *);

FBIADAPI int WINAPI AdMemSetSamplingConfig(HANDLE, PADMEMSMPLREQ);
FBIADAPI int WINAPI AdMemGetSamplingConfig(HANDLE, PADMEMSMPLREQ);
FBIADAPI int WINAPI AdSetOutMode(HANDLE, ULONG, ULONG);
FBIADAPI int WINAPI AdGetOutMode(HANDLE, ULONG *, ULONG *);
FBIADAPI int WINAPI AdMemSetDiPattern(HANDLE, ULONG, ULONG *);
FBIADAPI int WINAPI AdMemGetDiPattern(HANDLE, ULONG *);

FBIADAPI int WINAPI AdBmStartSampling(HANDLE, PVOID, ULONG);


FBIADAPI int WINAPI AdCalibration(HANDLE, ULONG, ULONG, ULONG);
FBIADAPI int WINAPI AdOutputSync(HANDLE, ULONG, ULONG);

FBIADAPI int WINAPI AdSetEndEventConfig(HANDLE, int);

#ifdef __cplusplus
}
#endif

//#endif
