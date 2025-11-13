// -----------------------------------------------------------------------
//
//     GPC/GPF-3100 Header file
//
//		File Name: GpcAd.h
//
//		Ver 1.02
//
//	Copyright 1999, 2002 Interface Corporation. All rights reserved.
// -----------------------------------------------------------------------

#if !defined( _FbiAd_H_ )
#define _FbiAd_H_

#ifdef __cplusplus
extern	"C" {
#endif

//-----------------------------------------------------------------------------------------------
//
//     GPC/GPF-3100 Overlapped Process Identifier
//
//-----------------------------------------------------------------------------------------------
#define FLAG_SYNC	1	// Sampling as an non-overlapped operation
#define FLAG_ASYNC	2	// Sampling as overlapped operation

//-----------------------------------------------------------------------------------------------
//
//     GPC/GPF-3100 File Format Identifier
//
//-----------------------------------------------------------------------------------------------
#define FLAG_BIN	1	// Binary format file
#define FLAG_CSV	2	// CSV format file

//-----------------------------------------------------------------------------------------------
//
//     GPC/GPF-3100 Sampling Status Identifier
//
//-----------------------------------------------------------------------------------------------
#define AD_STATUS_STOP_SAMPLING		1	// The sampling is stopped.
#define AD_STATUS_WAIT_TRIGGER		2	// The sampling is waiting for a trigger.
#define AD_STATUS_NOW_SAMPLING		3	// The sampling is running.
#define AD_STATUS_NOT_SAMPLING		0x101	// サンプリング未実行です

//-----------------------------------------------------------------------------------------------
//
//     GPC/GPF-3100 Event Factor Identifier
//
//-----------------------------------------------------------------------------------------------
#define AD_EVENT_SMPLNUM			1	// The specified number of samples has been acquired.
#define AD_EVENT_STOP_TRIGGER		2	// The sampling has been stopped because a trigger is asserted.
#define AD_EVENT_STOP_FUNCTION		3	// The sampling has been stopped by software.
#define AD_EVENT_STOP_TIMEOUT		4	// The sampling has been stopped because a time-out interval elapsed.
#define AD_EVENT_STOP_SAMPLING		5	// The sampling is stopped.
#define	AD_EVENT_STOP_SCER			6	// The sampling is stopped by a clock error.
#define	AD_EVENT_STOP_ORER			7	// The sampling is stopped by an overrun error.
#define	AD_EVENT_SCER				8	// The sampling pacer clock error is occurred.
#define	AD_EVENT_ORER				9	// The overrun error is occurred.
#define AD_EVENT_STOP_LV_1			10	// The channel 1 sampling is stopped. (Only applicable to the PCI-3179)
#define AD_EVENT_STOP_LV_2			11	// The channel 2 sampling is stopped. (Only applicable to the PCI-3179)
#define AD_EVENT_STOP_LV_3			12	// The channel 3 sampling is stopped. (Only applicable to the PCI-3179)
#define AD_EVENT_STOP_LV_4			13	// The channel 4 sampling is stopped. (Only applicable to the PCI-3179)
#define AD_EVENT_RANGE				14	// The AD conversion value reached the full-scale range.
#define AD_EVENT_STOP_RANGE			15	// The sampling is stopped by the full-scale range detection.
#define AD_EVENT_OVPM				16	// The AD conversion value reached the OVPM.
#define AD_EVENT_STOP_OVPM			17	// The sampling is stopped by theOVPM.
#define AD_EVENT_EX_INT				18	// Ex Int
#define AD_EVENT_DISCONNECTION		0x100	// Solcon ADPU3215,3216 Disconnection.

//-----------------------------------------------------------------------------------------------
//
//     GPC/GPF-3100 Input Configuration Identifier
//
//-----------------------------------------------------------------------------------------------
#define AD_INPUT_SINGLE				1	// Single-ended input
#define AD_INPUT_DIFF				2	// Differential input

//-----------------------------------------------------------------------------------------------
//
//     GPC/GPF-3100 Volume Identifier
//
//-----------------------------------------------------------------------------------------------
#define AD_ADJUST_BIOFFSET			1	// Bipolar offset
#define AD_ADJUST_UNIOFFSET			2	// Unipolar offset
#define AD_ADJUST_BIGAIN			3	// Bipolar gain
#define AD_ADJUST_UNIGAIN			4	// Unipolar gain

//-----------------------------------------------------------------------------------------------
//
//     GPC/GPF-3100 Calibration Item Identifier
//
//-----------------------------------------------------------------------------------------------
#define AD_ADJUST_UP				1	// Increases the volume.
#define AD_ADJUST_DOWN				2	// Decreases the volumde.
#define AD_ADJUST_STORE				3	// Saves the present value to the non-volatile memory.
#define AD_ADJUST_STANDBY			4	// Places the electronic volume device into the standby mode.
#define AD_ADJUST_NOT_STORE			5	// Not save the value.
#define AD_ADJUST_STORE_INITAREA	6 	// Saves the present value to the non-volatile memory.

//-----------------------------------------------------------------------------------------------
//
//     GPC/GPF-3100 Read Adjust Identifier
//
//-----------------------------------------------------------------------------------------------
#define AD_ADJUST_READ_FACTORY		1
#define AD_ADJUST_READ_USER			2

//-----------------------------------------------------------------------------------------------
//
//     GPC/GPF-3100 Data Identifier
//
//-----------------------------------------------------------------------------------------------
#define AD_DATA_PHYSICAL			1	// Physical value (voltage [V] or current [mA])
#define AD_DATA_BIN8				2	// 8-bit binary
#define AD_DATA_BIN12				3	// 12-bit binary
#define AD_DATA_BIN16				4	// 16-bit binary
#define AD_DATA_BIN24				5	// 24-bit binary
#define AD_DATA_BIN10				6	// 10-bit binary

//-----------------------------------------------------------------------------------------------
//
//     GPC/GPF-3100 Data Conversion Identifier
//
//-----------------------------------------------------------------------------------------------
#define AD_CONV_SMOOTH			1	// Converts the data with interpolation.
#define AD_CONV_AVERAGE1		0x100	// Converts the data with the simple averaging.
#define AD_CONV_AVERAGE2		0x200	// Converts the data with the shifted averaging.

//-----------------------------------------------------------------------------------------------
//
//     GPC/GPF-3100 Sampling Mode Identifier
//
//-----------------------------------------------------------------------------------------------
#define AD_IO_SAMPLING				1	// Programmed I/O
#define AD_FIFO_SAMPLING			2	// FIFO
#define AD_MEM_SAMPLING				4	// Memory
#define	AD_BM_SAMPLING				8	// Bus master

//-----------------------------------------------------------------------------------------------
//
//     GPC/GPF-3100 Trigger Timing Identifier
//
//-----------------------------------------------------------------------------------------------
#define AD_TRIG_START				1	// Start-trigger (default setting)
#define AD_TRIG_STOP				2	// Stop-trigger
#define AD_TRIG_START_STOP			3	// Start/stop-trigger

//-----------------------------------------------------------------------------------------------
//
//     GPC/GPF-3100 Trigger Level Identifier
//
//-----------------------------------------------------------------------------------------------
#define AD_FREERUN					1	// No trigger (default setting)
#define AD_EXTTRG					2	// External trigger
#define AD_EXTTRG_DI				3	// External trigger with mask using general purpose digital input pin
#define AD_LEVEL_P					4	// Analog trigger (low-to-high transition)
#define AD_LEVEL_M					5	// Analog trigger (high-to-low transition)
#define AD_LEVEL_D					6	// Analog trigger (low-to-high or high-to-low transition)
#define AD_INRANGE					7	// Analog trigger (into the range)
#define AD_OUTRANGE					8	// Analog trigger (out of the range)
#define AD_ETERNITY					9	// Infinite sampling
#define AD_SMPLNUM					10	// Specified number
#define AD_START_SIGTIMER			11	// Interval timer
#define AD_START_DA_START			12	// Analog output start (DaStartSampling)
#define AD_START_DA_STOP			13	// Analog output stop
#define AD_START_DA_IO				14	// Analog output (DaOutputDA)
#define AD_START_DA_SMPLNUM			15	// Analog output number
#define AD_STOP_DA_START			12	// Analog output start (DaStartSampling)
#define AD_STOP_DA_STOP				13	// Analog output stop
#define AD_STOP_DA_IO				14	// Analog output (DaOutputDA)
#define AD_STOP_DA_SMPLNUM			15	// Analog output number
#define AD_STOP_SIGTIMER			16	// Interval timer
#define AD_START_P1			0x00000010	// Start-trigger (Level 1): low-to- high transition
#define AD_START_M1			0x00000020	// Start-trigger (Level 1): high-to-low transition
#define AD_START_D1			0x00000040	// Start-trigger (Level 1): high-to-low or low-to-high transition (direction DON'T CARE)
#define AD_START_P2			0x00000080	// Start-trigger (Level 2): low-to- high transition
#define AD_START_M2			0x00000100	// Start-trigger (Level 2): high-to-low transition
#define AD_START_D2			0x00000200	// Start-trigger (Level 2): high-to-low or low-to-high transition (direction DON'T CARE)
#define AD_STOP_P1			0x00000400	// Stop-trigger (Level 1): low-to-high transition
#define AD_STOP_M1			0x00000800	// Stop-trigger (Level 1): high-to-low transition
#define AD_STOP_D1			0x00001000	// Stop-trigger (Level 1): high-to-low or low-to-high transition (direction DON'T CARE)
#define AD_STOP_P2			0x00002000	// Stop-trigger (Level 2): low-to-high transition
#define AD_STOP_M2			0x00004000	// Stop-trigger (Level 2): high-to-low transition
#define AD_STOP_D2			0x00008000	// Stop-trigger (Level 2): high-to-low or low-to-high transition (direction DON'T CARE)
#define AD_ANALOG_FILTER	0x00010000	// Uses an analog trigger filter.
#define AD_START_CNT_EQ		0x00020000	// Start-trigger: Counter equal
#define AD_STOP_CNT_EQ		0x00040000	// Stop-trigger: Counter equal
#define AD_START_DI_EQ		0x00080000	// Stop-trigger: DI equal
#define AD_STOP_DI_EQ		0x00100000	// Stop-trigger: DI equal
#define AD_STOP_SOFT		0x00200000	// Stop-trigger: Soft stop
#define AD_START_Z_CLR		0x00400000	// Start-trigger: Z clear
#define AD_STOP_Z_CLR		0x00800000	// Stop-trigger: Z clear
#define AD_START_SOFT		0x01000000	// Start-trigger: Soft start
#define AD_SOFTTRG			0x02000000	// Soft Trigger
#define AD_START_SYNC1		0x10000000	// Start-trigger: Internal Sync1 trigger
#define AD_START_SYNC2		0x20000000	// Start-trigger: Internal Sync2 trigger
#define AD_STOP_SYNC1		0x40000000	// Stop-trigger: Internal Sync1 trigger
#define AD_STOP_SYNC2		0x80000000	// Stop-trigger: Internal Sync2 trigger

//-----------------------------------------------------------------------------------------------
//
//     GPC/GPF-3100 Polarity Identifier
//
//-----------------------------------------------------------------------------------------------
#define AD_DOWN_EDGE				1	// Falling edge (default setting)
#define AD_UP_EDGE					2	// Rising edge
#define AD_EXTRG_IN					3	// External trigger input
#define AD_EXCLK_IN					4	// External clock input
#define AD_LOW_LEVEL				5	// Negative level (default setting)
#define AD_HIGH_LEVEL				6	// Positive level

#define AD_EDGE_P1				0x0010	// Level 1: low-to-high transition
#define AD_EDGE_M1				0x0020	// Level 1: high-to-low transition
#define AD_EDGE_D1				0x0040	// Level 1: high-to-low or low-to-high transition (direction DON'T CARE)
#define AD_EDGE_P2				0x0080	// Level 2: low-to-high transition
#define AD_EDGE_M2				0x0100	// Level 2: high-to-low transition
#define AD_EDGE_D2				0x0200	//  Level 2: high-to-low or low-to-high transition (direction DON'T CARE)

#define	AD_DISABLE			0x80000000	// 

#define	AD_TRIG_MODE				2	//
#define	AD_BUSY_MODE				3	//
#define	AD_POST_MODE				4	//
#define	AD_ENABLE					5	//
#define	AD_SMP1_MODE				6	//
#define	AD_SMP2_MODE				7	//
#define	AD_ATRIG_MODE				8	//

//-----------------------------------------------------------------------------------------------
//
//     GPC/GPF-3100 Pulse Polarity Identifier
//
//-----------------------------------------------------------------------------------------------
#define AD_LOW_PULSE				1	// Negative pulse (default setting)
#define AD_HIGH_PULSE				2	// Positive pulse

//-----------------------------------------------------------------------------------------------
//
//     GPC/GPF-3100 Double-Clocked Mode Identifier
//
//-----------------------------------------------------------------------------------------------
#define AD_NORMAL_MODE				1	// Normal mode (default setting)
#define AD_FAST_MODE				2	// Double-clocked mode


//-----------------------------------------------------------------------------------------------
//
//     GPC/GPF-3100 Status Identifier
//
//-----------------------------------------------------------------------------------------------
#define AD_NO_STATUS				1	// Adds no bus master sampling status. (default setting)
#define AD_ADD_STATUS				2	// Adds bus master sampling status.

//-----------------------------------------------------------------------------------------------
//
//     GPC/GPF-3100 Error Control Identifier
//
//-----------------------------------------------------------------------------------------------
#define AD_STOP_SCER				2	// Stops sampling by a sampling clock error.
#define AD_STOP_ORER				4	// Stops sampling by an overrun error.

//-----------------------------------------------------------------------------------------------
//
//     GPC/GPF-3100 Data Save Identifier
//
//-----------------------------------------------------------------------------------------------
#define AD_APPEND					1	// Adds new data at the end of the buffer. (default setting)
#define AD_OVERWRITE				2	// Overwrites new data on existing data.

//-----------------------------------------------------------------------------------------------
//
//		GPC/GPF-3100 Degital Filter Identifier
//
//-----------------------------------------------------------------------------------------------
#define AD_DF_8						0	// 8 (default setting)
#define AD_DF_16					1	// 16
#define AD_DF_32					2	// 32
#define AD_DF_64					3	// 64
#define AD_DF_128					4	// 128
#define AD_DF_256					5	// 256

//-----------------------------------------------------------------------------------------------
//
//     GPC/GPF-3100 Range Identifier
//
//-----------------------------------------------------------------------------------------------
#define AD_0_1V				0x00000001	// Voltage: unipolar 0 V to 1 V
#define AD_0_2P5V			0x00000002	// Voltage: unipolar 0 V to 2.5 V
#define AD_0_5V				0x00000004	// Voltage: unipolar 0 V to 5 V
#define AD_0_10V			0x00000008	// Voltage: unipolar 0 V to 10 V
#define AD_1_5V				0x00000010	// Voltage: unipolar 1 V to 5 V
#define AD_0_2V				0x00000020	// Voltage: unipolar 0 V to 2 V
#define AD_0_0P125V			0x00000040	// Voltage: unipolar 0 V to 0.125 V
#define AD_0_1P25V			0x00000080	// Voltage: unipolar 0 V to 1.25 V
#define AD_0_0P625V			0x00000100	// Voltage: unipolar 0 V to 0.625 V
#define AD_0_0P156V			0x00000200	// Voltage: unipolar 0 V to 0.156 V
#define AD_0_20mA			0x00001000	// Current: unipolar 0 mA to 20 mA
#define AD_4_20mA			0x00002000	// Current: unipolar 4 mA to 20 mA
#define AD_20mA				0x00004000	// Current: bipolar +/- 20 mA
#define AD_1V				0x00010000	// Voltage: bipolar +/- 1 V
#define AD_2P5V				0x00020000	// Voltage: bipolar +/- 2.5 V
#define AD_5V				0x00040000	// Voltage: bipolar +/- 5 V
#define AD_10V				0x00080000	// Voltage: bipolar +/- 10 V
#define AD_20V				0x00100000	// Voltage: bipolar +/- 20 V
#define AD_50V				0x00200000	// Voltage: bipolar +/- 50 V
#define AD_0P125V			0x00400000	// Voltage: bipolar +/- 0.125 V
#define AD_1P25V			0x00800000	// Voltage: bipolar +/- 1.25 V
#define AD_0P625V			0x01000000	// Voltage: bipolar +/- 0.625 V
#define AD_0P156V			0x02000000	// Voltage: bipolar +/- 0.156 V
#define AD_1P25V_AC			0x04000000	// Voltage: bipolar +/- 1.25 V(AC Coupling)
#define AD_0P625V_AC		0x08000000	// Voltage: bipolar +/- 0.625 V(AC Coupling)
#define AD_0P156V_AC		0x10000000	// Voltage: bipolar +/- 0.156 V(AC Coupling)
#define AD_AC_COUPLING		0x40000000	// AC Coupling
#define AD_GND				0x80000000	// Voltage: GND

#define AD_TYPE_K			0x00000001
#define AD_TYPE_B			0x00000002
#define AD_TYPE_R			0x00000004
#define AD_TYPE_S			0x00000008
#define AD_TYPE_N			0x00000010
#define AD_TYPE_E			0x00000020
#define AD_TYPE_J			0x00000040
#define AD_TYPE_T			0x00000080
#define AD_TYPE_AD			0x80000000

//-----------------------------------------------------------------------------------------------
//
//     GPC/GPF-3100 Isolation Identifier
//
//-----------------------------------------------------------------------------------------------
#define AD_ISOLATION				1	// Isolated
#define AD_NOT_ISOLATION			2	// Not isolated

//-----------------------------------------------------------------------------------------------
//
//     GPC/GPF-3100 Synchronous Mode Identifier
//
//-----------------------------------------------------------------------------------------------
#define AD_MASTER_MODE				1	// Master mode
#define AD_SLAVE_MODE				2	// Slave mode

//-----------------------------------------------------------------------------------------------
//
//     GPC/GPF-3100 Synchronous Number Identifier
//
//-----------------------------------------------------------------------------------------------
#define AD_SYNC_NUM_1				0x0100
#define AD_SYNC_NUM_2				0x0200
#define AD_SYNC_NUM_3				0x0400
#define AD_SYNC_NUM_4				0x0800
#define AD_SYNC_NUM_5				0x1000
#define AD_SYNC_NUM_6				0x2000
#define AD_SYNC_NUM_7				0x4000

//-----------------------------------------------------------------------------------------------
//
//     GPC/GPF-3100 Calibration Identifier
//
//-----------------------------------------------------------------------------------------------
#define AD_SELF_CALIBRATION			1	// Self calibration of the AD converter
#define AD_ZEROSCALE_CALIBRATION	2	// Zero voltage calibration (system calibration)
#define AD_FULLSCALE_CALIBRATION	3	// Full scale voltage calibration (system calibration)

//-----------------------------------------------------------------------------------------------
//
//     GPC/GPF-3100 Full-scale Identifier
//
//-----------------------------------------------------------------------------------------------
#define AD_STATUS_NEGATIVE_FULL_SCALE	1	// Negative full-scale
#define AD_STATUS_POSITIVE_FULL_SCALE	2	// Positive full-scale
#define AD_STATUS_UNDER_RANGE			1	// Negative full-scale
#define AD_STATUS_OVER_RANGE			2	// Positive full-scale

//-----------------------------------------------------------------------------------------------
//
//     GPC/GPF-3100 OVPM Identifier
//
//-----------------------------------------------------------------------------------------------
#define AD_STATUS_OVPM_NORMAL		0	// NormalRange
#define AD_STATUS_OVPM_HIGH_RANGE	1	// Voltage: bipolar +/- 50 V Range
#define AD_STATUS_OVPM_GND_RANGE	2	// GND

//-----------------------------------------------------------------------------------------------
//
//     GPC/GPF-3100 PCI-3525, CPZ-3525 CPZ-360810 Conecter Function Identifier
//
//-----------------------------------------------------------------------------------------------
#define AD_CN_FREE					0	// Not used
#define AD_CN_EXTRG_IN				1	// External trigger input
#define AD_CN_EXTRG_OUT				2	// External trigger output
#define AD_CN_EXCLK_IN				3	// External clock input
#define AD_CN_EXCLK_OUT				4	// External clock output
#define AD_CN_EXINT_IN				5	// External interrupt input
#define AD_CN_ATRG_OUT				6	// Analog trigger out
#define AD_CN_DI					7	// Digital input
#define AD_CN_DO					8	// Digital output
#define AD_CN_DAOUT					9	// Analog output
#define AD_CN_OPEN					10	// Open
#define AD_CN_EXSMP1_OUT			12	// Sampling Status1
#define AD_CN_EXSMP2_OUT			13	// Sampling Status2
#define AD_CN_DIO					1	// DIO
#define AD_CN_CONTROL				2	// Control used
#define AD_CN_CNT					3	// Counter used

//-----------------------------------------------------------------------------------------------
//
//		GPC/GPF-3100 CPZ-360810 DIN/DOUT Function Identifier
//
//-----------------------------------------------------------------------------------------------
#define AD_EX_DIO1				1	// DIN/DOUT1
#define AD_EX_DIO2				2	// DIN/DOUT2
#define AD_EX_DIO3				3	// DIN/DOUT3
#define AD_EX_DIO4				4	// DIN/DOUT4
#define AD_EX_DIO5				5	// DIN/DOUT5
#define AD_EX_DIO6				6	// DIN/DOUT6
#define AD_EX_DIO7				7	// DIN/DOUT7
#define AD_EX_DIO8				8	// DIN/DOUT8


//-----------------------------------------------------------------------------------------------
//
//     GPC/GPF-3100 Measure Temperature
//
//-----------------------------------------------------------------------------------------------
#define AD_GET_CURRENT_TEMPERATURE			1	// Current Temperature
#define AD_LOAD_TEMPERATURE					2	// User area Temperature
#define AD_LOAD_FACTORY_SETTING_TEMPERATURE	3	// Factory setting Temperature
#define AD_SAVE_TEMPERATURE_USER			4	// User area of Eeprom 

//-----------------------------------------------------------------------------------------------
//
//     GPC/GPF-3100 Error Identifier
//
//-----------------------------------------------------------------------------------------------
#define AD_ERROR_SUCCESS						0x00000000
#define AD_ERROR_NOT_DEVICE						0xC0000001
#define AD_ERROR_NOT_OPEN						0xC0000002
#define AD_ERROR_INVALID_HANDLE					0xC0000003
#define AD_ERROR_ALREADY_OPEN					0xC0000004
#define AD_ERROR_NOT_SUPPORTED					0xC0000009
#define AD_ERROR_NOW_SAMPLING					0xC0001001
#define AD_ERROR_STOP_SAMPLING					0xC0001002
#define AD_ERROR_START_SAMPLING					0xC0001003
#define AD_ERROR_SAMPLING_TIMEOUT				0xC0001004
#define AD_ERROR_SAMPLING_FREQ					0xC0001005
#define AD_ERROR_INVALID_PARAMETER				0xC0001021
#define AD_ERROR_ILLEGAL_PARAMETER				0xC0001022
#define AD_ERROR_NULL_POINTER					0xC0001023
#define AD_ERROR_GET_DATA						0xC0001024
#define AD_ERROR_USED_DA						0xC0001025
#define AD_ERROR_FILE_OPEN						0xC0001041
#define AD_ERROR_FILE_CLOSE						0xC0001042
#define AD_ERROR_FILE_READ						0xC0001043
#define AD_ERROR_FILE_WRITE						0xC0001044
#define AD_ERROR_INVALID_DATA_FORMAT			0xC0001061
#define AD_ERROR_INVALID_AVERAGE_OR_SMOOTHING	0xC0001062
#define AD_ERROR_INVALID_SOURCE_DATA			0xC0001063
#define AD_ERROR_NOT_ALLOCATE_MEMORY			0xC0001081
#define AD_ERROR_NOT_LOAD_DLL					0xC0001082
#define AD_ERROR_CALL_DLL						0xC0001083
#define AD_ERROR_CALIBRATION					0xC0001084

#define AD_ERROR_USB_TIMEOUT					0xC0001090
#define AD_ERROR_USBIO_FAILED					0xC0001091

// -----------------------------------------------------------------------
//
//     User-supplied Function
//
// -----------------------------------------------------------------------
typedef void (CALLBACK CONVPROC)(
	WORD wCh,
	DWORD dwCount,
	LPVOID lpData
);
typedef CONVPROC FAR *LPCONVPROC;

//@@@ Ver3.00-71 (64bit対応)
#ifdef _WIN64
typedef void (CALLBACK ADCALLBACK)(PVOID dwUser);
#else
typedef void (CALLBACK ADCALLBACK)(DWORD dwUser);
#endif
typedef ADCALLBACK FAR *LPADCALLBACK;

// -----------------------------------------------------------------------
//     Sampling Condition Structure for Each Channel
// -----------------------------------------------------------------------
typedef struct {
	ULONG			ulChNo;
	ULONG			ulRange;
} ADSMPLCHREQ, *PADSMPLCHREQ;

// -----------------------------------------------------------------------
//     Analog Trigger Condition Structure for Each Channel
// -----------------------------------------------------------------------
typedef struct {
	ULONG			ulChNo;
	float			fTrigLevel;
	float			fHysteresis;
} ADTRIGCHREQ, *PADTRIGCHREQ;

// -----------------------------------------------------------------------
//     Sampling Condition Structure
// -----------------------------------------------------------------------
typedef struct {
	ULONG			ulChCount;
	ADSMPLCHREQ		SmplChReq[256];
	ULONG			ulSamplingMode;
	ULONG			ulSingleDiff;
	ULONG			ulSmplNum;
	ULONG			ulSmplEventNum;
	float			fSmplFreq;
	ULONG			ulTrigPoint;
	ULONG			ulTrigMode;
	LONG			lTrigDelay;
	ULONG			ulTrigCh;
	float			fTrigLevel1;
	float			fTrigLevel2;
	ULONG			ulEClkEdge;
	ULONG			ulATrgPulse;
	ULONG			ulTrigEdge;
	ULONG			ulTrigDI;
	ULONG			ulFastMode;
} ADSMPLREQ, *PADSMPLREQ;

// -----------------------------------------------------------------------
//     Sampling Condition Structure for Bus Master
// -----------------------------------------------------------------------
typedef struct {
	ULONG			ulChCount;
	ADSMPLCHREQ		SmplChReq[256];
	ULONG			ulSingleDiff;
	ULONG			ulSmplNum;
	ULONG			ulSmplEventNum;
	ULONG			ulSmplRepeat;
	ULONG			ulBufferMode;
	float			fSmplFreq;
	float			fScanFreq;
	ULONG			ulStartMode;
	ULONG			ulStopMode;
	ULONG			ulPreTrigDelay;
	ULONG			ulPostTrigDelay;
	ADTRIGCHREQ		TrigChReq[2];
	ULONG			ulATrgMode;
	ULONG			ulATrgPulse;
	ULONG			ulStartTrigEdge;
	ULONG			ulStopTrigEdge;
	ULONG			ulTrigDI;
	ULONG			ulEClkEdge;
	ULONG			ulFastMode;
	ULONG			ulStatusMode;
	ULONG			ulErrCtrl;
} ADBMSMPLREQ, *PADBMSMPLREQ;

// -----------------------------------------------------------------------
//     Sampling Condition Structure for Memory
// -----------------------------------------------------------------------
typedef struct {
	ULONG			ulChCount;
	ADSMPLCHREQ		SmplChReq[256];
	ULONG			ulSingleDiff;
	float			fSmplFreq;
	ULONG			ulStopMode;
	ULONG			ulPreTrigDelay;
	ULONG			ulPostTrigDelay;
	ADTRIGCHREQ		TrigChReq[2];
	ULONG			ulATrgMode;
	ULONG			ulATrgPulse;
	ULONG			ulStopTrigEdge;
	ULONG			ulEClkEdge;
	ULONG			ulFastMode;
	ULONG			ulStatusMode;
	ULONG			ulErrCtrl;
} ADMEMSMPLREQ, *PADMEMSMPLREQ;

// -----------------------------------------------------------------------
//     Board Specification Structure
// -----------------------------------------------------------------------
typedef struct {
	ULONG			ulBoardType;
	ULONG			ulBoardID;
	DWORD			dwSamplingMode;
	ULONG			ulChCountS;
	ULONG			ulChCountD;
	ULONG			ulResolution;
	DWORD			dwRange;
	ULONG			ulIsolation;
	ULONG			ulDi;
	ULONG			ulDo;
} ADBOARDSPEC, *PADBOARDSPEC;

#ifdef __cplusplus
}
#endif

#endif
