// -----------------------------------------------------------------------
//
//		Heaader file
//
//		File name:	GpcDa.h
//
//		Ver 1.10
//
//		Copyright 1999, 2004 Interface Corporation. All rights reserved.
// -----------------------------------------------------------------------

#if !defined( _FbiDa_H_ )
#define _FbiDa_H_

#ifdef __cplusplus
extern	"C" {
#endif

//-----------------------------------------------------------------------------------------------
//
//		Overlapped Process Identifier
//
//-----------------------------------------------------------------------------------------------
#define FLAG_SYNC	1	// The analog output update is performed as an non-overlapped operation.
#define FLAG_ASYNC	2	// The analog output update is performed as an overlapped operation.

//-----------------------------------------------------------------------------------------------
//
//		File Format Identifier
//
//-----------------------------------------------------------------------------------------------
#define FLAG_BIN	1	// Binary format file
#define FLAG_CSV	2	// CSV format file

//-----------------------------------------------------------------------------------------------
//
//		Analog Output Status Identifier
//
//-----------------------------------------------------------------------------------------------
#define DA_STATUS_STOP_SAMPLING		1	// The analog output update is stopped.
#define DA_STATUS_WAIT_TRIGGER		2	// The analog output update is waiting for a trigger.
#define DA_STATUS_NOW_SAMPLING		3	// The analog output update is running.

//-----------------------------------------------------------------------------------------------
//
//		Event Factor Identifier
//
//-----------------------------------------------------------------------------------------------
#define DA_EVENT_STOP_TRIGGER		1	// The analog output has been stopped because a trigger is asserted.
#define DA_EVENT_STOP_FUNCTION		2	// The analog output has been stopped by software.
#define DA_EVENT_STOP_SAMPLING		3	// The Analog output terminated.
#define DA_EVENT_RESET_IN			4	// The reset input signal is asserted.
#define DA_EVENT_CURRENT_OFF		5	// The current loop fault has been detected.
#define DA_EVENT_COUNT				6	// The specified number of output has been acquired.
#define DA_EVENT_FIFO_EMPTY			7	// Fifo Empty
#define DA_EVENT_EX_INT				8	// EX INT
#define DA_EVENT_EXOV_OFF			9	// EX Over Voktage off
#define DA_EVENT_OV_OFF				10	// Over Voktage off

//-----------------------------------------------------------------------------------------------
//
//		Range Identifier
//
//-----------------------------------------------------------------------------------------------
#define DA_0_1V				0x00000001	// Voltage: unipolar 0 V to +1 V
#define DA_0_2P5V			0x00000002	// Voltage: unipolar 0 V to +2.5 V
#define DA_0_5V				0x00000004	// Voltage: unipolar 0 V to +5 V
#define DA_0_10V			0x00000008	// Voltage: unipolar 0 V to +10 V
#define DA_1_5V				0x00000010	// Voltage: unipolar +1 V to +5 V
#define DA_0_20mA			0x00001000	// Current: unipolar 0 mA to +20 mA
#define DA_4_20mA			0x00002000	// Current: unipolar +4 mA to +20 mA
#define DA_0_1mA			0x00004000	// Current: unipolar +0 mA to +1 mA
#define DA_0_100mA			0x00008000	// Current: unipolar +0 mA to +100 mA
#define DA_1V				0x00010000	// Voltage: bipolar +/-1 V
#define DA_2P5V				0x00020000	// Voltage: bipolar +/-2.5 V
#define DA_5V				0x00040000	// Voltage: bipolar +/-5 V
#define DA_10V				0x00080000	// Voltage: bipolar +/-10 V
#define DA_20mA				0x01000000	// Current: bipolar +/-20 mA

//-----------------------------------------------------------------------------------------------
//
//		Data Transfer Architecture Identifier
//
//-----------------------------------------------------------------------------------------------
#define DA_IO_SAMPLING				1	// Programmed I/O
#define DA_FIFO_SAMPLING			2	// FIFO
#define DA_MEM_SAMPLING				4	// Memory

//-----------------------------------------------------------------------------------------------
//
//		Trigger Point Identifier
//
//-----------------------------------------------------------------------------------------------
#define DA_TRIG_START				1	// Start-trigger (default setting)
#define DA_TRIG_STOP				2	// Stop-trigger
#define DA_TRIG_START_STOP			3	// Start/stop-trigger

//-----------------------------------------------------------------------------------------------
//
//		Trigger Level Identifier
//
//-----------------------------------------------------------------------------------------------
#define DA_FREERUN				1	// No-trigger (default setting)
#define DA_EXTTRG				2	// External trigger
#define DA_EXTTRG_DI			3	// External trigger with DI masking
#define DA_SOFTTRG				4	// Soft-trigger

//-----------------------------------------------------------------------------------------------
//
//		Polarity Identifier
//
//-----------------------------------------------------------------------------------------------
#define DA_DOWN_EDGE			1	// Falling edge (default)
#define DA_UP_EDGE				2	// Rising edge

//-----------------------------------------------------------------------------------------------
//
//		Isolation Identifier
//
//-----------------------------------------------------------------------------------------------
#define DA_ISOLATION				1	// Photo-isolated board
#define DA_NOT_ISOLATION			2	// Not isolated board

//-----------------------------------------------------------------------------------------------
//
//		Range Identifier
//
//-----------------------------------------------------------------------------------------------
#define DA_RANGE_UNIPOLAR			1	// Unipolar
#define DA_RANGE_BIPOLAR			2	// Bipolar

//-----------------------------------------------------------------------------------------------
//
//		Waveform Generation Mode Identifier
//
//-----------------------------------------------------------------------------------------------
#define DA_MODE_CUT					1	// Time-based waveform generation
#define DA_MODE_SYNTHE				2	// Frequency-based waveform generation

//-----------------------------------------------------------------------------------------------
//
//		Repeat Mode Identifier
//
//-----------------------------------------------------------------------------------------------
#define DA_REPEAT_NONINTERVAL		1	// Repeat without the wait state (default setting)
#define DA_REPEAT_INTERVAL			2	// Repeat with the wait state

//-----------------------------------------------------------------------------------------------
//
//		Counter Clear Identifier
//
//-----------------------------------------------------------------------------------------------
#define DA_COUNTER_CLEAR			1	// Cleared (default setting)
#define DA_COUNTER_NONCLEAR			2	// Not cleared

//-----------------------------------------------------------------------------------------------
//
//		DA Latch Identifier
//
//-----------------------------------------------------------------------------------------------
#define DA_LATCH_CLEAR				1	// The voltage is set to the lowest voltage of the range.
#define DA_LATCH_NONCLEAR			2	// The voltage is held.

//-----------------------------------------------------------------------------------------------
//
//		Clock Source Identifier
//
//-----------------------------------------------------------------------------------------------
#define DA_CLOCK_TIMER				1	// Internal programmable timer (8254 compatible)
#define DA_CLOCK_FIXED				2	// Fixed 5 MHz clock

//-----------------------------------------------------------------------------------------------
//
//		Configurations of the Connector CN3 Identifier
//
//-----------------------------------------------------------------------------------------------
#define DA_EXTRG_IN					1	// External trigger input (default setting)
#define DA_EXTRG_OUT				2	// External trigger output

//-----------------------------------------------------------------------------------------------
//
//		Configurations of the Connector CN4 Identifier
//
//-----------------------------------------------------------------------------------------------
#define DA_EXCLK_IN					1	// External clock input (default setting)
#define DA_EXCLK_OUT				2	// External clock output

//-----------------------------------------------------------------------------------------------
//
//		Reset Polarity Identifier
//
//-----------------------------------------------------------------------------------------------
#define DA_RESET_DOWN_EDGE			0x04	// Falling edge (default)
#define DA_RESET_UP_EDGE			0x08	// Rising edge

//-----------------------------------------------------------------------------------------------
//
//		External trigger Polarity Identifier
//
//-----------------------------------------------------------------------------------------------
#define DA_EXTRG_DOWN_EDGE			0x10	// Falling edge (default)
#define DA_EXTRG_UP_EDGE			0x20	// Rising edge

//-----------------------------------------------------------------------------------------------
//
//		Reset Identifier
//
//-----------------------------------------------------------------------------------------------
#define DA_RESET_ON					1	//Used
#define DA_RESET_OFF				2	// Not used (default setting)

//-----------------------------------------------------------------------------------------------
//
//		Filter Identifier
//
//-----------------------------------------------------------------------------------------------
#define DA_FILTER_OFF				1	// Not used (default setting)
#define DA_FILTER_ON				2	// Used

//-----------------------------------------------------------------------------------------------
//
//		Synchronous Analog Output Identifier
//
//-----------------------------------------------------------------------------------------------
#define DA_MASTER_MODE				1	// Master mode
#define DA_SLAVE_MODE				2	// Slave mode

//-----------------------------------------------------------------------------------------------
//
//     Synchronous Number Identifier
//
//-----------------------------------------------------------------------------------------------
#define DA_SYNC_NUM_1				0x0100
#define DA_SYNC_NUM_2				0x0200
#define DA_SYNC_NUM_3				0x0400
#define DA_SYNC_NUM_4				0x0800
#define DA_SYNC_NUM_5				0x1000
#define DA_SYNC_NUM_6				0x2000
#define DA_SYNC_NUM_7				0x4000

//-----------------------------------------------------------------------------------------------
//
//		GPC/GPF-3300 CPZ-360810 DIN/DOUT Function Identifier
//
//-----------------------------------------------------------------------------------------------
#define DA_EX_DIO1				1	// DIN/DOUT1
#define DA_EX_DIO2				2	// DIN/DOUT2
#define DA_EX_DIO3				3	// DIN/DOUT3
#define DA_EX_DIO4				4	// DIN/DOUT4
#define DA_EX_DIO5				5	// DIN/DOUT5

//-----------------------------------------------------------------------------------------------
//
//		PCI-3525 channel 3 and channel 4 Function Identifier
//
//-----------------------------------------------------------------------------------------------
#define DA_CN_FREE					0	// not used
#define DA_CN_EXTRG_IN				1	// External trigger input
#define DA_CN_EXTRG_OUT				2	// External trigger output
#define DA_CN_EXCLK_IN				3	// External clock input
#define DA_CN_EXCLK_OUT				4	// External clock output
#define DA_CN_EXINT_IN				5	// External interrupt input
#define DA_CN_ATRG_OUT				6	// Analog trigger out
#define DA_CN_DI					7	// Digital input
#define DA_CN_DO					8	// Digital output
#define DA_CN_DAOUT					9	// Analog output
#define DA_CN_OPEN					10	// open

//-----------------------------------------------------------------------------------------------
//
//		PCI-3525 External Trigger Polarity Identifier
//
//-----------------------------------------------------------------------------------------------
#define DA_START_DOWN_EDGE			1	// Start external trigger falling edge
#define DA_START_UP_EDGE			2	// Start external trigger rising edge
#define DA_STOP_DOWN_EDGE			4	// Stop external trigger falling edge
#define DA_STOP_UP_EDGE				8	// Stop external trigger rising edge

//-----------------------------------------------------------------------------------------------
//
//		Fifo Trigger Level Identifier
//
//-----------------------------------------------------------------------------------------------
#define DA_TRG_FREERUN				0		// No trigger
#define DA_TRG_EXTTRG				1		// External trigger
#define DA_TRG_ATRG					2		// Analog trigger
#define DA_TRG_SIGTIMER				3		// Interval timer
#define DA_TRG_CNT_EQ				4		// Counter equal
#define DA_TRG_Z_CLR				5		// Z clear
#define DA_TRG_AD_START				5		// Analog input start
#define DA_TRG_AD_STOP				6		// Analog input stop
#define DA_TRG_AD_PRETRG			7		// Analog input pre-trigger
#define DA_TRG_AD_POSTTRG			8		// Analog input post-trigger
#define DA_TRG_SMPLNUM				9		// Analog output stop number
#define DA_TRG_FIFO_EMPTY			10		// FIFO empty
#define DA_TRG_SOFT					11		// Soft trigger
#define DA_TRG_SYNC1				14		// Internel sync1 trigger
#define DA_TRG_SYNC2				15		// Internel sync2 trigger
#define DA_FIFORESET				0x0100	// FIFO reset
#define DA_RETRG					0x0200	// Retrigger

//-----------------------------------------------------------------------------------------------
//
//		Volume Identifier
//
//-----------------------------------------------------------------------------------------------
#define DA_ADJUST_BIOFFSET			1	// Bipolar offset calibration
#define DA_ADJUST_UNIOFFSET			2	// Unipolar offset calibration
#define DA_ADJUST_BIGAIN			3	// Bipolar gain calibration
#define DA_ADJUST_UNIGAIN			4	// Unipolar gain calibration

//-----------------------------------------------------------------------------------------------
//
//		Calibration Item Identifier
//
//-----------------------------------------------------------------------------------------------
#define DA_ADJUST_UP				1	// Increase the volume.
#define DA_ADJUST_DOWN				2	// Decrease the volume.
#define DA_ADJUST_STORE				3	// Save the present value 
                                        // to the non-volatile memory.
#define DA_ADJUST_STANDBY			4	// Place the electronic volume device into 
                                        // the standby mode.
#define DA_ADJUST_NOT_STORE			5	// Not save the value.

	
//-----------------------------------------------------------------------------------------------
//
//     Read Adjust Identifier
//
//-----------------------------------------------------------------------------------------------
#define DA_ADJUST_READ_FACTORY		1
#define DA_ADJUST_READ_USER			2
	
//-----------------------------------------------------------------------------------------------
//
//		Data Identifier
//
//-----------------------------------------------------------------------------------------------
#define DA_DATA_PHYSICAL			1	// Physical value (voltage [V], current [mA])
#define DA_DATA_BIN8				2	// 8-bit binary
#define DA_DATA_BIN12				3	// 12-bit binary
#define DA_DATA_BIN16				4	// 16-bit binary
#define DA_DATA_BIN24				5	// 24-bit binary
#define DA_DATA_BIN14				6	// 14-bit binary

//-----------------------------------------------------------------------------------------------
//
//		Data Conversion Identifier
//
//-----------------------------------------------------------------------------------------------
#define DA_CONV_SMOOTH				1	// Convert the data with interpolation.
#define DA_CONV_AVERAGE1			0x100	// Convert the data with the simple averaging.
#define DA_CONV_AVERAGE2			0x200	// Convert the data with the shifted averaging.

//-----------------------------------------------------------------------------------------------
//
//		Simultaneous Output Set Identifier
//
//-----------------------------------------------------------------------------------------------
#define DA_NORMAL_OUTPUT			1	// Not simultaneous output
#define DA_SYNC_OUTPUT				2	// Simultaneous output

//-----------------------------------------------------------------------------------------------
//
//		Error Identifier
//
//-----------------------------------------------------------------------------------------------
#define DA_ERROR_SUCCESS						0x00000000
#define DA_ERROR_NOT_DEVICE						0xC0000001
#define DA_ERROR_NOT_OPEN						0xC0000002
#define DA_ERROR_INVALID_HANDLE					0xC0000003
#define DA_ERROR_ALREADY_OPEN					0xC0000004
#define DA_ERROR_NOT_SUPPORTED					0xC0000009
#define DA_ERROR_NOW_SAMPLING					0xC0001001
#define DA_ERROR_STOP_SAMPLING					0xC0001002
#define DA_ERROR_START_SAMPLING					0xC0001003
#define DA_ERROR_SAMPLING_TIMEOUT				0xC0001004
#define DA_ERROR_INVALID_PARAMETER				0xC0001021
#define DA_ERROR_ILLEGAL_PARAMETER				0xC0001022
#define DA_ERROR_NULL_POINTER					0xC0001023
#define DA_ERROR_SET_DATA						0xC0001024
#define DA_ERROR_USED_AD						0xC0001025
#define DA_ERROR_FILE_OPEN						0xC0001041
#define DA_ERROR_FILE_CLOSE						0xC0001042
#define DA_ERROR_FILE_READ						0xC0001043
#define DA_ERROR_FILE_WRITE						0xC0001044
#define DA_ERROR_INVALID_DATA_FORMAT			0xC0001061
#define DA_ERROR_INVALID_AVERAGE_OR_SMOOTHING	0xC0001062
#define DA_ERROR_INVALID_SOURCE_DATA			0xC0001063
#define DA_ERROR_NOT_ALLOCATE_MEMORY			0xC0001081
#define DA_ERROR_NOT_LOAD_DLL					0xC0001082
#define DA_ERROR_CALL_DLL						0xC0001083
#define DA_ERROR_CALIBRATION					0xC0001084
#define DA_ERROR_USBIO_FAILED					0xC0001085
#define DA_ERROR_USBIO_TIMEOUT					0xC0001086

// -----------------------------------------------------------------------
//
//		Structure Declaration
//
// -----------------------------------------------------------------------
typedef void (CALLBACK DACONVPROC)(
	WORD wCh,
	DWORD dwCount,
	LPVOID lpData
);
typedef DACONVPROC FAR *LPDACONVPROC;

//@@@ Ver3.00-62 (64bit‘Î‰ž)
#ifdef _WIN64
typedef void (CALLBACK DACALLBACK)(PVOID dwUser);
#else
typedef void (CALLBACK DACALLBACK)(DWORD dwUser);
#endif
typedef DACALLBACK FAR *LPDACALLBACK;

// -----------------------------------------------------------------------
//	Analog Output Request Condition Structure for Each Channel
// -----------------------------------------------------------------------
typedef struct {
	ULONG			ulChNo;
	ULONG			ulRange;
} DASMPLCHREQ, *PDASMPLCHREQ;

// -----------------------------------------------------------------------
//	Analog Output Request Condition Structure
// -----------------------------------------------------------------------
typedef struct {
	ULONG			ulChCount;
	DASMPLCHREQ		SmplChReq[256];
	ULONG			ulSamplingMode;
	float			fSmplFreq;
	ULONG			ulSmplRepeat;
	ULONG			ulTrigMode;
	ULONG			ulTrigPoint;
	ULONG			ulTrigDelay;
	ULONG			ulEClkEdge;
	ULONG			ulTrigEdge;
	ULONG			ulTrigDI;
} DASMPLREQ, *PDASMPLREQ;


// -----------------------------------------------------------------------
//	Board Specification Structure
// -----------------------------------------------------------------------
typedef struct {
	ULONG			ulBoardType;
	ULONG			ulBoardID;
	ULONG			ulSamplingMode;
	ULONG			ulChCount;
	ULONG			ulResolution;
	ULONG			ulRange;
	ULONG			ulIsolation;
	ULONG			ulDi;
	ULONG			ulDo;
} DABOARDSPEC, *PDABOARDSPEC;

// -----------------------------------------------------------------------
//	Output Range Configurations Structure for Each Channel (for the PCI/PAZ-3305)
// -----------------------------------------------------------------------
typedef struct {
	ULONG		ulRange;
	float		fVolt;
	ULONG		ulFilter;
} DAMODECHREQ, *PDAMODECHREQ;

// -----------------------------------------------------------------------
//	Waveform Generation Mode Structure (for the PCI/PAZ-3305)
// -----------------------------------------------------------------------
typedef struct {
	DAMODECHREQ		ModeChReq[2];
	ULONG			ulPulseMode;
	ULONG			ulSyntheOut;
	ULONG			ulInterval;
	float			fIntervalCycle;
	ULONG			ulCounterClear;
	ULONG			ulDaLatch;
	ULONG			ulSamplingClock;
	ULONG			ulExControl;
	ULONG			ulExClock;
} DAMODEREQ, *PDAMODEREQ;

// -----------------------------------------------------------------------
//	Fifo Analog Output Request Condition Structure (for the PCI-3525)
// -----------------------------------------------------------------------
typedef struct {
	ULONG			ulChCount;
	DASMPLCHREQ	SmplChReq[256];
	float			fSmplFreq;
	ULONG			ulSmplRepeat;
	ULONG			ulSmplNum;
	ULONG			ulStartTrigCondition;
	ULONG			ulStopTrigCondition;
	ULONG			ulEClkEdge;
	ULONG			ulTrigEdge;
} DAFIFOREQ, *PDAFIFOREQ;

#ifdef __cplusplus
}
#endif

#endif
