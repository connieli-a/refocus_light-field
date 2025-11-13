Attribute VB_Name = "FbiDa"
'-----------------------------------------------------------------------------------------------
'
'   Overlapped Process Identifier
'
'-----------------------------------------------------------------------------------------------
Public Const FLAG_SYNC = 1                    ' The analog output update is performed as an
                                              ' non-overlapped operation.
Public Const FLAG_ASYNC = 2                   ' The analog output update is performed as an 
                                              ' overlapped operation.

'-----------------------------------------------------------------------------------------------
'
'   File Format Identifier
'
'-----------------------------------------------------------------------------------------------
Public Const FLAG_BIN = 1                     ' Binary format file
Public Const FLAG_CSV = 2                     ' CSV format file


'-----------------------------------------------------------------------------------------------
'
'   Analog Output Status Identifier
'
'-----------------------------------------------------------------------------------------------
Public Const DA_STATUS_STOP_SAMPLING = 1      ' The analog output update is stopped.
Public Const DA_STATUS_WAIT_TRIGGER = 2       ' The analog output update is waiting for a trigger.
Public Const DA_STATUS_NOW_SAMPLING = 3       ' The analog output update is running.


'-----------------------------------------------------------------------------------------------
'
'   Event Factor Identifier
'
'-----------------------------------------------------------------------------------------------
Public Const DA_EVENT_STOP_TRIGGER = 1        ' The analog output has been stopped 
                                              ' because a trigger is asserted.
Public Const DA_EVENT_STOP_FUNCTION = 2       ' The analog output has been stopped by software.
Public Const DA_EVENT_STOP_SAMPLING = 3       ' The analog output terminated.
Public Const DA_EVENT_RESET_IN = 4            ' The reset input signal is asserted.
Public Const DA_EVENT_CURRENT_OFF = 5         ' The current loop fault has been detected.
Public Const DA_EVENT_COUNT = 6               ' The specified number of samples has been acquired.
Public Const DA_EVENT_FIFO_EMPTY = 7          ' The fifo is Empty
Public Const DA_EVENT_EX_INT = 8              ' The fifo is Empty
Public Const DA_EVENT_EXOV_OFF = 9            ' The fifo is Empty
Public Const DA_EVENT_OV_OFF = 10             ' The fifo is Empty


'-----------------------------------------------------------------------------------------------
'
'   Range Identifier
'
'-----------------------------------------------------------------------------------------------
Public Const DA_0_1V = &H1                    ' Voltage: unipolar 0 V to +1 V
Public Const DA_0_2P5V = &H2                  ' Voltage: unipolar 0 V to +2.5 V
Public Const DA_0_5V = &H4                    ' Voltage: unipolar 0 V to +5 V
Public Const DA_0_10V = &H8                   ' Voltage: unipolar 0 V to +10 V
Public Const DA_1_5V = &H10                   ' Voltage: unipolar +1 V to +5 V
Public Const DA_0_20mA = &H1000               ' Current: unipolar 0 mA to +20 mA
Public Const DA_4_20mA = &H2000               ' Current: unipolar +4 mA to +20 mA
Public Const DA_0_1mA = &H4000                ' Current: unipolar 0 mA to +1 mA
Public Const DA_0_100mA = &H8000&             ' Current: unipolar 0 mA to +100 mA
Public Const DA_1V = &H10000                  ' Voltage: bipolar +/-1 V
Public Const DA_2P5V = &H20000                ' Voltage: bipolar +/-2.5 V
Public Const DA_5V = &H40000                  ' Voltage: bipolar +/-5 V
Public Const DA_10V = &H80000                 ' Voltage: bipolar +/-10 V
Public Const DA_20mA = &H1000000              ' Current: bipolar +/-20 mA


'-----------------------------------------------------------------------------------------------
'
'   Data Transfer Architecture Identifier
'
'-----------------------------------------------------------------------------------------------
Public Const DA_IO_SAMPLING = 1               ' Programmed I/O
Public Const DA_FIFO_SAMPLING = 2             ' FIFO
Public Const DA_MEM_SAMPLING = 4              ' Memory


'-----------------------------------------------------------------------------------------------
'
'   Trigger Point Identifier
'
'-----------------------------------------------------------------------------------------------
Public Const DA_TRIG_START = 1                ' Start-trigger (default setting)
Public Const DA_TRIG_STOP = 2                 ' Stop-trigger
Public Const DA_TRIG_START_STOP = 3           ' Start/stop-trigger


'-----------------------------------------------------------------------------------------------
'
'   Trigger Level Identifier
'
'-----------------------------------------------------------------------------------------------
Public Const DA_FREERUN = 1                   ' No-trigger (default setting)
Public Const DA_EXTTRG = 2                    ' External trigger
Public Const DA_EXTTRG_DI = 3                 ' External trigger with DI masking


'-----------------------------------------------------------------------------------------------
'
'   Polarity Identifier
'
'-----------------------------------------------------------------------------------------------
Public Const DA_DOWN_EDGE = 1                 ' Falling edge (default setting)
Public Const DA_UP_EDGE = 2                   ' Rising edge

'-----------------------------------------------------------------------------------------------
'
'   Isolation Identifier
'
'-----------------------------------------------------------------------------------------------
Public Const DA_ISOLATION = 1                 ' Photo-isolated board
Public Const DA_NOT_ISOLATION = 2             ' Not isolated board


'-----------------------------------------------------------------------------------------------
'
'   Range Identifier
'
'-----------------------------------------------------------------------------------------------
Public Const DA_RANGE_UNIPOLAR = 1            ' Unipolar
Public Const DA_RANGE_BIPOLAR = 2             ' Bipolar

'-----------------------------------------------------------------------------------------------
'
'   Waveform Generation Mode Identifier
'
'-----------------------------------------------------------------------------------------------
Public Const DA_MODE_CUT = 1                  ' Time-based waveform generation
Public Const DA_MODE_SYNTHE = 2               ' Frequency-based waveform generation

'-----------------------------------------------------------------------------------------------
'
'   Repeat Mode Identifier
'
'-----------------------------------------------------------------------------------------------
Public Const DA_REPEAT_NONINTERVAL = 1        ' Repeat without the wait state (default setting)
Public Const DA_REPEAT_INTERVAL = 2           ' Repeat with the wait state

'-----------------------------------------------------------------------------------------------
'
'   Counter Clear Identifier
'
'-----------------------------------------------------------------------------------------------
Public Const DA_COUNTER_CLEAR = 1             ' Cleared (default setting)
Public Const DA_COUNTER_NONCLEAR = 2          ' Not cleared

'-----------------------------------------------------------------------------------------------
'
'   DA Latch Identifier
'
'-----------------------------------------------------------------------------------------------
Public Const DA_LATCH_CLEAR = 1               ' The voltage is set to the lowest voltage of the range.
Public Const DA_LATCH_NONCLEAR = 2            ' The voltage is held.

'-----------------------------------------------------------------------------------------------
'
'   Clock Source Identifier
'
'-----------------------------------------------------------------------------------------------
Public Const DA_CLOCK_TIMER = 1               ' Internal programmable timer (8254 compatible)
Public Const DA_CLOCK_FIXED = 2               ' Fixed 5 MHz clock

'-----------------------------------------------------------------------------------------------
'
'   Configurations of the Connector CN3 Identifier
'
'-----------------------------------------------------------------------------------------------
Public Const DA_EXTRG_IN = 1                  ' External trigger input (default setting)
Public Const DA_EXTRG_OUT = 2                 ' External trigger output

'-----------------------------------------------------------------------------------------------
'
'   Configurations of the Connector CN4 Identifier
'
'-----------------------------------------------------------------------------------------------
Public Const DA_EXCLK_IN = 1                  ' External clock input (default setting)
Public Const DA_EXCLK_OUT = 2                 ' External clock output

'-----------------------------------------------------------------------------------------------
'
'		Reset Identifier
'
'-----------------------------------------------------------------------------------------------
Public Const DA_RESET_ON  = 1	               ' Used
Public Const DA_RESET_OFF = 2	               ' Not used (default setting)

'-----------------------------------------------------------------------------------------------
'
'   Filter Identifier
'
'-----------------------------------------------------------------------------------------------
Public Const DA_FILTER_OFF = 1                ' Not used (default setting)
Public Const DA_FILTER_ON = 2                 ' Used

'-----------------------------------------------------------------------------------------------
'
'   Synchronous Analog Output Identifier
'
'-----------------------------------------------------------------------------------------------
Public Const DA_MASTER_MODE = 1               ' Master mode
Public Const DA_SLAVE_MODE  = 2               ' Slave mode

'-----------------------------------------------------------------------------------------------
'
'     Synchronous Number Identifier
'
'-----------------------------------------------------------------------------------------------
Public Const DA_SYNC_NUM_1 = &H0100
Public Const DA_SYNC_NUM_2 = &H0200
Public Const DA_SYNC_NUM_3 = &H0400
Public Const DA_SYNC_NUM_4 = &H0800
Public Const DA_SYNC_NUM_5 = &H1000
Public Const DA_SYNC_NUM_6 = &H2000
Public Const DA_SYNC_NUM_7 = &H4000

'-----------------------------------------------------------------------------------------------
'
'		GPC/GPF-3100 CPZ-360810 DIN/DOUT Function Identifier
'
'-----------------------------------------------------------------------------------------------
Public Const DA_EX_DIO1	= 1	' DIN/DOUT1
Public Const DA_EX_DIO2 = 2	' DIN/DOUT2
Public Const DA_EX_DIO3 = 3	' DIN/DOUT3
Public Const DA_EX_DIO4 = 4	' DIN/DOUT4
Public Const DA_EX_DIO5 = 5	' DIN/DOUT5

'-----------------------------------------------------------------------------------------------
'
'		PCI-3525 channel 3 and channel 4 Function Identifier
'
'-----------------------------------------------------------------------------------------------
Public Const DA_CN_FREE		 = 0	' not used
Public Const DA_CN_EXTRG_IN	 = 1	' External trigger input
Public Const DA_CN_EXTRG_OUT = 2	' External trigger output
Public Const DA_CN_EXCLK_IN	 = 3	' External clock input
Public Const DA_CN_EXCLK_OUT = 4	' External clock output
Public Const DA_CN_EXINT_IN	 = 5	' External interrupt input
Public Const DA_CN_ATRG_OUT	 = 6	' Analog trigger out
Public Const DA_CN_DI		 = 7	' Digital input
Public Const DA_CN_DO		 = 8	' Digital output
Public Const DA_CN_DAOUT	 = 9	' Analog output
Public Const DA_CN_OPEN		 = 10	' open

'-----------------------------------------------------------------------------------------------
'
'   PCI-3525 External Trigger Polarity Identifier
'
'-----------------------------------------------------------------------------------------------
Public Const DA_START_DOWN_EDGE = 1           ' Start external trigger falling edge
Public Const DA_START_UP_EDGE = 2             ' Start external trigger rising edge
Public Const DA_STOP_DOWN_EDGE = 4            ' Stop external trigger falling edge
Public Const DA_STOP_UP_EDGE = 8              ' Stop external trigger rising edge

'-----------------------------------------------------------------------------------------------
'
'   Fifo Trigger Level Identifier
'
'-----------------------------------------------------------------------------------------------
Public Const DA_TRG_FREERUN = 0               ' No trigger
Public Const DA_TRG_EXTTRG = 1                ' External trigger
Public Const DA_TRG_ATRG = 2                  ' Analog trigger
Public Const DA_TRG_SIGTIMER = 3              ' Interval timer
Public Const DA_TRG_CNT_EQ = 4                ' Counter equal
Public Const DA_TRG_Z_CLR  = 5		          ' Z clear
Public Const DA_TRG_AD_START = 5              ' Analog input start
Public Const DA_TRG_AD_STOP = 6               ' Analog input stop
Public Const DA_TRG_AD_PRETRG = 7             ' Analog input pre-trigger
Public Const DA_TRG_AD_POSTTRG = 8            ' Analog input post-trigger
Public Const DA_TRG_SMPLNUM = 9               ' Analog output stop number
Public Const DA_TRG_FIFO_EMPTY = 10           ' FIFO empty
Public Const DA_TRG_SYNC1 = 14                ' Internel sync1 trigger
Public Const DA_TRG_SYNC2 = 15                ' Internel sync2 trigger
Public Const DA_FIFORESET = &H100             ' FIFO reset
Public Const DA_RETRG = &H200                 ' Retrigger

'-----------------------------------------------------------------------------------------------
'
'   Simultaneous Output Set Identifier
'
'-----------------------------------------------------------------------------------------------
Public Const DA_NORMAL_OUTPUT = 1       ' Not simultaneous output
Public Const DA_SYNC_OUTPUT = 2         ' Simultaneous output

'-----------------------------------------------------------------------------------------------
'
'   Volume Identifier
'
'-----------------------------------------------------------------------------------------------
Public Const DA_ADJUST_BIOFFSET  = 1          ' Bipolar offset calibration
Public Const DA_ADJUST_UNIOFFSET = 2          ' Unipolar offset calibration
Public Const DA_ADJUST_BIGAIN    = 3          ' Bipolar gain calibration
Public Const DA_ADJUST_UNIGAIN   = 4          ' Unipolar gain calibration


'-----------------------------------------------------------------------------------------------
'
'   Calibration Item Identifier
'
'-----------------------------------------------------------------------------------------------
Public Const DA_ADJUST_UP = 1                 ' Increase the volume.
Public Const DA_ADJUST_DOWN = 2               ' Decrease the volume.
Public Const DA_ADJUST_STORE = 3              ' Save the present value to the non-volatile memory.
Public Const DA_ADJUST_STANDBY = 4            ' Place the electronic volume device into the standby mode.
Public Const DA_ADJUST_NOT_STORE = 5          ' Not save the value.


'-----------------------------------------------------------------------------------------------
'
'     Read Adjust Identifier
'
'-----------------------------------------------------------------------------------------------
Public Const DA_ADJUST_READ_FACTORY = 1      ' Factory Setting
Public Const DA_ADJUST_READ_USER = 2         ' User Setting


'-----------------------------------------------------------------------------------------------
'
'   Data Identifier
'
'-----------------------------------------------------------------------------------------------
Public Const DA_DATA_PHYSICAL = 1             ' Physical value (voltage [V], current [mA])
Public Const DA_DATA_BIN8 = 2                 ' 8-bit binary
Public Const DA_DATA_BIN12 = 3                ' 12-bit binary
Public Const DA_DATA_BIN16 = 4                ' 16-bit binary
Public Const DA_DATA_BIN24 = 5                ' 24-bit binary
Public Const DA_DATA_BIN14 = 6                ' 14-bit binary

'-----------------------------------------------------------------------------------------------
'
'   Data Conversion Identifier
'
'-----------------------------------------------------------------------------------------------
Public Const DA_CONV_SMOOTH = 1               ' Convert the data with interpolation.
Public Const DA_CONV_AVERAGE1 = &H100         ' Convert the data with the simple averaging.
Public Const DA_CONV_AVERAGE2 = &H200         ' Convert the data with the shifted averaging.

'-----------------------------------------------------------------------------------------------
'
'   Error Identifier
'
'-----------------------------------------------------------------------------------------------
Public Const DA_ERROR_SUCCESS = 0
Public Const DA_ERROR_NOT_DEVICE = &HC0000001
Public Const DA_ERROR_NOT_OPEN = &HC0000002
Public Const DA_ERROR_INVALID_HANDLE = &HC0000003
Public Const DA_ERROR_ALREADY_OPEN = &HC0000004
Public Const DA_ERROR_NOT_SUPPORTED = &HC0000009
Public Const DA_ERROR_NOW_SAMPLING = &HC0001001
Public Const DA_ERROR_STOP_SAMPLING = &HC0001002
Public Const DA_ERROR_START_SAMPLING = &HC0001003
Public Const DA_ERROR_SAMPLING_TIMEOUT = &HC0001004
Public Const DA_ERROR_INVALID_PARAMETER = &HC0001021
Public Const DA_ERROR_ILLEGAL_PARAMETER = &HC0001022
Public Const DA_ERROR_NULL_POINTER = &HC0001023
Public Const DA_ERROR_SET_DATA = &HC0001024
Public Const DA_ERROR_USED_AD = &HC0001025
Public Const DA_ERROR_FILE_OPEN = &HC0001041
Public Const DA_ERROR_FILE_CLOSE = &HC0001042
Public Const DA_ERROR_FILE_READ = &HC0001043
Public Const DA_ERROR_FILE_WRITE = &HC0001044
Public Const DA_ERROR_INVALID_DATA_FORMAT = &HC0001061
Public Const DA_ERROR_INVALID_AVERAGE_OR_SMOOTHING = &HC0001062
Public Const DA_ERROR_INVALID_SOURCE_DATA = &HC0001063
Public Const DA_ERROR_NOT_ALLOCATE_MEMORY = &HC0001081
Public Const DA_ERROR_NOT_LOAD_DLL = &HC0001082
Public Const DA_ERROR_CALL_DLL = &HC0001083
Public Const DA_ERROR_CALIBRATION = &HC0001084
Public Const DA_ERROR_USBIO_FAILED = &HC0001085
Public Const DA_ERROR_USBIO_TIMEOUT = &HC0001086


'-------------------------------------------------------------------------------------------
'
'   Structure Declaration
'
'-------------------------------------------------------------------------------------------

' -----------------------------------------------------------------------
'  Analog Output Request Condition Structure for Each Channel
' -----------------------------------------------------------------------
Type DASMPLCHREQ
    ulChNo As Long
    ulRange As Long
End Type

' -----------------------------------------------------------------------
'  Analog Output Request Condition Structure
' -----------------------------------------------------------------------
Type DASMPLREQ
    ulChCount As Long
    SmplChReq(0 To 255) As DASMPLCHREQ
    ulSamplingMode As Long
    fSmplFreq As Single
    ulSmplRepeat As Long
    ulTrigMode As Long
    ulTrigPoint As Long
    ulTrigDelay As Long
    ulEClkEdge As Long
    ulTrigEdge As Long
    ulTrigDI As Long
End Type

' -----------------------------------------------------------------------
'  Board Specification Structure
' -----------------------------------------------------------------------
Type DABOARDSPEC
    ulBoardType As Long
    ulBoardID As Long
    ulSamplingMode As Long
    ulChCount As Long
    ulResolution As Long
    ulRange As Long
    ulIsolation As Long
    ulDi As Long
    ulDo As Long
End Type

' -----------------------------------------------------------------------
'  Output Range Configurations Structure for Each Channel (for the PCI/PAZ-3305)
' -----------------------------------------------------------------------
Type DAMODECHREQ
    ulRange As Long
    fVolt As Single
    ulFilter As Long
End Type

' -----------------------------------------------------------------------
'  Waveform Generation Mode Structure (for the PCI/PAZ-3305)
' -----------------------------------------------------------------------
Type DAMODEREQ
    ModeChReq(0 To 1) As DAMODECHREQ
    ulPulseMode As Long
    ulSyntheOut As Long
    ulInterval As Long
    fIntervalCycle As Single
    ulCounterClear As Long
    ulDaLatch As Long
    ulSamplingClock As Long
    ulExControl As Long
    ulExClock As Long
End Type

' -----------------------------------------------------------------------
'  Fifo Analog Output Request Condition Structure (for the PCI-3525)
' -----------------------------------------------------------------------
Type DAFIFOREQ
    ulChCount As Long
    SmplChReq(0 To 255) As DASMPLCHREQ
    fSmplFreq As Single
    ulSmplRepeat As Long
    ulSmplNum As Long
    ulStartTrigCondition As Long
    ulStopTrigCondition As Long
    ulEClkEdge As Long
    ulTrigEdge As Long
End Type


'-------------------------------------------------
'
'   Function Declaration
'
'-------------------------------------------------
Declare Function DaOpen Lib "FbiDa.DLL" (ByVal lpszName As String) As Long
Declare Function DaClose Lib "FbiDa.DLL" (ByVal hDeviceHandle As Long) As Long

Declare Function DaGetDeviceInfo Lib "FbiDa.DLL" (ByVal hDeviceHandle As Long, ByRef pDaBoardSpec As DABOARDSPEC) As Long

Declare Function DaSetBoardConfig Lib "FbiDa.DLL" (ByVal hDeviceHandle As Long, ByVal ulSmplBufferSize As Long, ByVal hEvent As Long, ByVal lpCallBackProc As Long, ByVal dwUser As Long) As Long
Declare Function DaSetCountEvent Lib "FbiDa.DLL" (ByVal hDeviceHandle As Long, ByVal ulEventNum As Long, ByVal hEvent As Long, ByVal lpCallBackProc As Long, ByVal dwUser As Long) As Long
Declare Function DaGetBoardConfig Lib "FbiDa.DLL" (ByVal hDeviceHandle As Long, ByRef ulSmplBufferSize As Long, ByRef ulDaSmplEventFactor As Long) As Long

Declare Function DaSetSamplingConfig Lib "FbiDa.DLL" (ByVal hDeviceHandle As Long, ByRef pDaSmplConfig As DASMPLREQ) As Long
Declare Function DaGetSamplingConfig Lib "FbiDa.DLL" (ByVal hDeviceHandle As Long, ByRef pDaSmplConfig As DASMPLREQ) As Long

Declare Function DaSetMode Lib "FbiDa.DLL" (ByVal hDeviceHandle As Long, ByRef pDaMode As DAMODEREQ) As Long
Declare Function DaGetMode Lib "FbiDa.DLL" (ByVal hDeviceHandle As Long, ByRef pDaMode As DAMODEREQ) As Long

Declare Function DaSetSamplingData Lib "FbiDa.DLL" (ByVal hDeviceHandle As Long, ByRef pSmplData As Any, ByVal ulSmplDataNum As Long) As Long
Declare Function DaClearSamplingData Lib "FbiDa.DLL" (ByVal hDeviceHandle As Long) As Long

Declare Function DaStartSampling Lib "FbiDa.DLL" (ByVal hDeviceHandle As Long, ByVal ulSyncFlag As Long) As Long
Declare Function DaStartFileSampling Lib "FbiDa.DLL" (ByVal hDeviceHandle As Long, ByVal pszPathName As String, ByVal ulFileFlag As Long, ByVal ulSmplNum As Long) As Long
Declare Function DaSyncSampling Lib "FbiDa.DLL" (ByVal hDeviceHandle As Long, ByVal ulMode As Long) As Long
Declare Function DaStopSampling Lib "FbiDa.DLL" (ByVal hDeviceHandle As Long) As Long

Declare Function DaGetStatus Lib "FbiDa.DLL" (ByVal hDeviceHandle As Long, ByRef ulDaSmplStatus As Long, ByRef ulDaSmplCount As Long, ByRef ulDaAvailCount As Long, ByRef ulDaAvailRepeat As Long) As Long

Declare Function DaGetOutputMode Lib "FbiDa.DLL" (ByVal hDeviceHandle As Long, ByRef ulMode As Long) As Long
Declare Function DaSetOutputMode Lib "FbiDa.DLL" (ByVal hDeviceHandle As Long, ByVal ulMode As Long) As Long

Declare Function DaOutputDA Lib "FbiDa.DLL" (ByVal hDeviceHandle As Long, ByVal nCh As Long, ByRef pDaSmplChReq As DASMPLCHREQ, ByRef pData As Any) As Long

Declare Function DaInputDI Lib "FbiDa.DLL" (ByVal hDeviceHandle As Long, ByRef dwData As Long) As Long
Declare Function DaOutputDO Lib "FbiDa.DLL" (ByVal hDeviceHandle As Long, ByVal dwData As Long) As Long

Declare Function DaAdjustVR Lib "FbiDa.DLL" (ByVal hDeviceHandle As Long, ByVal ulAdjustCh As Long, ByVal ulSelVolume As Long, ByVal ulDirection As Long, ByVal ulTap As Long) As Long
Declare Function DaReadAdjustVR Lib "FbiDa.DLL" (ByVal hDeviceHandle As Long, ByVal ulAdjustCh As Long) As Long
Declare Function DaReadAdjustVREx Lib "FbiDa.DLL" (ByVal hDeviceHandle As Long, ByVal ulAdjustCh As Long, ByVal ulControl As Long) As Long

Declare Function DaDataConv Lib "FbiDaDc.DLL" (ByVal uSrcFormCode As Long, ByRef pSrcData As Any, ByVal uSrcSmplDataNum As Long, ByRef pSrcSmplReq As DASMPLREQ, ByVal uDestFormCode As Long, ByRef pDestData As Any, ByRef puDestSmplDataNum As Long, ByRef pDestSmplReq As DASMPLREQ, ByVal uEffect As Long, ByVal uCount As Long, ByVal pfnConv As Long) As Long
Declare Function DaWriteFile Lib "FbiDaDc.DLL" (ByVal pszPathName As String, ByRef pSmplData As Any, ByVal ulFormCode As Long, ByVal ulSmplNum As Long, ByVal ulChCount As Long) As Long

Declare Function CreateEvent Lib "kernel32" Alias "CreateEventA" (ByVal lpEventAttributes As Long, ByVal ManualReset As Long, ByVal bInitialState As Long, ByVal lpName As String) As Long
Declare Function WaitForSingleObject Lib "kernel32" (ByVal hHandle As Long, ByVal dwMilliseconds As Long) As Long
Declare Function CloseHandle Lib "kernel32" (ByVal hObject As Long) As Long
Declare Function ResetEvent Lib "kernel32" (ByVal hEvent As Long) As Long

Declare Function DaSetFifoConfig Lib "FbiDa.DLL" (ByVal hDeviceHandle As Long, ByRef pDaFifoConfig As DAFIFOREQ) As Long
Declare Function DaGetFifoConfig Lib "FbiDa.DLL" (ByVal hDeviceHandle As Long, ByRef pDaFifoConfig As DAFIFOREQ) As Long

Declare Function DaSetInterval Lib "FbiDa.DLL" (ByVal hDeviceHandle As Long, ByVal ulInterval As Long) As Long
Declare Function DaGetInterval Lib "FbiDa.DLL" (ByVal hDeviceHandle As Long, ByRef ulInterval As Long) As Long

Declare Function DaSetFunction Lib "FbiDa.DLL" (ByVal hDeviceHandle As Long, ByVal ulChNo As Long, ByVal ulFunction As Long) As Long
Declare Function DaGetFunction Lib "FbiDa.DLL" (ByVal hDeviceHandle As Long, ByVal ulChNo As Long, ByRef ulFunction As Long) As Long

Declare Function DaCalibration Lib "FbiDa.DLL" (ByVal hDeviceHandle As Long, ByVal ulAdjustCh As Long, ByVal ulRange As Long) As Long

Declare Function DaSetCurrentDir Lib "FbiDa.DLL" (ByVal hDeviceHandle As Long, ByVal dwData As Long) As Long
Declare Function DaGetCurrentDir Lib "FbiDa.DLL" (ByVal hDeviceHandle As Long, ByRef dwData As Long) As Long
Declare Function DaSetPowerSupply Lib "FbiDa.DLL" (ByVal hDeviceHandle As Long, ByVal ExOnOff As Long) As Long
Declare Function DaGetPowerSupply Lib "FbiDa.DLL" (ByVal hDeviceHandle As Long, ByRef ExOnOff As Long) As Long
Declare Function DaSetExcessVoltage Lib "FbiDa.DLL" (ByVal hDeviceHandle As Long, ByVal ExOnOff As Long) As Long
Declare Function DaGetRelayStatus Lib "FbiDa.DLL" (ByVal hDeviceHandle As Long, ByRef Status As Long) As Long
Declare Function DaGetOVStatus Lib "FbiDa.DLL" (ByVal hDeviceHandle As Long, ByRef LowStatus As Long, ByRef HighStatus As Long) As Long

