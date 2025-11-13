Attribute VB_Name = "FbiAd"
'-----------------------------------------------------------------------------------------------
'
'   GPC/GPF-3100 Visual Basic  Overlapped Process Identifier
'
'-----------------------------------------------------------------------------------------------

Public Const FLAG_SYNC = 1               ' Sampling as an non-overlapped operation
Public Const FLAG_ASYNC = 2              ' Sampling as overlapped operation


'-----------------------------------------------------------------------------------------------
'
'   GPC/GPF-3100 Visual Basic  File Format Identifier
'
'-----------------------------------------------------------------------------------------------

Public Const FLAG_BIN = 1                   ' Binary format file
Public Const FLAG_CSV = 2                   ' CSV format file


'-----------------------------------------------------------------------------------------------
'
'   GPC/GPF-3100 Visual Basic  Sampling Status Identifier
'
'-----------------------------------------------------------------------------------------------
Public Const AD_STATUS_STOP_SAMPLING = 1      ' The sampling is stopped.
Public Const AD_STATUS_WAIT_TRIGGER = 2       ' The sampling is waiting for a trigger.
Public Const AD_STATUS_NOW_SAMPLING = 3       ' The sampling is running.


'-----------------------------------------------------------------------------------------------
'
'   GPC/GPF-3100 Visual Basic  Event Factor Identifier
'
'-----------------------------------------------------------------------------------------------
Public Const AD_EVENT_SMPLNUM = 1                  ' The specified number of samples has been acquired.
Public Const AD_EVENT_STOP_TRIGGER = 2             ' The sampling has been stopped because a trigger is asserted.
Public Const AD_EVENT_STOP_FUNCTION = 3            ' The sampling has been stopped by software.
Public Const AD_EVENT_STOP_TIMEOUT = 4             ' The sampling has been stopped because a time-out interval elapsed.
Public Const AD_EVENT_STOP_SAMPLING = 5            ' The sampling is stopped.
Public Const AD_EVENT_STOP_SCER = 6                ' The sampling is stopped by a clock error.
Public Const AD_EVENT_STOP_ORER = 7                ' The sampling is stopped by an overrun error.
Public Const AD_EVENT_SCER = 8                     ' The sampling pacer clock error is occurred.
Public Const AD_EVENT_ORER = 9                     ' The overrun error is occurred.
Public Const AD_EVENT_STOP_LV_1 = 10               ' The channel 1 sampling is stopped. (Only applicable to the PCI-3179)
Public Const AD_EVENT_STOP_LV_2 = 11               ' The channel 2 sampling is stopped. (Only applicable to the PCI-3179)
Public Const AD_EVENT_STOP_LV_3 = 12               ' The channel 3 sampling is stopped. (Only applicable to the PCI-3179)
Public Const AD_EVENT_STOP_LV_4 = 13               ' The channel 4 sampling is stopped. (Only applicable to the PCI-3179)
Public Const AD_EVENT_RANGE = 14                   ' The AD conversion value reached the full-scale range.
Public Const AD_EVENT_STOP_RANGE = 15              ' The sampling is stopped by the full-scale range detection.
Public Const AD_EVENT_OVPM = 16                    ' The AD conversion value reached the OVPM.
Public Const AD_EVENT_STOP_OVPM = 17               ' The sampling is stopped by theOVPM.
Public Const AD_EVENT_DISCONNECTION = &H100        ' Solcon ADPU3215,3216 Disconnection.


'-----------------------------------------------------------------------------------------------
'
'   GPC/GPF-3100 Visual Basic  Input Configuration Identifier
'
'-----------------------------------------------------------------------------------------------
Public Const AD_INPUT_SINGLE = 1            ' Single-ended input
Public Const AD_INPUT_DIFF = 2              ' Differential input


'-----------------------------------------------------------------------------------------------
'
'   GPC/GPF-3100 Visual Basic  Volume Identifier
'
'-----------------------------------------------------------------------------------------------
Public Const AD_ADJUST_BIOFFSET = 1            ' Bipolar offset
Public Const AD_ADJUST_UNIOFFSET = 2           ' Unipolar offset
Public Const AD_ADJUST_BIGAIN = 3              ' Bipolar gain
Public Const AD_ADJUST_UNIGAIN = 4             ' Unipolar gain


'-----------------------------------------------------------------------------------------------
'
'   GPC/GPF-3100 Visual Basic  Calibration Item Identifier
'
'-----------------------------------------------------------------------------------------------
Public Const AD_ADJUST_UP = 1              ' Increases the volume.
Public Const AD_ADJUST_DOWN = 2            ' Decreases the volume.
Public Const AD_ADJUST_STORE = 3           ' Saves the present value to the non-volatile memory.
Public Const AD_ADJUST_STANDBY = 4         ' Places the electronic volume device into the standby mode.
Public Const AD_ADJUST_NOT_STORE = 5       ' Not save the value.

'-----------------------------------------------------------------------------------------------
'
'  GPC/GPF-3100 Read Adjust Identifier
'
'-----------------------------------------------------------------------------------------------
Public Const AD_ADJUST_READ_FACTORY = 1
Public Const AD_ADJUST_READ_USER = 2

'-----------------------------------------------------------------------------------------------
'
'   GPC/GPF-3100 Visual Basic  Data Identifier
'
'-----------------------------------------------------------------------------------------------
Public Const AD_DATA_PHYSICAL = 1             ' Physical value (voltage [V] or current [mA])
Public Const AD_DATA_BIN8 = 2                 ' 8-bit binary
Public Const AD_DATA_BIN12 = 3                ' 12-bit binary
Public Const AD_DATA_BIN16 = 4                ' 16-bit binary
Public Const AD_DATA_BIN24 = 5                ' 24-bit binary
Public Const AD_DATA_BIN10 = 6                ' 10-bit binary

'-----------------------------------------------------------------------------------------------
'
'   GPC-3100 VisualBasic  Data Conversion Identifier
'
'-----------------------------------------------------------------------------------------------
Public Const AD_CONV_SMOOTH = 1               ' Converts the data with interpolation.
Public Const AD_CONV_AVERAGE1 = &H100         ' Converts the data with the simple averaging.
Public Const AD_CONV_AVERAGE2 = &H200         ' Converts the data with the shifted averaging.


'-----------------------------------------------------------------------------------------------
'
'   GPC/GPF-3100 Visual Basic  Sampling Mode Identifier
'
'-----------------------------------------------------------------------------------------------
Public Const AD_IO_SAMPLING = 1               ' Programmed I/O
Public Const AD_FIFO_SAMPLING = 2             ' FIFO
Public Const AD_MEM_SAMPLING = 4              ' Memory
Public Const AD_BM_SAMPLING = 8               ' Bus master

'-----------------------------------------------------------------------------------------------
'
'   GPC/GPF-3100 Visual Basic  Trigger Timing Identifier
'
'-----------------------------------------------------------------------------------------------
Public Const AD_TRIG_START = 1                ' Start-trigger (Default setting)
Public Const AD_TRIG_STOP = 2                 ' Stop-trigger
Public Const AD_TRIG_START_STOP = 3           ' Start/stop-trigger


'-----------------------------------------------------------------------------------------------
'
'   GPC/GPF-3100 Visual Basic  Trigger Level Identifier
'
'-----------------------------------------------------------------------------------------------
Public Const AD_FREERUN = 1                   ' No trigger (default setting)
Public Const AD_EXTTRG = 2                    ' External trigger
Public Const AD_EXTTRG_DI = 3                 ' External trigger with mask using general purpose digital input pin
Public Const AD_LEVEL_P = 4                   ' Analog trigger (low-to-high transition)
Public Const AD_LEVEL_M = 5                   ' Analog trigger (high-to-low transition)
Public Const AD_LEVEL_D = 6                   ' Analog trigger (low-to-high or high-to-low transition)
Public Const AD_INRANGE = 7                   ' Analog  trigger (into the range)
Public Const AD_OUTRANGE = 8                  ' Analog trigger (out of the range)
Public Const AD_ETERNITY = 9                  ' Infinite sampling
Public Const AD_SMPLNUM = 10                  ' Specified number
Public Const AD_START_SIGTIMER = 11           ' Interval timer
Public Const AD_START_DA_START = 12           ' Analog output start (DaStartSampling)
Public Const AD_START_DA_STOP = 13            ' Analog output stop
Public Const AD_START_DA_IO = 14              ' Analog output (DaOutputDA)
Public Const AD_START_DA_SMPLNUM = 15         ' Analog output number
Public Const AD_STOP_DA_START = 12            ' Analog output start (DaStartSampling)
Public Const AD_STOP_DA_STOP = 13             ' Analog output stop
Public Const AD_STOP_DA_IO = 14               ' Analog output (DaOutputDA)
Public Const AD_STOP_DA_SMPLNUM = 15          ' Analog output num
Public Const AD_START_P1 = &H10               ' Start-trigger (Level 1): low-to- high transition
Public Const AD_START_M1 = &H20               ' Start-trigger (Level 1): high-to-low transition
Public Const AD_START_D1 = &H40               ' Start-trigger (Level 1): high-to-low or low -to-high transition (direction DON'T CARE)
Public Const AD_START_P2 = &H80               ' Start-trigger (Level 2): low-to-high transition
Public Const AD_START_M2 = &H100              ' Start-trigger (Level 2): high-to-low transition
Public Const AD_START_D2 = &H200              ' Start-trigger (Level 2): high-to-low or low -to-high transition (direction DON'T CARE)
Public Const AD_STOP_P1 = &H400               ' Stop-trigger (Level 1): low-to-high transition
Public Const AD_STOP_M1 = &H800               ' Stop-trigger (Level 1): high-to-low transition
Public Const AD_STOP_D1 = &H1000              ' Stop-trigger (Level 1): high-to-low or low -to-high transition (direction DON'T CARE)
Public Const AD_STOP_P2 = &H2000              ' Stop-trigger (Level 2): low-to-high transition
Public Const AD_STOP_M2 = &H4000              ' Stop-trigger (Level 2): high-to-low transition
Public Const AD_STOP_D2 = &H8000&             ' Stop-trigger (Level 2): high-to-low or low -to-high transition (direction DON'T CARE)
Public Const AD_ANALOG_FILTER = &H10000       ' Uses an analog trigger filter.
Public Const AD_START_CNT_EQ = &H20000        ' Start-trigger: Counter equal
Public Const AD_STOP_CNT_EQ = &H40000         ' Stop-trigger: Counter equal
Public Const AD_START_DI_EQ = &H80000         ' Stop-trigger: DI equal
Public Const AD_STOP_DI_EQ = &H100000         ' Stop-trigger: DI equal
Public Const AD_STOP_SOFT = &H200000          ' Stop-trigger: Soft stop
Public Const AD_START_Z_CLR = &H400000        ' Start-trigger: Z clear
Public Const AD_STOP_Z_CLR = &H800000         ' Stop-trigger: Z clear
'MATLAB
Public Const AD_START_SOFT = &H1000000
Public Const AD_SOFTTRG = &H2000000

Public Const AD_START_SYNC1 = &H10000000      ' Start-trigger: Internal Sync1 trigger
Public Const AD_START_SYNC2 = &H20000000      ' Start-trigger: Interna2 Sync1 trigger
Public Const AD_STOP_SYNC1  = &H40000000      ' Stop-trigger: Internal Sync1 trigger
Public Const AD_STOP_SYNC2  = &H80000000      ' Stop-trigger: Interna2 Sync1 trigger

'-----------------------------------------------------------------------------------------------
'
'   GPC/GPF-3100 Visual Basic  Polarity Identifier
'
'-----------------------------------------------------------------------------------------------
Public Const AD_DOWN_EDGE = 1              ' Falling edge (default setting)
Public Const AD_UP_EDGE = 2                ' Rising edge
Public Const AD_EXTRG_IN = 3               ' External trigger input
Public Const AD_EXCLK_IN = 4               ' External clock input
Public Const AD_LOW_LEVEL = 5                      ' Negative level (default setting)
Public Const AD_HIGH_LEVEL = 6                     ' Positive level

Public Const AD_EDGE_P1 = &H10             ' Level 1: low-to-high transition
Public Const AD_EDGE_M1 = &H20             ' Level 1: high-to-low transition
Public Const AD_EDGE_D1 = &H40             ' Level 1: high-to-low or low -to-high transition (direction DON'T CARE)
Public Const AD_EDGE_P2 = &H80             ' Level 2: low-to-high transition
Public Const AD_EDGE_M2 = &H100            ' Level 2: high-to-low transition
Public Const AD_EDGE_D2 = &H200            ' Level 2: high-to-low or low -to-high transition (direction DON'T CARE)
Public Const AD_DISABLE = &H80000000       ' No pulse output (defult setting)

Public Const AD_TRIG_MODE = 2              '
Public Const AD_BUSY_MODE = 3              '
Public Const AD_POST_MODE = 4              '
Public Const AD_ENABLE = 5                 '
Public Const AD_SMP1_MODE = 6              '
Public Const AD_SMP2_MODE = 7              '
Public Const AD_ATRIG_MODE = 8             '

'-----------------------------------------------------------------------------------------------
'
'   GPC/GPF-3100 Visual Basic  Pulse Polarity Identifier
'
'-----------------------------------------------------------------------------------------------
Public Const AD_LOW_PULSE = 1                ' Negative pulse (default setting)
Public Const AD_HIGH_PULSE = 2               ' Positive pulse


'-----------------------------------------------------------------------------------------------
'
'   GPC/GPF-3100 Visual Basic  Double-Clocked Mode Identifier
'
'-----------------------------------------------------------------------------------------------
Public Const AD_NORMAL_MODE = 1            ' Normal mode (default setting)
Public Const AD_FAST_MODE = 2              ' Double-clocked mode


'-----------------------------------------------------------------------------------------------
'
'   GPC/GPF-3100 Visual Basic  Status Identifier
'
'-----------------------------------------------------------------------------------------------
Public Const AD_NO_STATUS = 1                ' Adds no bus master sampling status. (default setting)
Public Const AD_ADD_STATUS = 2               ' Adds bus master sampling status.


'-----------------------------------------------------------------------------------------------
'
'   GPC/GPF-3100 Visual Basic  Error Control Identifier
'
'-----------------------------------------------------------------------------------------------
Public Const AD_STOP_SCER = 2                 ' Stops sampling by a sampling clock error.
Public Const AD_STOP_ORER = 4                 ' Stops sampling by an overrun error.


'-----------------------------------------------------------------------------------------------
'
'   GPC/GPF-3100 Visual Basic  Data Save Identifier
'
'-----------------------------------------------------------------------------------------------
Public Const AD_APPEND = 1                  ' Adds new data at the end of the buffer. (default setting)
Public Const AD_OVERWRITE = 2               ' Oversrites new data on existing data.


'-----------------------------------------------------------------------------------------------
'
'   GPC/GPF-3100 Visual Basic  Range Identifier
'
'-----------------------------------------------------------------------------------------------
Public Const AD_0_1V = &H1                ' Voltage: unipolar 0 V to 1 V
Public Const AD_0_2P5V = &H2              ' Voltage: unipolar 0 V to 2.5 V
Public Const AD_0_5V = &H4                ' Voltage: unipolar 0 V to 5 V
Public Const AD_0_10V = &H8               ' Voltage: unipolar 0 V to 10 V
Public Const AD_1_5V = &H10               ' Voltage: unipolar 1 V to 5 V
Public Const AD_0_2V = &H20               ' Voltage: unipolar 0 V to 2 V
Public Const AD_0_0P125V = &H40           ' Voltage: unipolar 0 V to 0.125V
Public Const AD_0_1P25V = &H80            ' Voltage: unipolar 0 V to 1.25v
Public Const AD_0_0P625V = &H100          ' Voltage: unipolar 0 V to 0.625V
Public Const AD_0_0P156V = &H200          ' Voltage: unipolar 0 V to 0.156V
Public Const AD_0_20mA = &H1000           ' Current: unipolar 0 mA to 20 mA
Public Const AD_4_20mA = &H2000           ' Current: unipolar 4 mA to 20 mA
Public Const AD_20mA = &H4000             ' Current: bipolar +/- 20 mA
Public Const AD_1V = &H10000              ' Voltage: bipolar +/- 1 V
Public Const AD_2P5V = &H20000            ' Voltage: bipolar +/- 2.5 V
Public Const AD_5V = &H40000              ' Voltage: bipolar +/- 5 V
Public Const AD_10V = &H80000             ' Voltage: bipolar +/- 10 V
Public Const AD_20V = &H100000            ' Voltage: bipolar +/- 20 V
Public Const AD_50V = &H200000            ' Voltage: bipolar +/- 50 V
Public Const AD_0P125V = &H400000         ' Voltage: bipolar +/- 0.125 V
Public Const AD_1P25V = &H800000          ' Voltage: bipolar +/- 1.25 V
Public Const AD_0P625V = &H1000000        ' Voltage: bipolar +/- 0.625 V
Public Const AD_0P156V = &H2000000        ' Voltage: bipolar +/- 0.156 V
Public Const AD_1P25V_AC = &H4000000      ' Voltage: bipolar +/- 1.25 V (AC-Coupling)
Public Const AD_0P625V_AC = &H8000000     ' Voltage: bipolar +/- 0.625 V (AC-Coupling)
Public Const AD_0P156V_AC = &H10000000    ' Voltage: bipolar +/- 0.156 V (AC-Coupling)
Public Const AD_AC_COUPLING = &H40000000  ' AC-Coupling
Public Const AD_GNG = &H80000000          ' GNG

Public Const AD_TYPE_K = &H00000001       ' TYPE K
Public Const AD_TYPE_B = &H00000002       ' TYPE B
Public Const AD_TYPE_R = &H00000004       ' TYPE R
Public Const AD_TYPE_S = &H00000008       ' TYPE S
Public Const AD_TYPE_N = &H00000010       ' TYPE N
Public Const AD_TYPE_E = &H00000020       ' TYPE E
Public Const AD_TYPE_J = &H00000040       ' TYPE J
Public Const AD_TYPE_T = &H00000080       ' TYPE T

'-----------------------------------------------------------------------------------------------
'
'   GPC/GPF-3100 Visual Basic  Isolation Identifier
'
'-----------------------------------------------------------------------------------------------
Public Const AD_ISOLATION = 1              ' Isolated
Public Const AD_NOT_ISOLATION = 2          ' Not isolated


'-----------------------------------------------------------------------------------------------
'
'   GPC/GPF-3100 Visual Basic  Synchronous Mode Identifier
'
'-----------------------------------------------------------------------------------------------
Public Const AD_MASTER_MODE = 1             ' Master mode
Public Const AD_SLAVE_MODE = 2              ' Slave mode

'-----------------------------------------------------------------------------------------------
'
'     GPC/GPF-3100 Synchronous Number Identifier
'
'-----------------------------------------------------------------------------------------------
Public Const AD_SYNC_NUM_1 = &H100
Public Const AD_SYNC_NUM_2 = &H200
Public Const AD_SYNC_NUM_3 = &H400
Public Const AD_SYNC_NUM_4 = &H800
Public Const AD_SYNC_NUM_5 = &H1000
Public Const AD_SYNC_NUM_6 = &H2000
Public Const AD_SYNC_NUM_7 = &H4000

' -----------------------------------------------------------------------------------------------
'
'               GPC/GPF-3100 Degital Filter Identifier
'
' -----------------------------------------------------------------------------------------------
Public Const AD_DF_8 = 0          ' 8 (default setting)
Public Const AD_DF_16 = 1         ' 16
Public Const AD_DF_32 = 2         ' 32
Public Const AD_DF_64 = 3         ' 64
Public Const AD_DF_128 = 4        ' 128
Public Const AD_DF_256 = 5        ' 256

'-----------------------------------------------------------------------------------------------
'
'   GPC/GPF-3100 Visual Basic Calibration Identifier
'
'-----------------------------------------------------------------------------------------------
Public Const AD_SELF_CALIBRATION = 1           ' Self calibration of the AD converter
Public Const AD_ZEROSCALE_CALIBRATION = 2      ' Zero voltage calibration (system calibration)
Public Const AD_FULLSCALE_CALIBRATION = 3      ' Full scale voltage calibration (system calibration)

'-----------------------------------------------------------------------------------------------
'
'   GPC/GPF-3100 Visual Basic  Full-scale Identifier
'
'-----------------------------------------------------------------------------------------------
Public Const AD_STATUS_NEGATIVE_FULL_SCALE = 1    ' Negative full-scale
Public Const AD_STATUS_POSITIVE_FULL_SCALE = 2    ' Positive full-scale
Public Const AD_STATUS_UNDER_RANGE = 1            ' Negative full-scale
Public Const AD_STATUS_OVER_RANGE = 2             ' Positive full-scale


'-----------------------------------------------------------------------------------------------
'
'   GPC/GPF-3100 OVPM Identifier
'
'-----------------------------------------------------------------------------------------------
Public Const AD_STATUS_OVPM_NORMA = 0           ' NormalRange
Public Const AD_STATUS_OVPM_HIGH_RANGE = 1      ' Voltage: bipolar +/- 50 V Range
Public Const AD_STATUS_OVPM_GND_RANGE = 2       ' GND


'-----------------------------------------------------------------------------------------------
'
'   GPC/GPF-3100 Visual Basic  PCI-3525 CN3,4 Function Identifier
'
'-----------------------------------------------------------------------------------------------
Public Const AD_CN_FREE = 0                 ' Not used
Public Const AD_CN_EXTRG_IN = 1             ' External trigger input
Public Const AD_CN_EXTRG_OUT = 2            ' External trigger output
Public Const AD_CN_EXCLK_IN = 3             ' External clock input
Public Const AD_CN_EXCLK_OUT = 4            ' External clock output
Public Const AD_CN_EXINT_IN = 5             ' External interrupt input
Public Const AD_CN_ATRG_OUT = 6             ' Analog trigger out
Public Const AD_CN_DI = 7                   ' Digital input
Public Const AD_CN_DO = 8                   ' Digital output
Public Const AD_CN_DAOUT = 9                ' Analog output
Public Const AD_CN_OPEN = 10                ' Open
Public Const AD_CN_EXSMP1_OUT = 12          ' Sampling Status1
Public Const AD_CN_EXSMP2_OUT = 13          ' Sampling Status2
Public Const AD_CN_DIO = 0                  ' DIO used
Public Const AD_CN_CONTROL = 1              ' Control input
Public Const AD_CN_CNT = 2                  ' Counter output

' -----------------------------------------------------------------------------------------------
'
'               GPC/GPF-3100 CPZ-360810 DIN/DOUT Function Identifier
'
' -----------------------------------------------------------------------------------------------
Public Const AD_EX_DIO1 = 1     ' DIN/DOUT1
Public Const AD_EX_DIO2 = 2     ' DIN/DOUT2
Public Const AD_EX_DIO3 = 3     ' DIN/DOUT3
Public Const AD_EX_DIO4 = 4     ' DIN/DOUT4
Public Const AD_EX_DIO5 = 5     ' DIN/DOUT5
Public Const AD_EX_DIO6 = 6     ' DIN/DOUT6
Public Const AD_EX_DIO7 = 7     ' DIN/DOUT7
Public Const AD_EX_DIO8 = 8     ' DIN/DOUT8

'-----------------------------------------------------------------------------------------------
'
'     GPC/GPF-3100 Measure Temperature
'
'-----------------------------------------------------------------------------------------------
Public Const AD_GET_CURRENT_TEMPERATURE = 1            ' Current Temperature
Public Const AD_LOAD_TEMPERATURE = 2                   ' User area Temperature
Public Const AD_LOAD_FACTORY_SETTING_TEMPERATURE = 3   ' Factory setting Temperature
Public Const AD_SAVE_TEMPERATURE_USER = 4              ' User area of Eeprom 

'-------------------------------------------------
'
'   GPC/GPF-3100 Visual Basic  Structure Declaration
'
'-------------------------------------------------

' -----------------------------------------------------------------------
'  Sampling Condition Structure for Each Channel
' -----------------------------------------------------------------------
Type ADSMPLCHREQ
    ulChNo As Long
    ulRange As Long
End Type


' -----------------------------------------------------------------------
'  Analog Trigger Condition Structure for Each Channel
' -----------------------------------------------------------------------
Type ADTRIGCHREQ
    ulChNo As Long
    fTrigLevel As Single
    fHysteresis As Single
End Type


' -----------------------------------------------------------------------
'  Sampling Condition Structure
' -----------------------------------------------------------------------
Type ADSMPLREQ
    ulChCount As Long
    SmplChReq(0 To 255) As ADSMPLCHREQ
    ulSamplingMode As Long
    ulSingleDiff As Long
    ulSmplNum As Long
    ulSmplEventNum As Long
    fSmplFreq As Single
    ulTrigPoint As Long
    ulTrigMode As Long
    lTrigDelay As Long
    ulTrigCh As Long
    fTrigLevel1 As Single
    fTrigLevel2 As Single
    ulEClkEdge As Long
    ulATrgPulse As Long
    ulTrigEdge As Long
    ulTrigDI As Long
    ulFastMode As Long
End Type


' -----------------------------------------------------------------------
'  Sampling Condition Structure for Bus Master
' -----------------------------------------------------------------------
Type ADBMSMPLREQ
    ulChCount As Long
    SmplChReq(0 To 255) As ADSMPLCHREQ
    ulSingleDiff As Long
    ulSmplNum As Long
    ulSmplEventNum As Long
    ulSmplRepeat As Long
    ulBufferMode As Long
    fSmplFreq As Single
    fScanFreq As Single
    ulStartMode As Long
    ulStopMode As Long
    ulPreTrigDelay As Long
    ulPostTrigDelay As Long
        TrigChReq(0 To 1) As ADTRIGCHREQ
    ulATrgMode As Long
    ulATrgPulse As Long
    ulStartTrigEdge As Long
    ulStopTrigEdge As Long
    ulTrigDI As Long
    ulEClkEdge As Long
    ulFastMode As Long
    ulStatusMode As Long
    ulErrCtrl As Long
End Type

' -----------------------------------------------------------------------
'     Sampling Condition Structure for Memory
' -----------------------------------------------------------------------
Type ADMEMSMPLREQ
    ulChCount As Long
    SmplChReq(0 To 255) As ADSMPLCHREQ
    ulSingleDiff As Long
    fSmplFreq As Single
    ulStopMode As Long
    ulPreTrigDelay As Long
    ulPostTrigDelay As Long
        TrigChReq(0 To 1) As ADTRIGCHREQ
    ulATrgMode As Long
    ulATrgPulse As Long
    ulStopTrigEdge As Long
    ulEClkEdge As Long
    ulFastMode As Long
    ulStatusMode As Long
    ulErrCtrl As Long
End Type

' -----------------------------------------------------------------------
'  Board Specification Structure
' -----------------------------------------------------------------------
Type ADBOARDSPEC
    ulBoardType As Long
    ulBoardID As Long
    dwSamplingMode As Long
    ulChCountS As Long
    ulChCountD As Long
    ulResolution As Long
    dwRange As Long
    ulIsolation As Long
    ulDi As Long
    ulDo As Long
End Type


'-----------------------------------------------------------------------------------------------
'
'   GPC/GPF-3100 Visual Basic  Error Identifier
'
'-----------------------------------------------------------------------------------------------
Public Const AD_ERROR_SUCCESS = 0
Public Const AD_ERROR_NOT_DEVICE = &HC0000001
Public Const AD_ERROR_NOT_OPEN = &HC0000002
Public Const AD_ERROR_INVALID_HANDLE = &HC0000003
Public Const AD_ERROR_ALREADY_OPEN = &HC0000004
Public Const AD_ERROR_NOT_SUPPORTED = &HC0000009
Public Const AD_ERROR_NOW_SAMPLING = &HC0001001
Public Const AD_ERROR_STOP_SAMPLING = &HC0001002
Public Const AD_ERROR_START_SAMPLING = &HC0001003
Public Const AD_ERROR_SAMPLING_TIMEOUT = &HC0001004
Public Const AD_ERROR_INVALID_PARAMETER = &HC0001021
Public Const AD_ERROR_ILLEGAL_PARAMETER = &HC0001022
Public Const AD_ERROR_NULL_POINTER = &HC0001023
Public Const AD_ERROR_GET_DATA = &HC0001024
Public Const AD_ERROR_USED_DA = &HC0001025
Public Const AD_ERROR_FILE_OPEN = &HC0001041
Public Const AD_ERROR_FILE_CLOSE = &HC0001042
Public Const AD_ERROR_FILE_READ = &HC0001043
Public Const AD_ERROR_FILE_WRITE = &HC0001044
Public Const AD_ERROR_INVALID_DATA_FORMAT = &HC0001061
Public Const AD_ERROR_INVALID_AVERAGE_OR_SMOOTHING = &HC0001062
Public Const AD_ERROR_INVALID_SOURCE_DATA = &HC0001063
Public Const AD_ERROR_NOT_ALLOCATE_MEMORY = &HC0001081
Public Const AD_ERROR_NOT_LOAD_DLL = &HC0001082
Public Const AD_ERROR_CALL_DLL = &HC0001083

'-----------------------------------------------------------------------------------------------
'
'   GPC/GPF-3100 Visual Basic  Function Declaration
'
'-----------------------------------------------------------------------------------------------

Declare Function AdOpen Lib "fbiad.dll" (ByVal lpszName As String) As Long
Declare Function AdClose Lib "fbiad.dll" (ByVal hdevicehandle As Long) As Long
Declare Function AdGetDeviceInfo Lib "fbiad.dll" (ByVal hdevicehandle As Long, ByRef pBoardSpec As ADBOARDSPEC) As Long
Declare Function AdSetBoardConfig Lib "fbiad.dll" (ByVal hdevicehandle As Long, ByVal hEvent As Long, ByVal lpCallBackProc As Long, ByVal dwUser As Long) As Long
Declare Function AdGetBoardConfig Lib "fbiad.dll" (ByVal hdevicehandle As Long, ByRef ulAdSmplEventFactor As Long) As Long
Declare Function AdSetSamplingConfig Lib "fbiad.dll" (ByVal hdevicehandle As Long, ByRef pAdSmplConfig As ADSMPLREQ) As Long
Declare Function AdGetSamplingConfig Lib "fbiad.dll" (ByVal hdevicehandle As Long, ByRef pAdSmplConfig As ADSMPLREQ) As Long
Declare Function AdGetSamplingData Lib "fbiad.dll" (ByVal hdevicehandle As Long, ByRef pSmplData As Any, ByRef ulSmplNum As Long) As Long
Declare Function AdClearSamplingData Lib "fbiad.dll" (ByVal hdevicehandle As Long) As Long
Declare Function AdStartSampling Lib "fbiad.dll" (ByVal hdevicehandle As Long, ByVal ulSyncFlag As Long) As Long
Declare Function AdStartFileSampling Lib "fbiad.dll" (ByVal hdevicehandle As Long, ByVal pszPathName As String, ByVal ulFileFlag As Long) As Long
Declare Function AdTriggerSampling Lib "fbiad.dll" (ByVal hdevicehandle As Long, ByVal ulChNo As Long, ByVal ulRange As Long, ByVal ulSingleDiff As Long, ByVal ulTriggerMode As Long, ByVal ulTrigEdge As Long, ByVal ulSmplNum As Long) As Long
Declare Function AdMemTriggerSampling Lib "fbiad.dll" (ByVal hdevicehandle As Long, ByVal ulChCount As Long, ByRef lpSmplChReq As ADSMPLCHREQ, ByVal ulSmplNum As Long, ByVal ulRepeatCount As Long, ByVal ulTrigEdge As Long, ByVal fSmplFreq As Single, ByVal ulEClkEdge As Long, ByVal ulFastMode As Long) As Long
Declare Function AdSyncSampling Lib "fbiad.dll" (ByVal hdevicehandle As Long, ByVal ulMode As Long) As Long
Declare Function AdStopSampling Lib "fbiad.dll" (ByVal hdevicehandle As Long) As Long
Declare Function AdGetStatus Lib "fbiad.dll" (ByVal hdevicehandle As Long, ByRef ulAdSmplStatus As Long, ByRef ulAdSmplCount As Long, ByRef ulAdAvailCount As Long) As Long
Declare Function AdInputAD Lib "fbiad.dll" (ByVal hdevicehandle As Long, ByVal ulCh As Long, ByVal ulSingleDiff As Long, ByRef lpAdSmplChReq As ADSMPLCHREQ, ByRef lpData As Any) As Long
Declare Function AdInputDI Lib "fbiad.dll" (ByVal hdevicehandle As Long, ByRef dwData As Long) As Long
Declare Function AdOutputDO Lib "fbiad.dll" (ByVal hdevicehandle As Long, ByVal dwData As Long) As Long
Declare Function AdAdjustVR Lib "fbiad.dll" (ByVal hdevicehandle As Long, ByVal ulAdjustCh As Long, ByVal ulSingleDiff As Long, ByVal ulSelVolume As Long, ByVal ulControl As Long, ByVal ulTap As Long) As Long
Declare Function AdReadAdjustVR Lib "fbiad.dll" (ByVal hdevicehandle As Long, ByVal ulAdjustCh As Long) As Long
Declare Function AdReadAdjustVREx Lib "fbiad.dll" (ByVal hdevicehandle As Long, ByVal ulAdjustCh As Long, ByVal ulControl As Long) As Long
Declare Function AdDataConv Lib "FbiAdDC.DLL" (ByVal uSrcFormCode As Long, ByRef pSrcData As Any, ByVal uSrcSmplDataNum As Long, ByRef pSrcSmplReq As ADSMPLREQ, ByVal uDestFormCode As Long, ByRef pDestData As Any, ByRef puDestSmplDataNum As Long, ByRef pDestSmplReq As ADSMPLREQ, ByVal uEffect As Long, ByVal uCount As Long, ByVal lpfnConv As Long) As Long
Declare Function AdReadFile Lib "FbiAdDC.DLL" (ByVal pszPathName As String, ByRef pSmplData As Any, ByVal uFormCode As Long) As Long

Declare Function AdBmSetSamplingConfig Lib "fbiad.dll" (ByVal hdevicehandle As Long, ByRef pAdBmSmplConfig As ADBMSMPLREQ) As Long
Declare Function AdBmGetSamplingConfig Lib "fbiad.dll" (ByVal hdevicehandle As Long, ByRef pAdBmSmplConfig As ADBMSMPLREQ) As Long
Declare Function AdBmGetSamplingData Lib "fbiad.dll" (ByVal hdevicehandle As Long, ByRef pBmSmplData As Any, ByRef ulBmSmplNum As Long) As Long
Declare Function AdBmStartFileSampling Lib "fbiad.dll" (ByVal hdevicehandle As Long, ByVal pszPathName As String, ByVal ulFileFlag As Long) As Long

Declare Function AdLvSetSamplingConfig Lib "fbiad.dll" (ByVal hdevicehandle As Long, ByVal ulChNo As Long, ByVal ulSmplNum As Long, ByVal fSmplFreq As Single, ByVal ulRange As Long, ByVal hEvent As Long, ByVal lpCallBackProc As Long, ByVal dwUser As Long) As Long
Declare Function AdLvGetSamplingConfig Lib "fbiad.dll" (ByVal hdevicehandle As Long, ByVal ulChNo As Long, ByRef ulSmplNum As Long, ByRef fSmplFreq As Single, ByRef ulRange As Long) As Long
Declare Function AdLvCalibration Lib "fbiad.dll" (ByVal hdevicehandle As Long, ByVal ulChNo As Long, ByVal ulCalibration As Long) As Long
Declare Function AdLvGetSamplingData Lib "fbiad.dll" (ByVal hdevicehandle As Long, ByVal ulChNo As Long, ByRef pSmplData As Any, ByRef ulSmplNum As Long) As Long
Declare Function AdLvStartSampling Lib "fbiad.dll" (ByVal hdevicehandle As Long, ByVal ulChNo As Long) As Long
Declare Function AdLvStopSampling Lib "fbiad.dll" (ByVal hdevicehandle As Long, ByVal ulChNo As Long) As Long
Declare Function AdLvGetStatus Lib "fbiad.dll" (ByVal hdevicehandle As Long, ByVal ulChNo As Long, ByRef ulAdSmplStatus As Long, ByRef ulAdSmplCount As Long, ByRef ulAdAvailCount As Long) As Long
Declare Function AdMeasureTemperature Lib "fbiad.dll" (ByVal hdevicehandle As Long, ByRef fTemperature As Single) As Long
Declare Function AdMeasureTemperatureEx Lib "fbiad.dll" (ByVal hdevicehandle As Long, ByRef fTemperature As Single, ByVal ulControl As Long) As Long

Declare Function AdAllocateSamplingBuffer Lib "fbiad.dll" (ByVal hdevicehandle As Long) As Long
Declare Function AdReadSamplingBuffer Lib "fbiad.dll" (ByVal hdevicehandle As Long, ByVal lOffset As Long, ByRef ulSmplNum As Long, ByRef pSmplData As Any) As Long

Declare Function AdSetRangeEvent Lib "fbiad.dll" (ByVal hdevicehandle As Long, ByVal dwEventMask As Long, ByVal dwStopMode As Long) As Long
Declare Function AdGetRangeEventStatus Lib "fbiad.dll" (ByVal hdevicehandle As Long, ByRef ulEventChNo As Long, ByRef ulEventStatus As Long) As Long
Declare Function AdResetRangeEvent Lib "fbiad.dll" (ByVal hdevicehandle As Long) As Long
Declare Function AdGetOverRangeChStatus Lib "fbiad.dll" (ByVal hdevicehandle As Long, ByRef ulChStatusNo As Long) As Long
Declare Function AdResetOverRangeCh Lib "fbiad.dll" (ByVal hdevicehandle As Long) As Long

Declare Function AdSetInterval Lib "fbiad.dll" (ByVal hdevicehandle As Long, ByVal ulInterval As Long) As Long
Declare Function AdGetInterval Lib "fbiad.dll" (ByVal hdevicehandle As Long, ByRef ulInterval As Long) As Long
Declare Function AdSetFunction Lib "fbiad.dll" (ByVal hdevicehandle As Long, ByVal ulChNo As Long, ByVal ulFunction As Long) As Long
Declare Function AdGetFunction Lib "fbiad.dll" (ByVal hdevicehandle As Long, ByVal ulChNo As Long, ByRef ulFunction As Long) As Long

Declare Function AdSetFilter Lib "fbiad.dll" (ByVal hdevicehandle As Long, ByVal ulFilter As Long) As Long
Declare Function AdGetFilter Lib "fbiad.dll" (ByVal hdevicehandle As Long, ByRef ulFilter As Long) As Long

Declare Function AdMemSetSamplingConfig Lib "fbiad.dll" (ByVal hdevicehandle As Long, ByRef pAdMemSmplConfig As ADMEMSMPLREQ) As Long
Declare Function AdMemGetSamplingConfig Lib "fbiad.dll" (ByVal hdevicehandle As Long, ByRef pAdMemSmplConfig As ADMEMSMPLREQ) As Long
Declare Function AdSetOutMode Lib "fbiad.dll" (ByVal hdevicehandle As Long, ByVal ulExTrgMode As Long, ByVal ulExClkMode As Long) As Long
Declare Function AdGetOutMode Lib "fbiad.dll" (ByVal hdevicehandle As Long, ByRef ulExTrgMode As Long, ByRef ulExClkMode As Long) As Long
Declare Function AdMemSetDiPattern Lib "fbiad.dll" (ByVal hdevicehandle As Long, ByVal ulCh As Long, ByRef ulPatternTrig As Long) As Long
Declare Function AdMemGetDiPattern Lib "fbiad.dll" (ByVal hdevicehandle As Long, ByRef ulPatternTrig As Long) As Long

Declare Function AdCommonGetPciDeviceInfo Lib "fbiad.dll" (ByVal hdevicehandle As Long, ByRef dwDeviceID As Long, ByRef dwVendorID As Long, ByRef dwClassCode As Long, ByRef dwRevisionID As Long, ByRef dwBaseAddress0 As Long, ByRef dwBaseAddress1 As Long, ByRef dwBaseAddress2 As Long, ByRef dwBaseAddress3 As Long, ByRef dwBaseAddress4 As Long, ByRef dwBaseAddress5 As Long, ByRef dwSubsystemID As Long, ByRef dwSubsystemVendorID As Long, ByRef dwInterruptLine As Long, ByRef dwBoardID As Long) As Long
Declare Function AdFifoGetSamplingData Lib "fbiad.dll" (ByVal hdevicehandle As Long, ByRef pSmplData As Any, ByRef pDiData As Any, ByRef ulSmplNum As Long) As Long

Declare Function AdBmStartSampling Lib "fbiad.dll" (ByVal hdevicehandle As Long, ByRef pBmSmplData As Any, ByVal ulSize As Long) As Long

Declare Function AdCalibration Lib "fbiad.dll" (ByVal hdevicehandle As Long, ByVal ulChannel As Long, ByVal ulRange As Long, ByVal ulSingleDiff As Long) As Long
Declare Function AdOutputSync Lib "fbiad.dll" (ByVal hdevicehandle As Long, ByVal TrgLine As Long, ByVal TrgMode As Long) As Long

Declare Function AdSetEndConfig Lib "fbiad.dll" (ByVal hdevicehandle As Long, ByVal Mode As Long) As Long

Declare Function CreateEvent Lib "kernel32" Alias "CreateEventA" (ByVal lpEventAttributes As Long, ByVal ManualReset As Long, ByVal bInitialState As Long, ByVal lpName As String) As Long
Declare Function WaitForSingleObject Lib "kernel32" (ByVal hHandle As Long, ByVal dwMilliseconds As Long) As Long
Declare Function CloseHandle Lib "kernel32" (ByVal hObject As Long) As Long
Declare Function ResetEvent Lib "kernel32" (ByVal hEvent As Long) As Long

