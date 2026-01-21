#include "ADControl.h"

ADControl::ADControl(){
    initialized = (Initialization() == 0);
}
ADControl::~ADControl(){
    End();
	
}
// -------------------------------------------------------
// 初期化
// -------------------------------------------------------
int ADControl::Initialization()
{
	// デバイスのオープン
	
		hDeviceHandle = AdOpen((LPCSTR)"FBIAD1");
	
		if (hDeviceHandle == INVALID_HANDLE_VALUE) {
			printf("Failed to open the specified device.");
			return -1;
		}
	// ボード情報の取得（分解能など）
	if(AdGetDeviceInfo(hDeviceHandle, &BoardSpec)!= AD_ERROR_SUCCESS)
		return -1;
	
	return 0;
}
void ADControl::End()
{
	// デバイスのクローズ
	AdClose(hDeviceHandle);

}
// -------------------------------------------------------
// 取得
// -------------------------------------------------------
int ADControl::GetData(float& voltage)
{
	if(!initialized)
		return -1;
	int				nRet;
	ADSMPLCHREQ		AdSmplChReq[1];
	double vref = 10.0;
	unsigned long N = BoardSpec.ulResolution;
	
	// チャンネル数:2
	// 指定チャンネル:CH1,CH2
	// レンジ:±5V
	AdSmplChReq[0].ulChNo = 1;
	AdSmplChReq[0].ulRange = AD_10V;
	
	if (BoardSpec.ulResolution <= 8) {
		// 1件入力データを取得(BYTEデータ)
		nRet = AdInputAD(hDeviceHandle, 1, AD_INPUT_SINGLE, AdSmplChReq, bSmplData);
		if (nRet != AD_ERROR_SUCCESS) {
			printf("AdInputAD errr(%lx)", nRet);
			return -1;
		}

		// 取得したデータを表示
		printf("CH1=%02Xh  CH2=%02Xh\n", bSmplData[0], bSmplData[1]);
	}
	else if (BoardSpec.ulResolution > 8 && BoardSpec.ulResolution <= 16) {
		// 1件入力データを取得(WORDデータ)
		nRet = AdInputAD(hDeviceHandle, 1, AD_INPUT_SINGLE, AdSmplChReq, wSmplData);
		if (nRet != AD_ERROR_SUCCESS) {
			printf("AdInputAD errr(%lx)", nRet);
			return -1;
		}
		unsigned short raw = wSmplData[0];
		voltage = ((float)raw / ((1u << N) - 1)) * (2.0 * vref) - vref;
		// printf("channel 1Raw=%04u  Voltage=%.3f V\n", raw, voltage);
		// 取得したデータを表示
		// printf("CH1=%04Xh  CH2=%04Xh\n", wSmplData[0], wSmplData[1]);
	}
	else {
		// 1件入力データを取得(DWORDデータ)
		nRet = AdInputAD(hDeviceHandle, 1, AD_INPUT_SINGLE, AdSmplChReq, dwSmplData);
		if (nRet != AD_ERROR_SUCCESS) {
			printf("AdInputAD errr(%lx)", nRet);
			return -1;
		}

		// 取得したデータを表示
		printf("CH1=%08lXh  CH2=%08lXh\n", dwSmplData[0], dwSmplData[1]);
	}

	return 0;
}

// -------------------------------------------------------
// 終了
// -------------------------------------------------------


// -------------------------------------------------------
// メイン
// -------------------------------------------------------
// int main(void)
// {
// 	int		nRet;
// 	HANDLE	hDeviceHandle;		// デバイスハンドル

// 	// 初期化
// 	nRet = Initialization(&hDeviceHandle);
// 	if (nRet != 0) return -1;

// 	// 取得
// 	nRet = GetData(hDeviceHandle);
// 	if (nRet != 0) return -1;

// 	// 終了
// 	nRet = End(hDeviceHandle);
// 	if (nRet != 0) return -1;

// 	return 0;
// }


