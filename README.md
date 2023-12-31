# OpenCV_Parking
C#과 OpenCV를 활용한 주차장으로 Visual Studio 2022를 기반으로 만들었습니다.   

<br/>

### 프로젝트
|프로젝트|Visual Studio 플랫폼|설명|작성한 분|
|:---:|:---:|:---:|:---:|
|NotOCR.cs|콘솔 앱(.NET Framework)|OCR을 쓰지않고 자동차 번호판 추출|장지현님|
|NotOCRTesseract.cs|콘솔 앱(.NET Framework)|NotOCR + 글자 인식(Tesseract)|황형선님|
|Object detect.cs|콘솔 앱(.NET Framework)|객체(차종) 인식|장효원님|
|OcrTesseract.cs|콘솔 앱(.NET Framework)|UseOCR + 글자 인식(Tesseract)|황형선님|
|OpenCvForm|Windows Forms 앱(.NET Framework)|객체(차종) 인식|장효원님|
|TesseractWinform|Windows Forms 앱(.NET Framework)|NotOCR + 글자 인식(Tesseract)|황형선님|
|The_Last_One.cs|Windows Forms 앱(.NET Framework)|통합 - 객체(차종) 인식, NotOCR + 글자 인식(Tesseract), 차선 인식|황형선님|
|UseOCR.cs|콘솔 앱(.NET Framework)|OCR을 사용하고 자동차 번호판 추출|장지현님|
|lane_violation|콘솔 앱(.NET Framework)|차선 인식과 그 외 프로그램|문도원님|

<br/>

### 자료
|자료명|설명|
|:---:|:---:|
|darknet_model|객체(차종) 인식 데이터|
|image|자동차 번호판 인식 데이터|
|tesseract-5.2.0|tesseract 데이터|
|lane_parking|차선 인식 데이터|
|OpenCV_form.png|객체(차종) 인식 윈폼형태|  

<br/>

### Nuget 패키지 관리
* OpenCvSharp4 4.8.0
* OpenCvSharp4.Extensions 4.8.0
* OpenCvSharp4.runtime.win 4.8.0
* OpenCvSharp4.Windows 4.8.0
  
<br/><br/><br/>

<details>
<summary>기타 사항</summary>  
* 존재함 주석이 있는데 이는 NotOCR과 UseOCR 공통점을 찾는다고 개인적으로 표시한 것입니다  
<br/>
* UseOCR은 C++을 C#으로 변환하여 작업한 코드이기에 데이터 누수 등이 발생할 수 있습니다.  
</details>

