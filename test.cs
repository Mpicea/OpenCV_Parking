using OpenCvSharp;
using OpenCvSharp.ML;
using System;
using System.CodeDom;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Net;
using System.Numerics;
using System.Reflection.Emit;
using System.Text;
using System.Threading.Tasks;

namespace OpenCV_test01
{

    public class Parking
    {
        public void refine_candidate(Mat image, ref RotatedRect candi)
        {
            Size newSize = new Size(image.Cols + 2, image.Rows + 2);
            Mat fill = new Mat(newSize, MatType.CV_8UC1, Scalar.All(0));    // 채움 영역

            // 채움 색상
            Scalar dif1 = new Scalar(25, 25, 25);
            Scalar dif2 = new Scalar(25, 25, 25);
                                                                            // 범위 
            int flags = 4 + 0xff00;                                     // 채움 방향
            flags += (int)FloodFillFlags.FixedRange;
            flags += (int)FloodFillFlags.MaskOnly;

            // 후보영역 유사 컬러 채움
            // 랜덤 좌표 15개
            List<Point2f> rand_pt = new List<Point2f>(15);

            Random random = new Random();
            for (int i = 0; i < 15; i++)
            {
                float x = (float)(random.NextDouble() * 14 - 7);  // Mean 0, Standard Deviation 7
                float y = (float)(random.NextDouble() * 14 - 7);  // Mean 0, Standard Deviation 7
                rand_pt.Add(new Point2f(x, y));
            }

            // 입력영상 범위 사각형
            Rect img_rect = new Rect(new Point(0, 0), image.Size());
            for(int i = 0; i<rand_pt.Count; i++)
            {
                Point2f seed = new Point2f(candi.Center.X + rand_pt[i].X, candi.Center.Y + rand_pt[i].Y); // 랜덤좌표 평행이동
                if (img_rect.Contains((int)seed.X, (int)seed.Y)==true);
            }

            // 채움 영역 사각형 계산
            List<Point> fill_pts = new List<Point>();
            for(int i=0; i<fill.Rows; i++)
            {
                for(int j=0; j<fill.Cols; j++)
                {
                    if(fill.At<Byte>(i,j) == 255) // 채움 영역이면 
                    {
                        fill_pts.Add(new Point(j, i)); // 좌표 저장
                    }
                }
            }
            // 채움 좌표들로 최소영역 계산
            candi = Cv2.MinAreaRect(fill_pts);
        }

        public void rotate_plate(Mat image,ref Mat corp_img, RotatedRect candi)
        {
            // 종횡비 
            // 회전각도	
            float aspect = (float)candi.Size.Width / candi.Size.Height;
            float angle = candi.Angle;

            if (aspect < 1)
            {                                           // 1보다 작으면 세로로 긴 영역
                // 가로 세로 맞바꿈
                // 회전각도 조정
                var temp = candi.Size.Width;
                candi.Size.Width = candi.Size.Height;
                candi.Size.Height = temp;
                angle += 90;
            }

            // 회전 행렬 계산
            Mat rotmat = Cv2.GetRotationMatrix2D(candi.Center, angle, 1);
            // 회전변환 수행
            Cv2.WarpAffine(image, corp_img, rotmat, image.Size(), InterpolationFlags.Cubic);
            //getRectSubPix(corp_img, candi.size, candi.center, corp_img);
            Size testSize = new Size(candi.Size.Width, candi.Size.Height);
            Cv2.GetRectSubPix(corp_img, testSize, candi.Center, corp_img);
        }

        public List<Mat> make_candidates(Mat image, ref List<RotatedRect> candidates)
        {
            //vector<Mat> candidates_img;
            List<Mat> candidates_img = new List<Mat>();
            for (int i = 0; i < candidates.Count;)
            {   RotatedRect testrect = candidates[i];
                refine_candidate(image, ref testrect);     // 후보 영역 개선
                if (vertify_plate(candidates[i]))               // 개선 영역 재검증
                {
                    Mat corp_img = new Mat();
                    rotate_plate(image,ref corp_img, candidates[i]);   // 회전 및 후보영상 가져오기

                    Cv2.CvtColor(corp_img, corp_img, ColorConversionCodes.BGR2GRAY);              // 명암도 변환
                    Cv2.Resize(corp_img, corp_img, new Size(144, 28), 0, 0,InterpolationFlags.Cubic); // 크기 정규화
                    candidates_img.Add(corp_img);                     // 보정 영상 저장
                    i++;
                }
                else                                            // 재검증 탈락 
                    candidates.RemoveAt(i);   // 벡터 원소에서 제거

            }
            return candidates_img;
        }

        public void preprocessing_plate(Mat plate_img, ref Mat ret_img)
        {
            Cv2.Resize(plate_img, plate_img,new Size(180, 35));
            Cv2.Threshold(plate_img, plate_img, 32, 255,ThresholdTypes.Binary | ThresholdTypes.Otsu);

            Cv2.ImShow("plate_img", plate_img);
            Cv2.ImWrite("plate_img.png", plate_img);

            Point pt1 = new Point(6, 3);
            //Point pt2 = plate_img.Size() - pt1;
            Point pt2 = new Point(plate_img.Size().Width - pt1.X, plate_img.Size().Height - pt1.Y);
            //ret_img = plate_img(Rect(pt1, pt2)).clone();
            Size retsize = new Size(pt2.X - pt1.X, pt2.Y-pt1.Y);
            ret_img = new Mat(plate_img, new Rect(pt1, retsize)).Clone();

            Cv2.ImWrite("ret_img.png", ret_img);

        }

        public void find_objects(Mat sub_mat, ref List<Rect> object_rects)
        {
            //	Mat tmp = ~sub_mat;								
            //	cvtColor(tmp, tmp, CV_GRAY2BGR);
            HierarchyIndex[] hierarchyIndices;
            //List<List<Point>> contours = new List<List<Point>>();
            Point[][] contours;
            Cv2.FindContours(sub_mat,out contours,out hierarchyIndices,RetrievalModes.External,ContourApproximationModes.ApproxSimple);

            List<Rect> text_rects = new List<Rect>();
            for (int i = 0; i < contours.Length; i++)
            {
                Rect r = Cv2.BoundingRect(contours[i]);                     // 검출 객체 사각형

                if (r.Width / (float)r.Height > 2.5) continue;

                if (r.X > 45 && r.X < 80 && (r.Width*r.Height) > 60)
                    text_rects.Add(r);                        // 문자 객체 저장
                else if ((r.Width * r.Height) > 150)                                // 잡음 객체 크기
                {
                    object_rects.Add(r); // 숫자 객체 저장
                }
            }
            if (text_rects.Count > 0)
            {
                //		rectangle(tmp, text_rects[0], Scalar(0, 255, 0), 1);
                for (int i = 1; i < text_rects.Count; i++)
                {           // 문자 객체 범위 누적
                    text_rects[0] |= text_rects[i];
                    //			rectangle(tmp, text_rects[i], Scalar(0, 255, 0), 1);
                }
                object_rects.Add(text_rects[0]);                      // 문자 객체 저장
            }
            //	imwrite("tmp.png", tmp);
        }

        public void sort_rects(List<Rect> object_rects,ref List<Rect> sorted_rects)
        {
            Mat pos_x = new Mat();
            for (int i = 0; i < object_rects.Count; i++)
            {
                pos_x.Add(object_rects[i].X);
            }

            Cv2.SortIdx(pos_x, pos_x, SortFlags.EveryColumn | SortFlags.Ascending);
            //sortIdx(pos_x, pos_x, SORT_EVERY_COLUMN + SORT_ASCENDING);
            for (int i = 0; i < pos_x.Rows; i++)
            {
                int idx = pos_x.At<int>(i, 0);
                sorted_rects.Add(object_rects[idx]);
            }
        }

        public void classify_numbers(List<Mat> numbers, KNearest[] knn, int K1, int K2)
        {
            string[] text_value = {								// 인식할 문자 – 레이블값 대응 문자 
		"가", "나", "다", "라", "마", "거", "너", "더", "러", "머",
        "고", "노", "도", "로", "모", "구", "누", "두", "루", "무",
        "바", "사", "아", "자", "허",
    };

            Console.Write("분류 결과 : ");
            for (int i = 0; i < numbers.Count; i++)
            {
                Mat num = find_number(numbers[i]);              // 숫자객체 검색
                Mat data = place_middle(num, new Size(40, 40));     // 중앙 배치

                Mat results = new Mat();
                if (i == 2)
                {
                    knn[1].FindNearest(data, K1, results);             // 숫자 k-NN 분류 수행
                    Console.Write(text_value[(int)results.At<float>(0)] + " ");// 결과 출력
                }
                else
                {
                    knn[0].FindNearest(data, K2, results);             // 문자 k-NN 분류 수행
                    Console.Write(results.At<float>(0) + " ");                // 결과 출력
                }
                //		imshow("number_" + to_string(i - 1), num);
            }
        }

        public void find_histoPos(Mat img, ref int start, ref int end, int direct)
        {
            Cv2.Reduce(img, img, (ReduceDimension)direct, ReduceTypes.Avg, 0);
            int minFound = 0;
            for (int i = 0; i < img.Total(); i++)
            {
                if (img.At<Byte>(i) < 250)         // 빈라인이 아니면
                {
                    end = i;                        // 히스토그램 마지막 위치
                    if (minFound == 0)
                    {
                        start = i;                  // 히스토그램 첫 위치
                        minFound = 1;
                    }
                }
            }
        }

        /*
        public void find_histoPos(Mat img, ref Point start, ref Point end, int direct)
        {
            Mat reducedImg = new Mat();
            Cv2.Reduce(img, reducedImg, (ReduceDimension)direct, ReduceTypes.Avg, 0);

            int minFound = 0;
            for (int i = 0; i < reducedImg.Cols; i++)
            {
                if (reducedImg.At<byte>(0, i) < 250) // 빈 라인이 아니면
                {
                    end.X = i; // 히스토그램 마지막 위치
                    if (minFound == 0)
                    {
                        start.X = i; // 히스토그램 첫 위치
                        minFound = 1;
                    }
                }
            }
        }
        */
        public Mat find_number(Mat part)
        {
            Point start = new Point();
            Point end = new Point();
            //Size end = new Size();
            find_histoPos(part, ref start.X, ref end.X, 0);     // 수직 투영 
            find_histoPos(part, ref start.Y, ref end.Y, 1);     // 수평 투영 


            return part[new Rect(start, new Size(end.X-start.X, end.Y - start.Y))];              // 숫자객체 영상
            //return part.SubMat(new Rect(start, end));
        }
        /*
        public Mat find_number(Mat part)
        {
            Point start = new Point();
            Size end = new Size();
            find_histoPos(part, ref start.X, ref end.Width, 0);     // 수직 투영 
            find_histoPos(part, ref start.Y, ref end.Height, 1);     // 수평 투영 


            return part[new Rect(start, end)];              // 숫자객체 영상
            //return part.SubMat(new Rect(start, end));
        }
        */

        /*
        public Mat find_number(Mat part)
        {
            Point start = new Point();
            Point end = new Point();
            find_histoPos(part, ref start, ref end, 0); // 수직 투영 
            find_histoPos(part, ref start, ref end, 1); // 수평 투영 

            Rect roi = new Rect(start, new Size(end.X - start.X, end.Y - start.Y));
            if (roi.Width > 0 && roi.Height > 0)
            {
                Mat numberMat = new Mat(part, roi); // 숫자 객체 영상
                return numberMat.Clone();
            }
            else
            {
                return null; // 유효하지 않은 영역인 경우 null 또는 빈 Mat을 반환할 수 있습니다.
            }
        }
        */

        public Mat place_middle(Mat number, Size new_size)
        {
            int big = Math.Max(number.Cols, number.Rows);
            Mat square= new Mat(big, big, number.Type(), Scalar.All(255));   // 정방영상

            Point start = new Point((square.Width - number.Width) / 2, (square.Height - number.Height)/2);

            Rect middle_rect = new Rect(start, number.Size());         // 중심 사각형
            Mat middle = new Mat(square, middle_rect);
            number.CopyTo(middle);

            Cv2.Resize(square, square, new_size);               // 크기 변경
            square.ConvertTo(square, MatType.CV_32F);

            return square.Reshape(0, 1);
        }


        public KNearest kNN_train(string train_img, int K, int Nclass, int Nsample)
        {
            Size size = new Size(40, 40);                                  // 셀 크기
            Mat trainData = new Mat();
            Mat classLable = new Mat();
            Mat train_image = Cv2.ImRead(train_img, ImreadModes.Grayscale);         // 전체 학습영상 로드
            if (train_image.Empty())
            {
                Console.WriteLine("이미지를 불러오지 못하였습니다.");
                Environment.Exit(0);
            }
            
            Cv2.Threshold(train_image, train_image, 32, 255, ThresholdTypes.Binary);
            for (int i = 0, k = 0; i < Nclass; i++)
            {
                for (int j = 0; j < Nsample; j++, k++)
                {
                    Point pt = new Point(j* size.Width, i* size.Height);        // 셀 시작좌표
                    Rect roi = new Rect(pt, size);
                    Mat part = new Mat(train_image,roi);                    // 숫자 영상 분리

                    Mat num = find_number(part);            // 숫자객체 검출
                    Mat data = place_middle(num, size);         // 셀 중심에 숫자 배치 
                    trainData.Add(data);                      // 학습 데이터 수집
                    classLable.Add(i);                        // 레이블링
                }
            }

            KNearest knn = KNearest.Create();         // k-NN 객체 생성
            knn.Train(trainData, SampleTypes.RowSample, classLable);      // k-NN 학습
            return knn;
        }

        public Mat preprocessing(Mat image)
        {
            Mat gray = new Mat();
            Mat th_img = new Mat();
            Mat morph = new Mat();
            Mat kernel= new Mat(5, 25, MatType.CV_8UC1, Scalar.All(1));      // 닫힘 연산 마스크
            Cv2.CvtColor(image, gray, ColorConversionCodes.BGR2GRAY);     // 명암도 영상 변환

            Cv2.Blur(gray, gray, new Size(5, 5));               // 블러링
            Cv2.Sobel(gray, gray, MatType.CV_8U, 1, 0, 3);          // 소벨 에지 검출

            Cv2.Threshold(gray, th_img, 120, 255,ThresholdTypes.Binary);   // 이진화 수행
            Cv2.MorphologyEx(th_img, morph, MorphTypes.Close, kernel);   // 열림 연산 수행
                                                                //	imshow("th_img", th_img), imshow("morph", morph);
            return morph;
        }

        public bool vertify_plate(RotatedRect mr)
        {
            float size = mr.Size.Height * mr.Size.Width;
            float aspect = (float)mr.Size.Height / mr.Size.Width;   // 종횡비 계산
            if (aspect < 1) aspect = 1 / aspect;

            bool ch1 = size > 2000 && size < 30000;     // 번호판 넓이 조건
            bool ch2 = aspect > 1.3 && aspect < 6.4;        // 번호판 종횡비 조건

            return ch1 && ch2;
        }

        public void find_candidates(Mat img, ref List<RotatedRect> candidates)
        {
            Point[][] contours;             // 외곽선
            HierarchyIndex[] hierarchy;
                                                        // 외곽선 검출
            Cv2.FindContours(img.Clone(), out contours, out hierarchy, RetrievalModes.External, ContourApproximationModes.ApproxSimple);

            for (int i = 0; i < contours.Length; i++)  // 검출 외곽선 조회
            {
                RotatedRect rot_rect = Cv2.MinAreaRect(contours[i]);    // 외곽선 최소영역 회전사각형
                if (vertify_plate(rot_rect))                        // 번호판 검증
                    candidates.Add(rot_rect);             // 회전사각형 저장
            }
        }

        public void draw_rotatedRect(ref Mat img, RotatedRect mr, Scalar color, int thickness = 2)
        {
            Point[] pts = new Point[4];

            float angle = mr.Angle * (float)Math.PI / 180.0f;
            Size rectSize = new Size((int)mr.Size.Width, (int)mr.Size.Height);

            Point center = new Point((int)mr.Center.X, (int)mr.Center.Y);

            float cosA = (float)Math.Cos(angle);
            float sinA = (float)Math.Sin(angle);

            pts[0] = center + new Point(-rectSize.Width / 2, -rectSize.Height / 2);
            pts[1] = center + new Point(rectSize.Width / 2, -rectSize.Height / 2);
            pts[2] = center + new Point(rectSize.Width / 2, rectSize.Height / 2);
            pts[3] = center + new Point(-rectSize.Width / 2, rectSize.Height / 2);

            for (int i = 0; i < 4; ++i)
            {
                Cv2.Line(img, pts[i], pts[(i + 1) % 4], color, thickness);
            }
        }

        public void read_trainData(string fn, ref Mat trainingData, ref Mat lables)
        {
            using(FileStorage fs = new FileStorage(fn, FileStorage.Modes.Read))
            {
                if (!fs.IsOpened())
                {
                    throw new Exception("Failed to open file");
                }

                trainingData =  fs["TrainingData"].ReadMat();
                lables = fs["classes"].ReadMat();
            }
            //fs.release();

            trainingData.ConvertTo(trainingData, MatType.CV_32FC1);
        }

        public SVM SVM_create(int type, int max_iter, double epsilon)
        {
            SVM svm = SVM.Create();       // SVM 객체 선언
                                          // SVM 파라미터 지정
            svm.Type = SVM.Types.CSvc;          // C-Support Vector Classification				
            svm.KernelType = SVM.KernelTypes.Linear;            // 선형 SVM 
            svm.Gamma = 1;                           // 커널함수의 감마값
            svm.C = 1;                               // 최적화를 위한 C 파리미터
            TermCriteria criteria = new TermCriteria((CriteriaTypes)type, max_iter, epsilon);
            svm.TermCriteria = criteria;             // 반복 알고리즘의 조건
            return svm;

            
        }

        public int classify_plates(SVM svm, List<Mat> candi_img)
        {
            for (int i = 0; i < candi_img.Count; i++)
            {
                Mat onerow = candi_img[i].Reshape(1, 1);  // 1행 데이터 변환
                onerow.ConvertTo(onerow, MatType.CV_32F);

                Mat results = new Mat();                        // 분류 결과 저장 행렬
                svm.Predict(onerow, results);      // SVM 분류 수행

                if (results.At<float>(0) == 1)      // 분류결과가 번호판이면
                    return i;                       // 영상번호 반환
            }
            return -1;
        }
    }
    internal class Program
    {
        static void Main(string[] args)
        {
            Parking parking = new Parking();

            int K1 = 15, K2 = 15;
            KNearest[] knn = new KNearest[2];
            string path = "C:\\Users\\Admin\\source\\openCV_Parking\\image\\";
            knn[0] = parking.kNN_train(path+"trainimage\\train_numbers2001.png", K1, 10, 20);
            knn[1] = parking.kNN_train(path+"trainimage\\train_texts.png", K2, 25, 20);

            // 	학습된 데이터 로드
            SVM svm = SVM.Load(path+"\\SVMtrain.xml");

            int car_no;
            Console.Write("차량 영상 번호 (0-20) : ");
            car_no = int.Parse(Console.ReadLine());
            string fn = string.Format(path+"test_car\\{0:D2}.jpg", car_no);
            Mat image = Cv2.ImRead(fn, ImreadModes.Color);
            if (image.Empty())
            {
                Console.WriteLine("영상을 불러올 수 없습니다.");
                Environment.Exit(1);
            }

            Mat morph = parking.preprocessing(image);                               // 전처리
            List<RotatedRect> candidates = new List<RotatedRect>();
            parking.find_candidates(morph, ref candidates);                                 // 후보 영역 검출
            List<Mat> candidate_img = parking.make_candidates(image, ref candidates);// 후보 영상 생성

            int plate_no = parking.classify_plates(svm, candidate_img);         // SVM 분류

            if (plate_no >= 0)
            {
                List<Rect> obejct_rects = new List<Rect>();
                List<Rect> sorted_rects = new List<Rect>();
                List<Mat> numbers = new List<Mat>();                            // 숫자 객체 
                Mat plate_img = new Mat();
                Mat color_plate = new Mat();                             // 컬러 번호판 영상 

                parking.preprocessing_plate(candidate_img[plate_no], ref plate_img);    // 번호판 영상 전처리
                Cv2.CvtColor(plate_img, color_plate, ColorConversionCodes.GRAY2BGR);

                parking.find_objects(~plate_img, ref obejct_rects);     // 숫자 및 문자 검출  

                parking.sort_rects(obejct_rects, ref sorted_rects);         // 검출객체 정렬(x 좌표기준)

                for (int i = 0; i < sorted_rects.Count; i++)
                {
                    numbers.Add(plate_img[sorted_rects[i]]);  // 정렬된 숫자 영상
                    Cv2.Rectangle(color_plate, sorted_rects[i], new Scalar(0, 0, 255), 1); // 사각형 그리기
                }

                if (numbers.Count == 7)
                {
                    parking.classify_numbers(numbers, knn, K1, K2);     // kNN 분류 수행
                }
                else Console.WriteLine("숫자(문자) 객체를 정확히 검출하지 못했습니다.");

                Cv2.ImShow("번호판 영상", color_plate);                  // 번호판 영상 표시
                parking.draw_rotatedRect(ref image, candidates[plate_no], new Scalar(0, 0, 255), 2);
            }
            else Console.WriteLine("번호판을 검출하지 못하였습니다. ");

            Cv2.ImShow("image", image);
            Cv2.WaitKey();
        }
    }
}
