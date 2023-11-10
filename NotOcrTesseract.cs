using OpenCvSharp;
using OpenCvSharp.Extensions;
using OpenCvSharp.ML;
using System;
using System.Collections.Generic;
using Tesseract;

namespace ParkingTest02
{
    class NotOcr
    {
        public void read_trainData(string fn, ref Mat trainingData, ref Mat lables)
        {
            if (lables.Empty())
            {
                lables = new Mat();
            }

            using (FileStorage fs = new FileStorage(fn, FileStorage.Modes.Read))
            {
                if (!fs.IsOpened())
                {
                    throw new Exception("Failed to open file");
                }

                trainingData = fs["TrainingData"].ReadMat();
                lables = fs["classes"].ReadMat();
            }
            //fs.release();

            trainingData.ConvertTo(trainingData, MatType.CV_32FC1);
        }

        public Mat preprocessing(Mat image, ref Mat morph)
        {
            Mat gray = new Mat();
            Mat sobel = new Mat();
            Mat th_img = new Mat();
            Mat kernel = new Mat(5, 25, MatType.CV_8UC1, Scalar.All(1));      // 닫힘 연산 마스크
            Cv2.CvtColor(image, gray, ColorConversionCodes.BGR2GRAY);     // 명암도 영상 변환

            Cv2.Blur(gray, gray, new OpenCvSharp.Size(5, 5));               // 블러링
            Cv2.Sobel(gray, gray, MatType.CV_8U, 1, 0, 3);          // 소벨 에지 검출

            Cv2.Threshold(gray, th_img, 120, 255, ThresholdTypes.Binary);   // 이진화 수행
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
            OpenCvSharp.Point[][] contours;             // 외곽선
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
            OpenCvSharp.Point[] pts = new OpenCvSharp.Point[4];

            float angle = mr.Angle * (float)Math.PI / 180.0f;
            OpenCvSharp.Size rectSize = new OpenCvSharp.Size((int)mr.Size.Width, (int)mr.Size.Height);

            OpenCvSharp.Point center = new OpenCvSharp.Point((int)mr.Center.X, (int)mr.Center.Y);

            float cosA = (float)Math.Cos(angle);
            float sinA = (float)Math.Sin(angle);

            pts[0] = center + new OpenCvSharp.Point(-rectSize.Width / 2, -rectSize.Height / 2);
            pts[1] = center + new OpenCvSharp.Point(rectSize.Width / 2, -rectSize.Height / 2);
            pts[2] = center + new OpenCvSharp.Point(rectSize.Width / 2, rectSize.Height / 2);
            pts[3] = center + new OpenCvSharp.Point(-rectSize.Width / 2, rectSize.Height / 2);

            for (int i = 0; i < 4; ++i)
            {
                Cv2.Line(img, pts[i], pts[(i + 1) % 4], color, thickness);
            }
        }

        public void refine_candidate(Mat image, ref RotatedRect candidate)
        {
            OpenCvSharp.Size newSize = new OpenCvSharp.Size(image.Cols + 2, image.Rows + 2);
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
            List<Point2f> rand_pt = new List<Point2f>(10);

            Random random = new Random();
            for (int i = 0; i < 10; i++)
            {
                float x = (float)(random.NextDouble() * 14 - 7);  // Mean 0, Standard Deviation 7
                float y = (float)(random.NextDouble() * 14 - 7);  // Mean 0, Standard Deviation 7
                rand_pt.Add(new Point2f(x, y));
            }

            // 입력영상 범위 사각형
            OpenCvSharp.Rect img_rect = new OpenCvSharp.Rect(new OpenCvSharp.Point(0, 0), image.Size());
            for (int i = 0; i < rand_pt.Count; i++)
            {
                OpenCvSharp.Point seed = new OpenCvSharp.Point(candidate.Center.X + rand_pt[i].X, candidate.Center.Y + rand_pt[i].Y); // 랜덤좌표 평행이동
                if (img_rect.Contains((int)seed.X, (int)seed.Y) == true)
                    Cv2.FloodFill(image, fill, seed, Scalar.All(0), out OpenCvSharp.Rect rect, dif1, dif2, (FloodFillFlags)flags);
            }

            // 채움 영역 사각형 계산
            List<OpenCvSharp.Point> fill_pts = new List<OpenCvSharp.Point>();
            for (int i = 0; i < fill.Rows; i++)
            {
                for (int j = 0; j < fill.Cols; j++)
                {
                    if (fill.At<Byte>(i, j) == 255) // 채움 영역이면 
                    {
                        fill_pts.Add(new OpenCvSharp.Point(j, i)); // 좌표 저장
                    }
                }
            }
            // 채움 좌표들로 최소영역 계산
            candidate = Cv2.MinAreaRect(fill_pts);
        }


        public Mat correct_plate(Mat input, RotatedRect ro_rect)
        {
            OpenCvSharp.Size m_size = new OpenCvSharp.Size(ro_rect.Size.Width, ro_rect.Size.Height);
            float aspect = (float)m_size.Width / m_size.Height;
            float angle = ro_rect.Angle;

            if (aspect < 1)
            {
                angle += 90;
                var temp = m_size.Width;
                m_size.Width = m_size.Height;
                m_size.Height = temp;
            }

            Mat rot_img = new Mat();
            Mat correct_img = new Mat();
            Mat rotmat = Cv2.GetRotationMatrix2D(ro_rect.Center, angle, 1);
            Cv2.WarpAffine(input, rot_img, rotmat, input.Size(), InterpolationFlags.Cubic);

            Cv2.GetRectSubPix(rot_img, m_size, ro_rect.Center, correct_img);
            Cv2.Resize(correct_img, correct_img, new OpenCvSharp.Size(144, 28), 0, 0, InterpolationFlags.Cubic);

            return correct_img;
        }


        public SVM SVM_train(string fn)
        {
            Mat trainingData = new Mat();
            Mat labels = new Mat();
            read_trainData(fn, ref trainingData, ref labels);

            SVM svm = SVM.Create();
            svm.Type = SVM.Types.CSvc;
            svm.KernelType = SVM.KernelTypes.Linear;
            svm.Gamma = 1;
            svm.C = 1;
            TermCriteria criteria = new TermCriteria(CriteriaTypes.MaxIter, 1000, 0.01);
            svm.TermCriteria = criteria;
            //svm->setTermCriteria(TermCriteria(CV_TERMCRIT_ITER, 1000, 0.01));
            svm.Train(trainingData, SampleTypes.RowSample, labels);
            return svm;
        }

        public List<Mat> make_candidates(Mat img, ref List<RotatedRect> ro_rects)
        {
            List<Mat> candidates = new List<Mat>();
            for (int i = 0; i < ro_rects.Count;)
            {
                RotatedRect testrect = ro_rects[i];
                refine_candidate(img, ref testrect);     // 후보 영역 개선
                ro_rects[i] = testrect;

                if (vertify_plate(ro_rects[i]))
                {
                    Mat corr_img = correct_plate(img, ro_rects[i]);
                    Cv2.CvtColor(corr_img, corr_img, ColorConversionCodes.BGR2GRAY);
                    candidates.Add(corr_img);        // 보정 영상 저장

                    Cv2.ImShow("plate_img - " + i, corr_img);
                    Cv2.ResizeWindow("plate_img - " + i, 200, 50);   //윈도우 크기 조정
                    if (i == 0)
                    {
                        var ocr = new TesseractEngine("./tesseract-5.2.0/tessdata", "kor", EngineMode.Default);
                        var image = BitmapConverter.ToBitmap(corr_img);
                        var page = ocr.Process(image);
                        var text = page.GetText();
                        Console.WriteLine($"테서렉트 분류 결과 : {text}");
                    }
                    i++;
                }
                else ro_rects.RemoveAt(i);

            }
            return candidates;
        }

        public int classify_plates(SVM svm, List<Mat> candidate_img)
        {
            for (int i = 0; i < candidate_img.Count; i++)
            {
                Mat onerow = candidate_img[i].Reshape(1, 1);
                onerow.ConvertTo(onerow, MatType.CV_32F);

                Mat results = new Mat();
                svm.Predict(onerow, results);

                if (results.At<float>(0) == 1)
                    return i;
            }
            return -1;
        }
    }
    internal class Program
    {

        static void Main(string[] args)
        {
            NotOcr notOcr = new NotOcr();

            int no = 0;
            do
            {
                Cv2.DestroyAllWindows();
                string fn = string.Format("C:\\temp\\img\\{0:D2}.jpg", no);
                Mat image = Cv2.ImRead(fn, (ImreadModes)1);
                if (image.Empty())
                {
                    Console.WriteLine("비어있습니다");
                    return;
                }


                Mat morph = new Mat();
                Mat plate_img = new Mat();
                notOcr.preprocessing(image, ref morph);
                List<RotatedRect> candidates = new List<RotatedRect>();
                notOcr.find_candidates(morph, ref candidates);

                List<Mat> candidate_img = notOcr.make_candidates(image, ref candidates);

                Cv2.ImShow("image - " + no, image);

                no++;
                int key = Cv2.WaitKey();
                if (key == 27)  // ESC 키를 누르면 루프 종료
                    break;
            }
            while (true);
        }
    }
}
