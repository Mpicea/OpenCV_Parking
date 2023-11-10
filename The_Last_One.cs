using OpenCvSharp;
using OpenCvSharp.Dnn;
using OpenCvSharp.Extensions;
using OpenCvSharp.ML;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using System.Windows.Forms;
using Tesseract;

namespace OpenCVWinForm001
{
    public partial class Form1 : Form
    {
        private VideoCapture capture;
        private Mat frame;
        private bool isRunning = false;
        private bool FaceDetect = false;
        private bool plateDetect = false;
        private bool lane_violation = false;
        public Form1()
        {
            InitializeComponent();
        }

        private void Form1_Load(object sender, EventArgs e)
        {
            capture = new VideoCapture(0);
            frame = new Mat();
        }

        private async void button1_Click(object sender, EventArgs e)
        {
            if (isRunning)
            {
                isRunning = false;
                return;
            }
            isRunning = true;
            while (isRunning)
            {
                if (capture.IsOpened())
                {
                    capture.Read(frame);
                    if (FaceDetect)
                    {
                        const string cfgFile = "./yolov3.cfg";
                        const string darknetModel = "./yolov3.weights";
                        string[] classNames = File.ReadAllLines("./yolov3.txt");

                        using (Mat image = new Mat())
                        {
                            using (Net net = Net.ReadNetFromDarknet(cfgFile, darknetModel))
                            {
                                capture.Read(image); // 프레임 캡처

                                if (image.Empty())
                                    return;

                                List<string> labels = new List<string>();
                                List<float> scores = new List<float>();
                                List<OpenCvSharp.Rect> bboxes = new List<OpenCvSharp.Rect>();

                                Mat inputBlob = CvDnn.BlobFromImage(image, 1 / 255f, new OpenCvSharp.Size(416, 416), crop: false);

                                net.SetInput(inputBlob);
                                var outBlobNames = net.GetUnconnectedOutLayersNames();
                                var outputBlobs = outBlobNames.Select(toMat => new Mat()).ToArray();

                                net.Forward(outputBlobs, outBlobNames);
                                foreach (Mat prob in outputBlobs)
                                {
                                    for (int p = 0; p < prob.Rows; p++)
                                    {
                                        float confidence = prob.At<float>(p, 4);
                                        if (confidence > 0.9)
                                        {
                                            Cv2.MinMaxLoc(prob.Row(p).ColRange(5, prob.Cols), out _, out _, out _, out OpenCvSharp.Point classNumber);

                                            int classes = classNumber.X;
                                            float probability = prob.At<float>(p, classes + 5);

                                            //if (probability > 0.9 && classNames[classes] != "person") //사람감지X
                                            if (probability > 0.9)
                                            {
                                                float centerX = prob.At<float>(p, 0) * image.Width;
                                                float centerY = prob.At<float>(p, 1) * image.Height;
                                                float width = prob.At<float>(p, 2) * image.Width;
                                                float height = prob.At<float>(p, 3) * image.Height;

                                                labels.Add(classNames[classes]);
                                                scores.Add(probability);
                                                bboxes.Add(new OpenCvSharp.Rect((int)centerX - (int)width / 2, (int)centerY - (int)height / 2, (int)width, (int)height));
                                            }
                                        }
                                    }
                                }

                                CvDnn.NMSBoxes(bboxes, scores, 0.9f, 0.5f, out int[] indices);

                                foreach (int i in indices)
                                {
                                    Cv2.Rectangle(image, bboxes[i], Scalar.Red, 1);
                                    Cv2.PutText(image, labels[i], bboxes[i].Location, HersheyFonts.HersheyComplex, 1.0, Scalar.Red);

                                    // 감지된 데이터를 ListBox1에 추가
                                    listBox1.Items.Add("Label: " + labels[i]);
                                    listBox1.Items.Add("Confidence: " + scores[i]);
                                    listBox1.Items.Add("Bounding Box: " + bboxes[i]);
                                    listBox1.Items.Add("");
                                }

                                pictureBox1.Image = BitmapConverter.ToBitmap(image);
                            }
                        }
                    }
                    else // FaceDetect가 false일 때는 원본 프레임을 표시
                    {
                        pictureBox1.Image = BitmapConverter.ToBitmap(frame);
                    }
                    if (plateDetect)
                    {
                        NotOcr notOcr = new NotOcr();
                        using (Mat image = new Mat())
                        {
                            capture.Read(image);
                            if (image.Empty())
                                return;
                            Mat morph = new Mat();
                            Mat plate_img = new Mat();
                            notOcr.preprocessing(image, ref morph);
                            List<RotatedRect> candidates = new List<RotatedRect>();
                            notOcr.find_candidates(morph, ref candidates);

                            List<Mat> candidate_img = notOcr.make_candidates(image, ref candidates);
                            if (candidate_img.Count > 0)
                            {
                                var ocr = new TesseractEngine("./tesseract-5.2.0/tessdata", "kor", EngineMode.Default);
                                var img = BitmapConverter.ToBitmap(image);
                                var page = ocr.Process(img);
                                var text = page.GetText();
                                string fillter = notOcr.FilterText(text);
                                if (fillter != "")
                                {
                                    listBox2.Items.Add(fillter);
                                }

                            }

                        }

                    }
                    pictureBox1.Image = BitmapConverter.ToBitmap(frame);
                }
                await Task.Delay(33);
                Cv2.WaitKey(1);
            }
        }

        private void button2_Click(object sender, EventArgs e)
        {
            plateDetect = !plateDetect;
        }

        private void button3_Click(object sender, EventArgs e)
        {
            FaceDetect = !FaceDetect;
        }

        private void button4_Click(object sender, EventArgs e)
        {
            lane_violation = !lane_violation;
            if (lane_violation)
            {
                Mat image = new Mat("./parking2.jpg", ImreadModes.Color);

                // 이미지 처리
                Mat gray = new Mat();
                Cv2.CvtColor(image, gray, ColorConversionCodes.BGR2GRAY);
                Cv2.GaussianBlur(gray, gray, new OpenCvSharp.Size(5, 5), 0);
                Cv2.Canny(gray, gray, 50, 150);

                // 흰색 픽셀을 감지하는 마스크 생성
                Mat mask = new Mat();
                Cv2.InRange(gray, new Scalar(50), new Scalar(255), mask);

                //ROI 감지구역 설정
                Mat roi = new Mat(mask, new OpenCvSharp.Rect(0, mask.Rows / 2, mask.Cols, mask.Rows / 2));

                // 마스크를 사용하여 흰색 선만 추출
                Cv2.Canny(roi, gray, 50, 150);

                // 주차된 차량 판별
                LineSegmentPoint[] lines = Cv2.HoughLinesP(gray, 1, Math.PI / 180, 50, 50, 10);

                foreach (var line in lines)
                {
                    // 빨간색으로 선 그리기
                    Cv2.Line(image, new OpenCvSharp.Point(line.P1.X, line.P1.Y + mask.Rows / 2), new OpenCvSharp.Point(line.P2.X, line.P2.Y + mask.Rows / 2), Scalar.Red, 2);
                }
                Bitmap clone_img = BitmapConverter.ToBitmap(image);
                pictureBox2.Image = clone_img;
            }
        }

        private void button5_Click(object sender, EventArgs e)
        {
            isRunning = false;
            capture.Release();
            this.Close();
        }
    }
    class NotOcr
    {

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

        public string FilterText(string text)
        {
            // 원하는 패턴 또는 문자 추출 또는 유효성 검사를 수행
            // 예시: 공백과 알파벳 문자만 유지
            string filteredText = string.Concat(text.Split(new char[] { ' ' }, StringSplitOptions.RemoveEmptyEntries));
            filteredText = new string(filteredText.ToCharArray(), 0, Math.Min(filteredText.Length, 10)); // 처음 10자만 유지

            return filteredText;
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

                    //Cv2.ImShow("plate_img - " + i, corr_img);
                    //Cv2.ResizeWindow("plate_img - " + i, 200, 50);   //윈도우 크기 조정
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
}
