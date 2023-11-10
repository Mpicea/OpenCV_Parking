using OpenCvSharp;
using OpenCvSharp.Dnn;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace OpenCVdarknet
{
    internal class Program
    {
        static void Main(string[] args)
        {
            const string cfgFile = "darknet_model/yolov3.cfg";
            const string darknetModel = "darknet_model/yolov3.weights";
            string[] classNames = File.ReadAllLines("darknet_model/yolov3.txt");

            VideoCapture capture = new VideoCapture(0); 

            using (Mat image = new Mat())
            {
                using (Net net = Net.ReadNetFromDarknet(cfgFile, darknetModel))
                {
                    while (true)
                    {
                        capture.Read(image);

                        if (image.Empty())
                            break;

                        List<string> labels = new List<string>();
                        List<float> scores = new List<float>();
                        List<Rect> bboxes = new List<Rect>();

                        Mat inputBlob = CvDnn.BlobFromImage(image, 1 / 255f, new Size(416, 416), crop: false);

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
                                    Cv2.MinMaxLoc(prob.Row(p).ColRange(5, prob.Cols), out _, out _, out _, out Point classNumber);

                                    int classes = classNumber.X;
                                    float probability = prob.At<float>(p, classes + 5);

                                    if (className == "car" || className == "motorcycle" || className == "bus" || className == "truck")
                                    {
                                        float centerX = prob.At<float>(p, 0) * image.Width;
                                        float centerY = prob.At<float>(p, 1) * image.Height;
                                        float width = prob.At<float>(p, 2) * image.Width;
                                        float height = prob.At<float>(p, 3) * image.Height;

                                        labels.Add(classNames[classes]);
                                        scores.Add(probability);
                                        bboxes.Add(new Rect((int)centerX - (int)width / 2, (int)centerY - (int)height / 2, (int)width, (int)height));
                                    }
                                }
                            }
                        }

                        CvDnn.NMSBoxes(bboxes, scores, 0.9f, 0.5f, out int[] indices);

                        foreach (int i in indices)
                        {
                            Cv2.Rectangle(image, bboxes[i], Scalar.Red, 1);
                            Cv2.PutText(image, labels[i], bboxes[i].Location, HersheyFonts.HersheyComplex, 1.0, Scalar.Red);

                            // 감지된 데이터를 콘솔에 출력
                            Console.WriteLine("Label: " + labels[i]);
                            Console.WriteLine("Confidence: " + scores[i]);
                            Console.WriteLine("Bounding Box: " + bboxes[i]);
                        }

                        Cv2.ImShow("Camera", image);
                        Cv2.WaitKey(1);
                    }
                }
            }

            capture.Release(); 
            Cv2.DestroyAllWindows(); 
        }
    }
}
