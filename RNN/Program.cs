using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RNN
{
    class Program
    {
        private static RecurrentNeuralNetwork m_RNN;
        private static FileManager m_fileManager;

        static void Main(string[] args)
        {
            m_fileManager = new FileManager("memory.txt");

            int[] neuronsScheme = new int[3] { 50, 45, 26 };
            int receptorsNumber = 52;

            m_RNN = new RecurrentNeuralNetwork(neuronsScheme, receptorsNumber, m_fileManager);

            ShowMenu();

            Console.ReadKey();
        }

        private static void ShowMenu()
        {
            Console.WriteLine("If you want use (1) or teach (0) RNN (none input to exit): ");
            string anwser = Console.ReadLine();

            if (anwser == "1")
            {
                NetUseFunc();
                ShowMenu();
            }

            if (anwser == "0")
            {
                NetTeachFunc();
                ShowMenu();
            }
        }

        private static void NetUseFunc()
        {
            // Input core:
            Console.Write("Input core: ");
            char core = Convert.ToChar(Console.ReadLine());

            // Input count of chars:
            Console.Write("Input count of chars: ");
            int count = Convert.ToInt32(Console.ReadLine());

            m_RNN.Handle(core, count);
        }

        private static void NetTeachFunc()
        {
            // Input text:
            Console.Write("Input teach text: ");
            string text = Console.ReadLine();

            // Train net:
            Console.WriteLine("Training net...");

            double learningSpeed = 0;

            try
            {
                for (int i = 0; i < 64000; i++)
                {
                    // Пересчет величины скорости обучения:
                    learningSpeed = 0.01 * Math.Pow(0.1, i / 150000);

                    m_RNN.Teach(text, learningSpeed);
                }

                // Save network memory:
                m_RNN.SaveMemory(m_fileManager);

                Console.WriteLine("Training success!");
            }
            catch
            {
                Console.WriteLine("Training failed!");
            }
}
    }
}
