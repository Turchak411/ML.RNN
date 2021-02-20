using System.Collections.Generic;

namespace RNN
{
    public class RecurrentNeuralNetwork : NeuralNetwork
    {
        private RecurrentNeuralNetwork() { }

        public RecurrentNeuralNetwork(int[] neuronsNumberByLayers, int receptorsNumber, FileManager fileManager)
        {
            Layer firstLayer = new Layer(neuronsNumberByLayers[0], receptorsNumber, 0, fileManager);
            m_layerList.Add(firstLayer);

            for (int i = 1; i < neuronsNumberByLayers.Length; i++)
            {
                Layer layer = new Layer(neuronsNumberByLayers[i], neuronsNumberByLayers[i - 1], i, fileManager);
                m_layerList.Add(layer);
            }
        }

        public void Teach(string text, double learningSpeed)
        {
            // Инициализация прошлой и текущей буквы:
            double[] context;
            double[] currentChar;
            double[] nextChar;

            for (int i = 1; i < text.Length - 1; i++)
            {
                // Прошлая буква (контекст):
                context = GetCharVector(text[i - 1]);
                // Текущая буква:
                currentChar = GetCharVector(text[i]);
                // Следующая буква (ответный вектор)
                nextChar = GetCharVector(text[i + 1]);

                // Слияние контекста и текущей буквы:
                var list = new List<double>(context);
                list.AddRange(currentChar);
                double[] data = list.ToArray();

                base.Handle(data);
                base.Teach(data, nextChar, learningSpeed);
            }

        }

        public string Handle(char core, int count)
        {
            // Инициализация списка букв:
            List<double[]> charList = new List<double[]>();

            // Инициализация контекста с нулевыми значениями:
            double[] context = new double[26];

            // Инициализация текущего символа:
            double[] currentChar = GetCharVector(core);

            for (int i = 0; i < count; i++)
            {
                // Слияние контекста и текущей буквы:
                var list = new List<double>(context);
                list.AddRange(currentChar);
                double[] data = list.ToArray();

                currentChar = context;

                context = base.Handle(data);

                // Запись полученной буквы:
                charList.Add(context);
            }

            return DecodeChars(charList);
        }

        private string DecodeChars(List<double[]> charList)
        {
            string decodedString = "";

            for(int i = 0; i < charList.Count; i++)
            {
                decodedString += GetCharSymbol(charList[i]);
            }

            return decodedString;
        }

        private double[] GetCharVector(char textChar)
        {
            double[] charVector = new double[26];

            switch (textChar)
            {
                case 'a':
                    charVector[0] = 1;
                    break;
                case 'b':
                    charVector[1] = 1;
                    break;
                case 'c':
                    charVector[2] = 1;
                    break;
                case 'd':
                    charVector[3] = 1;
                    break;
                case 'e':
                    charVector[4] = 1;
                    break;
                case 'f':
                    charVector[5] = 1;
                    break;
                case 'g':
                    charVector[6] = 1;
                    break;
                case 'h':
                    charVector[7] = 1;
                    break;
                case 'i':
                    charVector[8] = 1;
                    break;
                case 'j':
                    charVector[9] = 1;
                    break;
                case 'k':
                    charVector[10] = 1;
                    break;
                case 'l':
                    charVector[11] = 1;
                    break;
                case 'm':
                    charVector[12] = 1;
                    break;
                case 'n':
                    charVector[13] = 1;
                    break;
                case 'o':
                    charVector[14] = 1;
                    break;
                case 'p':
                    charVector[15] = 1;
                    break;
                case 'q':
                    charVector[16] = 1;
                    break;
                case 'r':
                    charVector[17] = 1;
                    break;
                case 's':
                    charVector[18] = 1;
                    break;
                case 't':
                    charVector[19] = 1;
                    break;
                case 'u':
                    charVector[20] = 1;
                    break;
                case 'v':
                    charVector[21] = 1;
                    break;
                case 'w':
                    charVector[22] = 1;
                    break;
                case 'x':
                    charVector[23] = 1;
                    break;
                case 'y':
                    charVector[24] = 1;
                    break;
                case 'z':
                    charVector[25] = 1;
                    break;
                default:
                    break;
            }

            return charVector;
        }

        private string GetCharSymbol(double[] charVector)
        {
            // Нахождение символа в векторе:
            int indexOfMaxValue = 0;
            double maxValue = charVector[indexOfMaxValue];

            for (int i = 0; i < charVector.Length; i++)
            {
                if(charVector[i] > maxValue)
                {
                    indexOfMaxValue = i;
                    maxValue = charVector[i];
                }
            }

            // Непосредственно декодинг символа с соответствующим индексом:
            switch(indexOfMaxValue)
            {
                case 0:
                    return "a";
                case 1:
                    return "b";
                case 2:
                    return "c";
                case 3:
                    return "d";
                case 4:
                    return "e";
                case 5:
                    return "f";
                case 6:
                    return "g";
                case 7:
                    return "h";
                case 8:
                    return "i";
                case 9:
                    return "j";
                case 10:
                    return "k";
                case 11:
                    return "l";
                case 12:
                    return "m";
                case 13:
                    return "n";
                case 14:
                    return "o";
                case 15:
                    return "p";
                case 16:
                    return "q";
                case 17:
                    return "r";
                case 18:
                    return "s";
                case 19:
                    return "t";
                case 20:
                    return "u";
                case 21:
                    return "v";
                case 22:
                    return "w";
                case 23:
                    return "x";
                case 24:
                    return "y";
                case 25:
                    return "z";
                default:
                    return "";
            }
        }
    }

    public struct DataSet
    {
        public double[] _input;
        public double[] _output;
    }
}
