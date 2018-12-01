using System;
using System.Collections.Generic;
using System.IO;

namespace RNN
{
    public class FileManager
    {
        private string m_dataPath;

        private FileManager() { }

        public FileManager(string dataPath)
        {
            m_dataPath = dataPath;
        }

        public double[] LoadMemory(int layerNumber, int neuronNumber)
        {
            double[] memory = new double[0];

            using (StreamReader fileReader = new StreamReader(m_dataPath))
            {
                while (!fileReader.EndOfStream)
                {
                    string[] readedLine = fileReader.ReadLine().Split(' ');

                    if ((readedLine[0] == "layer_" + layerNumber) && (readedLine[1] == "neuron_" + neuronNumber))
                    {
                        memory = GetWeights(readedLine);
                    }
                }
            }

            return memory;
        }

        private double[] GetWeights(string[] readedLine)
        {
            double[] weights = new double[readedLine.Length - 2];

            for (int i = 0; i < weights.Length; i++)
            {
                weights[i] = double.Parse(readedLine[i + 2]);
            }

            return weights;
        }

        public void PrepareToSaveMemory()
        {
            File.Delete(m_dataPath);
        }

        public void SaveMemory(int layerNumber, int neuronNumber, double[] weights)
        {
            using (StreamWriter fileWriter = new StreamWriter(m_dataPath, true))
            {
                fileWriter.Write("layer_{0} neuron_{1}", layerNumber, neuronNumber);

                for (int i = 0; i < weights.Length; i++)
                {
                    fileWriter.Write(" " + weights[i]);
                }

                fileWriter.WriteLine("");
            }
        }

    }
}
