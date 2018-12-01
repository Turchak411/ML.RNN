using System;
using System.Collections.Generic;
using System.Text;

namespace RNN
{
    public class Neuron
    {
        private double[] m_weights;
        private double m_offsetValue;
        private double m_offsetWeight;

        private double m_lastAnwser;

        private double m_error;

        private Neuron() { }

        public Neuron(double[] weightsValues, double offsetValue, double offsetWeight, double activationThreshold)
        {
            m_weights = weightsValues;
            m_offsetValue = offsetValue;
            m_offsetWeight = offsetWeight;

            m_error = 1;
        }

        public double Handle(double[] data)
        {
            double x = CalcSum(data);
            double actFunc = ActivationFunction(x);

            m_lastAnwser = actFunc;
            return actFunc;
        }

        private double CalcSum(double[] data)
        {
            double x = 0;

            for (int i = 0; i < m_weights.Length; i++)
            {
                x += m_weights[i] * data[i];
            }

            return x + m_offsetValue * m_offsetWeight;
        }

        private double ActivationFunction(double x)
        {
            return 1 / (1 + Math.Pow(Math.E, -x));     // Sigmoid-func
        }

        // CALCULATING ERRORS:

        public void CalcErrorForOutNeuron(double rightAnwser)
        {
            m_error = (rightAnwser - m_lastAnwser) * m_lastAnwser * (1 - m_lastAnwser);
        }

        public double CalcErrorForHiddenNeuron(int neuronIndex, double[][] nextLayerWeights, double[] nextLayerErrors)
        {
            // Вычисление производной активационной функции:
            m_error = m_lastAnwser * (1 - m_lastAnwser);

            // Суммирование ошибок со следующего слоя:
            double sum = 0;

            for (int i = 0; i < nextLayerWeights.GetLength(0); i++)
            {
                sum += nextLayerWeights[i][neuronIndex] * nextLayerErrors[i];
            }

            m_error = m_error * sum;

            return m_error;
        }

        public double[] GetWeights()
        {
            return m_weights;
        }

        public double GetError()
        {
            return m_error;
        }

        // CHANGE WEIGHTS:

        public void ChangeWeights(double learnSpeed, double[] anwsersFromPrewLayer)
        {
            for (int i = 0; i < m_weights.Length; i++)
            {
                m_weights[i] = m_weights[i] + learnSpeed * m_error * anwsersFromPrewLayer[i];
            }

            // Изменение величины смещения:
            m_offsetWeight = m_offsetWeight + learnSpeed * m_error;
        }

        public double GetLastAnwser()
        {
            return m_lastAnwser;
        }

        // SAVE MEMORY:

        public void SaveMemory(FileManager fileManager, int layerNumber, int neuronNumber)
        {
            fileManager.SaveMemory(layerNumber, neuronNumber, m_weights);
        }

    }
}
