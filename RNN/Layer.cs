using System;
using System.Collections.Generic;
using System.Text;

namespace RNN
{
    public class Layer
    {
        private List<Neuron> m_neuronList = new List<Neuron>();

        private Layer() { }

        public Layer(int neuronCount, int weightCount, int layerNumber, FileManager fileManager)
        {
            double offsetValue = 0.5;

            for (int i = 0; i < neuronCount; i++)
            {
                double[] weights = fileManager.LoadMemory(layerNumber, i);
                Neuron neuron = new Neuron(weights, offsetValue, -1, 0.3);

                m_neuronList.Add(neuron);
            }
        }

        public double[] Handle(double[] data)
        {
            double[] layerResultVector = new double[m_neuronList.Count];

            for (int i = 0; i < layerResultVector.Length; i++)
            {
                layerResultVector[i] = m_neuronList[i].Handle(data);
            }

            return layerResultVector;
        }

        // CALCULATING ERRORS:

        public void CalcErrorAsOut(double[] rightAnwsersSet)
        {
            for (int i = 0; i < m_neuronList.Count; i++)
            {
                m_neuronList[i].CalcErrorForOutNeuron(rightAnwsersSet[i]);
            }
        }

        public void CalcErrorAsHidden(double[][] nextLayerWeights, double[] nextLayerErrors)
        {
            for (int i = 0; i < m_neuronList.Count; i++)
            {
                m_neuronList[i].CalcErrorForHiddenNeuron(i, nextLayerWeights, nextLayerErrors);
            }
        }

        // CHANGE WEIGHTS:

        public void ChangeWeights(double learnSpeed, double[] anwsersFromPrewLayer)
        {
            for (int i = 0; i < m_neuronList.Count; i++)
            {
                m_neuronList[i].ChangeWeights(learnSpeed, anwsersFromPrewLayer);
            }
        }

        public double[] GetLastAnwsers()
        {
            double[] lastAnwsers = new double[m_neuronList.Count];

            for (int i = 0; i < m_neuronList.Count; i++)
            {
                lastAnwsers[i] = m_neuronList[i].GetLastAnwser();
            }

            return lastAnwsers;
        }

        public double[][] GetWeights()
        {
            double[][] weights = new double[m_neuronList.Count][];

            for (int i = 0; i < m_neuronList.Count; i++)
            {
                weights[i] = m_neuronList[i].GetWeights();
            }

            return weights;
        }

        public double[] GetErrors()
        {
            double[] errors = new double[m_neuronList.Count];

            for (int i = 0; i < m_neuronList.Count; i++)
            {
                errors[i] = m_neuronList[i].GetError();
            }

            return errors;
        }

        // SAVE MEMORY:

        public void SaveMemory(FileManager fileManager, int layerNumber)
        {
            for (int i = 0; i < m_neuronList.Count; i++)
            {
                m_neuronList[i].SaveMemory(fileManager, layerNumber, i);
            }
        }
    }
}
