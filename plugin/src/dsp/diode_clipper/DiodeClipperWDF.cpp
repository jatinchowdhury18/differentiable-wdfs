#include "DiodeClipperWDF.h"

void DiodeClipperWDF::prepare (double sampleRate)
{
    C.prepare ((float) sampleRate);

    prevModelChoice = -1;
}

void DiodeClipperWDF::setParameters (float cutoffFreqHz, int modelIndex)
{
    const auto resVal = 1.0f / (MathConstants<float>::twoPi * cutoffFreqHz * capVal);
    Vs.setResistanceValue (resVal);

    modelChoice = modelIndex;
}

template <typename DiodeType, typename VoltageType, typename PType, typename CType>
void processDiodeClipper (AudioBuffer<float>& buffer, DiodeType& dp, VoltageType& Vs, PType& P1, CType& C)
{
    auto* x = buffer.getWritePointer (0); // mono only!
    for (int n = 0; n < buffer.getNumSamples(); ++n)
    {
        Vs.setVoltage (x[n]);

        dp.incident (P1.reflected());
        x[n] = wdft::voltage<float> (C);
        P1.incident (dp.reflected());
    }
}

void DiodeClipperWDF::process (AudioBuffer<float>& buffer)
{
    if (modelChoice == 0) // TOMS Diode Pair
    {
        if (prevModelChoice != modelChoice)
        {
            P1.connectToParent (&dpToms);
            dpToms.calcImpedance();
            prevModelChoice = modelChoice;
        }

        processDiodeClipper (buffer, dpToms, Vs, P1, C);
    }
    else if (modelChoice == 1) // D'Angelo Wright Omega Diode Pair
    {
        if (prevModelChoice != modelChoice)
        {
            P1.connectToParent (&dpApprox);
            dpApprox.calcImpedance();
            prevModelChoice = modelChoice;
        }

        processDiodeClipper (buffer, dpApprox, Vs, P1, C);
    }
    else if (modelChoice == 2) // Neural nets...
    {
        if (prevModelChoice != modelChoice)
        {
            P1.connectToParent (&dp2x4Model);
            dp2x4Model.calcImpedance();
            prevModelChoice = modelChoice;
        }

        processDiodeClipper (buffer, dp2x4Model, Vs, P1, C);
    }
    else if (modelChoice == 3)
    {
        if (prevModelChoice != modelChoice)
        {
            P1.connectToParent (&dp2x8Model);
            dp2x8Model.calcImpedance();
            prevModelChoice = modelChoice;
        }

        processDiodeClipper (buffer, dp2x8Model, Vs, P1, C);
    }
    else if (modelChoice == 4)
    {
        if (prevModelChoice != modelChoice)
        {
            P1.connectToParent (&dp2x16Model);
            dp2x16Model.calcImpedance();
            prevModelChoice = modelChoice;
        }

        processDiodeClipper (buffer, dp2x16Model, Vs, P1, C);
    }
    else if (modelChoice == 5)
    {
        if (prevModelChoice != modelChoice)
        {
            P1.connectToParent (&dp4x4Model);
            dp4x4Model.calcImpedance();
            prevModelChoice = modelChoice;
        }

        processDiodeClipper (buffer, dp4x4Model, Vs, P1, C);
    }
    else if (modelChoice == 6)
    {
        if (prevModelChoice != modelChoice)
        {
            P1.connectToParent (&dp4x8Model);
            dp4x8Model.calcImpedance();
            prevModelChoice = modelChoice;
        }

        processDiodeClipper (buffer, dp4x8Model, Vs, P1, C);
    }
}
