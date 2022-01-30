#include "HPFDiodeClipper.h"

namespace
{
const String gainTag = "gain";
const String cutoffTag = "cutoff";
const String modelTag = "model";
} // namespace

HPFDiodeClipper::HPFDiodeClipper (const String& prefix, AudioProcessorValueTreeState& vts)
{
    for (auto& paramTag : { gainTag, cutoffTag, modelTag })
        paramTags.add (prefix + paramTag);

    gainDBParam = vts.getRawParameterValue (prefix + gainTag);
    cutoffHzParam = vts.getRawParameterValue (prefix + cutoffTag);
    modelChoiceParam = vts.getRawParameterValue (prefix + modelTag);
}

void HPFDiodeClipper::addParameters (chowdsp::Parameters& params, const String& prefix)
{
    using namespace chowdsp::ParamUtils;
    emplace_param<VTSParam> (params, prefix + gainTag, "Gain", String(), NormalisableRange { 0.0f, 24.0f }, 0.0f, &gainValToString, &stringToGainVal);

    NormalisableRange cutoffRange { 200.0f, 20000.0f };
    cutoffRange.setSkewForCentre (2000.0f);
    emplace_param<VTSParam> (params, prefix + cutoffTag, "Cutoff", String(), cutoffRange, 4000.0f, &freqValToString, &stringToFreqVal);

    StringArray modelChoices { "1N4148 Ideal", "1N4148 Approx", "1N4148 2x16 Extrapolated", "1N4148 2x16 Trained" };
    emplace_param<AudioParameterChoice> (params, prefix + modelTag, "Model", modelChoices, 0);
}

void HPFDiodeClipper::prepare (double sampleRate, int samplesPerBlock)
{
    inputGain.prepare ({ sampleRate, (uint32) samplesPerBlock, 1 });
    inputGain.setRampDurationSeconds (0.02);

    C.prepare ((float) sampleRate);

    prevModelChoice = -1;
}

template <typename DiodeType, typename VoltageType, typename PType, typename RType>
void processDiodeClipper (AudioBuffer<float>& buffer, DiodeType& dp, VoltageType& Vs, PType& P1, RType& R)
{
    auto* x = buffer.getWritePointer (0); // mono only!
    for (int n = 0; n < buffer.getNumSamples(); ++n)
    {
        Vs.setVoltage (x[n]);

        dp.incident (P1.reflected());
        x[n] = wdft::voltage<float> (R);
        P1.incident (dp.reflected());
    }
}

void HPFDiodeClipper::process (AudioBuffer<float>& buffer)
{
    // process input gain
    inputGain.setGainDecibels (gainDBParam->load() - 6.0f);
    dsp::AudioBlock<float> block { buffer };
    inputGain.process (dsp::ProcessContextReplacing<float> { block });

    // set cutoff
    const auto resVal = 1.0f / (MathConstants<float>::twoPi * *cutoffHzParam * capVal);
    R.setResistanceValue (resVal);

    modelChoice = (int) *modelChoiceParam;
    if (modelChoice == 0) // TOMS Wright Omega Diode Pair
    {
        if (prevModelChoice != modelChoice)
        {
            P1.connectToParent (&dpToms);
            dpToms.calcImpedance();
            prevModelChoice = modelChoice;
        }

        processDiodeClipper (buffer, dpToms, Vs, P1, R);
    }
    else if (modelChoice == 1) // D'Angelo Wright Omega Diode Pair
    {
        if (prevModelChoice != modelChoice)
        {
            P1.connectToParent (&dpApprox);
            dpApprox.calcImpedance();
            prevModelChoice = modelChoice;
        }

        processDiodeClipper (buffer, dpApprox, Vs, P1, R);
    }
    else if (modelChoice == 2) // Extrapolated neural net
    {
        if (prevModelChoice != modelChoice)
        {
            P1.connectToParent (&dpExtrModel);
            dpExtrModel.calcImpedance();
            prevModelChoice = modelChoice;
        }

        processDiodeClipper (buffer, dpExtrModel, Vs, P1, R);
    }
    else if (modelChoice == 3) // Trained neural net
    {
        if (prevModelChoice != modelChoice)
        {
            P1.connectToParent (&dpTrainModel);
            dpTrainModel.calcImpedance();
            prevModelChoice = modelChoice;
        }

        processDiodeClipper (buffer, dpTrainModel, Vs, P1, R);
    }
}
