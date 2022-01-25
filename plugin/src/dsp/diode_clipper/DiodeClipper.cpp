#include "DiodeClipper.h"

namespace
{
const String gainTag = "gain";
const String cutoffTag = "cutoff";
const String modelTag = "model";
} // namespace

DiodeClipper::DiodeClipper (const String& prefix, AudioProcessorValueTreeState& vts)
{
    for (auto& paramTag : { gainTag, cutoffTag, modelTag })
        paramTags.add (prefix + paramTag);

    gainDBParam = vts.getRawParameterValue (prefix + gainTag);
    cutoffHzParam = vts.getRawParameterValue (prefix + cutoffTag);
    modelChoiceParam = vts.getRawParameterValue (prefix + modelTag);
}

void DiodeClipper::addParameters (chowdsp::Parameters& params, const String& prefix)
{
    using namespace chowdsp::ParamUtils;
    emplace_param<VTSParam> (params, prefix + gainTag, "Gain", String(), NormalisableRange { -18.0f, 18.0f }, 0.0f, &gainValToString, &stringToGainVal);

    NormalisableRange cutoffRange { 200.0f, 20000.0f };
    cutoffRange.setSkewForCentre (2000.0f);
    emplace_param<VTSParam> (params, prefix + cutoffTag, "Cutoff", String(), cutoffRange, 4000.0f, &freqValToString, &stringToFreqVal);

    StringArray modelChoices { "1N4148 Ideal", "1N4148 Approx", "1N4148 4x8" };
    emplace_param<AudioParameterChoice> (params, prefix + modelTag, "Model", modelChoices, 0);
}

void DiodeClipper::prepare (double sampleRate, int samplesPerBlock)
{
    inputGain.prepare ({ sampleRate, (uint32) samplesPerBlock, 1 });
    inputGain.setRampDurationSeconds (0.02);

    C.prepare ((float) sampleRate);
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

void DiodeClipper::process (AudioBuffer<float>& buffer)
{
    // process input gain
    inputGain.setGainDecibels (gainDBParam->load());
    dsp::AudioBlock<float> block { buffer };
    inputGain.process (dsp::ProcessContextReplacing<float> { block });

    // set cutoff frequency
    const auto resVal = 1.0f / (MathConstants<float>::twoPi * *cutoffHzParam * capVal);
    Vs.setResistanceValue (resVal);

    auto modelChoice = (int) *modelChoiceParam;
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
            P1.connectToParent (&dp4x8Model);
            dp4x8Model.calcImpedance();
            prevModelChoice = modelChoice;
        }

        processDiodeClipper (buffer, dp4x8Model, Vs, P1, C);
    }
}
