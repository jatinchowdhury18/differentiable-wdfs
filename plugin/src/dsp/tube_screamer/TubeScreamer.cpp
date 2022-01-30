#include "TubeScreamer.h"

namespace
{
const String gainTag = "gain";
const String driveTag = "drive";
const String modelTag = "model";
} // namespace

TubeScreamer::TubeScreamer (const String& prefix, AudioProcessorValueTreeState& vts)
{
    for (auto& paramTag : { driveTag, gainTag, modelTag })
        paramTags.add (prefix + paramTag);

    gainDBParam = vts.getRawParameterValue (prefix + gainTag);
    driveParam = vts.getRawParameterValue (prefix + driveTag);
    modelChoiceParam = vts.getRawParameterValue (prefix + modelTag);
}

void TubeScreamer::addParameters (chowdsp::Parameters& params, const String& prefix)
{
    using namespace chowdsp::ParamUtils;

    emplace_param<VTSParam> (params, prefix + gainTag, "Gain", String(), NormalisableRange { -12.0f, 12.0f }, 0.0f, &gainValToString, &stringToGainVal);
    emplace_param<VTSParam> (params, prefix + driveTag, "Drive", String(), NormalisableRange { 0.0f, 1.0f }, 0.5f, &percentValToString, &stringToPercentVal);

    StringArray modelChoices { "1N4148 Approx", "1N4148 2x16" };
    emplace_param<AudioParameterChoice> (params, prefix + modelTag, "Model", modelChoices, 0);
}

void TubeScreamer::prepare (double sampleRate, int samplesPerBlock)
{
    inputGain.prepare ({ sampleRate, (uint32) samplesPerBlock, 1 });
    inputGain.setRampDurationSeconds (0.02);

    C2.prepare ((float) sampleRate);
    C3.prepare ((float) sampleRate);
    C4.prepare ((float) sampleRate);

    prevModelChoice = -1;
}

template <typename DiodeType, typename VoltageType, typename PType, typename ResType>
void processTubeScreamer (AudioBuffer<float>& buffer, DiodeType& dp, VoltageType& Vin, PType& P3, ResType& RL)
{
    auto* x = buffer.getWritePointer (0); // mono only!
    for (int n = 0; n < buffer.getNumSamples(); ++n)
    {
        Vin.setVoltage (x[n]);

        dp.incident (P3.reflected());
        P3.incident (dp.reflected());

        x[n] = wdft::voltage<float> (RL);
    }
}

void TubeScreamer::process (AudioBuffer<float>& buffer)
{
    // process input gain
    inputGain.setGainDecibels (gainDBParam->load() - 6.0f);
    dsp::AudioBlock<float> block { buffer };
    inputGain.process (dsp::ProcessContextReplacing<float> { block });

    // drive resistor value
    R6_P1.setResistanceValue (R6 + Pot1 * *driveParam);

    modelChoice = (int) *modelChoiceParam;
    if (modelChoice == 0) // D'Angelo Wright Omega Diode Pair
    {
        if (prevModelChoice != modelChoice)
        {
            P3.connectToParent (&dpApprox);
            dpApprox.calcImpedance();
            prevModelChoice = modelChoice;
        }

        processTubeScreamer (buffer, dpApprox, Vin, P3, RL);
    }
    else if (modelChoice == 1) // Neural nets...
    {
        if (prevModelChoice != modelChoice)
        {
            P3.connectToParent (&dp2x8Model);
            dp2x8Model.calcImpedance();
            prevModelChoice = modelChoice;
        }

        processTubeScreamer (buffer, dp2x8Model, Vin, P3, RL);
    }

    buffer.applyGain (Decibels::decibelsToGain (-12.0f));
}
