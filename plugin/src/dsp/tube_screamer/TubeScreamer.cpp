#include "TubeScreamer.h"

namespace
{
const String gainTag = "gain";
const String driveTag = "drive";
} // namespace

TubeScreamer::TubeScreamer (const String& prefix, AudioProcessorValueTreeState& vts)
{
    for (auto& paramTag : { driveTag, gainTag })
        paramTags.add (prefix + paramTag);

    gainDBParam = vts.getRawParameterValue (prefix + gainTag);
    driveParam = vts.getRawParameterValue (prefix + driveTag);
}

void TubeScreamer::addParameters (chowdsp::Parameters& params, const String& prefix)
{
    using namespace chowdsp::ParamUtils;

    emplace_param<VTSParam> (params, prefix + gainTag, "Gain", String(), NormalisableRange { -12.0f, 12.0f }, 0.0f, &gainValToString, &stringToGainVal);
    emplace_param<VTSParam> (params, prefix + driveTag, "Drive", String(), NormalisableRange { 0.0f, 1.0f }, 0.5f, &percentValToString, &stringToPercentVal);
}

void TubeScreamer::prepare (double sampleRate, int samplesPerBlock)
{
    inputGain.prepare ({ sampleRate, (uint32) samplesPerBlock, 1 });
    inputGain.setRampDurationSeconds (0.02);

    C2.prepare ((float) sampleRate);
    C3.prepare ((float) sampleRate);
    C4.prepare ((float) sampleRate);
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

    processTubeScreamer (buffer, dp, Vin, P3, RL);

    buffer.applyGain (Decibels::decibelsToGain (-12.0f));
}
