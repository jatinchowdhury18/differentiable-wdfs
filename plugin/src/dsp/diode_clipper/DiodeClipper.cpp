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

    StringArray modelChoices { "1N4148 Ideal", "1N4148 Approx", "1N4148 2x4", "1N4148 2x8", "1N4148 2x16", "1N4148 4x4", "1N4148 4x8" };
    emplace_param<AudioParameterChoice> (params, prefix + modelTag, "Model", modelChoices, 0);
}

void DiodeClipper::prepare (double sampleRate, int samplesPerBlock)
{
    inputGain.prepare ({ sampleRate, (uint32) samplesPerBlock, 1 });
    inputGain.setRampDurationSeconds (0.02);

    wdf.prepare (sampleRate);
}

void DiodeClipper::process (AudioBuffer<float>& buffer)
{
    // process input gain
    inputGain.setGainDecibels (gainDBParam->load());
    dsp::AudioBlock<float> block { buffer };
    inputGain.process (dsp::ProcessContextReplacing<float> { block });

    wdf.setParameters (*cutoffHzParam, (int) *modelChoiceParam);
    wdf.process (buffer);
}
