#include "MultiDiodeClipper.h"

namespace
{
const String gainTag = "gain";
const String cutoffTag = "cutoff";
const String modelTag = "model";
} // namespace

MultiDiodeClipper::MultiDiodeClipper (const String& prefix, AudioProcessorValueTreeState& vts)
{
    for (auto& paramTag : { gainTag, cutoffTag, modelTag })
        paramTags.add (prefix + paramTag);

    gainDBParam = vts.getRawParameterValue (prefix + gainTag);
    cutoffHzParam = vts.getRawParameterValue (prefix + cutoffTag);
    modelChoiceParam = vts.getRawParameterValue (prefix + modelTag);
}

void MultiDiodeClipper::addParameters (chowdsp::Parameters& params, const String& prefix)
{
    using namespace chowdsp::ParamUtils;
    emplace_param<VTSParam> (params, prefix + gainTag, "Gain", String(), NormalisableRange { -18.0f, 18.0f }, 0.0f, &gainValToString, &stringToGainVal);

    NormalisableRange cutoffRange { 200.0f, 20000.0f };
    cutoffRange.setSkewForCentre (2000.0f);
    emplace_param<VTSParam> (params, prefix + cutoffTag, "Cutoff", String(), cutoffRange, 4000.0f, &freqValToString, &stringToFreqVal);

    StringArray modelChoices { "1up/2down 2x8", "2up/2down 2x8", "1up/3down 2x8", "2up/3down 2x8", "3up/3down 2x8" };
    emplace_param<AudioParameterChoice> (params, prefix + modelTag, "Model", modelChoices, 0);
}

void MultiDiodeClipper::prepare (double sampleRate, int samplesPerBlock)
{
    inputGain.prepare ({ sampleRate, (uint32) samplesPerBlock, 1 });
    inputGain.setRampDurationSeconds (0.02);

    wdf.prepare (sampleRate);
}

void MultiDiodeClipper::process (AudioBuffer<float>& buffer)
{
    // process input gain
    inputGain.setGainDecibels (gainDBParam->load());
    dsp::AudioBlock<float> block { buffer };
    inputGain.process (dsp::ProcessContextReplacing<float> { block });

    wdf.setParameters (*cutoffHzParam, (int) *modelChoiceParam + 7);
    wdf.process (buffer);
}
