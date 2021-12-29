#include "DifferentiableWDFPlugin.h"
#include "gui/PluginEditor.h"

DifferentiableWDFPlugin::DifferentiableWDFPlugin()
{
}

void DifferentiableWDFPlugin::addParameters (Parameters& params)
{
}

void DifferentiableWDFPlugin::prepareToPlay (double sampleRate, int samplesPerBlock)
{
}

void DifferentiableWDFPlugin::releaseResources()
{
}

void DifferentiableWDFPlugin::processAudioBlock (AudioBuffer<float>& buffer)
{
}

AudioProcessorEditor* DifferentiableWDFPlugin::createEditor()
{
    return new PluginEditor (*this);
}

// This creates new instances of the plugin
AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new DifferentiableWDFPlugin();
}
