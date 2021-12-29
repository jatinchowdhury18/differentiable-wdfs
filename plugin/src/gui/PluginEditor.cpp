#include "PluginEditor.h"

PluginEditor::PluginEditor (DifferentiableWDFPlugin& p) : AudioProcessorEditor (p),
                                                          plugin (p)
{
    setSize (400, 400);
}

void PluginEditor::paint (Graphics& g)
{
    g.fillAll (Colours::black);
}
