target_sources(DifferentiableWDFs PRIVATE
    DifferentiableWDFPlugin.cpp

    dsp/diode_clipper/DiodeClipper.cpp

    gui/CircuitModelGUI.cpp
    gui/PluginEditor.cpp
)

target_precompile_headers(DifferentiableWDFs PRIVATE pch.h)
