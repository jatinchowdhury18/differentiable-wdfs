target_sources(DifferentiableWDFs PRIVATE
    DifferentiableWDFPlugin.cpp

    gui/PluginEditor.cpp
)

target_precompile_headers(DifferentiableWDFs PRIVATE pch.h)
