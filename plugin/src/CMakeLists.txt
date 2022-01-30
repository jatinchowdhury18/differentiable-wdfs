target_sources(DifferentiableWDFs PRIVATE
    DifferentiableWDFPlugin.cpp

    dsp/diode_clipper/DiodeClipper.cpp
    dsp/diode_clipper/DiodeClipperWDF.cpp
    dsp/diode_clipper/HPFDiodeClipper.cpp
    dsp/diode_clipper/MultiDiodeClipper.cpp
    dsp/tube_screamer/TubeScreamer.cpp

    gui/CircuitModelGUI.cpp
    gui/PluginEditor.cpp
)

target_precompile_headers(DifferentiableWDFs PRIVATE pch.h)

juce_add_binary_data(ModelData
    NAMESPACE ModelData
    HEADER_NAME ModelData.h
    SOURCES
        "../../wdf_py/diode_clipper/models/1N4148 (1U-1D)_2x4_training_3.json"
        "../../wdf_py/diode_clipper/models/1N4148 (1U-1D)_2x8_training_3.json"
        "../../wdf_py/diode_clipper/models/1N4148 (1U-1D)_2x16_training_2000.json"
        "../../wdf_py/diode_clipper/models/1N4148 (1U-1D)_4x4_training_1.json"
        "../../wdf_py/diode_clipper/models/1N4148 (1U-1D)_4x8_training_500.json"

        "../../wdf_py/diode_clipper/models/1N4148 (1U-2D)_2x16_training_3.json"
        "../../wdf_py/diode_clipper/models/1N4148 (2U-2D)_2x16_training_500.json"
        "../../wdf_py/diode_clipper/models/1N4148 (1U-3D)_2x16_training_2.json"
        "../../wdf_py/diode_clipper/models/1N4148 (2U-3D)_2x16_training_3.json"
        "../../wdf_py/diode_clipper/models/1N4148 (3U-3D)_2x16_training_3.json"

        "../../wdf_py/diode_clipper/models/1N4148 (1U-1D)_2x16_training_1_hpf.json"
)

# Need to build BinaryData with -fPIC flag on Linux
set_target_properties(ModelData PROPERTIES
    POSITION_INDEPENDENT_CODE TRUE)
