from collections import namedtuple

DiodeConfig = namedtuple('DiodeConfig', ['name', 'Is', 'nabla', 'Vt', 'N_up', 'N_down'], defaults=['', 1.0e-9, 1.0, 25.85e-3, 1, 1])

default_diode = DiodeConfig('DefaultDiode')
diode_1n4148_1u1d = DiodeConfig('1N4148 (1U-1D)', Is=4.352e-9, nabla=1.906) # borrowed from: https://github.com/neiser/spice-padiwa-amps/blob/master/1N4148.lib
diode_1n4148_1u2d = DiodeConfig('1N4148 (1U-2D)', Is=4.352e-9, nabla=1.906, N_up=1, N_down=2)
diode_1n4148_1u3d = DiodeConfig('1N4148 (1U-3D)', Is=4.352e-9, nabla=1.906, N_up=1, N_down=3)
diode_1n4148_2u2d = DiodeConfig('1N4148 (2U-2D)', Is=4.352e-9, nabla=1.906, N_up=2, N_down=2)
diode_1n4148_2u3d = DiodeConfig('1N4148 (2U-3D)', Is=4.352e-9, nabla=1.906, N_up=2, N_down=3)
diode_1n4148_3u3d = DiodeConfig('1N4148 (3U-3D)', Is=4.352e-9, nabla=1.906, N_up=3, N_down=3)
