# Define

- Linguagem da lib + distribuição -> python 3.10.2 wheel instalado com pip em cada container
- precisão -> Default FP32 options={"precision": "FP32" | "FP16"}
- engine_version -> "<engine_ref>-fp32" ex: yolo11n-1.0.0-fp32
- I/O -> dict[str, np.ndarray]
- input_shape e output_shape no info -> info(h)["output_shape"] {nome: [dims]} {"images": [1, 3, 640, 640]}
- latency_ms -> tempo total da chamada
- gpu_memory_mb -> memória do handle, medir no load
- pasta de engines -> /opt/ods/engines
- Para decidir backend -> TensorRT sempre q disponível, senão onnx cpu
- Erro -> P2Error com atributo de code
- Concorrência -> (Provisório) usar um handle por thread
- Diferença entre backs -> IoU 0,98 em FP32 e 0,96 em FP16. Dif confiança é 0,1 e 0,3 para FP32 e FP16 respectivamente

# Environment

###### R36 (release), REVISION: 4.7, GCID: 42132812, BOARD: generic, EABI: aarch64, DATE: Thu Sep 18 22:54:44 UTC 2025
###### KERNEL_VARIANT: oot
TARGET_USERSPACE_LIB_DIR=nvidia
TARGET_USERSPACE_LIB_DIR_PATH=usr/lib/aarch64-linux-gnu/nvidia
10.3.0
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2024 NVIDIA Corporation
Built on Wed_Aug_14_10:14:07_PDT_2024
Cuda compilation tools, release 12.6, V12.6.68
Build cuda_12.6.r12.6/compiler.34714021_0


