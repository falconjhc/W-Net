load_dir=tfModels2019April_WNet/checkpoint/Exp20190423-WNet-DenseMixer-AdaIN-Multi_StyleHw50_ContentPf64_GenEncDec6-Des1@Lyr3_DisMdy6conv/generator/
# load_dir=tfModels2019April_WNet/checkpoint/Exp20190520-WNet-DenseMixer-BN_StyleHw50_ContentPf64_GenEncDec6-Des1@Lyr3_DisMdy6conv/generator/
# load_dir=tfModels2019April_WNet/checkpoint/Exp20190520-WNet-ResidualMixer-BN_StyleHw50_ContentPf64_GenEncDec6-Res1@Lyr3_DisMdy6conv/generator/
# load_dir=tfModels2019April_WNet/checkpoint/Exp20190606-WNet-ResidualMixer-AdaIN-Multi_StyleHw50_ContentPf64_GenEncDec6-Res1@Lyr3_DisMdy6conv/generator/

python Hw50_MultiContent_InferAllScripts.py --style_input_number=4  --evaluating_generator_dir=$load_dir
python Hw50_MultiContent_InferAllScripts.py --style_input_number=1  --evaluating_generator_dir=$load_dir