load_dir=tfModels2019April_WNet/checkpoint/Exp20190423-WNet-DenseMixer-BN_StyleHw50_ContentPfStd1_GenEncDec6-Des1@Lyr3_DisMdy6conv/generator/
# load_dir=tfModels2019April_WNet/checkpoint/Exp20190508-WNet-ResidualMixer-BN_StyleHw50_ContentPfStd1_GenEncDec6-Res1@Lyr3_DisMdy6conv/generator/
# load_dir=tfModels2019April_WNet/checkpoint/Exp20190606-WNet-DenseMixer-AdaIN-Multi_StyleHw50_ContentPfStd1_GenEncDec6-Des1@Lyr3_DisMdy6conv/generator/
# load_dir=tfModels2019April_WNet/checkpoint/Exp20190606-WNet-ResidualMixer-AdaIN-Multi_StyleHw50_ContentPfStd1_GenEncDec6-Res1@Lyr3_DisMdy6conv/generator/


python Hw50-SingleContent.py --style_input_number=4  --evaluating_generator_dir=$load_dir
python Hw50-SingleContent.py --style_input_number=1  --evaluating_generator_dir=$load_dir