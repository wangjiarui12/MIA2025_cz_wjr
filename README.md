模型权重百度网盘链接：
1. cyclegan + pix2pix: https://pan.baidu.com/s/1kT5eF0sfioJPdM9bHMOgnw?pwd=iddw 提取码: iddw





一.&nbsp;&nbsp;[测试数据](./test_data)与环境   
test_LQ为选取的100张低质量超声图像，test_HQ为编号对应的100张高质量超声图像。  
环境为environment1.yml 



二.&nbsp;&nbsp;模型、权重与测试文件  
权重放至checkpoints或者train_ckpt
| 方法名    | 权重路径&&链接                                                  | 测试命令行                                                                                                                                                                                                                                    | 输出路径                                           |
|-----------|---------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------|
| [ccyclegan](./more_models/ccyclegan/) | [放置路径](./more_models/ccyclegan/checkpoints/UsEnhance/)&&[权重](https://drive.google.com/drive/folders/1GrehMGcFn4TDFrf-tgzcFGDqEmoCq1Cw?usp=drive_link )   | 1. `cd ./more_models/ccyclegan`<br>2. `python test.py --dataroot ../../test_data --dataset_mode unaligned --name UsEnhance --model cycle_gan`                                                                                       | [./more_models/ccyclegan/results](./more_models/ccyclegan/results)            |
| [DDIB](./more_models/DDIB)      | [放置路径](./more_models/DDIB/checkpoints)&&[权重](https://drive.google.com/drive/folders/1DwFxtpdRTreGfuWuzRFNSOPd_T1XEJn2?usp=drive_link)    | 1. `cd ./more_models/DDIB`<br>2. `python image_translation.py --attention_resolutions 16 --class_cond False --diffusion_steps 1000 --dropout 0.0 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 128 --num_head_channels 64 --num_res_blocks 1 --resblock_updown True --use_fp16 False --use_scale_shift_norm True --timestep_respacing 200` | [./more_models/DDIB/translated_images](./more_models/DDIB/translated_images) |
| [promptIR](./more_models/promptIR)  | [放置路径](./more_models/promptIR/train_ckpt)&&[权重](https://drive.google.com/drive/folders/1nKSIvsUEC5u4w8A82Wl7iQ9jKpHltUQy?usp=drive_link) | 1. `cd ./more_models/promptIR`<br>2. `python demo.py --test_path ../../test_data/test_LQ --output_path ./results/ --ckpt_name ./train_ckpt/promptir.ckpt`                                                                                   | [./more_models/promptIR/results](./more_models/promptIR/results)             |



三.&nbsp;&nbsp;指标计算  
1. `python cal.py --HQ image_folder1 --LQ image_folder2`