## Steps :

- For environment setup, execute the setup.sh file.
- For downloading and setting up the pretrained LLaMa model, follow the instructions given in `Dataset and LLaMa preparation` section of the original [README.md](https://github.com/jena-shreyas/Flipped-VQA/blob/master/README.md).
- Download the CLIP-extracted video features from [here](https://iitkgpacin-my.sharepoint.com/:f:/g/personal/shreyasjena_kgpian_iitkgp_ac_in/EtYpcn3skvxClLMwprdvMwQB2HvQeRac1sU4kntsW9mnPw?e=ejdQSx) and put the .pth file in data/causalvidqa/.
- Now, run the train.py. 

### Sample command :

torchrun --nproc_per_node 2 train.py --model 7B --max_seq_len 256 --batch_size 2 --epochs 5 --warmup_epochs 2 --bias 3.5 --tau 100. --max_feats 10 --dataset causalvidqa --blr 9e-2 --weight_decay 0.14 --output_dir ./checkpoint/causalvidqa --accum_iter 1 --num_workers 0 --no_pin_mem --vaq --qav
