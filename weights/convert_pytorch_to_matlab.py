import torch 
from scipy.io import savemat

"wget https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth"

pytorch_mae_dict = torch.load("mae_pretrain_vit_base.pth")['model']

matlab_dict = {}

for i in pytorch_mae_dict:
    name = i.replace(".", "_")
    matlab_dict[name] = pytorch_mae_dict[i].numpy()
    print(name, matlab_dict[name].shape)

savemat("mae_pretrain_vit_base.mat", matlab_dict)
