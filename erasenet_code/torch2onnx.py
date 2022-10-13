import torch
from torch import nn
import torch.onnx
from models.sa_gan_onnx import STRnet2
from pdb import set_trace as stx
import os

def remove_all_spectral_norm(item):
    
    if isinstance(item, nn.Module):
        try:
            nn.utils.remove_spectral_norm(item)
        except Exception:
            pass
        
        for child in item.children():  
            remove_all_spectral_norm(child)

    if isinstance(item, nn.ModuleList):
        for module in item:
            remove_all_spectral_norm(module)

    if isinstance(item, nn.Sequential):
        modules = item.children()
        for module in modules:
            remove_all_spectral_norm(module)
            
def test(pthfile,name):
  
    # create model
    model =STRnet2(3)
    model_dict=model.state_dict()
    model.cuda()
    #pthfile = './checkpoint/model_best.pth'
    loaded_model = torch.load(pthfile)
    model.load_state_dict(loaded_model['state_dict'])
    remove_all_spectral_norm(model)
    model.eval()
    #x = torch.randn(1, 3, 1024 ,1024,device="cuda:0")
    # x = torch.randn(1, 3, 1024 ,1024,device="cpu")
    x = (torch.randn(1, 3, 1024 ,1024, device='cuda'))
    input_name = 'input'
    output_name = 'output'
    torch.onnx.export(model,               # model being run
                  x,                         # model input 
                  "./onnx/{}.onnx".format(name), 
                  verbose=True,# where to save the model (can be a file or file-like object)                  
                  opset_version=11,          # the ONNX version to export the model to                  
                  input_names = [input_name],   # the model's input names
                  output_names = [output_name], # the model's output names
                  dynamic_axes= {
                        input_name: {0: 'batch_size',2 : 'in_width', 3: 'int_height'},
                        output_name: {0: 'batch_size',2: 'out_width', 3:'out_height'}
                       }
                  )



if __name__ == "__main__":
    load_path='checkpoint'
    filename=os.listdir(load_path)
    for name in filename:
        path=os.path.join(load_path,name)
        test(path,name)
