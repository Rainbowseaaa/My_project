import torch
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

def gen_single_SLM_mask(SLM_size: tuple = (1200, 1920), bits: int = 8, Mask: torch.tensor = None, shiftx: int = 0,
                        shifty: int = 0) -> torch.tensor:
    '''
    默认将mask放在SLM的中心。
    shiftx>0：向右移动
    shifty>0：向上移动
    '''
    if Mask is None:
        Mask = torch.zeros(SLM_size)
    Mask = Mask.cpu()
    Mask = torch.where(Mask < 0, Mask + torch.pi * 2, Mask)  # 将范围从-pi~pi转换到0~2pi
    if bits == 8:
        Mask = torch.round(Mask/(torch.pi*2)*(2**8-1))   # 0~255 int
    if bits == 10:
        Mask = torch.round(Mask/(torch.pi*2)*(2**10-1))  # 0~1023 int
    mask_size = Mask.shape  # (H, W)
    if SLM_size[0] < mask_size[0] or SLM_size[1] < mask_size[1]:
        raise ValueError("mask size should be smaller than SLM size!")
    bottom = (SLM_size[0] - mask_size[0]) // 2
    left = (SLM_size[1] - mask_size[1]) // 2
    SLM_mask = torch.zeros(SLM_size)
    if (SLM_size[0] - mask_size[0]) // 2 < -shifty:
        raise ValueError('shifty is too small!')  # 向下移动过多，SLM不能完全显示
    if (SLM_size[0] - mask_size[0]) // 2 < shifty:
        raise ValueError('shifty is too large!')  # 向上移动过多，SLM不能完全显示
    if (SLM_size[1] - mask_size[1]) // 2 < -shiftx:
        raise ValueError('shiftx is too small!')  # 向左移动过多，SLM不能完全显示
    if (SLM_size[1] - mask_size[1]) // 2 < shiftx:
        raise ValueError('shiftx is too large!')  # 向右移动过多，SLM不能完全显示
    SLM_mask[bottom - shifty:bottom + mask_size[0] - shifty, left + shiftx:left + mask_size[1] + shiftx] = Mask
    return SLM_mask.to(torch.uint8)
def gen_multi_SLM_mask(SLM_size=(1200, 1920), masks=None, shiftx_list=None, shifty_list=None, bits=8):
    slm_masks = torch.zeros(SLM_size)
    for i, mask in enumerate(masks):
        slm_masks += gen_single_SLM_mask(SLM_size=SLM_size, Mask=mask, shiftx=shiftx_list[i], shifty=shifty_list[i], bits=bits)
    return slm_masks.to(torch.uint8)



if __name__ == '__main__':
    mask = torch.ones(500, 500)
    slm_mask = gen_single_SLM_mask(SLM_size=(1200, 1920), Mask=mask, shiftx=200, shifty=0)
    plt.figure()
    plt.imshow(slm_mask)
    plt.show()
