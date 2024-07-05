import torch  
  
class LowFrequencyDenoiser:  
    def __init__(self, threshold_param=0.5):  
        """  
        初始化低频去噪器。  
          
        参数:  
        - threshold_param: 用于确定哪些频率分量被视为低频的分位数（介于0和1之间）。  
        较低的值会保留更多的低频分量，而较高的值会去除更多的低频分量。  
        """  
        self.threshold_param = threshold_param  
    # @staticmethod
    def create_adaptive_low_freq_mask(self, x_fft):  
        """  
        创建自适应低频掩码以去噪。  
  
        参数:  
        - x_fft: 输入信号的傅里叶变换，形状为 (B, H, W, 2)，其中2表示复数实部和虚部。  
  
        返回:  
        - adaptive_mask: 一个与x_fft形状相同的掩码，低频分量被标记为0（去除），其他为1（保留）。  
        """  
        B, _, _ = x_fft.shape  # 假设x_fft是复数，因此有4个维度  
  
        # 计算频率域的能量  
        energy = torch.abs(x_fft).pow(2).sum(dim=-1)  # 对复数取模后平方，然后沿复数维度求和  
  
        # 扁平化能量并计算中位数  
        flat_energy = energy.view(B, -1)  
        median_energy = flat_energy.median(dim=1, keepdim=True)[0]  
  
        # 标准化能量  
        normalized_energy = energy / (median_energy + 1e-6)  
  
        # 计算阈值，低于此阈值的能量被认为是低频分量  
        # print(f"threshold_param_low_freq:: {self.threshold_param}")
        threshold = torch.quantile(normalized_energy, self.threshold_param)  
        low_frequencies = normalized_energy <= threshold  # 注意这里是 <=  
        # print(f"Low frequency threshold: {threshold}")
        # print(f"Number of low frequency components: {low_frequencies.sum()}")  
        # print(f"normalized_energy shape: {normalized_energy.shape}")
        # print(f"low_frequencies_normalized_energy:{normalized_energy}")
        # 初始化自适应掩码  
        adaptive_mask = torch.ones_like(x_fft, device=x_fft.device)  # 使用1填充掩码，因为我们要去除低频  
        # adaptive_mask[..., :2] = 0  # 假设x_fft的最后一个维度是复数维度，我们不希望修改它  
        # adaptive_mask[low_frequencies[..., None].expand_as(x_fft[..., :2])] = 0  # 将低频分量设为0  
        adaptive_mask[low_frequencies] = 1
        # 注意：由于我们处理的是复数，我们实际上只需要修改掩码的一部分（实部或虚部），  
        # 但由于我们只想标记哪些频率分量被去除，这里选择修改整个x_fft形状相同的掩码。  
        # 在实际应用中，你可能需要调整这部分以符合你的具体需求。  
  
        return adaptive_mask  
  
# # 使用示例  
# if __name__ == "__main__":  
#     # 假设我们有一个复数的FFT输出  
#     B, H, W = 1, 10, 10  
#     x_fft = torch.randn(B, H, W, 2, dtype=torch.complex64)  # 随机生成复数FFT数据  
#     print(f"x_fft shape: {x_fft.shape}")
#     print(f"x_fft dtype: {x_fft.dtype}")  
#     print(f"x_fft device: {x_fft.device}")  
#     print(f"x_fft real part: {x_fft[..., 0]}")  # 打印实部  
#     print(f"x_fft imag part: {x_fft[..., 1]}")  # 打印虚部      
#     print(f"x_fft abs part: {torch.abs(x_fft)}")  # 打印复数模  
#     print(f"x_fft abs^2 part: {torch.abs(x_fft).pow(2)}")  # 打印复数模平方  
#     denoiser = LowFrequencyDenoiser(threshold_param=0.2)  # 创建一个低频去噪器  
#     mask = denoiser.create_adaptive_low_freq_mask(x_fft)  # 创建低频掩码  
#     print(f"mask shape: {mask.shape}")  # 打印掩码形状  
#     print(f"mask dtype: {mask.dtype}")  # 打印掩码数据类型  
#     print(f"mask device: {mask.device}")  # 打印掩码所在设备  
#     print(f"mask values: {mask}")  # 打印掩码的值