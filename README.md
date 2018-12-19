# pytorch-partial-conv-based-padding
Simple Code Implementation of ["Partial Convolution based Padding"](https://arxiv.org/abs/1811.11718) with ResNet architecture using PyTorch. 

For simplicity, i write codes in `ipynb`. So, you can easliy test my code.

*Last update : 2018/12/19*

## Contributor
* hoya012

## Requirements
Python 3.5
```
numpy
matplotlib
torch=1.0.0
torchvision
```

## Usage
You only run `PCB_padding_ResNet.ipynb`.

If you change ResNet Architecture, try this.

``` 
net = ResNet(BasicBlock, [2, 2, 2, 2], 10) #ResNet-18
net = ResNet(BasicBlock, [3, 4, 6, 3], 10) #ResNet-34
net = ResNet(Bottleneck, [3, 4, 6, 3], 10) #ResNet-50
net = ResNet(Bottleneck, [3, 4, 23, 3], 10) #ResNet-101
net = ResNet(Bottleneck, [3, 8, 36, 3], 10) #ResNet-152
```

## Partial Convolution based Padding Implementation
You can only use `partial=True` if you want to use partial_conv_based_padding, else, same with torch.nn.Conv2d.

```
class Conv2d_partial(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, partial=False):
        super(Conv2d_partial, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
     
        self.partial = partial
        
    def forward(self, input):
        if self.partial:
            self.padding = 0

            pad_val = (self.kernel_size[0] - 1) // 2
            if pad_val > 0:
                if (self.kernel_size[0] - self.stride[0]) % 2 == 0:
                    pad_top = pad_val
                    pad_bottom = pad_val
                    pad_left = pad_val
                    pad_right = pad_val
                else:
                    pad_top = pad_val
                    pad_bottom = self.kernel_size[0] - self.stride[0] - pad_top
                    pad_left = pad_val
                    pad_right = self.kernel_size[0] - self.stride[0] - pad_left
                
                p0 = torch.ones_like(input) 
                p0 = p0.sum()
                                
                input = F.pad(input, (pad_left, pad_right, pad_top, pad_bottom) , mode='constant', value=0)
                
                p1 = torch.ones_like(input) 
                p1 = p1.sum()

                ratio = torch.div(p1, p0 + 1e-8) 
                input = torch.mul(input, ratio)  
            
        return F.conv2d(input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
```

## Reference
- [TensorFlow implementation](https://github.com/taki0112/partial_conv-Tensorflow)
- [PyTorch ResNet implementation](https://github.com/jack-willturner/batchnorm-pruning/blob/master/models/resnet.py)
