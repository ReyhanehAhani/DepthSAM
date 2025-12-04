import re

path = "/home/ram112/projects/def-jieliang/ram112/All_DEPTHCLIP/DepthCLIP/DepthCLIP_code/monoclip.py"
print("Applying final resolution patch (1024 -> 1008)...")

with open(path, 'r') as f:
    content = f.read()

# 1. تغییر سایز دامی در MonoCLIP.__init__ (خط 139)
content = re.sub(r'torch\.randn\(1, 3, 1024, 1024\)', 
                 r'torch.randn(1, 3, 1008, 1008)', 
                 content, 
                 count=1)

# 2. تغییر شرط چک کردن سایز در SAM3Encoder.forward
content = re.sub(r'!= \(1024, 1024\)', 
                 r'!= (1008, 1008)', 
                 content, 
                 count=1)

# 3. تغییر سایز هدف F.interpolate در SAM3Encoder.forward
content = re.sub(r'size=\(1024, 1024\)', 
                 r'size=(1008, 1008)', 
                 content, 
                 count=1)

with open(path, 'w') as f:
    f.write(content)
    
print("✅ monoclip.py updated to 1008x1008 resolution!")

