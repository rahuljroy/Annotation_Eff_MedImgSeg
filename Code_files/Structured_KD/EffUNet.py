import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from efficientnet_pytorch import EfficientNet
from utils import print_model_parm_nums


# Utility Functions for the model
def double_conv(in_,out_,drop): # Double convolution layer for decoder 
	conv = nn.Sequential(
		nn.Conv2d(in_,out_,kernel_size=3,padding=(1,1)),
		nn.ReLU(inplace=True),
		nn.Conv2d(out_,out_,kernel_size=3,padding=(1,1)),
		nn.ReLU(inplace=True),
		nn.Dropout(drop)
		)
	return conv

def crop(tensor,target_tensor): # Crop tensor to target tensor size
	target_shape1, target_shape2 = target_tensor.shape[2], target_tensor.shape[3]
	return T.CenterCrop(target_shape, target_shape2)(tensor)


# Hook functions to get values of intermediate layers for cross connection
hook_values = []
def hook(_, input, output):
	global hook_values
	hook_values.append(output) # stores values of each layers in hook_values

indices = []
shapes = []
def init_hook(model,device):
	
	global shapes, indices, hook_values
	indices = []
	shapes = []
	hook_values = []

	for i in range(len(model._blocks)):
		model._blocks[i].register_forward_hook(hook) #register hooks
	
	image = torch.rand([1,1,64,64])
	image = image.cuda()
	out = model(image) # generate hook values to get shapes
	
	shape = [i.shape for i in hook_values] # get shape of all layers
	# print(shape)
	for i in range(len(shape)-1):
		if shape[i][2]!=shape[i+1][2]: # get indices of layers only where output dimension change
			indices.append(i)
	indices.append(len(shape)-1) # get last layer index
	
	shapes = [shape[i] for i in indices] # get shapes of required layers
	shapes = shapes[::-1]  

encoder_out = []
def epoch_hook(model, image):
	global encoder_out, indices, hook_values
	hook_values = []

	out = model(image) # generate layer outputs with current image
	encoder_out = [hook_values[i] for i in indices] # get layer outputs for selected indices


class EffUNet(nn.Module):

	def __init__(self,model='b0',out_channels=3,dropout=0.1,freeze_backbone=True,pretrained=False,device='cuda'):
		super(EffUNet,self).__init__()
		global layers, shapes

		if model not in set(['b0','b1','b2','b3','b4','b5','b6','b7']):
			raise Exception(f'{model} unavailable.')
		if pretrained:
			self.encoder = EfficientNet.from_pretrained(f'efficientnet-{model}')
			self.encoder._change_in_channels(in_channels = 1)
		else:
			self.encoder = EfficientNet.from_name(f'efficientnet-{model}')
			self.encoder._change_in_channels(in_channels = 1)

		# Disable non required layers by replacing them with identity
		self.encoder._conv_head=torch.nn.Identity()
		self.encoder._bn1=torch.nn.Identity()
		self.encoder._avg_pooling=torch.nn.Identity()
		self.encoder._dropout=torch.nn.Identity()
		self.encoder._fc=torch.nn.Identity()
		self.encoder._swish=torch.nn.Identity()

		if isinstance(device, str):
			self.device = torch.device(device)
		else:
			self.device = device
		self.encoder.cuda()
		self.encoder._conv_stem.stride=1 # change stride of first layer from 2 to 1 to increase o/p size
		self.encoder._conv_stem.kernel_size=(1,1) # 

		# freeze encoder
		if freeze_backbone:
			for param in self.encoder.parameters():
				param.requires_grad = False

		# register hooks & get shapes
		init_hook(self.encoder,self.device)

		# Building decoder
		self.decoder = torch.nn.modules.container.ModuleList()
		for i in range(len(shapes)-1):
			self.decoder.append(torch.nn.modules.container.ModuleList())
			self.decoder[i].append(nn.ConvTranspose2d(shapes[i][1],shapes[i][1]-shapes[i+1][1],kernel_size=2,stride=2)).cuda()
			self.decoder[i].append(double_conv(shapes[i][1],shapes[i+1][1],dropout)).cuda()

		#output layer
		self.out = nn.Conv2d(shapes[-1][1],out_channels,kernel_size=1).cuda()

	def forward(self, image):
		global layers
		
		h=image.shape[2]
		w=image.shape[3]
		if h%8!=0 or w%8!=0:
			new_h = round(h/8)*8
			new_w = round(w/8)*8
			image =  T.Resize((new_h,new_w))(image)

		# Encoder
		epoch_hook(self.encoder, image) # required outputs accumulate in "encoder_out"

		#Decoder
		x = encoder_out.pop()
		enc = x
		# print(x.shape)
		for i in range(len(self.decoder)):
			x = self.decoder[i][0](x) # conv transpose
			# print(x.shape)
			prev = encoder_out.pop()
			# print(x.shape, prev.shape)
			if x.shape[2] < prev.shape[2]:
				x = F.pad(x, (1,0,1,0))
				# prev = crop(prev, x) # croping for cross connection
			elif x.shape[2] > prev.shape[2]:
				prev = F.pad(prev,(1,0,1,0))
				# x = crop(x, prev) # croping for cross connection
			# print('Pad: ', x.shape, prev.shape)
			prev = torch.cat([x,prev],axis=1) # concatenating 
			# print(x.shape)
			x = self.decoder[i][1](prev) # double conv
			if i ==0:
				dec = x
		
		#out
		x = self.out(x)
		return [x, enc, dec]

# img = torch.rand([1,1,64,64]).cuda()
# model = EffUNet()
# print_model_parm_nums(model, 'model')
# # print(model)
# out = model(img)
# print(out.shape)