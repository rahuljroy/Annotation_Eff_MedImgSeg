import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.autograd import Variable
import os
from models.unet_3_enc import UNet2D
from torch.utils.data.sampler import SubsetRandomSampler
import copy


import numpy as np

def get_prior(model, img):
    with torch.no_grad():
        outputs = model(img)
        outputs = torch.argmax(outputs, dim=1)
    return outputs
    

class Prior_MSD(Dataset):
    def __init__(self, root, imgdir, labeldir, labeldir_left, labeldir_right, prior_list, device, slices=1, train_ratio = 1):
        model_fn = []
        prior_type = ['comp', 'left', 'right']
        for i in prior_type:
            for x in prior_list:
                if i in x:
                    model_fn.append('../prior_models/best_model/' + x)

        self.slices = slices
        assert self.slices % 2 == 1, "slices must be odd!"

        es = int(self.slices/2) # floor operation

        img_fn = [f for f in sorted(os.listdir(root+imgdir)) if f.endswith('.npy')]
        seg_fn = [f for f in sorted(os.listdir(root+labeldir)) if f.endswith('.npy')]
        seg_left = [f for f in sorted(os.listdir(root+labeldir_left)) if f.endswith('.npy')]
        seg_right = [f for f in sorted(os.listdir(root+labeldir_right)) if f.endswith('.npy')]

        n_classes = []
        for i in prior_type:
            if i == 'comp':
                n_classes.append(3)
            else:
                n_classes.append(2)
        
        models = []

        for i in range(len(model_fn)):
            prior_model = UNet2D(n_classes[i], 1).cuda()
            prior_model.load_state_dict(torch.load(model_fn[i]))
            models.append(prior_model)

        self.datalist = []
        
        for i, (img_file, seg_file, seg_left_file, seg_right_file) in enumerate(zip(img_fn, seg_fn, seg_left, seg_right)):

            filename_img = os.path.join(root, imgdir, img_file)
            filename_seg_full = os.path.join(root, labeldir, seg_file)
            filename_seg_left = os.path.join(root, labeldir_left, seg_left_file)
            filename_seg_right = os.path.join(root, labeldir_right, seg_right_file)

            assert img_file.replace('.','_').split('_')[1] == seg_file.replace('.','_').split('_')[1], "Image and label files not from the same patient"
            assert img_file.replace('.','_').split('_')[1] == seg_left_file.replace('.','_').split('_')[1], "Image and label files not from the same patient"
            assert img_file.replace('.','_').split('_')[1] == seg_right_file.replace('.','_').split('_')[1], "Image and label files not from the same patient"
            
            seg_img_data = np.load(filename_seg_full)
            seg_left = np.load(filename_seg_left)
            seg_right = np.load(filename_seg_right)
            img_data = np.load(filename_img)

            output = np.unique(seg_img_data, return_counts=True)
            counts = output[1]
            if i==0:
                weights = counts
            else:
                weights += counts


            assert img_data.shape == seg_img_data.shape, "Image and Labels have different shapes"
            assert img_data.shape == seg_left.shape, "Image and Labels have different shapes"
            assert img_data.shape == seg_right.shape, "Image and Labels have different shapes"
            
            for j in range(img_data.shape[2]):
                image2d = torch.from_numpy(img_data[:,:,j]).unsqueeze(dim=0).type(torch.FloatTensor)
                label2d = torch.from_numpy(seg_img_data[:,:,j]).unsqueeze(dim=0).type(torch.LongTensor)
                label_left = torch.from_numpy(seg_left[:,:,j]).unsqueeze(dim=0).type(torch.LongTensor)
                label_right = torch.from_numpy(seg_right[:,:,j]).unsqueeze(dim=0).type(torch.LongTensor)
                
                input_img = Variable(image2d).unsqueeze(dim=1).cuda()
                output_full = get_prior(models[0], input_img).type(torch.FloatTensor)
                output_left = get_prior(models[1], input_img).type(torch.FloatTensor)
                output_right = get_prior(models[2], input_img).type(torch.FloatTensor)
                cat_full = torch.cat((image2d, output_full), axis=0)
                cat_left = torch.cat((image2d, output_left), axis=0)
                cat_right = torch.cat((image2d, output_right), axis=0)

                if (len(torch.unique(label2d))>=2):
                        self.datalist.append([ cat_full, cat_left, cat_right, image2d , label2d , label_left, label_right])
        self.weights = torch.Tensor(weights)
        self.weights = torch.sum(self.weights) - self.weights
        self.weights = self.weights / torch.sum(self.weights)

    def __getitem__(self, index):
        [ cat_full, cat_left, cat_right, img, seg, seg_left, seg_right] = self.datalist[index]
        seg_full = copy.deepcopy(seg)
        seg_full[seg_full==2]=1
        cat_hip = copy.deepcopy(cat_full)
        cat_hip[cat_hip==2]=1
        return {'cat_hip' : cat_hip, 'cat_full' : cat_full, 'cat_left' : cat_left, 'cat_right' : cat_right, \
            'img' : img, 'seg' : seg, 'seg_left' : seg_left, 'seg_right' : seg_right, 'seg_full' : seg_full}

    def __len__(self):
        return len(self.datalist)



class Mean_Prior_MSD(Dataset):
    def __init__(self, root, imgdir, labeldir, labeldir_left, labeldir_right, prior_list, device, slices=1, train_ratio = 1):
        model_fn = []
        prior_type = ['comp', 'left', 'right']
        for i in prior_type:
            for x in prior_list:
                if i in x:
                    model_fn.append('../prior_models/best_model/' + x)

        self.slices = slices
        assert self.slices % 2 == 1, "slices must be odd!"

        es = int(self.slices/2) # floor operation

        img_fn = [f for f in sorted(os.listdir(root+imgdir)) if f.endswith('.npy')]
        seg_fn = [f for f in sorted(os.listdir(root+labeldir)) if f.endswith('.npy')]
        seg_left = [f for f in sorted(os.listdir(root+labeldir_left)) if f.endswith('.npy')]
        seg_right = [f for f in sorted(os.listdir(root+labeldir_right)) if f.endswith('.npy')]

        n_classes = []
        for i in prior_type:
            if i == 'comp':
                n_classes.append(3)
            else:
                n_classes.append(2)
        
        prior = np.load('../mean_priors/meanSP.npy')
        
        prior_left = copy.deepcopy(prior)
        prior_right = copy.deepcopy(prior)
        prior_full = copy.deepcopy(prior)
        prior_left[prior_left==2] = 0
        prior = torch.from_numpy(prior).unsqueeze(dim=0).type(torch.FloatTensor)
        prior_left = torch.from_numpy(prior_left).unsqueeze(dim=0).type(torch.FloatTensor)
        
        prior_right[prior_right==1] = 0
        prior_right[prior_right==2] = 1
        prior_right = torch.from_numpy(prior_right).unsqueeze(dim=0).type(torch.FloatTensor)
        
        prior_full[prior_full==2] = 1
        prior_full = torch.from_numpy(prior_full).unsqueeze(dim=0).type(torch.FloatTensor)

        self.datalist = []
        
        for i, (img_file, seg_file, seg_left_file, seg_right_file) in enumerate(zip(img_fn, seg_fn, seg_left, seg_right)):

            filename_img = os.path.join(root, imgdir, img_file)
            filename_seg_full = os.path.join(root, labeldir, seg_file)
            filename_seg_left = os.path.join(root, labeldir_left, seg_left_file)
            filename_seg_right = os.path.join(root, labeldir_right, seg_right_file)

            assert img_file.replace('.','_').split('_')[1] == seg_file.replace('.','_').split('_')[1], "Image and label files not from the same patient"
            assert img_file.replace('.','_').split('_')[1] == seg_left_file.replace('.','_').split('_')[1], "Image and label files not from the same patient"
            assert img_file.replace('.','_').split('_')[1] == seg_right_file.replace('.','_').split('_')[1], "Image and label files not from the same patient"
            
            seg_img_data = np.load(filename_seg_full)
            seg_left = np.load(filename_seg_left)
            seg_right = np.load(filename_seg_right)
            img_data = np.load(filename_img)

            output = np.unique(seg_img_data, return_counts=True)
            counts = output[1]
            if i==0:
                weights = counts
            else:
                weights += counts


            assert img_data.shape == seg_img_data.shape, "Image and Labels have different shapes"
            assert img_data.shape == seg_left.shape, "Image and Labels have different shapes"
            assert img_data.shape == seg_right.shape, "Image and Labels have different shapes"
            
            for j in range(img_data.shape[2]):
                image2d = torch.from_numpy(img_data[:,:,j]).unsqueeze(dim=0).type(torch.FloatTensor)
                label2d = torch.from_numpy(seg_img_data[:,:,j]).unsqueeze(dim=0).type(torch.LongTensor)
                label_left = torch.from_numpy(seg_left[:,:,j]).unsqueeze(dim=0).type(torch.LongTensor)
                label_right = torch.from_numpy(seg_right[:,:,j]).unsqueeze(dim=0).type(torch.LongTensor)
                
                cat_full = torch.cat((image2d, prior), axis=0)
                cat_left = torch.cat((image2d, prior_left), axis=0)
                cat_right = torch.cat((image2d, prior_right), axis=0)

                if (len(torch.unique(label2d))>=2):
                        self.datalist.append([ cat_full, cat_left, cat_right, image2d , label2d , label_left, label_right])
        self.weights = torch.Tensor(weights)
        self.weights = torch.sum(self.weights) - self.weights
        self.weights = self.weights / torch.sum(self.weights)

    def __getitem__(self, index):
        [ cat_full, cat_left, cat_right, img, seg, seg_left, seg_right] = self.datalist[index]
        seg_full = copy.deepcopy(seg)
        seg_full[seg_full==2]=1
        cat_hip = copy.deepcopy(cat_full)
        cat_hip[cat_hip==2]=1
        return {'cat_hip' : cat_hip, 'cat_full' : cat_full, 'cat_left' : cat_left, 'cat_right' : cat_right, \
            'img' : img, 'seg' : seg, 'seg_left' : seg_left, 'seg_right' : seg_right, 'seg_full' : seg_full}

    def __len__(self):
        return len(self.datalist)


class GT_Prior_MSD(Dataset):
    def __init__(self, root, imgdir, labeldir, labeldir_left, labeldir_right, prior_list, device, slices=1, train_ratio = 1):
        model_fn = []
        prior_type = ['comp', 'left', 'right']
        for i in prior_type:
            for x in prior_list:
                if i in x:
                    model_fn.append('../prior_models/best_model/' + x)

        self.slices = slices
        assert self.slices % 2 == 1, "slices must be odd!"

        es = int(self.slices/2) # floor operation

        img_fn = [f for f in sorted(os.listdir(root+imgdir)) if f.endswith('.npy')]
        seg_fn = [f for f in sorted(os.listdir(root+labeldir)) if f.endswith('.npy')]
        seg_left = [f for f in sorted(os.listdir(root+labeldir_left)) if f.endswith('.npy')]
        seg_right = [f for f in sorted(os.listdir(root+labeldir_right)) if f.endswith('.npy')]

        n_classes = []
        for i in prior_type:
            if i == 'comp':
                n_classes.append(3)
            else:
                n_classes.append(2)
        
        prior = np.load('../mean_priors/meanSP.npy')
        
        prior_left = copy.deepcopy(prior)
        prior_right = copy.deepcopy(prior)
        prior_full = copy.deepcopy(prior)
        prior_left[prior_left==2] = 0
        prior = torch.from_numpy(prior).unsqueeze(dim=0).type(torch.FloatTensor)
        prior_left = torch.from_numpy(prior_left).unsqueeze(dim=0).type(torch.FloatTensor)
        
        prior_right[prior_right==1] = 0
        prior_right[prior_right==2] = 1
        prior_right = torch.from_numpy(prior_right).unsqueeze(dim=0).type(torch.FloatTensor)
        
        prior_full[prior_full==2] = 1
        prior_full = torch.from_numpy(prior_full).unsqueeze(dim=0).type(torch.FloatTensor)

        self.datalist = []
        
        for i, (img_file, seg_file, seg_left_file, seg_right_file) in enumerate(zip(img_fn, seg_fn, seg_left, seg_right)):

            filename_img = os.path.join(root, imgdir, img_file)
            filename_seg_full = os.path.join(root, labeldir, seg_file)
            filename_seg_left = os.path.join(root, labeldir_left, seg_left_file)
            filename_seg_right = os.path.join(root, labeldir_right, seg_right_file)

            assert img_file.replace('.','_').split('_')[1] == seg_file.replace('.','_').split('_')[1], "Image and label files not from the same patient"
            assert img_file.replace('.','_').split('_')[1] == seg_left_file.replace('.','_').split('_')[1], "Image and label files not from the same patient"
            assert img_file.replace('.','_').split('_')[1] == seg_right_file.replace('.','_').split('_')[1], "Image and label files not from the same patient"
            
            seg_img_data = np.load(filename_seg_full)
            seg_left = np.load(filename_seg_left)
            seg_right = np.load(filename_seg_right)
            img_data = np.load(filename_img)

            output = np.unique(seg_img_data, return_counts=True)
            counts = output[1]
            if i==0:
                weights = counts
            else:
                weights += counts


            assert img_data.shape == seg_img_data.shape, "Image and Labels have different shapes"
            assert img_data.shape == seg_left.shape, "Image and Labels have different shapes"
            assert img_data.shape == seg_right.shape, "Image and Labels have different shapes"
            
            for j in range(img_data.shape[2]):
                image2d = torch.from_numpy(img_data[:,:,j]).unsqueeze(dim=0).type(torch.FloatTensor)
                label2d = torch.from_numpy(seg_img_data[:,:,j]).unsqueeze(dim=0).type(torch.LongTensor)
                label_left = torch.from_numpy(seg_left[:,:,j]).unsqueeze(dim=0).type(torch.LongTensor)
                label_right = torch.from_numpy(seg_right[:,:,j]).unsqueeze(dim=0).type(torch.LongTensor)
                
                cat_full = torch.cat((image2d, label2d), axis=0)
                cat_left = torch.cat((image2d, label_left), axis=0)
                cat_right = torch.cat((image2d, label_right), axis=0)

                if (len(torch.unique(label2d))>=2):
                        self.datalist.append([ cat_full, cat_left, cat_right, image2d , label2d , label_left, label_right])
        self.weights = torch.Tensor(weights)
        self.weights = torch.sum(self.weights) - self.weights
        self.weights = self.weights / torch.sum(self.weights)

    def __getitem__(self, index):
        [ cat_full, cat_left, cat_right, img, seg, seg_left, seg_right] = self.datalist[index]
        seg_full = copy.deepcopy(seg)
        seg_full[seg_full==2]=1
        cat_hip = copy.deepcopy(cat_full)
        cat_hip[cat_hip==2]=1
        return {'cat_hip' : cat_hip, 'cat_full' : cat_full, 'cat_left' : cat_left, 'cat_right' : cat_right, \
            'img' : img, 'seg' : seg, 'seg_left' : seg_left, 'seg_right' : seg_right, 'seg_full' : seg_full}

    def __len__(self):
        return len(self.datalist)

class Mean_Prior_Spine(Dataset):
    def __init__(self, root, imgdir, labeldir, labeldir_left, labeldir_right, device):
        self.root = root
        self.imgdir = imgdir
        self.labeldir = labeldir
        self.labeldir_left = labeldir_left
        self.labeldir_right = labeldir_right
        self.img_fn = [f for f in sorted(os.listdir(root+imgdir)) if f.endswith('.npy')]
        self.seg_fn = [f for f in sorted(os.listdir(root+labeldir)) if f.endswith('.npy')]
        self.seg_left = [f for f in sorted(os.listdir(root+labeldir_left)) if f.endswith('.npy')]
        self.seg_right = [f for f in sorted(os.listdir(root+labeldir_right)) if f.endswith('.npy')]
        
        prior = np.load('../mean_priors_spine/meanSP.npy')
        self.prior = torch.from_numpy(prior).unsqueeze(dim=0).type(torch.FloatTensor)
        prior_left = copy.deepcopy(prior)
        prior_right = copy.deepcopy(prior)
        prior_full = copy.deepcopy(prior)
        prior_left[prior_left==2] = 0
        prior = torch.from_numpy(prior).unsqueeze(dim=0).type(torch.FloatTensor)
        self.prior_left = torch.from_numpy(prior_left).unsqueeze(dim=0).type(torch.FloatTensor)
        
        prior_right[prior_right==1] = 0
        prior_right[prior_right==2] = 1
        self.prior_right = torch.from_numpy(prior_right).unsqueeze(dim=0).type(torch.FloatTensor)
        
        prior_full[prior_full==2] = 1
        self.prior_full = torch.from_numpy(prior_full).unsqueeze(dim=0).type(torch.FloatTensor)

        # self.datalist = []
        
        # for i, (img_file, seg_file, seg_left_file, seg_right_file) in enumerate(zip(img_fn, seg_fn, seg_left, seg_right)):

        #     filename_img = os.path.join(root, imgdir, img_file)
        #     filename_seg_full = os.path.join(root, labeldir, seg_file)
        #     filename_seg_left = os.path.join(root, labeldir_left, seg_left_file)
        #     filename_seg_right = os.path.join(root, labeldir_right, seg_right_file)

        #     assert img_file.replace('.','_').split('_')[1] == seg_file.replace('.','_').split('_')[1], "Image and label files not from the same patient"
        #     assert img_file.replace('.','_').split('_')[1] == seg_left_file.replace('.','_').split('_')[1], "Image and label files not from the same patient"
        #     assert img_file.replace('.','_').split('_')[1] == seg_right_file.replace('.','_').split('_')[1], "Image and label files not from the same patient"
            
        #     seg_img_data = np.load(filename_seg_full)
        #     seg_left = np.load(filename_seg_left)
        #     seg_right = np.load(filename_seg_right)
        #     img_data = np.load(filename_img)

        #     output = np.unique(seg_img_data, return_counts=True)
        #     counts = output[1]
        #     if i==0:
        #         weights = counts
        #     else:
        #         weights += counts


        #     assert img_data.shape == seg_img_data.shape, "Image and Labels have different shapes"
        #     assert img_data.shape == seg_left.shape, "Image and Labels have different shapes"
        #     assert img_data.shape == seg_right.shape, "Image and Labels have different shapes"
            
        #     for j in range(img_data.shape[2]):
        #         image2d = torch.from_numpy(img_data[:,:,j]).unsqueeze(dim=0).type(torch.FloatTensor)
        #         label2d = torch.from_numpy(seg_img_data[:,:,j]).unsqueeze(dim=0).type(torch.LongTensor)
        #         label_left = torch.from_numpy(seg_left[:,:,j]).unsqueeze(dim=0).type(torch.LongTensor)
        #         label_right = torch.from_numpy(seg_right[:,:,j]).unsqueeze(dim=0).type(torch.LongTensor)
                
        #         cat_full = torch.cat((image2d, prior), axis=0)
        #         cat_left = torch.cat((image2d, prior_left), axis=0)
        #         cat_right = torch.cat((image2d, prior_right), axis=0)

        #         if (len(torch.unique(label2d))>=2):
        #                 self.datalist.append([ cat_full, cat_left, cat_right, image2d , label2d , label_left, label_right])
        # self.weights = torch.Tensor(weights)
        # self.weights = torch.sum(self.weights) - self.weights
        # self.weights = self.weights / torch.sum(self.weights)

    def __getitem__(self, index):
        filename_img = os.path.join(self.root, self.imgdir, str(index+1)+'.npy')
        img =  torch.from_numpy(np.load(filename_img)).unsqueeze(dim=0).type(torch.FloatTensor)
        filename_seg = os.path.join(self.root, self.labeldir, str(index+1)+'.npy')
        seg =  torch.from_numpy(np.load(filename_seg)).unsqueeze(dim=0).type(torch.FloatTensor)
        filename_seg_left = os.path.join(self.root, self.labeldir_left, str(index+1)+'.npy')
        seg_left =  torch.from_numpy(np.load(filename_seg_left)).unsqueeze(dim=0).type(torch.FloatTensor)
        filename_seg_right = os.path.join(self.root, self.labeldir_right, str(index+1)+'.npy')
        seg_right =  torch.from_numpy(np.load(filename_seg_right)).unsqueeze(dim=0).type(torch.FloatTensor)
        seg_full = copy.deepcopy(seg)
        seg_full[seg_full==2]=1
        cat_full = torch.cat((img, self.prior), axis=0)
        cat_left = torch.cat((img, self.prior_left), axis=0)
        cat_right = torch.cat((img, self.prior_right), axis=0)
        cat_hip = copy.deepcopy(cat_full)
        cat_hip[cat_hip==2]=1
        # [ cat_full, cat_left, cat_right, img, seg, seg_left, seg_right] = self.datalist[index]
        
        return {'cat_hip' : cat_hip, 'cat_full' : cat_full, 'cat_left' : cat_left, 'cat_right' : cat_right, \
            'img' : img, 'seg' : seg, 'seg_left' : seg_left, 'seg_right' : seg_right, 'seg_full' : seg_full}

    def __len__(self):
        return len(self.img_fn)


class Prior_Spine(Dataset):
    def __init__(self, root, imgdir, labeldir, labeldir_left, labeldir_right, device):
        self.root = root
        self.imgdir = imgdir
        self.labeldir = labeldir
        self.labeldir_left = labeldir_left
        self.labeldir_right = labeldir_right
        self.img_fn = [f for f in sorted(os.listdir(root+imgdir)) if f.endswith('.npy')]
        self.seg_fn = [f for f in sorted(os.listdir(root+labeldir)) if f.endswith('.npy')]
        self.seg_left = [f for f in sorted(os.listdir(root+labeldir_left)) if f.endswith('.npy')]
        self.seg_right = [f for f in sorted(os.listdir(root+labeldir_right)) if f.endswith('.npy')]
        
        # self.prior_dir = '/home/rahul/MIG/M.Tech/Code/Segmentation_DA_MSD/data/SpineCT_processed/train/prior/'
        self.prior_dir = '../data/SpineCT_processed/train/prior/'
        self.prior_fn = [f for f in sorted(os.listdir(self.prior_dir+'complete/')) if f.endswith('.npy')]
        # self.prior = torch.from_numpy(prior).unsqueeze(dim=0).type(torch.FloatTensor)
        # prior_left = copy.deepcopy(prior)
        # prior_right = copy.deepcopy(prior)
        # prior_full = copy.deepcopy(prior)
        # prior_left[prior_left==2] = 0
        # prior = torch.from_numpy(prior).unsqueeze(dim=0).type(torch.FloatTensor)
        # self.prior_left = torch.from_numpy(prior_left).unsqueeze(dim=0).type(torch.FloatTensor)
        
        # prior_right[prior_right==1] = 0
        # prior_right[prior_right==2] = 1
        # self.prior_right = torch.from_numpy(prior_right).unsqueeze(dim=0).type(torch.FloatTensor)
        
        # prior_full[prior_full==2] = 1
        # self.prior_full = torch.from_numpy(prior_full).unsqueeze(dim=0).type(torch.FloatTensor)

        # self.datalist = []
        
        # for i, (img_file, seg_file, seg_left_file, seg_right_file) in enumerate(zip(img_fn, seg_fn, seg_left, seg_right)):

        #     filename_img = os.path.join(root, imgdir, img_file)
        #     filename_seg_full = os.path.join(root, labeldir, seg_file)
        #     filename_seg_left = os.path.join(root, labeldir_left, seg_left_file)
        #     filename_seg_right = os.path.join(root, labeldir_right, seg_right_file)

        #     assert img_file.replace('.','_').split('_')[1] == seg_file.replace('.','_').split('_')[1], "Image and label files not from the same patient"
        #     assert img_file.replace('.','_').split('_')[1] == seg_left_file.replace('.','_').split('_')[1], "Image and label files not from the same patient"
        #     assert img_file.replace('.','_').split('_')[1] == seg_right_file.replace('.','_').split('_')[1], "Image and label files not from the same patient"
            
        #     seg_img_data = np.load(filename_seg_full)
        #     seg_left = np.load(filename_seg_left)
        #     seg_right = np.load(filename_seg_right)
        #     img_data = np.load(filename_img)

        #     output = np.unique(seg_img_data, return_counts=True)
        #     counts = output[1]
        #     if i==0:
        #         weights = counts
        #     else:
        #         weights += counts


        #     assert img_data.shape == seg_img_data.shape, "Image and Labels have different shapes"
        #     assert img_data.shape == seg_left.shape, "Image and Labels have different shapes"
        #     assert img_data.shape == seg_right.shape, "Image and Labels have different shapes"
            
        #     for j in range(img_data.shape[2]):
        #         image2d = torch.from_numpy(img_data[:,:,j]).unsqueeze(dim=0).type(torch.FloatTensor)
        #         label2d = torch.from_numpy(seg_img_data[:,:,j]).unsqueeze(dim=0).type(torch.LongTensor)
        #         label_left = torch.from_numpy(seg_left[:,:,j]).unsqueeze(dim=0).type(torch.LongTensor)
        #         label_right = torch.from_numpy(seg_right[:,:,j]).unsqueeze(dim=0).type(torch.LongTensor)
                
        #         cat_full = torch.cat((image2d, prior), axis=0)
        #         cat_left = torch.cat((image2d, prior_left), axis=0)
        #         cat_right = torch.cat((image2d, prior_right), axis=0)

        #         if (len(torch.unique(label2d))>=2):
        #                 self.datalist.append([ cat_full, cat_left, cat_right, image2d , label2d , label_left, label_right])
        # self.weights = torch.Tensor(weights)
        # self.weights = torch.sum(self.weights) - self.weights
        # self.weights = self.weights / torch.sum(self.weights)

    def __getitem__(self, index):
        prior_left_fn = os.path.join(self.prior_dir, 'left', self.prior_fn[index])
        prior_left = torch.from_numpy(np.load(prior_left_fn)).unsqueeze(dim=0).type(torch.FloatTensor)
        prior_right_fn = os.path.join(self.prior_dir, 'right', self.prior_fn[index])
        prior_right = torch.from_numpy(np.load(prior_right_fn)).unsqueeze(dim=0).type(torch.FloatTensor)
        prior_full_fn = os.path.join(self.prior_dir, 'full', self.prior_fn[index])
        prior_full = torch.from_numpy(np.load(prior_full_fn)).unsqueeze(dim=0).type(torch.FloatTensor)
        prior_comp_fn = os.path.join(self.prior_dir, 'complete', self.prior_fn[index])
        prior_comp = torch.from_numpy(np.load(prior_comp_fn)).unsqueeze(dim=0).type(torch.FloatTensor)
        filename_img = os.path.join(self.root, self.imgdir, self.prior_fn[index])
        img =  torch.from_numpy(np.load(filename_img)).unsqueeze(dim=0).type(torch.FloatTensor)
        filename_seg = os.path.join(self.root, self.labeldir,self.prior_fn[index])
        seg =  torch.from_numpy(np.load(filename_seg)).unsqueeze(dim=0).type(torch.FloatTensor)
        filename_seg_left = os.path.join(self.root, self.labeldir_left, self.prior_fn[index])
        seg_left =  torch.from_numpy(np.load(filename_seg_left)).unsqueeze(dim=0).type(torch.FloatTensor)
        filename_seg_right = os.path.join(self.root, self.labeldir_right, self.prior_fn[index])
        seg_right =  torch.from_numpy(np.load(filename_seg_right)).unsqueeze(dim=0).type(torch.FloatTensor)
        seg_full = copy.deepcopy(seg)
        seg_full[seg_full==2]=1
        
        cat_full = torch.cat((img, prior_full), axis=0)
        cat_left = torch.cat((img, prior_left), axis=0)
        cat_right = torch.cat((img, prior_right), axis=0)
        cat_hip = torch.cat((img, prior_comp), axis=0)
        # cat_hip[cat_hip==2]=1
        # [ cat_full, cat_left, cat_right, img, seg, seg_left, seg_right] = self.datalist[index]
        
        return {'cat_hip' : cat_hip, 'cat_full' : cat_full, 'cat_left' : cat_left, 'cat_right' : cat_right, \
            'img' : img, 'seg' : seg, 'seg_left' : seg_left, 'seg_right' : seg_right, 'seg_full' : seg_full}

    def __len__(self):
        return len(self.prior_fn)

def get_train_val_loader(dataset_obj, validation_split, batch_size = 1, train_ratio = 1.0, n_cpus = 4,\
    numpy_seed = 0, torch_seed = 0, shuffle_dataset=True):
    

    dataset_size = len(dataset_obj)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset :
        np.random.seed(numpy_seed)
        torch.manual_seed(torch_seed)
    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices[0:int(train_ratio*len(train_indices))])
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(dataset_obj, batch_size=batch_size, 
                                            sampler=train_sampler, num_workers=n_cpus)
    validation_loader = DataLoader(dataset_obj, batch_size=batch_size,
                                                sampler=valid_sampler, num_workers=n_cpus)
    return train_loader, validation_loader#, dataset_obj.weights


# root_dir = '../data/SpineCT_processed/train/'
# imgdir = 'imagesTr'
# labeldir = 'labelsTr'
# labeldir_left = 'labels_left'
# labeldir_right = 'labels_right'

# validation_split = 0.2
# shuffle_dataset = True
# random_seed= 42
# # train_ratio = args.train_ratio

# # # dataset = Mean_Prior_MSD(root = root_dir, imgdir = imgdir, labeldir = labeldir, labeldir_left = labeldir_left, \
# # # 	labeldir_right = labeldir_right, prior_list = prior_list, \
# # # 		device = torch.device("cuda:1"), train_ratio = train_ratio)
# dataset = Prior_Spine(root = root_dir, imgdir = imgdir, labeldir = labeldir, labeldir_left = labeldir_left, labeldir_right = labeldir_right, device = torch.device("cuda:0"))

# train_loader, validation_loader = get_train_val_loader(dataset, validation_split, batch_size = 4, train_ratio = 1.0, n_cpus = 8,\
#     numpy_seed = 0, torch_seed = 0, shuffle_dataset=True)

# for i, batch in enumerate(train_loader):
#     for key in batch.keys():
#         print(key, ' : ', batch[key].shape)