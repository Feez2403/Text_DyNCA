import clip
import torch
import torchvision
# import kornia.augmentation as K
from torchvision import transforms


class CLIPLoss(torch.nn.Module):
    def __init__(self, args):
        super(CLIPLoss, self).__init__()
        self.args = args

        self.model, preprocess = clip.load(args.clip_model, device=args.DEVICE)
        self.model = self.model.eval()
        self.normalize = torchvision.transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                                          std=[0.26862954, 0.26130258, 0.27577711])

        with torch.no_grad():
            self.text_features = self.model.encode_text(clip.tokenize([args.prompt]).to(args.DEVICE))

        self.augemntations = []
        if "jitter" in args.clip_augemntations:
            self.augemntations.append(transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1))
        
        if "perspective" in args.clip_augemntations:
            self.augemntations.append(transforms.RandomPerspective(fill=0, p=1.0, distortion_scale=0.5))
            self.augemntations.append(transforms.RandomResizedCrop(336, scale=(0.8, 0.8), ratio=(1.0, 1.0)))
        else : 
            self.augemntations.append(transforms.Resize(336))
 
        if "affine" in args.clip_augemntations:
            self.augemntations.append(transforms.RandomAffine(degrees=15, translate=(0.08,0.12), shear=5, fill=0.0))

        self.augment_trans = transforms.Compose(self.augemntations)
            
        self._create_losses()
        

    def _create_losses(self):
        self.loss_mapper = {}
        self.loss_weights = {}

    def update_losses_to_apply(self, epoch):
        pass

    def forward(self, input_dict, return_summary=True):
        for generated_images in input_dict["generated_image_list"]:
            
            augmented_images = self.augment_trans(generated_images)
            normalized_images = self.normalize(augmented_images)
            image_features = self.model.encode_image(normalized_images)
            loss = 1.0 - torch.nn.functional.cosine_similarity(image_features, self.text_features, dim=1).mean()
        loss /= len(input_dict["generated_image_list"])

        return loss, None, None


# class MakeCutouts(torch.nn.Module):
#     def __init__(self, cut_size, args):
#         super(MakeCutouts, self).__init__()
#         self.args = args
#         self.cut_size = cut_size
#         self.num_augs = args.clip_num_augs

#         # Pick your own augments & their order
#         augment_list = []
#         for item in args.clip_augments[0]:
#             if item == 'Ji':
#                 augment_list.append(K.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, p=0.7))
#             elif item == 'Sh':
#                 augment_list.append(K.RandomSharpness(sharpness=0.3, p=0.5))
#             elif item == 'Gn':
#                 augment_list.append(K.RandomGaussianNoise(mean=0.0, std=1., p=0.5))
#             elif item == 'Pe':
#                 augment_list.append(K.RandomPerspective(distortion_scale=0.7, p=0.7))
#             elif item == 'Ro':
#                 augment_list.append(K.RandomRotation(degrees=15, p=0.7))
#             elif item == 'Af':
#                 augment_list.append(K.RandomAffine(degrees=15, translate=0.1, shear=5, p=0.7, padding_mode='zeros'))  # border, reflection, zeros
#             elif item == 'Et':
#                 augment_list.append(K.RandomElasticTransform(p=0.7))
#             elif item == 'Ts':
#                 augment_list.append(K.RandomThinPlateSpline(scale=0.8, same_on_batch=True, p=0.7))
#             elif item == 'Cr':
#                 augment_list.append(
#                     K.RandomCrop(size=(self.cut_size, self.cut_size), pad_if_needed=True, padding_mode='reflect',
#                                  p=0.5))
#             elif item == 'Er':
#                 augment_list.append(K.RandomErasing(scale=(.1, .4), ratio=(.3, 1 / .3), same_on_batch=True, p=0.7))
#             elif item == 'Re':
#                 augment_list.append(
#                     K.RandomResizedCrop(size=(self.cut_size, self.cut_size), scale=(0.1, 1), ratio=(0.75, 1.333),
#                                         cropping_mode='resample', p=0.5))

#         self.augs = torch.nn.Sequential(*augment_list)
#         self.noise_fac = 0.0

#         # Pooling
#         self.av_pool = torch.nn.AdaptiveAvgPool2d((self.cut_size, self.cut_size))
#         self.max_pool = torch.nn.AdaptiveMaxPool2d((self.cut_size, self.cut_size))

#     def forward(self, input):
#         cutouts = []

#         for _ in range(self.num_augs):
#             # Use Pooling
#             cutout = (self.av_pool(input) + self.max_pool(input)) / 2
#             cutouts.append(cutout)

#         batch = self.augs(torch.cat(cutouts, dim=0))

#         if self.noise_fac:
#             facs = batch.new_empty([len(batch), 1, 1, 1]).uniform_(0, self.noise_fac)
#             batch = batch + facs * torch.randn_like(batch)
#         return batch
