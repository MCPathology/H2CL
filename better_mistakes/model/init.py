import torch.cuda
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import timm
import hyptorch.nn as hypnn
import hyptorch.pmath as pm
import hyptorch.loss as L


def init_model_on_gpu(gpus_per_node, opts):
    arch_dict = models.__dict__
    pretrained = False if not hasattr(opts, "pretrained") else opts.pretrained
    distributed = False if not hasattr(opts, "distributed") else opts.distributed
    print("=> using model '{}', pretrained={}".format(opts.arch, False))
    # model = arch_dict[opts.arch](pretrained=pretrained)
    if opts.arch == "swinT":
        model = timm.create_model('swin_large_patch4_window12_384_in22k', pretrained=False)

        
        
    else:
        model = arch_dict[opts.arch](pretrained=False)
    feature_dim = opts.feature_dim
    
        
    if opts.devise or opts.barzdenzler:
        if opts.pretrained or opts.pretrained_folder:
            for param in model.parameters():
                if opts.train_backbone_after == 0:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        if opts.use_2fc:
            if opts.use_fc_batchnorm:
                model.fc = torch.nn.Sequential(
                    torch.nn.Linear(in_features=feature_dim, out_features=opts.fc_inner_dim, bias=True),
                    torch.nn.ReLU(),
                    torch.nn.BatchNorm1d(opts.fc_inner_dim),
                    torch.nn.Linear(in_features=opts.fc_inner_dim, out_features=opts.embedding_size, bias=True),
                )
            else:
                model.fc = torch.nn.Sequential(
                    torch.nn.Linear(in_features=feature_dim, out_features=opts.fc_inner_dim, bias=True),
                    torch.nn.ReLU(),
                    torch.nn.Linear(in_features=opts.fc_inner_dim, out_features=opts.embedding_size, bias=True),
                )
        else:
            if opts.use_fc_batchnorm:
                model.fc = torch.nn.Sequential(
                    torch.nn.BatchNorm1d(feature_dim), torch.nn.Linear(in_features=feature_dim, out_features=opts.embedding_size, bias=True)
                )
            else:
                model.fc = torch.nn.Sequential(
                    torch.nn.Linear(in_features=feature_dim, out_features=opts.embedding_size, bias=True),
                    hypnn.ToPoincare(
                        c=1.0,
                        ball_dim=opts.embedding_size,
                        riemannian=False,
                        clip_r=None,
                    ))
    else:
        if opts.arch == "swinT":
            c=0.05 #
            model.head =torch.nn.Linear(in_features=feature_dim, out_features=opts.num_classes)
            
            # loading pretrained checkpoint
            local_weights_path = 'hierswin_alpha0.4/HierSwin_best.pth.tar'  
        
            state_dict = torch.load(local_weights_path)
            model.load_state_dict(state_dict['state_dict'])
            for param in model.parameters():
                param.requires_grad = False

            model.head =torch.nn.Linear(in_features=feature_dim, out_features=opts.num_classes)
            
            #TLM  0.09 0.028    0.02 0.03
            model.head = torch.nn.Sequential(
                    torch.nn.Linear(in_features=feature_dim, out_features=feature_dim, bias=True),
                    torch.nn.ReLU6(),
                    nn.Dropout(0.25),
                    hypnn.ToPoincare(
                       c=c,
                       ball_dim=feature_dim,
                       riemannian=False,
                       clip_r=None,
                       ),
                    hypnn.HypLinear(in_features=feature_dim, out_features=feature_dim, c=c),
        
                    )
            
            for layer in [ model.layers[3]]:
                for param in layer.parameters():
                    param.requires_grad = True
        
        elif opts.arch == "resnet50":
            local_weights_path = "hierswin_alpha0.4/resnet50.pth"
        
            state_dict = torch.load(local_weights_path)
            model.load_state_dict(state_dict)
            
            
            model.fc = torch.nn.Sequential(torch.nn.Linear(in_features=feature_dim, out_features=feature_dim),
                                           torch.nn.ReLU(),
                                           nn.Dropout(0.25),
                                           hypnn.ToPoincare(
                                                            c=0.05,
                                                            ball_dim=feature_dim,
                                                            riemannian=False,
                                                            clip_r=None,
                                                            ),
                                           hypnn.HypLinear(in_features=feature_dim, out_features=feature_dim, c=0.05),
                                           )
        elif opts.arch == "mobilenet_v2":
            # 
            
            local_weights_path = "hierswin_alpha0.4/mobilenet_v2.pth"
            state_dict = torch.load(local_weights_path)
            model.load_state_dict(state_dict)
            
            model.classifier[1] = torch.nn.Sequential(torch.nn.Linear(in_features=feature_dim, out_features=feature_dim),
                                           torch.nn.ReLU(),
                                           nn.Dropout(0.25),
                                           hypnn.ToPoincare(
                                                            c=0.05,
                                                            ball_dim=feature_dim,
                                                            riemannian=False,
                                                            clip_r=None,
                                                            ),
                                           hypnn.HypLinear(in_features=feature_dim, out_features=feature_dim, c=0.05),
                                           )
            # print(model)
        elif opts.arch == "regnet_y_16gf":
            # 
            model.fc = torch.nn.Sequential(torch.nn.Linear(in_features=feature_dim, out_features=opts.num_classes))
            
        else:
            local_weights_path = "hierswin_alpha0.4/vit.pth"
        
            state_dict = torch.load(local_weights_path)
            model.load_state_dict(state_dict)
            model.heads = torch.nn.Sequential(torch.nn.Linear(in_features=feature_dim, out_features=feature_dim),
                                           torch.nn.ReLU(),
                                           nn.Dropout(0.25),
                                           hypnn.ToPoincare(
                                                            c=0.05,
                                                            ball_dim=feature_dim,
                                                            riemannian=False,
                                                            clip_r=None,
                                                            ),
                                           hypnn.HypLinear(in_features=feature_dim, out_features=feature_dim, c=0.05),
                                           )
            
    if distributed:

        if opts.gpu is not None:
            torch.cuda.set_device(opts.gpu)
            model.cuda(opts.gpu)

            opts.batch_size = int(opts.batch_size / gpus_per_node)
            opts.workers = int(opts.workers / gpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[opts.gpu])
        else:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif opts.gpu is not None:
        torch.cuda.set_device(opts.gpu)
        model = model.cuda(opts.gpu)
    
    else:
        model = torch.nn.DataParallel(model).cuda()

    return model


    
class Hier(nn.Module):
    def __init__(self, backbone, attention):
        super().__init__()
        self.backbone = backbone
        self.classification = attention
        self.i=0
    def forward(self, x, order, family, mask):
        order_embedding_H, family_embedding_H, order_embedding_E, family_embedding_E = self.backbone(x)
        self.i=self.i+1

        output, vis, text_loss, order_hyp, family_hyp, order_euc, family_euc = self.classification(order_embedding_H, family_embedding_H, 
                                                order_embedding_E, family_embedding_E, 
                                                order, family, mask)
        # output = self.backbone(x)
        return output, text_loss, order_hyp, family_hyp, order_euc, family_euc
        
class GlobalClassification(nn.Module):
    def __init__(self, dim, text_dim, num_classes, c=0.05):
        super().__init__()
        self.c = c
        self.curv = torch.tensor(self.c) 
        self.attention = nn.MultiheadAttention(dim, num_heads=4, batch_first=True)
        self.logit_scale = nn.Parameter(torch.ones([]) * 2.6592)
        self.family2euc = hypnn.FromPoincare(c=c,
                                            ball_dim=dim)
        self.order2euc = hypnn.FromPoincare(c=c,
                                            ball_dim=dim)
        self.family2hyp = hypnn.ToPoincare(c=c,
                                           ball_dim=dim,
                                           riemannian=False,
                                           clip_r=None)
        self.order2hyp = hypnn.ToPoincare(c=c,
                                          ball_dim=dim,
                                          riemannian=False,
                                          clip_r=None)
                                             
        self.texto2hyp = nn.Linear(text_dim, dim)
        self.textf2hyp = nn.Linear(text_dim, dim)
        
        self.norm = nn.LayerNorm(dim)
        self.classifier_hyp = hypnn.ToPoincare(c=c,
                                          ball_dim=dim,
                                          riemannian=False,
                                          clip_r=None)
        self.classifier = hypnn.HyperbolicMLR(dim, num_classes,c=c)
    
    def forward(self, order_hyp, family_hyp, order_euc, family_euc, order, family, mask):
        batch, dim = order_hyp.shape
        family_hyp = self.family2euc(family_hyp)
        order_hyp = self.order2euc(order_hyp)

        global_features = torch.stack((order_hyp, family_hyp, order_euc, family_euc), dim=1)
        global_attn, _ = self.attention(global_features, global_features, global_features)
        global_features = self.norm(global_features+global_attn)
        
        order_hyp, family_hyp, order_euc, family_euc = [f.squeeze(1) for f in torch.chunk(global_features, 4, dim=1)]
        
        family_hyp = L.exp_map0(family_hyp, self.curv)
        order_hyp = L.exp_map0(order_hyp, self.curv)
        
        # hyperbolic loss
        order = order.to('cuda')
        family = family.to('cuda')
        # euc_loss = self.euc_loss(order_euc, family_euc, order, family, mask)
        hyp_loss = self.hyp_loss(order_hyp, family_hyp, order, family, mask)
        order_hyp_out = L.log_map0(order_hyp, self.curv)
        output = self.classifier_hyp(order_hyp_out)
        output = self.classifier(output)        
        return output, hyp_loss, order_hyp, family_hyp, order_euc, family_euc
        
    def _compute_clip_loss(self, image_feats, text_feats):

        image_feats = F.normalize(image_feats, p=2, dim=-1)
        text_feats = F.normalize(text_feats, p=2, dim=-1)

        logit_scale = self.logit_scale.exp()
        
        # N = batch_size, D = feature_dim
        # logits_per_image: (N, N)
        logits_per_image = logit_scale * image_feats @ text_feats.t()
        logits_per_text = logits_per_image.t()

        batch_size = image_feats.shape[0]
        labels = torch.arange(batch_size, device=image_feats.device)

        loss_img = F.cross_entropy(logits_per_image, labels)
        loss_txt = F.cross_entropy(logits_per_text, labels)
        
        return (loss_img + loss_txt) / 2

    def euc_loss(self, order_euc, family_euc, order, family, mask):

        order_text_feats = self.texto2euc(order)
        order_loss = self._compute_clip_loss(order_euc, order_text_feats)
        
        family_loss = torch.tensor(0.0, device=order_euc.device)
        valid_mask = mask.bool()

        if valid_mask.any():
            valid_family_img_feats = family_euc[valid_mask]
            valid_family_text_feats = self.textf2euc(family[valid_mask])
            
            if valid_family_img_feats.shape[0] > 1:
                family_loss = self._compute_clip_loss(valid_family_img_feats, valid_family_text_feats)

        return order_loss + family_loss    
        
        
    def hyp_loss(self, order_hyp, family_hyp, order, family, mask):
        order_feats = L.exp_map0(self.texto2hyp(order), self.curv)
        order_angle = L.oxy_angle(order_feats, family_hyp, self.curv)
        order_aperture = L.half_aperture(order_feats, self.curv)
        order_loss = torch.clamp(order_angle - order_aperture, min=0).mean()
        
        family_loss = torch.tensor(0.0, device=order.device) #
        valid_mask = mask.bool()

        if valid_mask.any():  
            valid_family_feats = L.exp_map0(self.textf2hyp(family[valid_mask]), self.curv)
            valid_family_hyp = order_hyp[valid_mask]
    
            family_angle = L.oxy_angle(valid_family_feats, valid_family_hyp, self.curv)
            family_aperture = L.half_aperture(valid_family_feats, self.curv)
    
            family_loss = torch.clamp(family_angle - family_aperture, min=0).mean()
        return order_loss + family_loss
        
        
class CustomModel(nn.Module):
    def __init__(self, backbone, hyperbolic):
        super().__init__()
        self.backbone = backbone
        self.hyperbolic = hyperbolic
        

    def forward(self, x):
        x = self.backbone(x)
        x = self.hyperbolic(x)  #
        
        
        return x
        
class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.5, scale_by_temperature=True):
        super().__init__()
        self.temperature = temperature
        self.scale_by_temperature = scale_by_temperature

    def forward(self, features, labels=None, mask=None):
        
        device = torch.device('cuda')
                  
        features = F.normalize(features, p=2, dim=1)
        batch_size = features.shape[0]
        if labels is not None and mask is not None:  
            raise ValueError('Cannot define both `labels` and `mask`') 
        elif labels is None and mask is None: 
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None: 
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to('cuda')
        else:
            mask = mask.float().to(device)

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature)  #
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        exp_logits = torch.exp(logits)
     
        logits_mask = torch.ones_like(mask) - torch.eye(batch_size).to('cuda')     
        positives_mask = mask * logits_mask
        negatives_mask = 1. - mask

        num_positives_per_row  = torch.sum(positives_mask , axis=1) # 
        valid_samples = num_positives_per_row > 0

        if not valid_samples.any():
            return torch.tensor(0.0, device=device)
        denominator = torch.sum(
        exp_logits * negatives_mask, axis=1, keepdims=True) + torch.sum(
            exp_logits * positives_mask, axis=1, keepdims=True)  
        # print(denominator)
        log_probs = logits - torch.log(denominator)
        if torch.any(torch.isnan(log_probs)):
            # print(positives_mask)
            print(features)
            
            raise ValueError("Log_prob has nan!")
        

        log_probs = torch.sum(
            log_probs*positives_mask , axis=1)[num_positives_per_row > 0] / num_positives_per_row[num_positives_per_row > 0]
        
        # loss
        loss = -log_probs
        if self.scale_by_temperature:
            loss *= self.temperature
        loss = loss.mean()
        if torch.isnan(loss):
            loss = torch.tensor(0.0, device=loss.device)
        
        return loss
        
class HypSupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.5, scale_by_temperature=True):
        super().__init__()
        self.temperature = temperature
        self.scale_by_temperature = scale_by_temperature

    def forward(self, features, labels=None, mask=None):
        # print(features)
        device = features.device
        batch_size = features.shape[0]
        c = 0.01

        if labels is not None and mask is not None:  
            raise ValueError('Cannot define both `labels` and `mask`') 
        elif labels is None and mask is None: 
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None: 
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

    # Compute pairwise hyperbolic distances
        features_expanded_1 = features.unsqueeze(0).expand(batch_size, batch_size, -1)
        features_expanded_2 = features.unsqueeze(1).expand(batch_size, batch_size, -1)

        mobius_diff = (c**0.5) * pm._mobius_add(-features_expanded_1, features_expanded_2, c=c)
        norm_diff = mobius_diff.norm(dim=-1, p=2)
        distance_matrix = pm.artanh(norm_diff) * 2 / (c**0.5)

    # Convert distances to similarities by negating them
        anchor_dot_contrast = -distance_matrix / self.temperature

    # For numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        exp_logits = torch.exp(logits)
        # print(exp_logits)

        logits_mask = torch.ones_like(mask) - torch.eye(batch_size).to(device)     
        positives_mask = mask * logits_mask
        negatives_mask = 1. - mask
        # print(positives_mask)
        num_positives_per_row = torch.sum(positives_mask, axis=1)

    # Avoid division by zero by only considering rows with positive samples
        valid_rows_mask = num_positives_per_row > 0
        
        denominator = torch.sum(
            exp_logits * negatives_mask, axis=1, keepdims=True) + torch.sum(
            exp_logits * positives_mask, axis=1, keepdims=True)

        log_probs = logits - torch.log(denominator)
        
    # Only compute log_probs for rows with positive samples
        log_probs_valid = log_probs[valid_rows_mask]
        # print(log_probs_valid)
        if log_probs_valid.size(0) > 0:
            log_probs_valid_sum = torch.sum(
                log_probs_valid * positives_mask[valid_rows_mask], axis=1) / num_positives_per_row[valid_rows_mask]
        
        # Loss calculation
            loss = -log_probs_valid_sum
        
            if self.scale_by_temperature:
                loss *= self.temperature
        
        # Handle NaNs
            loss[torch.isnan(loss)] = 0.0
        
            loss = loss.mean()
        
            return loss
    
        else:
        # If no valid rows are found (all negative examples), return zero loss or handle accordingly
            return torch.tensor(0.0, device=device)

        
class HGContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.5, scale_by_temperature=True):
        super().__init__()
        self.temperature = temperature
        self.scale_by_temperature = scale_by_temperature

    def forward(self, features, primary_labels, secondary_labels):
        device = features.device
        features = F.normalize(features, p=2, dim=1)
        batch_size = features.shape[0]

        if primary_labels.shape[0] != batch_size or secondary_labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')

        primary_labels = primary_labels.contiguous().view(-1, 1)
        secondary_labels = secondary_labels.contiguous().view(-1, 1)

        primary_mask = torch.eq(primary_labels, primary_labels.T).float().to(device)
        secondary_mask = torch.eq(secondary_labels, secondary_labels.T).float().to(device)

        combined_mask = torch.clamp(primary_mask + secondary_mask, max=1.0)

        anchor_dot_contrast = torch.matmul(features, features.T) / self.temperature
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()  
        
        logits = torch.clamp(logits, min=-50, max=50)
        exp_logits = torch.exp(logits)

        logits_mask = torch.ones_like(combined_mask) - torch.eye(batch_size, device=device)
        positives_mask = combined_mask * logits_mask
        negatives_mask = 1.0 - combined_mask

        num_positives_per_row = torch.sum(positives_mask, dim=1)
        valid_samples = num_positives_per_row > 0

        if not valid_samples.any():
            return torch.tensor(0.0, device=device)

        pos_contrib = torch.sum(exp_logits * positives_mask, dim=1, keepdim=True)
        neg_contrib = torch.sum(exp_logits * negatives_mask, dim=1, keepdim=True)
        denominator = pos_contrib + neg_contrib + 1e-8

        log_probs = logits - torch.log(denominator)
        log_probs = torch.sum(log_probs * positives_mask, dim=1)[valid_samples] / num_positives_per_row[valid_samples]

        loss = -log_probs
        if self.scale_by_temperature:
            loss *= self.temperature
        loss = loss.mean()

        return torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)
        
class HGHypContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.5, scale_by_temperature=True, c=0.05):
        super().__init__()
        self.temperature = temperature
        self.scale_by_temperature = scale_by_temperature
        self.c = c
    def forward(self, features, primary_labels, secondary_labels):
        device = features.device
        features = F.normalize(features, p=2, dim=1)
        batch_size = features.shape[0]

        if primary_labels.shape[0] != batch_size or secondary_labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')

        primary_labels = primary_labels.contiguous().view(-1, 1)
        secondary_labels = secondary_labels.contiguous().view(-1, 1)

        primary_mask = torch.eq(primary_labels, primary_labels.T).float().to(device)
        secondary_mask = torch.eq(secondary_labels, secondary_labels.T).float().to(device)

        combined_mask = torch.clamp(primary_mask + secondary_mask, max=1.0)
        
        features_expanded_1 = features.unsqueeze(0).expand(batch_size, batch_size, -1)
        features_expanded_2 = features.unsqueeze(1).expand(batch_size, batch_size, -1)

        mobius_diff = (c**0.5) * pm._mobius_add(-features_expanded_1, features_expanded_2, c=self.c)
        norm_diff = mobius_diff.norm(dim=-1, p=2)
        distance_matrix = pm.artanh(norm_diff) * 2 / (c**0.5)

    # Convert distances to similarities by negating them
        anchor_dot_contrast = -distance_matrix / self.temperature

    # For numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        exp_logits = torch.exp(logits)

        logits_mask = torch.ones_like(combined_mask) - torch.eye(batch_size, device=device)
        positives_mask = combined_mask * logits_mask
        negatives_mask = 1.0 - combined_mask

        num_positives_per_row = torch.sum(positives_mask, dim=1)
        valid_samples = num_positives_per_row > 0

        if not valid_samples.any():
            return torch.tensor(0.0, device=device)

        pos_contrib = torch.sum(exp_logits * positives_mask, dim=1, keepdim=True)
        neg_contrib = torch.sum(exp_logits * negatives_mask, dim=1, keepdim=True)
        denominator = pos_contrib + neg_contrib + 1e-8

        log_probs = logits - torch.log(denominator)
        log_probs = torch.sum(log_probs * positives_mask, dim=1)[valid_samples] / num_positives_per_row[valid_samples]

        loss = -log_probs
        if self.scale_by_temperature:
            loss *= self.temperature
        loss = loss.mean()

        return torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)
