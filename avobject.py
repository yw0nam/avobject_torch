import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
import time
import sys
from model import SyncNetModel
import os
from tqdm import tqdm
class avobject_model(nn.Module):

    def __init__(self, learning_rate=0.0001, nOut=1024, n_neg=4):
        super(avobject_model, self).__init__();
        
        self.n_neg = n_neg
        self.__S__ = SyncNetModel(nOut=nOut).cuda();
        self.__optimizer__ = torch.optim.SGD(self.parameters(), lr = learning_rate);
        self.cos = nn.CosineSimilarity(dim=1).cuda()
        self.logits_scale = nn.Linear(1, 1, bias=False).cuda()
        torch.nn.init.ones_(self.logits_scale.weight)
        
    def train_network(self,loader=None,evalmode=None):

        if evalmode:
            self.eval();
        else:
            self.train();
            
        loss    = 0
        counter = 0
        index = 0
        stepsize = loader.batch_size
        criterion = ContrastiveLoss()
        
        for data in loader:
            
            self.__optimizer__.zero_grad();
            video, audio = data['video'], data['audio']
            
            # padding for cover all image
            pad_len = 4
            video = torch.nn.ConstantPad3d([pad_len, pad_len, pad_len, pad_len],
                                            0)(video)
            if evalmode:
                with torch.no_grad():
                    out_vid = self.__S__.forward_vid(video.cuda(), return_feat=False)
                    out_aud = self.__S__.forward_aud(audio.cuda())
            else:
                out_vid = self.__S__.forward_vid(video.cuda(), return_feat=False)
                out_aud = self.__S__.forward_aud(audio.cuda())
                
            norm_vid = F.normalize(out_vid, p=2, dim=1)
            norm_aud = F.normalize(out_aud, p=2, dim=1)
            norm_aud = norm_aud.permute(0, 3, 1, 2)
            vid_sync, aud_sync, label = self.create_online_sync_negatives(norm_vid, norm_aud)
            score, _ = self.calc_av_scores(vid_sync, aud_sync)
            
            nloss = criterion(score, label)
            if not evalmode:
                nloss.backward()
                self.__optimizer__.step();
            
            counter+=1
            loss += nloss.detach().cpu();
            sys.stdout.write("\r progress: %d / %d , Loss %.5f"%(counter*stepsize, len(loader)*stepsize, loss/counter))
            sys.stdout.flush();
        sys.stdout.write("\n");
        return loss/counter
    def predict(self, data):
        video, audio = data['video'], data['audio']
        pad_len = 4
        video = torch.nn.ConstantPad3d([pad_len, pad_len, pad_len, pad_len],
                                        0)(video)
        with torch.no_grad():
            out_vid = self.__S__.forward_vid(video.cuda(), return_feat=False)
            out_aud = self.__S__.forward_aud(audio.cuda())
            
            norm_vid = F.normalize(out_vid, p=2, dim=1)
            norm_aud = F.normalize(out_aud, p=2, dim=1)
            norm_aud = norm_aud.permute(0, 3, 1, 2)
            vid_sync, aud_sync, label = self.create_online_sync_negatives(norm_vid, norm_aud)
            score, att_map = self.calc_av_scores(vid_sync, aud_sync)
        return score, att_map, vid_sync, aud_sync, label
    def create_online_sync_negatives(self, vid_emb, aud_emb, n_neg=4):

        assert n_neg % 2 == 0
        ww = n_neg // 2

        fr_trunc, to_trunc = ww, aud_emb.shape[-1] - ww
        vid_emb_pos = vid_emb[:, :, fr_trunc:to_trunc]
        slice_size = to_trunc - fr_trunc

        aud_emb_posneg = aud_emb.squeeze(1).unfold(-1, slice_size, 1)
        aud_emb_posneg = aud_emb_posneg.permute([0, 2, 1, 3])

        # this is the index of the positive samples within the posneg bundle
        pos_idx = n_neg // 2
        aud_emb_pos = aud_emb[:, 0, :, fr_trunc:to_trunc]

        # make sure that we got the indices correctly
        assert torch.all(aud_emb_posneg[:, pos_idx] == aud_emb_pos)

        return vid_emb_pos, aud_emb_posneg, pos_idx
        
    def calc_av_scores(self, vid, aud):
        """
        :return: aggregated scores over T, h, w
        """
        vid = vid[:, :, None]
        aud = aud.transpose(1, 2)[..., None, None]

        cos_similarity = self.cos(vid, aud)
        att_map = self.logits_scale(cos_similarity[..., None]).squeeze(-1)
        scores = torch.nn.MaxPool3d(kernel_size=(1, att_map.shape[-2],
                                                att_map.shape[-1]))(att_map)
        scores = scores.squeeze()
        return scores, att_map

    def updateLearningRate(self, alpha):

        learning_rate = []
        for param_group in self.__optimizer__.param_groups:
            param_group['lr'] = param_group['lr']*alpha
            learning_rate.append(param_group['lr'])

        return learning_rate;
   
    def saveParameters(self, path):

        torch.save(self.state_dict(), path);


    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Load parameters
    ## ===== ===== ===== ===== ===== ===== ===== =====

    def loadParameters(self, path):

        self_state = self.state_dict();
        loaded_state = torch.load(path);
        for name, param in loaded_state.items():
            origname = name;
            if name not in self_state:
                name = name.replace("module.", "");

                if name not in self_state:
                    print("%s is not in the model."%origname);
                    continue;

            if self_state[name].size() != loaded_state[origname].size():
                print("Wrong parameter length: %s, model: %s, loaded: %s"%(origname, self_state[name].size(), loaded_state[origname].size()));
                continue;

            self_state[name].copy_(param);
            
class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    """

    def __init__(self):
        super(ContrastiveLoss, self).__init__()
        
    def forward(self, scores, label):
        
        loss_contrastive = -torch.log(torch.exp(scores[:, label].sum()) / torch.exp(scores.sum(2).sum(0)).sum())

        return loss_contrastive
    