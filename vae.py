import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.utils import data


from src.zeroshot_networks.vae import EncoderTemplate, DecoderTemplate
from src.dataset_loaders._data_loader import DATA_LOADER as dataloader



class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()

        self.device = config.device
        self.auxiliary_data_source = config.specific_parameters.auxiliary_data_source
        self.all_data_sources = ['resnet_features', self.auxiliary_data_source]
        self.dataset = config.dataset_name
        self.generalized = config.generalized
        self.classifier_batch_size = 32
        self.img_seen_samples = config.samples_per_class[0]          
        self.att_seen_samples = config.samples_per_class[1]
        self.att_unseen_samples = config.samples_per_class[2]
        self.img_unseen_samples = config.samples_per_class[3]
        self.reco_loss_function = config.specific_parameters.loss
        self.nepoch = config.nepoch
        self.lr_cls = config.specific_parameters.lr_cls
        self.cross_reconstruction = config.specific_parameters.warmup.cross_reconstruction
        self.cls_train_epochs = config.specific_parameters.cls_train_steps

        self.dataset = dataloader(self.dataset,
                                  copy.deepcopy(self.auxiliary_data_source),
                                  device=self.device)

        if self.dataset == 'google_speech_commands':
            self.num_classes = 20
            self.num_novel_classes = 10

        self.encoder = {}
        for datatype, dim in zip(self.all_data_sources, feature_dimensions):
            self.encoder[datatype] = EncoderTemplate(
                dim, self.latent_size, self.hidden_size_rule[datatype], self.device)
            print(str(datatype) + ' ' + str(dim))

        self.decoder = {}
        for datatype, dim in zip(self.all_data_sources, feature_dimensions):
            self.decoder[datatype] = DecoderTemplate(
                self.latent_size, dim, self.hidden_size_rule[datatype], self.device)

        # An optimizer       
        parameters_to_optimize = list(self.parameters())
        for datatype in self.all_data_sources:
            parameters_to_optimize += list(self.encoder[datatype].parameters())
            parameters_to_optimize += list(self.decoder[datatype].parameters())

        self.optimizer = optim.Adam(parameters_to_optimize, lr=config.specific_parameters.lr_gen_model, betas=(
            0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=True)

        if self.reco_loss_function == 'l2':
            self.reconstruction_criterion = nn.MSELoss(size_average=False)

        elif self.reco_loss_function == 'l1':
            self.reconstruction_criterion = nn.L1Loss(size_average=False)

    def reparameterize(self, mu, logvar):
        if self.reparameterize_with_noise:
            sigma = torch.exp(logvar)
            eps = torch.FloatTensor(logvar.size()[0],1).normal_(0,1)
            eps  = eps.expand(sigma.size())
            return mu + sigma*eps
        else:
            return mu

    def trainstep(self, img, att):

        mu_img, logvar_img = self.encoder['resnet_features'](img)
        z_from_img = self.reparameterize(mu_img, logvar_img)

        mu_att, logvar_att = self.encoder[self.auxiliary_data_source](att)
        z_from_att = self.reparameterize(mu_att, logvar_att)

        img_from_img = self.decoder['jasper_features'](z_from_img)
        att_from_att = self.decoder[self.auxiliary_data_source](z_from_att)

        reconstruction_loss = self.reconstruction_criterion(img_from_img, img) \
                              + self.reconstruction_criterion(att_from_att, att)

        # Loss

        img_from_att = self.decoder['jasper_features'](z_from_att)
        att_from_img = self.decoder[self.auxiliary_data_source](z_from_img)

        cross_reconstruction_loss = self.reconstruction_criterion(img_from_att, img) \
                                    + self.reconstruction_criterion(att_from_img, att)

        # KL Divergence

        KLD = (0.5 * torch.sum(1 + logvar_att - mu_att.pow(2) - logvar_att.exp())) \
              + (0.5 * torch.sum(1 + logvar_img - mu_img.pow(2) - logvar_img.exp()))

        # Distribution Alignment

        distance = torch.sqrt(torch.sum((mu_img - mu_att) ** 2, dim=1) + \
                              torch.sum((torch.sqrt(logvar_img.exp()) - torch.sqrt(logvar_att.exp())) ** 2, dim=1))

        distance = distance.sum()

        f1 = 1.0*(self.current_epoch - self.warmup.cross_reconstruction.start_epoch)/(1.0*( self.warmup.cross_reconstruction.end_epoch- self.warmup.cross_reconstruction.start_epoch))
        f1 = f1*(1.0*self.warmup.cross_reconstruction.factor)
        cross_reconstruction_factor = torch.FloatTensor([min(max(f1,0),self.warmup.cross_reconstruction.factor)])

        f2 = 1.0 * (self.current_epoch - self.warmup.beta.start_epoch) / ( 1.0 * (self.warmup.beta.end_epoch - self.warmup.beta.start_epoch))
        f2 = f2 * (1.0 * self.warmup.beta.factor)
        beta = torch.FloatTensor([min(max(f2, 0), self.warmup.beta.factor)])

        f3 = 1.0*(self.current_epoch - self.warmup.distance.start_epoch )/(1.0*( self.warmup.distance.end_epoch- self.warmup.distance.start_epoch))
        f3 = f3*(1.0*self.warmup.distance.factor)
        distance_factor = torch.FloatTensor([min(max(f3,0),self.warmup.distance.factor)])


        self.optimizer.zero_grad()

        loss = reconstruction_loss - beta * KLD

        if cross_reconstruction_loss > 0:
            loss += cross_reconstruction_factor*cross_reconstruction_loss
        if distance_factor > 0:
            loss += distance_factor*distance
        loss.backward()

        self.optimizer.step()

        return loss.item()

    def train_vae(self):
        losses = []

        self.dataloader = data.DataLoader(
            self.dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)  # ,num_workers = 4)

        self.dataset.novelclasses = self.dataset.novelclasses.long()
        self.dataset.seenclasses = self.dataset.seenclasses.long()
        # leave both statements
        self.train()
        self.reparameterize_with_noise = True

        print('Train for reconstruction')
        for epoch in range(0, self.nepoch):
            self.current_epoch = epoch

            i = -1
            for iters in range(0, self.dataset.ntrain, self.batch_size):
                i += 1

                label, data_from_modalities = self.dataset.next_batch(
                    self.batch_size)

                label = label.long().to(self.device)
                for j in range(len(data_from_modalities)):
                    data_from_modalities[j] = data_from_modalities[j].to(
                        self.device)
                    data_from_modalities[j].requires_grad = False

                loss = self.trainstep(
                    data_from_modalities[0], data_from_modalities[1])

                if i % 50 == 0:

                    print('epoch ' + str(epoch) + ' | iter ' + str(i) + '\t' +
                          ' | loss ' + str(loss)[:5])

                if i % 50 == 0 and i > 0:
                    losses.append(loss)

                for key, value in self.encoder.items():
                    self.encoder[key].eval()
        for key, value in self.decoder.items():
            self.decoder[key].eval()

        return losses

    def predict(self):

        iter_idx = 0
        embeddings = torch.Tensor()
        for batch in self.dataset.gen_next_batch(self.batch_size, dset_part='test'):
            iter_idx += 1
            label, data_from_modalities = batch

            label = label.long().to(self.device)
            for j in range(len(data_from_modalities)):
                data_from_modalities[j] = data_from_modalities[j].to(
                    self.device)
            
            mu_att, logvar_att = self.encoder[self.auxiliary_data_source](data_from_modalities[1])
            z_from_att = self.reparameterize(mu_att, logvar_att)

            embeddings = torch.cat((embeddings, z_from_att), 0)
        
        return embeddings