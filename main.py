import fire
import os
from model import WaveGANGenerator, WaveGANDiscriminator, PitchClassifier
import torch
from torch import nn
from torch import autograd
import numpy as np
from utils import AudioDirectoryDataset, get_data_loader, natural_keys, ToTensor, Normalize
import torch.distributions as dist
from tqdm import tqdm
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from specgrams_helper import SpecgramsHelper

print("Done imports")


class Main(object):
    def __init__(self, models_folder='./models', model='main', latent_dim=128, data_folder='./data/nsynth-train/audio', csv_file='files_processed.txt',
                 valid_data_folder='./data/nsynth-valid/audio', valid_csv_file='files_processed.txt',
                 cuda=torch.cuda.is_available(), fps=16384,
                 logs_folder='./logs', n_epochs=5000,
                 batch_size=64, n_cpu=1, sample_interval=50, lambda_gp=10):
        super(Main, self).__init__()
        assert os.path.exists(data_folder), f"{data_folder} does not exists."
        self.n_cpu = n_cpu
        self.lambda_gp = lambda_gp
        self.sample_interval = sample_interval
        self.fps = fps
        self.Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        self.batch_size = batch_size * 1# torch.cuda.device_count()
        self.n_epochs = n_epochs
        self.data_folder = data_folder
        self.csv_file = csv_file
        self.valid_data_folder = valid_data_folder
        self.valid_csv_file = valid_csv_file
        self.models_folder = os.path.join(models_folder, model)
        self.logs_folder = os.path.join(logs_folder, model)
        os.makedirs(self.models_folder, exist_ok=True)
        os.makedirs(self.logs_folder, exist_ok=True)
        self.cuda = cuda
        self.model = model
        self.latent_dim = latent_dim
        print("creating models")
        self.generator = WaveGANGenerator()
        self.discriminator = WaveGANDiscriminator()
        self.pitch_classifier = PitchClassifier()

        print("models created")
        if cuda:
            # if torch.cuda.device_count() > 1:
            self.generator = nn.DataParallel(self.generator)
            self.discriminator = nn.DataParallel(self.discriminator)
            self.generator.cuda()
            self.discriminator.cuda()
        self.distribution = dist.normal.Normal(0, 1)
        self.writer = SummaryWriter(log_dir=self.logs_folder)
        print(f"tensorboard --logdir={self.logs_folder}")

    def load_model(self, train=False):
        if len(os.listdir(self.models_folder)) > 0:
            files = [f for f in os.listdir(self.models_folder) if os.path.isfile(os.path.join(self.models_folder, f))]
            files.sort(key=natural_keys)
            print(f'Loading Model {files[-1]}....')
            checkpoint = torch.load(os.path.join(self.models_folder, files[-1]),
                                    map_location='cuda' if self.cuda else 'cpu')
            self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'], strict=False)
            self.generator.load_state_dict(checkpoint['generator_state_dict'], strict=False)
            if train:
                self.discriminator.train()
                self.generator.train()
            else:
                self.discriminator.eval()
                self.generator.eval()
            return checkpoint['epoch']
        return 0

    def compute_gradient_penalty(self, D, real_samples, fake_samples):
        """Calculates the gradient penalty loss for WGAN GP"""
        # Random weight term for interpolation between real and fake samples
        alpha = self.Tensor(np.random.random((real_samples.size(0), 1, 1)))
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        d_interpolates = D(interpolates)
        fake = Variable(self.Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
        # Get gradient w.r.t. interpolates
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    # Adapted from https://github.com/caogang/wgan-gp/blob/master/gan_toy.py
    def calc_gradient_penalty(self, net_dis, real_data, fake_data, batch_size, lmbda, use_cuda=False):
        # Compute interpolation factors
        alpha = torch.rand(batch_size, 1, 1)
        alpha = alpha.expand(real_data.size())
        alpha = alpha.cuda() if use_cuda else alpha

        # Interpolate between real and fake data.
        interpolates = alpha * real_data + (1 - alpha) * fake_data
        if use_cuda:
            interpolates = interpolates.cuda()
        interpolates = autograd.Variable(interpolates, requires_grad=True)

        # Evaluate discriminator
        disc_interpolates = net_dis(interpolates)

        # Obtain gradients of the discriminator with respect to the inputs
        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                grad_outputs=torch.ones(disc_interpolates.size()).cuda() if use_cuda else
                                torch.ones(disc_interpolates.size()),
                                create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradients = gradients.view(gradients.size(0), -1)

        # Compute MSE between 1.0 and the gradient of the norm penalty to make discriminator
        # to be a 1-Lipschitz function.
        gradient_penalty = lmbda * ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty


    def train_pitch(self, lr=1e-4, betas=(0.5, 0.9), model_save=1):
        spec_helper = SpecgramsHelper(self.fps*4,  [1025, 512], 0.75, self.fps, 0)
        optimizer = torch.optim.Adam(self.pitch_classifier.parameters(), lr, betas)
        dataset = AudioDirectoryDataset(self.data_folder, self.csv_file, fps=self.fps)
        dataloader = get_data_loader(dataset, batch_size=self.batch_size, num_workers=self.n_cpu)
        criterion = nn.CrossEntropyLoss()

        if(self.cuda):
            criterion = criterion.cuda()

        for epoch in range(self.n_epochs):
            for i, (audio, target) in enumerate(dataloader):
                if(self.cuda):
                    audio = audio.cuda()
                    target = target.cuda()

                audio = Variable(audio.type(self.Tensor), requires_grad=False)
                audio = spec_helper.waves_to_stfts(audio)
                print(audio.shape)
                target = Variable(target)
                optimizer.zero_grad()
                output = self.pitch_classifier(audio)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                max_vals, max_indices = torch.max(output, 1)
                acc = (max_indices == target).sum().item() / max_indices.size()[0]

                n_iter = (epoch + offset) * len(dataloader) + i
                print(loss.item())
                print(acc)
                #self.writer.add_scalar('Train_Loss/loss', loss.item(), n_iter)
                #self.writer.add_scalar('Train_Acc/acc', acc, n_iter)
            if (epoch + offset + 1) % model_save == 0:
                torch.save({
                    'epoch': epoch + offset + 1,
                    'state_dict': self.pitch_classifier.state_dict()
                })

    def train(self, g_lr=1e-4, d_lr=1e-4, betas=(0.5, 0.9), n_critic=5, model_save=1):
        optimizer_G = torch.optim.Adam(self.generator.parameters(), g_lr,
                                       betas)
        optimizer_D = torch.optim.Adam(self.discriminator.parameters(), d_lr,
                                       betas)
        print('Optimizers Ready....')
        dataset = AudioDirectoryDataset(self.data_folder, self.csv_file, fps=self.fps)
        valid_dataset = AudioDirectoryDataset(self.valid_data_folder, self.valid_csv_file, fps=self.fps)
        print('Dataset Ready....')
        dataloader = get_data_loader(dataset, batch_size=self.batch_size, num_workers=self.n_cpu)
        valid_dataloader = get_data_loader(valid_dataset, batch_size=self.batch_size, num_workers=self.n_cpu)
        print('Dataloader Ready....')
        offset = self.load_model(train=True)
        print('Training is starting....')
        loss_log = tqdm(total=0, position=0, bar_format='{desc}')
        for epoch in range(self.n_epochs):
            for i, (audio, target) in enumerate(dataloader):
                if(i > 3490):
                    break
                for p in self.discriminator.parameters():
                    p.requires_grad = True

                self.discriminator.zero_grad()
                one = torch.tensor(1, dtype=torch.float)
                neg_one = one * -1
                if self.cuda:
                    one = one.cuda()
                    neg_one = neg_one.cuda()
                    audio = audio.cuda()

                #############################
                # (1) Train Discriminator
                #############################
                noise = torch.Tensor(self.batch_size, self.latent_dim).uniform_(-1, 1)
                if self.cuda:
                    noise = noise.cuda()

                noise_Var = Variable(noise, requires_grad=False)
                real_data = Variable(audio.type(self.Tensor), requires_grad=False)

                # a) compute loss contribution from real training data
                D_real = self.discriminator(real_data)
                D_real = D_real.mean()
                D_real.backward(neg_one)

                # b) compute loss contribution from generated data, then backprop.
                fake = Variable(self.generator(noise_Var).data)
                D_fake = self.discriminator(fake)
                D_fake = D_fake.mean()
                D_fake.backward(one)

                # c) compute gradient penalty and backprop
                gradient_penalty = self.calc_gradient_penalty(self.discriminator, real_data.data,
                                                     fake.data, self.batch_size, self.lambda_gp,
                                                     use_cuda=self.cuda)
                gradient_penalty.backward(one)

                # Compute cost * Wassertein loss..
                D_cost_train = D_fake - D_real + gradient_penalty
                D_wass_train = D_real - D_fake

                # Update gradient of discriminator.
                optimizer_D.step()

                # Train the generator every n_critic steps
                if i % n_critic == 0:
                    for p in self.discriminator.parameters():
                        p.requires_grad = False
                    self.generator.zero_grad()
                    noise = torch.Tensor(self.batch_size, self.latent_dim).uniform_(-1, 1)
                    if self.cuda:
                        noise = noise.cuda()
                    noise_Var = Variable(noise, requires_grad=False)
                    fake = self.generator(noise_Var)
                    G = self.discriminator(fake)
                    G = G.mean()
                    # Update gradients.
                    G.backward(neg_one)
                    G_cost = -G
                    optimizer_G.step()

                loss_log.set_description_str(
                    '[Epoch %d/%d] [Batch %d/%d] [G loss: %f] [D loss: %f]' % (
                        epoch + offset, self.n_epochs, i, len(dataloader), G_cost.item(), D_cost_train.item())
                )
                n_iter = (epoch + offset) * len(dataloader) + i
                self.writer.add_scalar('Loss/g_loss', G_cost.item(), n_iter)
                self.writer.add_scalar('Loss/d_loss', D_cost_train.item(), n_iter)

            #############################
            # (2) Compute Valid data
            #############################
            D_cost = []
            D_wass = []
            for i, (audio, target) in enumerate(valid_dataloader):
                if(i > 150):
                    break
                self.discriminator.zero_grad()
                if(self.cuda):
                    audio = audio.cuda()

                valid_data_Var = Variable(audio, requires_grad=False)
                D_real_valid = self.discriminator(valid_data_Var)
                D_real_valid = D_real_valid.mean()  # avg loss

                # b) compute loss contribution from generated data, then backprop.
                fake_valid = self.generator(noise_Var)
                D_fake_valid = self.discriminator(fake_valid)
                D_fake_valid = D_fake_valid.mean()

                # c) compute gradient penalty and backprop
                gradient_penalty_valid = self.calc_gradient_penalty(self.discriminator, valid_data_Var.data,
                                                            fake_valid.data, self.batch_size, self.lambda_gp,
                                                            use_cuda=self.cuda)
                # Compute metrics and record in batch history.
                D_cost_valid = (D_fake_valid - D_real_valid + gradient_penalty_valid)
                D_wass_valid = D_real_valid - D_fake_valid

                if self.cuda:
                    D_cost_valid = D_cost_valid.cpu()
                    D_wass_valid = D_wass_valid.cpu()

                D_cost.append(D_cost_valid.data.numpy())
                D_wass.append(D_wass_valid.data.numpy())

            self.writer.add_scalar('Validation/d_cost', sum(D_cost) / float(len(D_cost)), epoch)
            self.writer.add_scalar('Validation/d_wass', sum(D_wass) / float(len(D_wass)), epoch)


            if (epoch + offset + 1) % model_save == 0:
                self.save_model(epoch + offset + 1)

    def save_model(self, epoch):
        os.makedirs(self.models_folder, exist_ok=True)
        torch.save({
            'epoch': epoch,
            'discriminator_state_dict': self.discriminator.state_dict(),
            'generator_state_dict': self.generator.state_dict(),
        }, os.path.join(self.models_folder, f"{epoch}"))

    def generate_samples(self):
        offset = self.load_model(train=False)
        z = self.distribution.sample([self.batch_size, self.latent_dim])
        if self.cuda:
            z = z.cuda()
        self.writer.add_histogram('gen_distribution', z)
        z = self.generator(z)
        for i in range(z.shape[0]):
            np.save("samples/epoch_"+str(offset)+"_sample_"+str(i), z.cpu().detach().numpy()[i].reshape((-1, )))
            self.writer.add_audio(f'Samples {i}', z.cpu().detach().numpy()[i].reshape((-1, )), offset,
                                  sample_rate=self.fps)


if __name__ == '__main__':
    fire.Fire(Main)
