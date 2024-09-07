from model import VAE
import torch
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
import numpy as np
from tensorboardX import SummaryWriter

from GPyOpt.methods.bayesian_optimization import BayesianOptimization



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)

def kl_loss(mu, log_var):
    # TODO: dividir entre el numero de batches? 
    return -0.5 * torch.mean(1 + log_var - mu.pow(2) - torch.exp(log_var))

def r_loss(y_train, y_pred):
    r_loss = torch.mean((y_train - y_pred) ** 2)
    return r_loss


def to_numpy(pair_loader: DataLoader, vae):
    left_latent_list = []
    right_latent_list = []
    value_list = []
    for left_images, right_images, value_batch in pair_loader:
        left_images = left_images.to(device)
        right_images = right_images.to(device)
        left_images_latent = vae.forward_encoder(left_images)[0].detach().numpy()
        right_images_latent = vae.forward_encoder(right_images)[0].detach().numpy()
        values = value_batch.detach().numpy()
        left_latent_list.append(left_images_latent)
        right_latent_list.append(right_images_latent)
        value_list.append(values)
    left_latent = np.concatenate(left_latent_list, axis=0)
    right_latent = np.concatenate(right_latent_list, axis=0)
    values = np.concatenate(value_list, axis=0)
    return left_latent, right_latent, values

def evaluate(dist_array: np.array, true: np.array, threshold: float) -> float:
    decision = (dist_array < threshold).astype(int)
    return np.mean(decision == true)

def train_eval(train_loader: DataLoader,
          val_loader: DataLoader,
          train_pair_loader: DataLoader,
          val_pair_loader: DataLoader,
          hyperparameters: dict,
          routine_tag: str) -> dict:
    size = hyperparameters['size']
    input_shape = (size, size)
    z_dim = hyperparameters['z_dim']
    conv_blocks = hyperparameters['conv_blocks']
    r_loss_factor = hyperparameters['r_loss_factor']
    model_name = routine_tag + '-' + str(input_shape) + '-' + str(z_dim) + '-' + str(conv_blocks) + '-' + str(r_loss_factor)
    writer = SummaryWriter(comment= '-' + model_name)

    EPOCHS = 20
    LR = .0001
    SAMPLE_SIZE = 32

    train_sample, _ = next(iter(train_loader))
    val_sample, _ = next(iter(val_loader))

    val = 'val' if 'HPO' in routine_tag else 'test'

    img_grid = utils.make_grid(train_sample).numpy()
    writer.add_image('real-train-sample', img_grid)
    img_grid = utils.make_grid(val_sample).numpy()
    writer.add_image('real-' + val + '-sample', img_grid)

    train_sample = train_sample.to(device)
    val_sample = val_sample.to(device)

    latent_space_test_points = np.random.normal(scale=1.0, size=(SAMPLE_SIZE, z_dim))
    latent_space_test_points_v = torch.Tensor(latent_space_test_points).to(device)

    vae = VAE(input_shape, z_dim, conv_blocks).to(device)
    optimizer = optim.Adam(vae.parameters(), LR)

    training_losses = []
    val_losses = []
    vae.train()

    for e in range(EPOCHS):
        epoch_loss = []
        for images, _ in train_loader:
            images_v = images.to(device)

            optimizer.zero_grad()

            mu_v, log_var_v, images_out_v = vae(images_v)
            r_loss_v = r_loss(images_out_v, images_v)
            kl_loss_v = kl_loss(mu_v, log_var_v)
            loss = kl_loss_v + r_loss_v * r_loss_factor
            loss.backward()
            optimizer.step()

            epoch_loss.append(loss.item())

        vae.eval()

        mean_epoch_loss = np.mean(epoch_loss)
        writer.add_scalar('train-loss', mean_epoch_loss, e + 1)
        training_losses.append(mean_epoch_loss)
        if min(training_losses) == training_losses[-1]:
            vae.save(f'trained/{model_name}.dat')

        epoch_loss = []
        for images, _ in val_loader:
            images_v = images.to(device)
            mu_v, log_var_v, images_out_v = vae(images_v)
            r_loss_v = r_loss(images_out_v, images_v)
            kl_loss_v = kl_loss(mu_v, log_var_v)
            loss = kl_loss_v + r_loss_v * r_loss_factor

            epoch_loss.append(loss.item())

        mean_epoch_loss = np.mean(epoch_loss)
        val_losses.append(mean_epoch_loss)
        writer.add_scalar(val + '-loss', mean_epoch_loss, e + 1)

        generated_imgs_v = vae.forward_decoder(latent_space_test_points_v).detach()
        imgs_grid = utils.make_grid(generated_imgs_v)
        writer.add_image('latent-sample-decoded', imgs_grid.cpu().numpy(), e + 1)

        reconstructed_real_sample = vae.forward(train_sample)[2].detach()
        imgs_grid = utils.make_grid(reconstructed_real_sample)
        writer.add_image('real-train-sample-reconstructed', imgs_grid.cpu().numpy(), e + 1)

        reconstructed_real_sample = vae.forward(train_sample)[2].detach()
        imgs_grid = utils.make_grid(reconstructed_real_sample)
        writer.add_image('real-' + val + '-sample-reconstructed', imgs_grid.cpu().numpy(), e + 1)

        vae.train()

    vae.eval()
    
    vae = vae.load_state_dict(torch.load(f'trained/{model_name}.dat')['state_dict'])

    train_left_latent, train_right_latent, train_values = to_numpy(train_pair_loader, vae)
    val_left_latent, val_right_latent, val_values = to_numpy(val_pair_loader, vae)
    

    dif = train_left_latent - train_right_latent
    sq_dif = dif ** 2
    sq_dif_sum = np.sum(sq_dif, axis=1)
    train_dist_array = np.sqrt(sq_dif_sum)
    train_min = train_dist_array.min()
    train_max = train_dist_array.max()

    dif = val_left_latent - val_right_latent
    sq_dif = dif ** 2
    sq_dif_sum = np.sum(sq_dif, axis=1)
    val_dist_array = np.sqrt(sq_dif_sum)

    def obj_func(threshold: float) -> float:
        decision = (train_dist_array < threshold).astype(int)
        accuracy = np.mean(decision == train_values)
        return -accuracy

    domain = [{'name': 'threshold', 'type': 'continuous', 'domain': (train_min, train_max)}]
    max_iter = 50
    BO = BayesianOptimization(f = obj_func, domain = domain)
    BO.run_optimization(max_iter=max_iter)
    opt_threshold = BO.x_opt[0]

    train_acc = evaluate(train_dist_array, train_values, opt_threshold)
    writer.add_scalar('train-accuracy', train_acc)
    val_acc = evaluate(val_dist_array, val_values, opt_threshold)
    writer.add_scalar(val + '-accuracy', val_acc)

    return train_acc, val_acc, training_losses, val_losses